import logging
from pathlib import Path
from typing import Sequence, Optional, TypeAlias, Any, Dict, List, Tuple
from mcp.server import Server
from mcp.server.session import ServerSession
from mcp.server.sse import SseServerTransport
from mcp.types import (
    ClientCapabilities,
    TextContent,
    ImageContent,
    EmbeddedResource,
    Tool,
    ListRootsResult,
    RootsCapability,
)
Content: TypeAlias = TextContent | ImageContent | EmbeddedResource

from enum import Enum
import git
from git.exc import GitCommandError
from pydantic import BaseModel
import asyncio
import tempfile
import os
import re
import difflib
import shlex
import json
import subprocess
import yaml

logging.basicConfig(level=logging.DEBUG)

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

def find_git_root(path: str) -> Optional[str]:
    current = os.path.abspath(path)
    while current != os.path.dirname(current):
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        current = os.path.dirname(current)
    return None

def load_aider_config(repo_path: Optional[str] = None, config_file: Optional[str] = None) -> Dict[str, Any]:
    config = {}
    search_paths = []
    repo_path = os.path.abspath(repo_path or os.getcwd())
    
    logger.debug(f"Searching for Aider configuration in and around: {repo_path}")
    
    workdir_config = os.path.join(repo_path, ".aider.conf.yml")
    if os.path.exists(workdir_config):
        logger.debug(f"Found Aider config in working directory: {workdir_config}")
        search_paths.append(workdir_config)
    
    git_root = find_git_root(repo_path)
    if git_root and git_root != repo_path:
        git_config = os.path.join(git_root, ".aider.conf.yml")
        if os.path.exists(git_config) and git_config != workdir_config:
            logger.debug(f"Found Aider config in git root: {git_config}")
            search_paths.append(git_config)
    
    if config_file and os.path.exists(config_file):
        logger.debug(f"Using specified config file: {config_file}")
        if config_file not in search_paths:
            search_paths.append(config_file)
    
    home_config = os.path.expanduser("~/.aider.conf.yml")
    if os.path.exists(home_config) and home_config not in search_paths:
        logger.debug(f"Found Aider config in home directory: {home_config}")
        search_paths.append(home_config)
    
    for path in reversed(search_paths):
        try:
            with open(path, 'r') as f:
                logger.info(f"Loading Aider config from {path}")
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    logger.debug(f"Config from {path}: {yaml_config}")
                    config.update(yaml_config)
        except Exception as e:
            logger.warning(f"Error loading config from {path}: {e}")
    
    logger.debug(f"Final merged Aider configuration: {config}")
    return config

def load_dotenv_file(repo_path: Optional[str] = None, env_file: Optional[str] = None) -> Dict[str, str]:
    env_vars = {}
    search_paths = []
    repo_path = os.path.abspath(repo_path or os.getcwd())
    
    logger.debug(f"Searching for .env files in and around: {repo_path}")
    
    workdir_env = os.path.join(repo_path, ".env")
    if os.path.exists(workdir_env):
        logger.debug(f"Found .env in working directory: {workdir_env}")
        search_paths.append(workdir_env)
    
    git_root = find_git_root(repo_path)
    if git_root and git_root != repo_path:
        git_env = os.path.join(git_root, ".env")
        if os.path.exists(git_env) and git_env != workdir_env:
            logger.debug(f"Found .env in git root: {git_env}")
            search_paths.append(git_env)
    
    if env_file and os.path.exists(env_file):
        logger.debug(f"Using specified .env file: {env_file}")
        if env_file not in search_paths:
            search_paths.append(env_file)
    
    home_env = os.path.expanduser("~/.env")
    if os.path.exists(home_env) and home_env not in search_paths:
        logger.debug(f"Found .env in home directory: {home_env}")
        search_paths.append(home_env)
    
    for path in reversed(search_paths):
        try:
            with open(path, 'r') as f:
                logger.info(f"Loading .env from {path}")
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
                    except ValueError:
                        logger.warning(f"Invalid line in .env file {path}: {line}")
        except Exception as e:
            logger.warning(f"Error loading .env from {path}: {e}")
    
    logger.debug(f"Loaded environment variables: {list(env_vars.keys())}")
    return env_vars

async def run_command(command: List[str], input_data: Optional[str] = None) -> Tuple[str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if input_data else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    if input_data:
        stdout, stderr = await process.communicate(input_data.encode())
    else:
        stdout, stderr = await process.communicate()
    
    return stdout.decode(), stderr.decode()

def prepare_aider_command(
    base_command: List[str], 
    files: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None
) -> List[str]:
    command = base_command.copy()
    
    if options:
        for key, value in options.items():
            arg_key = key.replace('_', '-')
            
            if isinstance(value, bool):
                if value:
                    command.append(f"--{arg_key}")
                else:
                    command.append(f"--no-{arg_key}")
            
            elif isinstance(value, list):
                for item in value:
                    command.append(f"--{arg_key}")
                    command.append(str(item))
            
            elif value is not None:
                command.append(f"--{arg_key}")
                command.append(str(value))
    
    if files:
        command.extend(files)
    
    command = [c for c in command if c]
    
    return command

class GitStatus(BaseModel):
    repo_path: str

class GitDiffAll(BaseModel):
    repo_path: str

class GitDiff(BaseModel):
    repo_path: str
    target: str

class GitCommit(BaseModel):
    repo_path: str
    message: str
    files: Optional[List[str]] = None  # Optional: files to stage before commit

class GitReset(BaseModel):
    repo_path: str

class GitLog(BaseModel):
    repo_path: str
    max_count: int = 10

class GitCreateBranch(BaseModel):
    repo_path: str
    branch_name: str
    base_branch: str | None = None

class GitCheckout(BaseModel):
    repo_path: str
    branch_name: str

class GitShow(BaseModel):
    repo_path: str
    revision: str

class GitApplyDiff(BaseModel):
    repo_path: str
    diff_content: str

class GitReadFile(BaseModel):
    repo_path: str
    file_path: str

class SearchAndReplace(BaseModel):
    repo_path: str
    file_path: str
    search_string: str
    replace_string: str
    ignore_case: bool = False
    start_line: Optional[int] = None
    end_line: Optional[int] = None

class WriteToFile(BaseModel):
    repo_path: str
    file_path: str
    content: str

class ExecuteCommand(BaseModel):
    repo_path: str
    command: str

class AiEdit(BaseModel):
    repo_path: str
    message: str
    files: List[str] # Make files mandatory
    options: Optional[list[str]] = None

class AiderStatus(BaseModel):
    repo_path: str
    check_environment: bool = True

class GitTools(str, Enum):
    STATUS = "git_status"
    DIFF_ALL = "git_diff_all"
    DIFF = "git_diff"
    STAGE_AND_COMMIT = "git_stage_and_commit"
    RESET = "git_reset"
    LOG = "git_log"
    CREATE_BRANCH = "git_create_branch"
    CHECKOUT = "git_checkout"
    SHOW = "git_show"
    APPLY_DIFF = "git_apply_diff"
    READ_FILE = "git_read_file"
    SEARCH_AND_REPLACE = "search_and_replace"
    WRITE_TO_FILE = "write_to_file"
    EXECUTE_COMMAND = "execute_command"
    AI_EDIT = "ai_edit"
    AIDER_STATUS = "aider_status"

def git_status(repo: git.Repo) -> str:
    return repo.git.status()

def git_diff_all(repo: git.Repo) -> str:
    return repo.git.diff("HEAD")

def git_diff(repo: git.Repo, target: str) -> str:
    return repo.git.diff(target)

def git_stage_and_commit(repo: git.Repo, message: str, files: Optional[List[str]] = None) -> str:
    if files:
        repo.index.add(files)
        staged_message = f"Files {', '.join(files)} staged successfully."
    else:
        repo.git.add(A=True)
        staged_message = "All changes staged successfully."

    commit = repo.index.commit(message)
    return f"{staged_message}\nChanges committed successfully with hash {commit.hexsha}"

def git_reset(repo: git.Repo) -> str:
    repo.index.reset()
    return "All staged changes reset"

def git_log(repo: git.Repo, max_count: int = 10) -> list[str]:
    commits = list(repo.iter_commits(max_count=max_count))
    log = []
    for commit in commits:
        log.append(
            f"Commit: {commit.hexsha}\n"
            f"Author: {commit.author}\n"
            f"Date: {commit.authored_datetime}\n"
            f"Message: {str(commit.message)}\n"
        )
    return log

def git_create_branch(repo: git.Repo, branch_name: str, base_branch: str | None = None) -> str:
    if base_branch:
        base = repo.refs[base_branch]
    else:
        base = repo.active_branch

    repo.create_head(branch_name, base)
    return f"Created branch '{branch_name}' from '{base.name}'"

def git_checkout(repo: git.Repo, branch_name: str) -> str:
    repo.git.checkout(branch_name)
    return f"Switched to branch '{branch_name}'"

def git_show(repo: git.Repo, revision: str) -> str:
    commit = repo.commit(revision)
    output = [
        f"Commit: {commit.hexsha}\n"
        f"Author: {commit.author}\n"
        f"Date: {commit.authored_datetime}\n"
        f"Message: {str(commit.message)}\n"
    ]
    if commit.parents:
        parent = commit.parents[0]
        diff = parent.diff(commit, create_patch=True)
    else:
        diff = commit.diff(git.NULL_TREE, create_patch=True)
    for d in diff:
        output.append(f"\n--- {d.a_path}\n+++ {d.b_path}\n")
        if d.diff is not None:
            if isinstance(d.diff, bytes):
                output.append(d.diff.decode('utf-8'))
            else:
                output.append(str(d.diff))
    return "".join(output)

async def git_apply_diff(repo: git.Repo, diff_content: str) -> str:
    tmp_file_path = None
    affected_file_path = None
    original_content = ""

    match = re.search(r"--- a/(.+)", diff_content)
    if match:
        affected_file_path = match.group(1).strip()
    else:
        match = re.search(r"\+\+\+ b/(.+)", diff_content)
        if match:
            affected_file_path = match.group(1).strip()

    if affected_file_path:
        full_affected_path = Path(repo.working_dir) / affected_file_path
        if full_affected_path.exists():
            with open(full_affected_path, 'r') as f:
                original_content = f.read()

    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(diff_content)
            tmp_file_path = tmp.name
        
        repo.git.apply(
            '--check',
            '-3',
            '--whitespace=fix',
            '--allow-overlap',
            tmp_file_path
        )
            
        result_message = "Diff applied successfully"

        if affected_file_path:
            with open(full_affected_path, 'r') as f:
                new_content = f.read()

            result_message += await _generate_diff_output(original_content, new_content, affected_file_path)
            result_message += await _run_tsc_if_applicable(str(repo.working_dir), affected_file_path)

        return result_message
    except GitCommandError as gce:
        return f"GIT_COMMAND_FAILED: Failed to apply diff. Details: {gce.stderr}. AI_HINT: Check if the diff is valid and applies cleanly to the current state of the repository."
    except Exception as e:
        return f"UNEXPECTED_ERROR: An unexpected error occurred while applying diff: {e}. AI_HINT: Check the server logs for more details or review your input."
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def git_read_file(repo: git.Repo, file_path: str) -> str:
    try:
        full_path = Path(repo.working_dir) / file_path
        with open(full_path, 'r') as f:
            content = f.read()
        return f"Content of {file_path}:\n{content}"
    except FileNotFoundError:
        return f"Error: file wasn't found or out of cwd: {file_path}"
    except Exception as e:
        return f"UNEXPECTED_ERROR: Failed to read file '{file_path}': {e}. AI_HINT: Check if the file exists, is accessible, and not corrupted. Review server logs for more details."

async def _generate_diff_output(original_content: str, new_content: str, file_path: str) -> str:
    diff_lines = list(difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=""
    ))
    
    if len(diff_lines) > 1000:
        return f"\nDiff was too large (over 1000 lines)."
    else:
        diff_output = "".join(diff_lines)
        return f"\nDiff:\n{diff_output}" if diff_output else "\nNo changes detected (file content was identical)."

async def _run_tsc_if_applicable(repo_path: str, file_path: str) -> str:
    file_extension = os.path.splitext(file_path)[1]
    if file_extension in ['.ts', '.js', '.mjs']:
        tsc_command = f" tsc --noEmit --allowJs {file_path}"
        tsc_output = await execute_custom_command(repo_path, tsc_command)
        return f"\n\nTSC Output for {file_path}:\n{tsc_output}"
    return ""

async def _search_and_replace_python_logic(
    repo_path: str,
    search_string: str,
    replace_string: str,
    file_path: str,
    ignore_case: bool,
    start_line: Optional[int],
    end_line: Optional[int]
) -> str:
    try:
        full_file_path = Path(repo_path) / file_path
        with open(full_file_path, 'r') as f:
            lines = f.readlines()

        flags = 0
        if ignore_case:
            flags |= re.IGNORECASE

        literal_search_string = re.escape(search_string)
        logging.info(f"Attempting literal search with: {literal_search_string}")

        modified_lines_literal = []
        changes_made_literal = 0

        for i, line in enumerate(lines):
            line_num = i + 1
            if (start_line is None or line_num >= start_line) and \
               (end_line is None or line_num <= end_line):
                new_line, num_subs = re.subn(literal_search_string, replace_string, line, flags=flags)
                
                if new_line != line:
                    changes_made_literal += num_subs
                    modified_lines_literal.append(new_line)
                else:
                    modified_lines_literal.append(line)
            else:
                modified_lines_literal.append(line)

        if changes_made_literal > 0:
            original_content = "".join(lines)
            with open(full_file_path, 'w') as f:
                f.writelines(modified_lines_literal)
            
            result_message = f"Successfully replaced '{search_string}' with '{replace_string}' in {file_path} using literal search. Total changes: {changes_made_literal}."
            result_message += await _generate_diff_output(original_content, "".join(modified_lines_literal), file_path)
            result_message += await _run_tsc_if_applicable(repo_path, file_path)
            return result_message
        else:
            logging.info(f"Literal search failed. Attempting regex search with: {search_string}")
            modified_lines_regex = []
            changes_made_regex = 0
            
            for i, line in enumerate(lines):
                line_num = i + 1
                if (start_line is None or line_num >= start_line) and \
                   (end_line is None or line_num <= end_line):
                    new_line, num_subs = re.subn(search_string, replace_string, line, flags=flags)
                    
                    if new_line != line:
                        changes_made_regex += num_subs
                        modified_lines_regex.append(new_line)
                    else:
                        modified_lines_regex.append(line)
                else:
                    modified_lines_regex.append(line)

            if changes_made_regex > 0:
                original_content = "".join(lines)
                with open(full_file_path, 'w') as f:
                    f.writelines(modified_lines_regex)
                
                result_message = f"Successfully replaced '{search_string}' with '{replace_string}' in {file_path} using regex search. Total changes: {changes_made_regex}."
                result_message += await _generate_diff_output(original_content, "".join(modified_lines_regex), file_path)
                result_message += await _run_tsc_if_applicable(repo_path, file_path)
                return result_message
            else:
                return f"No changes made. '{search_string}' not found in {file_path} within the specified range using either literal or regex search."

    except FileNotFoundError:
        return f"Error: File not found at {full_file_path}"
    except re.error as e:
        return f"Error: Invalid regex pattern '{search_string}': {e}"
    except Exception as e:
        return f"UNEXPECTED_ERROR: An unexpected error occurred during search and replace: {e}. AI_HINT: Check your search/replace patterns and review server logs for more details."

async def search_and_replace_in_file(
    repo_path: str,
    search_string: str,
    replace_string: str,
    file_path: str,
    ignore_case: bool,
    start_line: Optional[int],
    end_line: Optional[int]
) -> str:
    full_file_path = Path(repo_path) / file_path

    sed_command_parts = ["sed", "-i"]

    sed_pattern = search_string.replace('#', r'\#')
    sed_replacement = replace_string.replace('#', r'\#').replace('&', r'\&').replace('\\', r'\\\\')

    sed_flags = "g"
    if ignore_case:
        sed_flags += "i"

    sed_sub_command = f"s#{sed_pattern}#{sed_replacement}#{sed_flags}"

    if start_line is not None and end_line is not None:
        sed_sub_command = f"{start_line},{end_line}{sed_sub_command}"
    elif start_line is not None:
        sed_sub_command = f"{start_line},${sed_sub_command}"
    elif end_line is not None:
        sed_sub_command = f"1,{end_line}{sed_sub_command}"

    sed_full_command = f"{' '.join(sed_command_parts)} '{sed_sub_command}' {shlex.quote(str(full_file_path))}"

    try:
        with open(full_file_path, 'r') as f:
            original_content = f.read()

        sed_result = await execute_custom_command(repo_path, sed_full_command)
        logging.info(f"Sed command result: {sed_result}")

        if "Command failed with exit code" in sed_result or "Error executing command" in sed_result:
            logging.warning(f"Sed command failed: {sed_result}. Falling back to Python logic.")
            return await _search_and_replace_python_logic(repo_path, search_string, replace_string, file_path, ignore_case, start_line, end_line)
        
        with open(full_file_path, 'r') as f:
            modified_content_sed = f.read()

        if original_content != modified_content_sed:
            result_message = f"Successfully replaced '{search_string}' with '{replace_string}' in {file_path} using sed."
            result_message += await _generate_diff_output(original_content, modified_content_sed, file_path)
            result_message += await _run_tsc_if_applicable(repo_path, file_path)
            return result_message
        else:
            logging.info(f"Sed command executed but made no changes. Falling back to Python logic.")
            return await _search_and_replace_python_logic(repo_path, search_string, replace_string, file_path, ignore_case, start_line, end_line)

    except FileNotFoundError:
        return f"Error: File not found at {full_file_path}"
    except Exception as e:
        logging.error(f"An unexpected error occurred during sed attempt: {e}. Falling back to Python logic.")
        return f"UNEXPECTED_ERROR: An unexpected error occurred during sed-based search and replace: {e}. AI_HINT: Check your search/replace patterns, file permissions, and review server logs for more details."

async def write_to_file_content(repo_path: str, file_path: str, content: str) -> str:
    try:
        full_file_path = Path(repo_path) / file_path
        
        original_content = ""
        file_existed = full_file_path.exists()
        if file_existed:
            with open(full_file_path, 'r') as f:
                original_content = f.read()

        full_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        with open(full_file_path, 'rb') as f_read_back:
            written_bytes = f_read_back.read()
        
        logging.debug(f"Content input to write_to_file (repr): {content!r}")
        logging.debug(f"Raw bytes written to file: {written_bytes!r}")
        logging.debug(f"Input content encoded (UTF-8): {content.encode('utf-8')!r}")

        if written_bytes != content.encode('utf-8'):
            logging.error("Mismatch between input content and written bytes! File corruption detected during write.")
            return "Mismatch between input content and written bytes! File corruption detected during write."

        result_message = ""
        if not file_existed:
            result_message = f"Successfully created new file: {file_path}."
        else:
            result_message += await _generate_diff_output(original_content, content, file_path)

        result_message += await _run_tsc_if_applicable(repo_path, file_path)

        return result_message
    except Exception as e:
        return f"UNEXPECTED_ERROR: Failed to write to file '{file_path}': {e}. AI_HINT: Check file permissions, disk space, and review server logs for more details."

async def execute_custom_command(repo_path: str, command: str) -> str:
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        output = ""
        if stdout:
            output += f"STDOUT:\n{stdout.decode().strip()}\n"
        if stderr:
            output += f"STDERR:\n{stderr.decode().strip()}\n"
        if process.returncode != 0:
            output += f"Command failed with exit code {process.returncode}"
        
        return output if output else "Command executed successfully with no output."
    except Exception as e:
        return f"UNEXPECTED_ERROR: Failed to execute command '{command}': {e}. AI_HINT: Check the command syntax, permissions, and review server logs for more details."

async def ai_edit_files(
    repo_path: str,
    message: str,
    session: ServerSession,
    files: List[str], # Make files mandatory
    options: Optional[list[str]],
    aider_path: Optional[str] = None,
    config_file: Optional[str] = None,
    env_file: Optional[str] = None,
) -> str:
    """
    AI pair programming tool for making targeted code changes using Aider.
    This function encapsulates the logic from aider_mcp/server.py's edit_files tool.
    """
    aider_path = aider_path or "aider"

    # DEBUG LOG: Print session type and available methods
    logger.error(f"ai_edit_files: session type={type(session)}, dir={dir(session)}")
    
    logger.info(f"Running aider in directory: {repo_path}")
    logger.debug(f"Message length: {len(message)} characters")
    logger.debug(f"Additional options: {options}")
    
    directory_path = os.path.abspath(repo_path)
    if not os.path.exists(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        return f"Error: Directory does not exist: {directory_path}"
    
    # AI-actionable error: Check if files list is empty
    if not files:
        error_message = (
            "ERROR: No files were provided for ai_edit. "
            "The 'files' argument is now mandatory and must contain a list of file paths "
            "that Aider should operate on. Please specify the files to edit."
        )
        logger.error(error_message)
        return error_message

    aider_config = load_aider_config(directory_path, config_file)
    load_dotenv_file(directory_path, env_file)
    
    aider_options: Dict[str, Any] = {}
    aider_options["yes_always"] = True
    
    additional_opts: Dict[str, Any] = {}
    if options:
        for opt in options:
            if opt.startswith("--"):
                if "=" in opt:
                    key, value_str = opt[2:].split("=", 1)
                    # Convert "true"/"false" strings to actual booleans
                    if value_str.lower() == "true":
                        additional_opts[key.replace("-", "_")] = True
                    elif value_str.lower() == "false":
                        additional_opts[key.replace("-", "_")] = False
                    else:
                        additional_opts[key.replace("-", "_")] = value_str
                else:
                    additional_opts[opt[2:].replace("-", "_")] = True
            elif opt.startswith("--no-"):
                key = opt[5:].replace("-", "_")
                additional_opts[key] = False

    # Remove unsupported options that are not recognized by aider itself.
    unsupported_options = ["base_url", "base-url"]  # Remove both underscore and dash variants
    for opt_key in unsupported_options:
        if opt_key in additional_opts:
            logger.warning(f"Removing unsupported Aider option: --{opt_key.replace('_', '-')}")
            del additional_opts[opt_key]
    
    aider_options.update(additional_opts)

    # --- BEGIN: LOGGING AND FILE DETECTION FOR DEBUGGING ---
    # The previous file detection logic from message is now removed as 'files' is mandatory.
    # We still log the files that are explicitly passed.
    for fname in files:
        fpath = os.path.join(directory_path, fname)
        if os.path.isfile(fpath):
            logger.info(f"[ai_edit_files] Provided file exists: {fname}")
            # Check if file is tracked by git
            try:
                repo = git.Repo(directory_path)
                tracked = fname in repo.git.ls_files().splitlines()
                logger.info(f"[ai_edit_files] Provided file {fname} tracked by git: {tracked}")
            except Exception as e:
                logger.warning(f"[ai_edit_files] Could not check git tracking for {fname}: {e}")
        else:
            logger.error(f"[ai_edit_files] Provided file not found in repo: {fname}. Aider may fail.")

    # Log if stdin is a TTY (should be False in non-interactive mode)
    try:
        import sys
        is_tty = sys.stdin.isatty()
        logger.info(f"[ai_edit_files] sys.stdin.isatty(): {is_tty}")
    except Exception as e:
        logger.warning(f"[ai_edit_files] Could not check sys.stdin.isatty(): {e}")

    # --- END: LOGGING AND FILE DETECTION FOR DEBUGGING ---

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(message)
        instructions_file = f.name
        logger.debug(f"Instructions written to temporary file: {instructions_file}")
    
    try:
        original_dir = os.getcwd()
        
        os.chdir(directory_path)
        logger.debug(f"Changed working directory to: {directory_path}")
        
        base_command = [aider_path]
        command = prepare_aider_command(
            base_command,
            files, # Use the files argument directly
            aider_options
        )
        logger.info(f"[ai_edit_files] Files passed to aider: {files}")
        logger.info(f"Running aider command: {' '.join(command)}")
        
        with open(instructions_file, 'r') as f_read:
            instructions_content_str = f_read.read()
            
        logger.debug("Executing Aider with the instructions...")
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=directory_path,
        )

        if instructions_content_str:
            try:
                if process.stdin is not None:
                    process.stdin.write(instructions_content_str.encode())
                    await process.stdin.drain()
                    process.stdin.close()
                else:
                    logger.error("process.stdin is None; cannot write instructions to process.")
            except Exception as e:
                logger.error(f"Error writing to or closing process stdin: {e}")

        async def read_stream_and_send(stream, stream_name):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded_line = line.decode().strip()
                await session.send_progress_notification(
                    progress_token="ai_edit",
                    progress=0.0,
                    message=f"[{stream_name}] {decoded_line}"
                )

        stdout_task = asyncio.create_task(read_stream_and_send(process.stdout, "AIDER_STDOUT"))
        stderr_task = asyncio.create_task(read_stream_and_send(process.stderr, "AIDER_STDERR"))

        await asyncio.gather(stdout_task, stderr_task)
        await process.wait()
        
        os.chdir(original_dir) # Restore original directory after process finishes

        return_code = process.returncode
        if return_code != 0:
            logger.error(f"Aider process exited with code {return_code}")
            await session.send_progress_notification(
                progress_token="ai_edit",
                progress=1.0,
                message=f"Aider process exited with code {return_code}"
            )
            return f"Error: Aider process exited with code {return_code}"
        else:
            logger.info("Aider process completed successfully")
            await session.send_progress_notification(
                progress_token="ai_edit",
                progress=1.0,
                message="Aider process completed successfully."
            )
            return "Code changes completed successfully."
    finally:
        logger.debug(f"Cleaning up temporary file: {instructions_file}")
        os.unlink(instructions_file)
        
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
            logger.debug(f"Restored working directory to: {original_dir}")

async def aider_status_tool(
    repo_path: str,
    check_environment: bool = True,
    aider_path: Optional[str] = None,
    config_file: Optional[str] = None
) -> str:
    """
    Check the status of Aider and its environment.
    This function encapsulates the logic from aider_mcp/server.py's aider_status tool.
    """
    aider_path = aider_path or "aider"

    logger.info("Checking Aider status")
    
    result: Dict[str, Any] = {}
    
    try:
        command = [aider_path, "--version"]
        stdout, stderr = await run_command(command)
        
        version_info = stdout.strip() if stdout else "Unknown version"
        logger.info(f"Detected Aider version: {version_info}")
        
        result["aider"] = {
            "installed": bool(stdout and not stderr),
            "version": version_info,
            "executable_path": aider_path,
        }
        
        directory_path = os.path.abspath(repo_path)
        result["directory"] = {
            "path": directory_path,
            "exists": os.path.exists(directory_path),
        }
        
        git_root = find_git_root(directory_path)
        result["git"] = {
            "is_git_repo": bool(git_root),
            "git_root": git_root,
        }
        
        if git_root:
            try:
                original_dir = os.getcwd()
                
                os.chdir(directory_path)
                
                name_cmd = ["git", "config", "--get", "remote.origin.url"]
                name_stdout, _ = await run_command(name_cmd)
                result["git"]["remote_url"] = name_stdout.strip() if name_stdout else None
                
                branch_cmd = ["git", "branch", "--show-current"]
                branch_stdout, _ = await run_command(branch_cmd)
                result["git"]["current_branch"] = branch_stdout.strip() if branch_stdout else None
                
                os.chdir(original_dir)
            except Exception as e:
                logger.warning(f"Error getting git details: {e}")
        
        if check_environment:
            
            config = load_aider_config(directory_path, config_file)
            if config:
                result["config"] = config
            
            result["config_files"] = {
                "searched": [
                    os.path.expanduser("~/.aider.conf.yml"),
                    os.path.join(git_root, ".aider.conf.yml") if git_root else None,
                    os.path.join(directory_path, ".aider.conf.yml"),
                ],
                "used": os.path.join(directory_path, ".aider.conf.yml")
                if os.path.exists(os.path.join(directory_path, ".aider.conf.yml")) else None
            }
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error checking Aider status: {e}")
        return f"UNEXPECTED_ERROR: Failed to check Aider status: {e}. AI_HINT: Ensure Aider is installed, environment variables are set, and review server logs for more details."

mcp_server: Server = Server("mcp-git")

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name=GitTools.STATUS,
            description="Shows the working tree status",
            inputSchema=GitStatus.model_json_schema(),
        ),
        Tool(
            name=GitTools.DIFF_ALL,
            description="Shows all changes in the working directory (staged and unstaged, compared to HEAD)",
            inputSchema=GitDiffAll.model_json_schema(),
        ),
        Tool(
            name=GitTools.DIFF,
            description="Shows differences between branches or commits",
            inputSchema=GitDiff.model_json_schema(),
        ),
        Tool(
            name=GitTools.STAGE_AND_COMMIT,
            description="Records changes to the repository",
            inputSchema=GitCommit.model_json_schema(),
        ),
        Tool(
            name=GitTools.RESET,
            description="Unstages all staged changes",
            inputSchema=GitReset.model_json_schema(),
        ),
        Tool(
            name=GitTools.LOG,
            description="Shows the commit logs",
            inputSchema=GitLog.model_json_schema(),
        ),
        Tool(
            name=GitTools.CREATE_BRANCH,
            description="Creates a new branch from an optional base branch",
            inputSchema=GitCreateBranch.model_json_schema(),
        ),
        Tool(
            name=GitTools.CHECKOUT,
            description="Switches branches",
            inputSchema=GitCheckout.model_json_schema(),
        ),
        Tool(
            name=GitTools.SHOW,
            description="Shows the contents of a commit",
            inputSchema=GitShow.model_json_schema(),
        ),
        Tool(
            name=GitTools.APPLY_DIFF,
            description="Applies a diff to the working directory",
            inputSchema=GitApplyDiff.model_json_schema(),
        ),
        Tool(
            name=GitTools.READ_FILE,
            description="Reads the content of a file in the repository",
            inputSchema=GitReadFile.model_json_schema(),
        ),
        Tool(
            name=GitTools.SEARCH_AND_REPLACE,
            description="Searches for a string or regex pattern in a file and replaces it with another string.",
            inputSchema=SearchAndReplace.model_json_schema(),
        ),
        Tool(
            name=GitTools.WRITE_TO_FILE,
            description="Writes content to a specified file, creating it if it doesn't exist or overwriting it if it does.",
            inputSchema=WriteToFile.model_json_schema(),
        ),
        Tool(
            name=GitTools.EXECUTE_COMMAND,
            description="Executes a custom shell command within the specified repository path.",
            inputSchema=ExecuteCommand.model_json_schema(),
        ),
        Tool(
            name=GitTools.AI_EDIT,
            description="AI pair programming tool for making targeted code changes using Aider. Use this tool to:\n\n"
                        "1. Implement new features or functionality in existing code\n"
                        "2. Add tests to an existing codebase\n"
                        "3. Fix bugs in code\n"
                        "4. Refactor or improve existing code\n"
                        "5. Make structural changes across multiple files\n\n"
                        "The tool requires:\n"
                        "- A repository path where the code exists\n"
                        "- A detailed message describing what changes to make. Please only describe one change per message. "
                        "If you need to make multiple changes, please submit multiple requests.\n\n"
                        "Best practices for messages:\n"
                        "- Be specific about what files or components to modify\n"
                        "- Describe the desired behavior or functionality clearly\n"
                        "- Provide context about the existing codebase structure\n"
                        "- Include any constraints or requirements to follow\n\n"
                        "Examples of good messages:\n"
                        "- \"Add unit tests for the Customer class in src/models/customer.rb testing the validation logic\"\n"
                        "- \"Implement pagination for the user listing API in the controllers/users_controller.js file\"\n"
                        "- \"Fix the bug in utils/date_formatter.py where dates before 1970 aren't handled correctly\"\n"
                        "- \"Refactor the authentication middleware in middleware/auth.js to use async/await instead of callbacks\"",
            inputSchema=AiEdit.model_json_schema(),
        ),
        Tool(
            name=GitTools.AIDER_STATUS,
            description="Check the status of Aider and its environment. Use this to:\n\n"
                        "1. Verify Aider is correctly installed\n"
                        "2. Check API keys for OpenAI/Anthropic are set up\n"
                        "3. View the current configuration\n"
                        "4. Diagnose connection or setup issues",
            inputSchema=AiderStatus.model_json_schema(),
        )
    ]

async def list_repos() -> Sequence[str]:
    async def by_roots() -> Sequence[str]:
        if not isinstance(mcp_server.request_context.session, ServerSession):
            raise TypeError("mcp_server.request_context.session must be a ServerSession")

        if not mcp_server.request_context.session.check_client_capability(
            ClientCapabilities(roots=RootsCapability())
        ):
            return []

        roots_result: ListRootsResult = await mcp_server.request_context.session.list_roots()
        logger.debug(f"Roots result: {roots_result}")
        repo_paths = []
        for root in roots_result.roots:
            path = root.uri.path
            try:
                git.Repo(path)
                repo_paths.append(str(path))
            except git.InvalidGitRepositoryError:
                pass
        return repo_paths

    return await by_roots()

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[Content]:
    # Explicitly check if the tool name is valid
    try:
        if name not in set(item.value for item in GitTools):
            raise ValueError(f"Unknown tool: {name}")

        repo_path_arg = arguments.get("repo_path", ".")
        if repo_path_arg == ".":
            return [
                TextContent(
                    type="text",
                    text=(
                        "ERROR: The repo_path parameter cannot be '.'. Please provide the full absolute path to the repository. "
                        "You must always resolve and pass the full path, not a relative path like '.'. This is required for correct operation."
                    )
                )
            ]
        repo_path = Path(repo_path_arg)
        
        repo = None
        # Enhanced: catch InvalidGitRepositoryError and check for home directory
        try:
            # --- Begin original match/case block ---
            match name:
                case GitTools.STATUS:
                    repo = git.Repo(repo_path)
                    status = git_status(repo)
                    return [TextContent(
                        type="text",
                        text=f"Repository status:\n{status}"
                    )]
                case GitTools.DIFF_ALL:
                    repo = git.Repo(repo_path)
                    diff = git_diff_all(repo)
                    return [TextContent(
                        type="text",
                        text=f"All changes (staged and unstaged):\n{diff}"
                    )]
                case GitTools.DIFF:
                    repo = git.Repo(repo_path)
                    diff = git_diff(repo, arguments["target"])
                    return [TextContent(
                        type="text",
                        text=f"Diff with {arguments['target']}:\n{diff}"
                    )]
                case GitTools.STAGE_AND_COMMIT:
                    repo = git.Repo(repo_path)
                    result = git_stage_and_commit(repo, arguments["message"])
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case GitTools.RESET:
                    repo = git.Repo(repo_path)
                    result = git_reset(repo)
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case GitTools.LOG:
                    repo = git.Repo(repo_path)
                    log = git_log(repo, arguments.get("max_count", 10))
                    return [TextContent(
                        type="text",
                        text="Commit history:\n" + "\n".join(log)
                    )]
                case GitTools.CREATE_BRANCH:
                    repo = git.Repo(repo_path)
                    result = git_create_branch(
                        repo,
                        arguments["branch_name"],
                        arguments.get("base_branch")
                    )
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case GitTools.CHECKOUT:
                    repo = git.Repo(repo_path)
                    result = git_checkout(repo, arguments["branch_name"])
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case GitTools.SHOW:
                    repo = git.Repo(repo_path)
                    result = git_show(repo, arguments["revision"])
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case GitTools.APPLY_DIFF:
                    repo = git.Repo(repo_path)
                    result = await git_apply_diff(repo, arguments["diff_content"])
                    return [TextContent(
                        type="text",
                        text=f"<![CDATA[{result}]]>"
                    )]
                case GitTools.READ_FILE:
                    repo = git.Repo(repo_path)
                    result = git_read_file(repo, arguments["file_path"])
                    return [TextContent(
                        type="text",
                        text=f"<![CDATA[{result}]]>"
                    )]
                case GitTools.SEARCH_AND_REPLACE:
                    result = await search_and_replace_in_file(
                        repo_path=str(repo_path),
                        file_path=arguments["file_path"],
                        search_string=arguments["search_string"],
                        replace_string=arguments["replace_string"],
                        ignore_case=arguments.get("ignore_case", False),
                        start_line=arguments.get("start_line"),
                        end_line=arguments.get("end_line")
                    )
                    return [TextContent(
                        type="text",
                        text=f"<![CDATA[{result}]]>"
                    )]
                case GitTools.WRITE_TO_FILE:
                    logging.debug(f"Content input to write_to_file: {arguments['content']}")
                    result = await write_to_file_content(
                        repo_path=str(repo_path),
                        file_path=arguments["file_path"],
                        content=arguments["content"]
                    )
                    logging.debug(f"Content before TextContent: {result}")
                    return [TextContent(
                        type="text",
                        text=f"<![CDATA[{result}]]>"
                    )]
                case GitTools.EXECUTE_COMMAND:
                    result = await execute_custom_command(
                        repo_path=str(repo_path),
                        command=arguments["command"]
                    )
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case GitTools.AI_EDIT:
                    message = arguments.get("message", "")
                    files = arguments["files"] # files is now mandatory
                    options = arguments.get("options", [])
                    # Retrieve OpenAI API key and base from environment variables
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    openai_api_base = os.environ.get("OPENAI_API_BASE")
                    result = await ai_edit_files(
                        repo_path=str(repo_path),
                        message=message,
                        session=mcp_server.request_context.session,
                        files=files, # Pass files to ai_edit_files
                        options=options,
                    )
                    return [TextContent(
                        type="text",
                        text=f"<![CDATA[{result}]]>"
                    )]
                case GitTools.AIDER_STATUS:
                    check_environment = arguments.get("check_environment", True)
                    result = await aider_status_tool(
                        repo_path=str(repo_path),
                        check_environment=check_environment
                    )
                    return [TextContent(
                        type="text",
                        text=f"<![CDATA[{result}]]>"
                    )]
                case _:
                    raise ValueError(f"Unknown tool: {name}")
            # --- End original match/case block ---

        except git.InvalidGitRepositoryError:
            # If the path is the user's home directory, return the specific warning
            home_dir = Path(os.path.expanduser("~"))
            if repo_path.resolve() == home_dir.resolve():
                return [
                    TextContent(
                        type="text",
                        text=(
                            "ERROR: The repo_path parameter cannot be '.'. Please provide the full absolute path to the repository. "
                            "You must always resolve and pass the full path, not a relative path like '.'. This is required for correct operation."
                        )
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"ERROR: Not a valid Git repository: {repo_path}"
                    )
                ]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"UNEXPECTED_ERROR: An unexpected exception occurred: {e}. AI_HINT: Check the server logs for more details or review your input for possible mistakes."
                )
            ]
    except ValueError as ve:
        return [
            TextContent(
                type="text",
                text=f"INVALID_TOOL_NAME: {ve}. AI_HINT: Check the tool name and ensure it matches one of the supported tools."
            )
        ]


POST_MESSAGE_ENDPOINT = "/messages/"

sse_transport = SseServerTransport(POST_MESSAGE_ENDPOINT)

async def handle_sse(request):
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        options = mcp_server.create_initialization_options()
        await mcp_server.run(read_stream, write_stream, options, raise_exceptions=True)
    return Response()

async def handle_post_message(scope, receive, send):
    await sse_transport.handle_post_message(scope, receive, send)

routes = [
    Route("/sse", endpoint=handle_sse, methods=["GET"]),
    Mount(POST_MESSAGE_ENDPOINT, app=handle_post_message),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    pass
