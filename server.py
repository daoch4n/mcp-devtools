"""
MCP Git Server

This module implements a server for the MCP (Multi-Agent Collaboration Platform)
that provides a set of Git-related tools and AI-powered code editing capabilities
using Aider. It allows clients to interact with Git repositories, perform file
operations, execute commands, and initiate AI-driven code modifications.

Key Components:
- Git Operations: Functions for common Git commands like status, diff, commit,
  reset, log, branch creation, checkout, and applying diffs.
- File Operations: Tools for reading, writing, and searching/replacing content
  within files.
- Command Execution: A general-purpose tool to execute arbitrary shell commands.
- AI-Powered Editing (Aider): Integration with the Aider tool for advanced
  code modifications based on natural language instructions.
- Configuration Loading: Utilities to load Aider-specific configurations and
  environment variables from various locations (.aider.conf.yml, .env).
- MCP Server Integration: Exposes these functionalities as MCP tools, allowing
  them to be called by agents.
- Starlette Application: Sets up an HTTP server with SSE (Server-Sent Events)
  for communication with MCP clients.
"""

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
Content: TypeAlias = TextContent | ImageContent | EmbeddedResource # type: ignore

from enum import Enum
import git # type: ignore
from git.exc import GitCommandError
from pydantic import BaseModel, Field
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
    """
    Finds the root directory of a Git repository by traversing up from the given path.

    Args:
        path: The starting path to search from.

    Returns:
        The absolute path to the Git repository root, or None if not found.
    """
    current = os.path.abspath(path)
    while current != os.path.dirname(current):
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        current = os.path.dirname(current)
    return None

def load_aider_config(repo_path: Optional[str] = None, config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads Aider configuration from various possible locations, merging them
    in a specific order of precedence (home dir < git root < working dir < specified file).

    Args:
        repo_path: The path to the repository or working directory. Defaults to current working directory.
        config_file: An optional specific path to an Aider configuration file to load.

    Returns:
        A dictionary containing the merged Aider configuration.
    """
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
    
    # Load in reverse order of precedence, so later files override earlier ones
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
    """
    Loads environment variables from .env files found in various locations,
    merging them in a specific order of precedence (home dir < git root < working dir < specified file).

    Args:
        repo_path: The path to the repository or working directory. Defaults to current working directory.
        env_file: An optional specific path to a .env file to load.

    Returns:
        A dictionary containing the loaded environment variables.
    """
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
    
    # Load in reverse order of precedence, so later files override earlier ones
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
    """
    Executes a shell command asynchronously.

    Args:
        command: A list of strings representing the command and its arguments.
        input_data: Optional string data to pass to the command's stdin.

    Returns:
        A tuple containing the stdout and stderr of the command as strings.
    """
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
    """
    Prepares the full Aider command by adding files and options to the base command.

    Args:
        base_command: The initial Aider command (e.g., ["aider"]).
        files: An optional list of file paths to include in the command.
        options: An optional dictionary of Aider options (e.g., {"yes_always": True}).

    Returns:
        A list of strings representing the complete Aider command.
    """
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
    
    command = [c for c in command if c]

    if files:
        command.extend(files)
    
    return command

class GitStatus(BaseModel):
    """
    Represents the input schema for the `git_status` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")

class GitDiffAll(BaseModel):
    """
    Represents the input schema for the `git_diff_all` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")

class GitDiff(BaseModel):
    """
    Represents the input schema for the `git_diff` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    target: str = Field(description="The target (e.g., branch name, commit hash, tag) to diff against. For example, 'main', 'HEAD~1', or a full commit SHA.")

class GitCommit(BaseModel):
    """
    Represents the input schema for the `git_stage_and_commit` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    message: str = Field(description="The commit message for the changes.")
    files: Optional[List[str]] = Field(
        None,
        description="An optional list of specific file paths (relative to the repository root) to stage before committing. If not provided, all changes will be staged."
    )

class GitReset(BaseModel):
    """
    Represents the input schema for the `git_reset` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")

class GitLog(BaseModel):
    """
    Represents the input schema for the `git_log` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    max_count: int = Field(10, description="The maximum number of commit entries to retrieve. Defaults to 10.")

class GitCreateBranch(BaseModel):
    """
    Represents the input schema for the `git_create_branch` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    branch_name: str = Field(description="The name of the new branch to create.")
    base_branch: Optional[str] = Field(
        None,
        description="Optional. The name of the branch or commit hash to base the new branch on. If not provided, the new branch will be based on the current active branch."
    )

class GitCheckout(BaseModel):
    """
    Represents the input schema for the `git_checkout` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    branch_name: str = Field(description="The name of the branch to checkout.")

class GitShow(BaseModel):
    """
    Represents the input schema for the `git_show` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    revision: str = Field(description="The commit hash or reference (e.g., 'HEAD', 'main', 'abc1234') to show details for.")

class GitApplyDiff(BaseModel):
    """
    Represents the input schema for the `git_apply_diff` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    diff_content: str = Field(description="The diff content string to apply to the repository. This should be in a unified diff format.")

class GitReadFile(BaseModel):
    """
    Represents the input schema for the `git_read_file` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    file_path: str = Field(description="The path to the file to read, relative to the repository's working directory.")

class SearchAndReplace(BaseModel):
    """
    Represents the input schema for the `search_and_replace` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    file_path: str = Field(description="The path to the file to modify, relative to the repository's working directory.")
    search_string: str = Field(description="The string or regex pattern to search for within the file.")
    replace_string: str = Field(description="The string to replace all matches of the search string with.")
    ignore_case: bool = Field(False, description="If true, the search will be case-insensitive. Defaults to false.")
    start_line: Optional[int] = Field(
        None,
        description="Optional. The 1-based starting line number for the search and replace operation (inclusive). If not provided, search starts from the beginning of the file."
    )
    end_line: Optional[int] = Field(
        None,
        description="Optional. The 1-based ending line number for the search and replace operation (inclusive). If not provided, search continues to the end of the file."
    )

class WriteToFile(BaseModel):
    """
    Represents the input schema for the `write_to_file` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    file_path: str = Field(description="The path to the file to write to, relative to the repository's working directory. The file will be created if it doesn't exist, or overwritten if it does.")
    content: str = Field(description="The string content to write to the specified file.")

class ExecuteCommand(BaseModel):
    """
    Represents the input schema for the `execute_command` tool.
    """
    repo_path: str = Field(description="The absolute path to the directory where the command should be executed.")
    command: str = Field(description="The shell command string to execute (e.g., 'ls -l', 'npm install').")

class EditFormat(str, Enum):
    """
    An enumeration of supported Aider edit formats.
    """
    DIFF = "diff"
    DIFF_FENCED = "diff-fenced"
    UDIFF = "udiff"
    WHOLE = "whole"

class AiEdit(BaseModel):
    """
    Represents the input schema for the `ai_edit` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory where the AI edit should be performed.")
    message: str = Field(description="A detailed natural language message describing the code changes to make. Be specific about files, desired behavior, and any constraints.")
    files: List[str] = Field(description="A list of file paths (relative to the repository root) that Aider should operate on. This argument is mandatory.")
    options: Optional[List[str]] = Field(
        None,
        description="Optional. A list of additional command-line options to pass directly to Aider (e.g., ['--model=gpt-4o', '--dirty-diff']). Each option should be a string."
    )
    edit_format: EditFormat = Field(
        EditFormat.DIFF,
        description="Optional. The format Aider should use for edits. Defaults to 'diff'. Options: 'diff', 'diff-fenced', 'udiff', 'whole'."
    )

class AiderStatus(BaseModel):
    """
    Represents the input schema for the `aider_status` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository or working directory to check Aider's status within.")
    check_environment: bool = Field(
        True,
        description="If true, the tool will also check Aider's configuration, environment variables, and Git repository details. Defaults to true."
    )

class GitTools(str, Enum):
    """
    An enumeration of all available Git and related tools.
    """
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
    """
    Gets the status of the Git working tree.

    Args:
        repo: The Git repository object.

    Returns:
        A string representing the output of `git status`.
    """
    return repo.git.status()

def git_diff_all(repo: git.Repo) -> str:
    """
    Shows all changes in the working directory (staged and unstaged, compared to HEAD).

    Args:
        repo: The Git repository object.

    Returns:
        A string representing the output of `git diff HEAD`.
    """
    return repo.git.diff("HEAD")

def git_diff(repo: git.Repo, target: str) -> str:
    """
    Shows differences between branches or commits.

    Args:
        repo: The Git repository object.
        target: The target (branch, commit hash, etc.) to diff against.

    Returns:
        A string representing the output of `git diff <target>`.
    """
    return repo.git.diff(target)

def git_stage_and_commit(repo: git.Repo, message: str, files: Optional[List[str]] = None) -> str:
    """
    Stages changes and commits them to the repository.

    Args:
        repo: The Git repository object.
        message: The commit message.
        files: An optional list of specific files to stage. If None, all changes are staged.

    Returns:
        A string indicating the success of the staging and commit operation.
    """
    if files:
        repo.index.add(files)
        staged_message = f"Files {', '.join(files)} staged successfully."
    else:
        repo.git.add(A=True)
        staged_message = "All changes staged successfully."

    commit = repo.index.commit(message)
    return f"{staged_message}\nChanges committed successfully with hash {commit.hexsha}"

def git_reset(repo: git.Repo) -> str:
    """
    Unstages all staged changes in the repository.

    Args:
        repo: The Git repository object.

    Returns:
        A string indicating that all staged changes have been reset.
    """
    repo.index.reset()
    return "All staged changes reset"

def git_log(repo: git.Repo, max_count: int = 10) -> list[str]:
    """
    Shows the commit logs for the repository.

    Args:
        repo: The Git repository object.
        max_count: The maximum number of commits to retrieve.

    Returns:
        A list of strings, where each string represents a formatted commit entry.
    """
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
    """
    Creates a new branch in the repository.

    Args:
        repo: The Git repository object.
        branch_name: The name of the new branch.
        base_branch: Optional. The name of the branch to base the new branch on.
                     If None, the new branch is based on the current active branch.

    Returns:
        A string indicating the successful creation of the branch.
    """
    if base_branch:
        base = repo.refs[base_branch]
    else:
        base = repo.active_branch

    repo.create_head(branch_name, base)
    return f"Created branch '{branch_name}' from '{base.name}'"

def git_checkout(repo: git.Repo, branch_name: str) -> str:
    """
    Switches the current branch to the specified branch.

    Args:
        repo: The Git repository object.
        branch_name: The name of the branch to checkout.

    Returns:
        A string indicating the successful checkout of the branch.
    """
    repo.git.checkout(branch_name)
    return f"Switched to branch '{branch_name}'"

def git_show(repo: git.Repo, revision: str) -> str:
    """
    Shows the contents (metadata and diff) of a specific commit.

    Args:
        repo: The Git repository object.
        revision: The commit hash or reference to show.

    Returns:
        A string containing the commit details and its diff.
    """
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
    """
    Applies a given diff content to the working directory of the repository.
    Includes a check for successful application and generates a new diff output.
    Also runs TSC if applicable after applying the diff.

    Args:
        repo: The Git repository object.
        diff_content: The diff string to apply.

    Returns:
        A string indicating the result of the diff application, including
        any new diff generated and TSC output if applicable, or an error message.
    """
    tmp_file_path = None
    affected_file_path = None
    original_content = ""

    # Attempt to extract affected file path from the diff content
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
    """
    Reads the content of a specified file within the repository.

    Args:
        repo: The Git repository object.
        file_path: The path to the file relative to the repository's working directory.

    Returns:
        A string containing the file's content, or an error message if the file
        is not found or cannot be read.
    """
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
    """
    Generates a unified diff string between two versions of file content.

    Args:
        original_content: The original content of the file.
        new_content: The new content of the file.
        file_path: The path of the file, used for diff headers.

    Returns:
        A string containing the unified diff, or a message indicating no changes
        or that the diff was too large.
    """
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
    """
    Runs TypeScript compiler (tsc) with --noEmit if the file has a .ts, .js, or .mjs extension.

    Args:
        repo_path: The path to the repository's working directory.
        file_path: The path to the file that was modified.

    Returns:
        A string containing the TSC output, or an empty string if TSC is not applicable.
    """
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
    """
    Performs search and replace in a file using Python's re module.
    Attempts literal search first, then falls back to regex search if no changes are made.

    Args:
        repo_path: The path to the repository's working directory.
        search_string: The string or regex pattern to search for.
        replace_string: The string to replace matches with.
        file_path: The path to the file to modify.
        ignore_case: If True, the search is case-insensitive.
        start_line: Optional. The 1-based starting line number for the search.
        end_line: Optional. The 1-based ending line number for the search.

    Returns:
        A string indicating the result of the operation, including diff and TSC output,
        or an error message.
    """
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
    """
    Searches for a string or regex pattern in a file and replaces it with another string.
    Attempts to use `sed` for efficiency, falling back to Python logic if `sed` fails or makes no changes.

    Args:
        repo_path: The path to the repository's working directory.
        search_string: The string or regex pattern to search for.
        replace_string: The string to replace matches with.
        file_path: The path to the file to modify.
        ignore_case: If True, the search is case-insensitive.
        start_line: Optional. The 1-based starting line number for the search.
        end_line: Optional. The 1-based ending line number for the search.

    Returns:
        A string indicating the result of the operation, including diff and TSC output,
        or an error message.
    """
    full_file_path = Path(repo_path) / file_path

    sed_command_parts = ["sed", "-i"]

    # Escape sed special characters in search and replacement strings
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
    """
    Writes content to a specified file, creating it if it doesn't exist or overwriting it if it does.
    Includes a check to ensure the content was written correctly and generates a diff.

    Args:
        repo_path: The path to the repository's working directory.
        file_path: The path to the file to write to, relative to the repository.
        content: The string content to write to the file.

    Returns:
        A string indicating the success of the write operation, including diff and TSC output,
        or an error message.
    """
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
    """
    Executes a custom shell command within the specified repository path.

    Args:
        repo_path: The path to the directory where the command should be executed.
        command: The shell command string to execute.

    Returns:
        A string containing the stdout and stderr of the command, and an indication
        if the command failed.
    """
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
    files: List[str],
    options: Optional[list[str]],
    edit_format: EditFormat = EditFormat.DIFF,
    aider_path: Optional[str] = None,
    config_file: Optional[str] = None,
    env_file: Optional[str] = None,
) -> str:
    """
    AI pair programming tool for making targeted code changes using Aider.
    This function encapsulates the logic from aider_mcp/server.py's edit_files tool.
    """
    aider_path = aider_path or "aider"
    edit_format_str = edit_format.value

    logger.info(f"Running aider in directory: {repo_path}")
    logger.debug(f"Message length: {len(message)} characters")
    logger.debug(f"Additional options: {options}")

    directory_path = os.path.abspath(repo_path)
    if not os.path.exists(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        return f"Error: Directory does not exist: {directory_path}"

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
    # Use the specified edit format
    aider_options["edit_format"] = edit_format_str
    # Pass the message directly as a command-line option
    aider_options["message"] = message

    additional_opts: Dict[str, Any] = {}
    if options:
        for opt in options:
            if opt.startswith("--"):
                if "=" in opt:
                    key, value_str = opt[2:].split("=", 1)
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

    unsupported_options = ["base_url", "base-url"]
    for opt_key in unsupported_options:
        if opt_key in additional_opts:
            logger.warning(f"Removing unsupported Aider option: --{opt_key.replace('_', '-')}")
            del additional_opts[opt_key]

    aider_options.update(additional_opts)

    for fname in files:
        fpath = os.path.join(directory_path, fname)
        if not os.path.isfile(fpath):
            logger.error(f"[ai_edit_files] Provided file not found in repo: {fname}. Aider may fail.")

    original_dir = os.getcwd()
    pre_aider_commit_hash = None
    try:
        # Capture the current HEAD commit hash before Aider runs
        try:
            repo = git.Repo(directory_path)
            if repo.head.is_valid() and repo.head.is_set(): # Check if HEAD points to a valid ref
                pre_aider_commit_hash = repo.head.commit.hexsha
                logger.debug(f"Pre-Aider HEAD commit: {pre_aider_commit_hash}")
            else:
                logger.debug("Repository has no commits yet or detached HEAD before Aider.")
        except git.InvalidGitRepositoryError:
            logger.warning(f"Directory {directory_path} is not a valid Git repository. Cannot capture pre-Aider commit hash.")
        except Exception as e:
            logger.warning(f"Error capturing pre-Aider commit hash: {e}")
        os.chdir(directory_path)
        logger.debug(f"Changed working directory to: {directory_path}")

        base_command = [aider_path]
        command_list = prepare_aider_command(
            base_command,
            files,
            aider_options
        )
        command_str = ' '.join(shlex.quote(part) for part in command_list)
        
        logger.info(f"[ai_edit_files] Files passed to aider: {files}")
        logger.info(f"Running aider command: {command_str}")

        logger.debug("Executing Aider with the instructions...")

        process = await asyncio.create_subprocess_shell(
            command_str,
            stdin=None, # No need for stdin anymore
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=directory_path,
        )

        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode('utf-8')
        stderr = stderr_bytes.decode('utf-8')

        await session.send_progress_notification(
            progress_token="ai_edit",
            progress=0.5,
            message=f"AIDER STDOUT:\n{stdout}"
        )
        if stderr:
            await session.send_progress_notification(
                progress_token="ai_edit",
                progress=0.5,
                message=f"AIDER STDERR:\n{stderr}"
            )

        return_code = process.returncode
        if return_code != 0:
            logger.error(f"Aider process exited with code {return_code}")
            return f"Error: Aider process exited with code {return_code}.\nSTDERR:\n{stderr}"
        else:
            logger.info("Aider process completed successfully")
            
            result_message = "Aider completed successfully."
            if "Applied edit to" in stdout:
                result_message = "Code changes completed and committed successfully."
                
                try:
                    # Re-initialize repo object to get latest state after Aider potentially made changes
                    repo = git.Repo(directory_path)
                    
                    post_aider_commit_hash = None
                    if repo.head.is_valid() and repo.head.is_set():
                        post_aider_commit_hash = repo.head.commit.hexsha
                        logger.debug(f"Post-Aider HEAD commit: {post_aider_commit_hash}")
                    else:
                        logger.debug("Repository has no commits or detached HEAD after Aider.")

                    if pre_aider_commit_hash and post_aider_commit_hash and pre_aider_commit_hash != post_aider_commit_hash:
                        # Generate diff between the two commit hashes
                        diff_output = repo.git.diff(pre_aider_commit_hash, post_aider_commit_hash)
                        if diff_output:
                            result_message += f"\n\nDiff of changes made by Aider:\n```diff\n{diff_output}\n```"
                        else:
                            result_message += "\n\nNo diff generated between pre and post Aider commits (perhaps no changes were made or it's an empty commit)."
                    elif not pre_aider_commit_hash and post_aider_commit_hash:
                        # Case: Repo was empty before, now has commits. Diff against NULL_TREE.
                        diff_output = repo.git.diff(git.NULL_TREE, post_aider_commit_hash)
                        if diff_output:
                            result_message += f"\n\nDiff of changes made by Aider (initial commit):\n```diff\n{diff_output}\n```"
                        else:
                            result_message += "\n\nNo diff generated for the initial commit (perhaps no changes were made or it's an empty commit)."
                    else:
                        result_message += "\n\nNo new commit detected or no changes made by Aider."

                except git.InvalidGitRepositoryError:
                    result_message += "\n\nCould not access Git repository to get diff after Aider run."
                except Exception as e:
                    result_message += f"\n\nError generating diff for Aider changes: {e}"
            else:
                 result_message += f"\nIt's unclear if changes were applied. Please verify the file manually.\nSTDOUT:\n{stdout}"
            
            return result_message

    except Exception as e:
        logger.error(f"An unexpected error occurred during ai_edit_files: {e}")
        return f"UNEXPECTED_ERROR: An unexpected error occurred during AI edit: {e}. AI_HINT: Check the server logs for more details."
    finally:
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
    Checks the status of Aider and its environment, including installation,
    configuration, and Git repository details.

    Args:
        repo_path: The path to the repository or working directory to check.
        check_environment: If True, also checks Aider configuration and Git details.
        aider_path: Optional. The path to the Aider executable. Defaults to "aider".
        config_file: Optional. Path to a specific Aider configuration file.

    Returns:
        A JSON string containing the status information, or an error message.
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
    """
    Lists all available tools provided by this MCP Git server.

    Returns:
        A list of Tool objects, each describing a callable tool with its name,
        description, and input schema.
    """
    return [
        Tool(
            name=GitTools.STATUS,
            description="Shows the current status of the Git working tree, including untracked, modified, and staged files.",
            inputSchema=GitStatus.model_json_schema(),
        ),
        Tool(
            name=GitTools.DIFF_ALL,
            description="Shows all changes in the working directory, including both staged and unstaged modifications, compared to the HEAD commit. This provides a comprehensive view of all local changes.",
            inputSchema=GitDiffAll.model_json_schema(),
        ),
        Tool(
            name=GitTools.DIFF,
            description="Shows differences between the current working directory and a specified Git target (e.g., another branch, a specific commit hash, or a tag).",
            inputSchema=GitDiff.model_json_schema(),
        ),
        Tool(
            name=GitTools.STAGE_AND_COMMIT,
            description="Stages specified files (or all changes if no files are specified) and then commits them to the repository with a given message. This creates a new commit in the Git history.",
            inputSchema=GitCommit.model_json_schema(),
        ),
        Tool(
            name=GitTools.RESET,
            description="Unstages all currently staged changes in the repository, moving them back to the working directory without discarding modifications. This is equivalent to `git reset` without arguments.",
            inputSchema=GitReset.model_json_schema(),
        ),
        Tool(
            name=GitTools.LOG,
            description="Shows the commit history for the repository, listing recent commits with their hash, author, date, and message. The number of commits can be limited.",
            inputSchema=GitLog.model_json_schema(),
        ),
        Tool(
            name=GitTools.CREATE_BRANCH,
            description="Creates a new Git branch with the specified name. Optionally, you can base the new branch on an existing branch or commit, otherwise it defaults to the current active branch.",
            inputSchema=GitCreateBranch.model_json_schema(),
        ),
        Tool(
            name=GitTools.CHECKOUT,
            description="Switches the current active branch to the specified branch name. This updates the working directory to reflect the state of the target branch.",
            inputSchema=GitCheckout.model_json_schema(),
        ),
        Tool(
            name=GitTools.SHOW,
            description="Shows the metadata (author, date, message) and the diff of a specific commit. This allows inspection of changes introduced by a particular commit.",
            inputSchema=GitShow.model_json_schema(),
        ),
        Tool(
            name=GitTools.APPLY_DIFF,
            description="Applies a given diff content (in unified diff format) to the working directory of the repository. This can be used to programmatically apply patches or changes.",
            inputSchema=GitApplyDiff.model_json_schema(),
        ),
        Tool(
            name=GitTools.READ_FILE,
            description="Reads and returns the entire content of a specified file within the Git repository's working directory. The file path must be relative to the repository root.",
            inputSchema=GitReadFile.model_json_schema(),
        ),
        Tool(
            name=GitTools.SEARCH_AND_REPLACE,
            description="Searches for a specified string or regex pattern within a file and replaces all occurrences with a new string. Supports case-insensitive search and line-range restrictions. It attempts to use `sed` for efficiency, falling back to Python logic if `sed` fails or makes no changes.",
            inputSchema=SearchAndReplace.model_json_schema(),
        ),
        Tool(
            name=GitTools.WRITE_TO_FILE,
            description="Writes the provided content to a specified file within the repository. If the file does not exist, it will be created. If it exists, its content will be completely overwritten. Includes a check to ensure content was written correctly and generates a diff.",
            inputSchema=WriteToFile.model_json_schema(),
        ),
        Tool(
            name=GitTools.EXECUTE_COMMAND,
            description="Executes an arbitrary shell command within the context of the specified repository's working directory. This tool can be used for tasks not covered by other specific Git tools, such as running build scripts, linters, or other system commands.",
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
    """
    Lists all Git repositories known to the MCP client.
    This function leverages the client's `list_roots` capability.

    Returns:
        A sequence of strings, where each string is the absolute path to a Git repository.
    """
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
    """
    Executes a requested tool based on its name and arguments.
    This is the main entry point for clients to interact with the server's tools.

    Args:
        name: The name of the tool to call (must be one of the `GitTools` enum values).
        arguments: A dictionary of arguments specific to the tool being called.

    Returns:
        A list of Content objects (typically TextContent) containing the result
        or an error message.
    """
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
        try:
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
                    result = git_stage_and_commit(repo, arguments["message"], arguments.get("files"))
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
                        text=result
                    )]
                case GitTools.READ_FILE:
                    repo = git.Repo(repo_path)
                    result = git_read_file(repo, arguments["file_path"])
                    return [TextContent(
                        type="text",
                        text=result
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
                        text=result
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
                        text=result
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
                    result = await ai_edit_files(
                        repo_path=str(repo_path),
                        message=message,
                        session=mcp_server.request_context.session,
                        files=files,
                        options=options,
                        edit_format=arguments.get("edit_format", EditFormat.DIFF),
                    )
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case GitTools.AIDER_STATUS:
                    check_environment = arguments.get("check_environment", True)
                    result = await aider_status_tool(
                        repo_path=str(repo_path),
                        check_environment=check_environment
                    )
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case _:
                    raise ValueError(f"Unknown tool: {name}")

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
    """
    Handles Server-Sent Events (SSE) connections from MCP clients.
    Establishes a communication channel for the MCP server to send events.

    Args:
        request: The Starlette Request object.

    Returns:
        A Starlette Response object for the SSE connection.
    """
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        options = mcp_server.create_initialization_options()
        await mcp_server.run(read_stream, write_stream, options, raise_exceptions=True)
    return Response()

async def handle_post_message(scope, receive, send):
    """
    Handles incoming POST messages from MCP clients, typically used for client-to-server communication.

    Args:
        scope: The ASGI scope dictionary.
        receive: The ASGI receive callable.
        send: The ASGI send callable.
    """
    await sse_transport.handle_post_message(scope, receive, send)

routes = [
    Route("/sse", endpoint=handle_sse, methods=["GET"]),
    Mount(POST_MESSAGE_ENDPOINT, app=handle_post_message),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    # To run the server, you would typically use uvicorn:
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    pass
