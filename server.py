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
from pathlib import Path, PurePath
from typing import Sequence, Optional, TypeAlias, Any, Dict, List, Tuple, Union, cast
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
Content: TypeAlias = Union[TextContent, ImageContent, EmbeddedResource]

from enum import Enum
import git # type: ignore
from git.exc import GitCommandError # type: ignore
from pydantic import BaseModel, Field
import asyncio
import tempfile
import os
import re
import difflib
import shlex
import json
import yaml

logging.basicConfig(level=logging.DEBUG)

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response
from starlette.requests import Request
from starlette.types import Scope, Receive, Send

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


def _get_last_aider_reply(directory_path: str) -> Optional[str]:
    """
    Read the Aider chat history, extract the last session, clean SEARCH/REPLACE noise,
    and return the snippet or None if unavailable.
    """
    try:
        history_path = Path(directory_path) / ".aider.chat.history.md"
        if not history_path.exists():
            return None
        history_content = history_path.read_text(encoding="utf-8", errors="ignore")
        
        # Find the last session
        anchor = "# aider chat started at"
        last_anchor_pos = history_content.rfind(anchor)
        session_content = history_content[last_anchor_pos:] if last_anchor_pos != -1 else history_content
        
        # Find the last ASSISTANT block in the session
        assistant_blocks = re.findall(r"## ASSISTANT.*?(?=\n## |\Z)", session_content, re.DOTALL)
        if assistant_blocks:
            # Use the last ASSISTANT block
            snippet = assistant_blocks[-1]
        else:
            # Fallback to the entire session if no ASSISTANT block found
            snippet = session_content
        
        # Remove noisy lines but keep useful ones
        lines: List[str] = snippet.split('\n')
        cleaned_lines: List[str] = []
        
        for line in lines:
            # Remove session header
            if line.startswith("# aider chat started at"):
                continue
            # Remove command invocation lines
            if line.startswith("> ") and ("/aider" in line or "Aider v" in line or "Git repo:" in line):
                continue
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        
        # Remove SEARCH/REPLACE noise blocks using regex
        result = re.sub(r"<<<<<<< SEARCH.*?>>>>>>> REPLACE", "", result, flags=re.DOTALL)
        
        return result or None
    except Exception as e:
        logger.debug(f"Failed to get Aider chat history: {e}")
        return None

# === AI_HINT helper builders (keep terse, agent-friendly) ===

def ai_hint_git_apply_diff_error(stderr: str | None, affected_file_path: str | None) -> str:
    extra: List[str] = []
    low = (stderr or "").lower()
    if "patch failed" in low or "could not apply" in low:
        extra.append("The diff may not match the current repo state. Ensure the working tree is clean and the diff was created against the current HEAD.")
    if "whitespace" in low:
        extra.append("Whitespace conflicts detected. Consider re-generating the diff or removing trailing spaces.")
    if affected_file_path:
        extra.append(f"Verify that the file path in the diff exists: {affected_file_path} (relative to repo root).")
    extra_hint = (" More info: " + " ".join(extra)) if extra else ""
    return (
        f"GIT_COMMAND_FAILED: Failed to apply diff. Details: {stderr}. "
        f"Check if the diff is valid and applies cleanly to the current state of the repository.{extra_hint}"
    )

def ai_hint_read_file_error(file_path: str, repo_working_dir: str, e: Exception) -> str:
    return (
        "UNEXPECTED_ERROR: Failed to read file "
        f"'{file_path}': {e}. Confirm the file path is relative to the repo root under '{repo_working_dir}'. "
        "Ensure the file exists and is readable, and that you passed an absolute repo_path."
    )

def ai_hint_write_error(repo_path: str, file_path: str, e: Exception) -> str:
    return (
        "UNEXPECTED_ERROR: Failed to write to file "
        f"'{file_path}': {e}. Ensure parent directories exist under '{repo_path}'. "
        "Confirm write permissions and available disk space, and pass an absolute repo_path."
    )

def ai_hint_exec_error(repo_path: str, command: str, e: Exception) -> str:
    return (
        "UNEXPECTED_ERROR: Failed to execute command "
        f"'{command}': {e}. Commands run with cwd set to '{repo_path}'. "
        "Verify the command is installed and on PATH, and start with a simple echo to validate the environment."
    )

def ai_hint_ai_edit_unexpected(e: Exception) -> str:
    return (
        "UNEXPECTED_ERROR: An unexpected error occurred during AI edit: "
        f"{e}. Verify aider is installed (try the 'aider_status' tool), pass absolute repo_path, "
        "and ensure 'files' and 'continue_thread' are provided."
    )

def ai_hint_aider_status_error(e: Exception) -> str:
    return (
        "UNEXPECTED_ERROR: Failed to check Aider status: "
        f"{e}. Ensure Aider is installed and on PATH (try 'aider --version'), or use 'aider_status' with a custom aider_path if needed."
    )

def ai_hint_unexpected_call_tool(e: Exception) -> str:
    return (
        "UNEXPECTED_ERROR: An unexpected exception occurred: "
        f"{e}. Re-check the tool name and arguments. Use 'list_tools' to inspect schemas, "
        "and ensure repo_path is an absolute path valid for your workspace."
    )

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
    config: Dict[str, Any] = {}
    search_paths: List[str] = []
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
    env_vars: Dict[str, str] = {}
    search_paths: List[str] = []
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

async def run_command(command: List[str], input_data: Optional[str] = None) -> Tuple[str, str, int]:
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
    
    return stdout.decode(), stderr.decode(), cast(int, process.returncode)

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
                for item in cast(List[Any], value):
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


class GitDiff(BaseModel):
    """
    Represents the input schema for the `git_diff` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    target: Optional[str] = Field(
        None,
        description="Optional. If omitted, behaves like `git diff` (worktree vs index). Pass 'HEAD' or another ref to compare against a commit or branch."
    )
    path: Optional[str] = Field(
        None,
        description="Optional. Limit the diff to a specific file or directory path."
    )

class GitStageAndCommit(BaseModel):
    """
    Represents the input schema for the `git_stage_and_commit` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    message: str = Field(description="The commit message for the changes.")
    files: Optional[List[str]] = Field(
        None,
        description="An optional list of specific file paths (relative to the repository root) to stage before committing. If not provided, all changes will be staged."
    )

class GitLog(BaseModel):
    """
    Represents the input schema for the `git_log` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    max_count: int = Field(10, description="The maximum number of commit entries to retrieve. Defaults to 10.")

class GitBranch(BaseModel):
    """
    Input schema for the `git_branch` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    action: str = Field(description="The branch operation to perform: 'create', 'checkout', 'rename', or 'list'.")
    branch_name: Optional[str] = Field(None, description="The name of the branch to create, checkout, or rename. Required for 'create', 'checkout', and 'rename' actions; optional for 'list'.")
    base_branch: Optional[str] = Field(None, description="Optional. The base branch to create from when action='create'. If omitted, creates from the current HEAD.")
    new_name: Optional[str] = Field(None, description="Optional. The new name for the branch when action='rename'.")

class GitShow(BaseModel):
    """
    Represents the input schema for the `git_show` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    revision: str = Field(description="The commit hash/reference or range (e.g., 'HEAD', 'abc1234', 'rev1..rev2', or 'rev1...rev2') to show.")
    path: Optional[str] = Field(
        None,
        description="Optional. Filter the output to show only changes for a specific file path."
    )
    show_metadata_only: bool = Field(
        False,
        description="Optional. If true, only show commit metadata (author, date, message) without diff content. Defaults to false."
    )
    show_diff_only: bool = Field(
        False,
        description="Optional. If true, only show diff content without commit metadata. Defaults to false. If both show_metadata_only and show_diff_only are true, both sections will be included."
    )

class GitApplyDiff(BaseModel):
    """
    Represents the input schema for the `git_apply_diff` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    diff_content: str = Field(description="The diff content string to apply to the repository. This should be in a unified diff format.")

class GitReadFile(BaseModel):
    """
    Represents the input schema for the `read_file` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    file_path: str = Field(description="The path to the file to read, relative to the repository's working directory.")

class WriteToFile(BaseModel):
    """
    Represents the input schema for the `write_to_file` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory.")
    file_path: str = Field(description="The path to the file to write to, relative to the repository's working directory. The file will be created if it doesn't exist. Existing files are never overwritten unless overwrite=true is explicitly provided.")
    content: str = Field(description="The string content to write to the specified file.")
    overwrite: bool = Field(
        default=False,
        description="If true, allows overwriting an existing file. Defaults to false."
    )

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
    continue_thread: bool = Field(description="Required. Whether to continue the Aider thread by restoring chat history. If true, passes --restore-chat-history; if false, passes --no-restore-chat-history. Clients must explicitly choose.")
    options: Optional[List[str]] = Field(
        None,
        description="Optional. A list of additional command-line options to pass directly to Aider. Each option should be a string."
    )
    edit_format: EditFormat = Field(
        EditFormat.DIFF,
        description=(
            "Optional. The format Aider should use for edits. "
            "If not explicitly provided, the default is selected based on the model name: "
            "if the model includes 'gemini', defaults to 'diff-fenced'; "
            "if the model includes 'gpt', defaults to 'udiff'; "
            "otherwise defaults to 'diff'. "
            "Options: 'diff', 'diff-fenced', 'udiff', 'whole'."
        ),
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
    DIFF = "git_diff"
    STAGE_AND_COMMIT = "git_stage_and_commit"
    LOG = "git_log"
    BRANCH = "git_branch"
    SHOW = "git_show"
    APPLY_DIFF = "git_apply_diff"
    READ_FILE = "read_file"
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


def git_diff(repo: git.Repo, target: Optional[str] = None, path: Optional[str] = None) -> str:
    """
    Shows differences in the working directory. If target is None, shows worktree vs index.
    If target is provided, shows differences against that target.
    If path is provided, limits the diff to that specific file or directory.

    Args:
        repo: The Git repository object.
        target: Optional. The target (branch, commit hash, etc.) to diff against.
                If None, behaves like `git diff` (worktree vs index).
        path: Optional. Limit the diff to a specific file or directory path.

    Returns:
        A string representing the output of `git diff` or `git diff <target>`.
    """
    args = [target] if target else []
    if path:
        args.extend(['--', path])
    return repo.git.diff(*args)

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
    log: List[str] = []
    for commit in commits:
        log.append(
            f"Commit: {commit.hexsha}\n"
            f"Author: {commit.author}\n"
            f"Date: {commit.authored_datetime}\n"
            f"Message: {str(commit.message)}\n"
        )
    return log

def git_branch(repo: git.Repo, action: str, branch_name: str | None = None, base_branch: str | None = None, new_name: str | None = None) -> str:
    """Create, checkout, rename, or list branches on the given repo.

    - action='create': creates branch_name at base_branch (or current HEAD if None)
    - action='checkout': checks out branch_name
    - action='rename': renames branch_name to new_name
    - action='list': lists all branches with current branch marked
    """
    if action == 'create':
        if not branch_name:
            raise ValueError("branch_name is required for 'create' action")
        if base_branch:
            repo.create_head(branch_name, base_branch)
            return f"Created branch '{branch_name}' from '{base_branch}'"
        else:
            repo.create_head(branch_name)
            return f"Created branch '{branch_name}'"
    elif action == 'checkout':
        if not branch_name:
            raise ValueError("branch_name is required for 'checkout' action")
        repo.git.checkout(branch_name)
        return f"Switched to branch '{branch_name}'"
    elif action == 'rename':
        if not branch_name:
            raise ValueError("branch_name is required for 'rename' action")
        if not new_name:
            raise ValueError("new_name is required for 'rename' action")
        repo.git.branch('-m', branch_name, new_name)
        return f"Renamed branch '{branch_name}' to '{new_name}'"
    elif action == 'list':
        branches: List[str] = []
        try:
            current_branch = repo.active_branch.name
        except TypeError:  # detached HEAD
            current_branch = None
            
        for head in repo.heads:
            if head.name == current_branch:
                branches.append(f"* {head.name}")
            else:
                branches.append(f"  {head.name}")
        return "Branches:\n" + "\n".join(branches)
    else:
        raise ValueError("Invalid action. Must be 'create', 'checkout', 'rename', or 'list'.")

def git_show(repo: git.Repo, revision: str, path: Optional[str] = None, show_metadata_only: bool = False, show_diff_only: bool = False) -> str:
    """
    Shows the contents (metadata and diff) of a specific commit or range of commits.
    For commit ranges (e.g., "A..B" or "A...B"), returns the raw git show output.

    Args:
        repo: The Git repository object.
        revision: The commit hash/reference or range to show.
        path: Optional. Filter the output to show only changes for a specific file path.
        show_metadata_only: If true, only show commit metadata without diff content.
        show_diff_only: If true, only show diff content without commit metadata.

    Returns:
        A string containing the commit details and its diff, or raw git show output for ranges.
    """
    if ".." in revision:
        # Handle commit ranges with raw git show
        args = [revision]
        
        # Apply flags based on options
        if show_metadata_only and not show_diff_only:
            args.append("--no-patch")  # Only show commit metadata
        elif show_diff_only and not show_metadata_only:
            args.append("--format=")   # Suppress commit metadata, show only diff
        
        # Add path filter if provided
        if path:
            args.extend(["--", path])
            
        return repo.git.show(*args)
    else:
        # Handle single commit with structured output
        commit = repo.commit(revision)
        
        # Build metadata section
        metadata = [
            f"Commit: {commit.hexsha}\n"
            f"Author: {commit.author}\n"
            f"Date: {commit.authored_datetime}\n"
            f"Message: {str(commit.message)}\n"
        ]
        metadata_str = "".join(metadata)
        
        # If only metadata requested, return early
        if show_metadata_only and not show_diff_only:
            return metadata_str
        
        # Compute diff section
        if commit.parents:
            parent = commit.parents[0]
            diff = parent.diff(commit, create_patch=True, paths=path)
        else:
            diff = commit.diff(git.NULL_TREE, create_patch=True, paths=path)
        
        diff_lines: List[str] = []
        for d in diff:
            diff_lines.append(f"\n--- {d.a_path}\n+++ {d.b_path}\n")
            if d.diff is not None:
                if isinstance(d.diff, bytes):
                    diff_lines.append(d.diff.decode('utf-8'))
                else:
                    diff_lines.append(str(d.diff))
        diff_str = "".join(diff_lines)
        
        # Return based on options
        if show_diff_only and not show_metadata_only:
            return diff_str
        else:
            return metadata_str + diff_str

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
    full_affected_path = None
    original_content = ""

    # Attempt to extract affected file path from the diff content
    match = re.search(r"--- a/(.+)", diff_content)
    if match:
        affected_file_path = match.group(1).strip()
    else:
        match = re.search(r"\+\+\+ b/(.+)", diff_content)
        if match:
            affected_file_path = match.group(1).strip()
        else:
            affected_file_path = None

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

        if affected_file_path and full_affected_path:
            with open(full_affected_path, 'r') as f:
                new_content = f.read()

            result_message += await _generate_diff_output(original_content, new_content, affected_file_path)
            result_message += await _run_tsc_if_applicable(str(repo.working_dir), affected_file_path)

        return result_message
    except GitCommandError as gce:
        return ai_hint_git_apply_diff_error(gce.stderr, affected_file_path)
    except Exception as e:
        return f"UNEXPECTED_ERROR: An unexpected error occurred while applying diff: {e}. AI_HINT: Check the server logs for more details or review your input."
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def read_file_content(repo: git.Repo, file_path: str) -> str:
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
        return ai_hint_read_file_error(file_path, str(repo.working_dir), e)

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


async def write_to_file_content(repo_path: str, file_path: str, content: str, overwrite: bool = False) -> str:
    """
    Writes content to a specified file, creating it if it doesn't exist.
    Includes a check to ensure the content was written correctly and generates a diff.

    Args:
        repo_path: The path to the repository's working directory.
        file_path: The path to the file to write to, relative to the repository.
        content: The string content to write to the file.
        overwrite: If True, allows overwriting existing files. Defaults to False.

    Returns:
        A string indicating the success of the write operation, including diff and TSC output,
        or an error message.
    """
    try:
        full_file_path = Path(repo_path) / file_path
        
        # Check if file exists and overwrite protection is enabled
        file_existed = full_file_path.exists()
        if file_existed and not overwrite:
            return f"OVERWRITE_PROTECTED: File already exists: {file_path}. Pass overwrite=true to overwrite."
        
        original_content = ""
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
        return ai_hint_write_error(repo_path, file_path, e)

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
        return ai_hint_exec_error(repo_path, command, e)

async def ai_edit(
    repo_path: str,
    message: str,
    session: ServerSession,
    files: List[str],
    options: Optional[list[str]],
    continue_thread: bool,
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

    load_aider_config(directory_path, config_file)
    load_dotenv_file(directory_path, env_file)

    aider_options: Dict[str, Any] = {}
    aider_options["yes_always"] = True

    # Prune Aider chat history if not continuing thread
    if not continue_thread:
        history_path = Path(directory_path) / ".aider.chat.history.md"
        if history_path.is_file():
            try:
                history_path.write_text("", encoding="utf-8")
                logger.info(f"[ai_edit] Cleared Aider chat history at: {history_path}")
            except OSError as e:
                logger.warning(f"[ai_edit] Failed to clear Aider chat history: {e}")

    # Determine the default edit format based on the model if not explicitly provided
    if edit_format == EditFormat.DIFF:
        model_name = aider_options.get("model", "").lower()
        if "gemini" in model_name:
            edit_format_str = EditFormat.DIFF_FENCED.value
        elif "gpt" in model_name:
            edit_format_str = EditFormat.UDIFF.value
        else:
            edit_format_str = EditFormat.DIFF.value

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

    # Enforce explicit restore_chat_history flag based on required parameter (continue_thread),
    # overriding any contradictory option passed via `options`.
    aider_options["restore_chat_history"] = continue_thread

    for fname in files:
        fpath = os.path.join(directory_path, fname)
        if not os.path.isfile(fpath):
            logger.error(f"[ai_edit] Provided file not found in repo: {fname}. Aider may fail.")

    original_dir = os.getcwd()
    pre_aider_commit_hash = None
    try:
        # Capture the current HEAD commit hash before Aider runs
        try:
            repo = git.Repo(directory_path)
            try:
                if repo.head.is_valid():
                    try:
                        pre_aider_commit_hash = repo.head.commit.hexsha
                        logger.debug(f"Pre-Aider HEAD commit: {pre_aider_commit_hash}")
                    except (ValueError, AttributeError, IndexError):
                        # Fallback: use git_log to get last commit hash
                        log_entries = git_log(repo, max_count=1)
                        if log_entries:
                            # Parse hash from "Commit: <hash>" line
                            first_line = log_entries[0].splitlines()[0]
                            if first_line.startswith("Commit: "):
                                pre_aider_commit_hash = first_line.split("Commit: ")[1].strip()
                                logger.debug(f"Pre-Aider HEAD commit (from git_log): {pre_aider_commit_hash}")
                            else:
                                logger.debug("git_log did not return a commit hash line.")
                        else:
                            logger.debug("git_log returned no entries; repository may be empty.")
                else:
                    logger.debug("Repository has no commits yet or detached HEAD before Aider.")
            except Exception as e:
                logger.debug(f"Error retrieving pre-Aider HEAD commit: {e}")
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
        
        logger.info(f"[ai_edit] Files passed to aider: {files}")
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
            "ai_edit",
            f"AIDER STDOUT:\n{stdout}",
            0.5
        )
        if stderr:
            await session.send_progress_notification(
                "ai_edit",
                f"AIDER STDERR:\n{stderr}",
                0.5
            )

        return_code = process.returncode
        if return_code != 0:
            logger.error(f"Aider process exited with code {return_code}")
            return f"Error: Aider process exited with code {return_code}.\nSTDERR:\n{stderr}"
        else:
            result_message = "Aider process completed."
            if "Applied edit to" in stdout:
                result_message = "Code changes completed and committed successfully."
                
                try:
                    # Re-initialize repo object to get latest state after Aider potentially made changes
                    repo = git.Repo(directory_path)
                    
                    post_aider_commit_hash = None
                    try:
                        if repo.head.is_valid():
                            try:
                                post_aider_commit_hash = repo.head.commit.hexsha
                                logger.debug(f"Post-Aider HEAD commit: {post_aider_commit_hash}")
                            except (ValueError, AttributeError, IndexError):
                                # Fallback: use git_log to get last commit hash
                                log_entries = git_log(repo, max_count=1)
                                if log_entries:
                                    first_line = log_entries[0].splitlines()[0]
                                    if first_line.startswith("Commit: "):
                                        post_aider_commit_hash = first_line.split("Commit: ")[1].strip()
                                        logger.debug(f"Post-Aider HEAD commit (from git_log): {post_aider_commit_hash}")
                                    else:
                                        logger.debug("git_log did not return a commit hash line.")
                                else:
                                    logger.debug("git_log returned no entries; repository may be empty.")
                        else:
                            logger.debug("Repository has no commits or detached HEAD after Aider.")
                    except Exception as e:
                        logger.debug(f"Error retrieving post-Aider HEAD commit: {e}")

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
                 result_message += (f"\nIt's unclear if changes were applied. Please verify the file manually.\n"
                                     f"You can also inspect .aider.chat.history.md in the repo root for Aider's chat log.\n"
                                     f"STDOUT:\n{stdout}")
            
            # Append the last Aider reply from chat history in all cases
            last_reply = _get_last_aider_reply(directory_path)
            if last_reply:
                result_message += f"\n\nAider's last reply:\n{last_reply}"

            return result_message

    except Exception as e:
        logger.error(f"An unexpected error occurred during ai_edit: {e}")
        return ai_hint_ai_edit_unexpected(e)
    finally:
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
            logger.debug(f"Restored working directory to: {original_dir}")

async def get_aider_status(
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
        stdout, stderr, _ = await run_command(command)
        
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
                name_stdout, _, _ = await run_command(name_cmd)
                result["git"]["remote_url"] = name_stdout.strip() if name_stdout else None
                
                branch_cmd = ["git", "branch", "--show-current"]
                branch_stdout, _, _ = await run_command(branch_cmd)
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
        return ai_hint_aider_status_error(e)

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
            name=GitTools.DIFF,
            description="Shows differences in the working directory. By default (without target), shows worktree vs index like `git diff`. Pass target='HEAD' for previous 'all changes vs HEAD' behavior. Optional path filter available.",
            inputSchema=GitDiff.model_json_schema(),
        ),
        Tool(
            name=GitTools.STAGE_AND_COMMIT,
            description="Stages specified files (or all changes if no files are specified) and then commits them to the repository with a given message. This creates a new commit in the Git history.",
            inputSchema=GitStageAndCommit.model_json_schema(),
        ),
        Tool(
            name=GitTools.LOG,
            description="Shows the commit history for the repository, listing recent commits with their hash, author, date, and message. The number of commits can be limited.",
            inputSchema=GitLog.model_json_schema(),
        ),
        Tool(
            name=GitTools.BRANCH,
            description="Create, checkout, rename, or list Git branches. Action may be 'create' with optional base_branch, 'checkout', 'rename' with new_name, or 'list' to show all branches with current marked by '*'.",
            inputSchema=GitBranch.model_json_schema(),
        ),
        Tool(
            name=GitTools.SHOW,
            description="Shows the metadata (author, date, message) and the diff of a specific commit. This allows inspection of changes introduced by a particular commit. Supports commit ranges like 'A..B' or 'A...B' as well. Optional path filter and metadata/diff-only options available.",
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
            name=GitTools.WRITE_TO_FILE,
            description="Writes the provided content to a specified file within the repository. If the file does not exist, it will be created. Existing files are never overwritten unless overwrite=true is explicitly provided. Includes a check to ensure content was written correctly and generates a diff.",
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
        if not mcp_server.request_context.session.check_client_capability(
            ClientCapabilities(roots=RootsCapability())
        ):
            return []

        roots_result: ListRootsResult = await mcp_server.request_context.session.list_roots()
        logger.debug(f"Roots result: {roots_result}")
        repo_paths: List[str] = []
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
async def call_tool(name: str, arguments: Dict[str, Any]) -> list[Content]:
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

        def _repo_path_error(bad_value: str) -> list[Content]:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"ERROR: The repo_path parameter cannot be '{bad_value}'. Please provide the full absolute path to the repository. "
                        f"You must always resolve and pass the full path, not a value like '{bad_value}'. This is required for correct operation."
                    )
                )
            ]

        repo_path_arg = str(arguments.get("repo_path", ".")).strip()

        # Common agent mistakes and heuristics
        # 1) Relative cwd
        if repo_path_arg == ".":
            return _repo_path_error(repo_path_arg)
        # 2) Container default working dir (not the actual project path)
        if repo_path_arg in {"/workspace", "/workspace/"}:
            return _repo_path_error(repo_path_arg)
        # 3) Tilde shortcuts (require expansion on client side)
        if repo_path_arg.startswith("~"):
            return [
                TextContent(
                    type="text",
                    text=(
                        f"ERROR: The repo_path '{repo_path_arg}' uses '~' which must be expanded on your side. "
                        "Please pass the full absolute path (e.g., /home/you/project), not a value like '~' or '~/project'."
                    ),
                )
            ]
        # 4) URL/URI style or placeholders/env vars or relative (AI patterns)
        if repo_path_arg.startswith("file://"):
            return [
                TextContent(
                    type="text",
                    text=(
                        f"ERROR: The repo_path '{repo_path_arg}' looks like a URI. "
                        "Please pass a plain absolute filesystem path (e.g., /abs/path/to/project)."
                    ),
                )
            ]
        # 5) Relative paths like './repo', '../repo', 'repo'
        if not PurePath(repo_path_arg).is_absolute():
            return [
                TextContent(
                    type="text",
                    text=(
                        f"ERROR: The repo_path '{repo_path_arg}' is a relative path. "
                        "Always pass the full absolute path to the repository (e.g., /abs/path/to/project)."
                    ),
                )
            ]
        # 6) Env var or placeholder patterns that AIs sometimes emit
        if any(token in repo_path_arg for token in ["${", "$PWD", "$CWD", "<", ">", "{", "}"]):
            return [
                TextContent(
                    type="text",
                    text=(
                        f"ERROR: The repo_path '{repo_path_arg}' appears to contain a placeholder or environment variable reference. "
                        "Resolve it to a concrete absolute path before calling this tool."
                    ),
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
                case GitTools.DIFF:
                    repo = git.Repo(repo_path)
                    target = arguments.get("target")
                    path = arguments.get("path")
                    diff = git_diff(repo, target, path)
                    diff_header = f"Diff with {target}:" if target else "Diff of unstaged changes (worktree vs index):"
                    if path:
                        diff_header = f"Diff with {target} for path {path}:" if target else f"Diff of unstaged changes (worktree vs index) for path {path}:"
                    return [TextContent(
                        type="text",
                        text=f"{diff_header}\n{diff}"
                    )]
                case GitTools.STAGE_AND_COMMIT:
                    repo = git.Repo(repo_path)
                    result = git_stage_and_commit(repo, arguments["message"], arguments.get("files"))
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
                case GitTools.BRANCH:
                    repo = git.Repo(repo_path)
                    result = git_branch(
                        repo,
                        arguments["action"],
                        arguments.get("branch_name"),
                        arguments.get("base_branch"),
                        arguments.get("new_name")
                    )
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case GitTools.SHOW:
                    repo = git.Repo(repo_path)
                    result = git_show(
                        repo, 
                        arguments["revision"],
                        path=arguments.get("path"),
                        show_metadata_only=arguments.get("show_metadata_only", False),
                        show_diff_only=arguments.get("show_diff_only", False)
                    )
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
                    result = read_file_content(repo, arguments["file_path"])
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case GitTools.WRITE_TO_FILE:
                    logging.debug(f"Content input to write_to_file: {arguments['content']}")
                    result = await write_to_file_content(
                        repo_path=str(repo_path),
                        file_path=arguments["file_path"],
                        content=arguments["content"],
                        overwrite=arguments.get("overwrite", False)
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
                    if "continue_thread" not in arguments:
                        return [TextContent(
                            type="text",
                            text=(
                                "ERROR: The 'continue_thread' boolean parameter is required for ai_edit. "
                                "Set it to true to pass --restore-chat-history (continue Aider thread), "
                                "or false to pass --no-restore-chat-history (start without restoring chat)."
                            )
                        )]
                    continue_thread = bool(arguments["continue_thread"])
                    result = await ai_edit(
                        repo_path=str(repo_path),
                        message=message,
                        session=mcp_server.request_context.session,
                        files=files,
                        options=options,
                        continue_thread=continue_thread,
                        edit_format=EditFormat(arguments.get("edit_format", EditFormat.DIFF.value)),
                    )
                    return [TextContent(
                        type="text",
                        text=result
                    )]
                case GitTools.AIDER_STATUS:
                    check_environment = arguments.get("check_environment", True)
                    result = await get_aider_status(
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
                # Reuse the dynamic error for '.' since that's the implicit case here
                return _repo_path_error(".")
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
                    text=ai_hint_unexpected_call_tool(e)
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

async def handle_sse(request: Request):
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

async def handle_post_message(scope: Scope, receive: Receive, send: Send):
    """
    Handles incoming POST messages from MCP clients, typically used for client-to-server communication.

    Args:
        scope: The ASGI scope dictionary.
        receive: The ASGI receive callable.
        send: The ASGI send callable.
    """
    await sse_transport.handle_post_message(scope, receive, send)

routes: List[Union[Route, Mount]] = [
    Route("/sse", endpoint=handle_sse, methods=["GET"]),
    Mount(POST_MESSAGE_ENDPOINT, app=handle_post_message),
]

app = Starlette(routes=cast(Any, routes))

if __name__ == "__main__":
    # To run the server, you would typically use uvicorn:
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    pass
