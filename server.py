import logging
from pathlib import Path
from typing import Sequence, Optional
from mcp.server import Server
from mcp.server.session import ServerSession
from mcp.server.sse import SseServerTransport
from mcp.types import (
    ClientCapabilities,
    TextContent,
    Tool,
    ListRootsResult,
    RootsCapability,
)
from enum import Enum
import git
from git.exc import GitCommandError
from pydantic import BaseModel
import asyncio
import tempfile
import os # Ensure os is imported
import re
import difflib # Import difflib
import shlex # Import shlex for shell quoting

# Configure logging to show DEBUG messages
logging.basicConfig(level=logging.DEBUG)

# Import Starlette and Route
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response

logger = logging.getLogger(__name__)

class GitStatus(BaseModel):
    repo_path: str

class GitDiffUnstaged(BaseModel):
    repo_path: str

class GitDiffStaged(BaseModel):
    repo_path: str

class GitDiff(BaseModel):
    repo_path: str
    target: str

class GitCommit(BaseModel):
    repo_path: str
    message: str

class GitAdd(BaseModel):
    repo_path: str
    files: list[str]

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

class GitStageAll(BaseModel):
    repo_path: str

class SearchAndReplace(BaseModel):
    repo_path: str # Added repo_path to the model
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

class GitTools(str, Enum):
    STATUS = "git_status"
    DIFF_UNSTAGED = "git_diff_unstaged"
    DIFF_STAGED = "git_diff_staged"
    DIFF = "git_diff"
    COMMIT = "git_commit"
    ADD = "git_add"
    RESET = "git_reset"
    LOG = "git_log"
    CREATE_BRANCH = "git_create_branch"
    CHECKOUT = "git_checkout"
    SHOW = "git_show"
    APPLY_DIFF = "git_apply_diff"
    READ_FILE = "git_read_file"
    STAGE_ALL = "git_stage_all"
    SEARCH_AND_REPLACE = "search_and_replace"
    WRITE_TO_FILE = "write_to_file"
    EXECUTE_COMMAND = "execute_command"

def git_status(repo: git.Repo) -> str:
    return repo.git.status()

def git_diff_unstaged(repo: git.Repo) -> str:
    return repo.git.diff()

def git_diff_staged(repo: git.Repo) -> str:
    return repo.git.diff("--cached")

def git_diff(repo: git.Repo, target: str) -> str:
    return repo.git.diff(target)

def git_commit(repo: git.Repo, message: str) -> str:
    commit = repo.index.commit(message)
    return f"Changes committed successfully with hash {commit.hexsha}"

def git_add(repo: git.Repo, files: list[str]) -> str:
    repo.index.add(files)
    return "Files staged successfully"

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
            f"Message: {commit.message}\n"
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
        f"Message: {commit.message}\n"
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
                output.append(str(d.diff)) # Fallback for unexpected string type
    return "".join(output)

def git_apply_diff(repo: git.Repo, diff_content: str) -> str:
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(diff_content)
            tmp_file_path = tmp.name
        
        # Apply with more relaxed settings to handle potential issues
        repo.git.apply(
            '--check',
            '-3', # Changed from '--threeway' to '-3'
            '--whitespace=fix',
            '--allow-overlap',
            tmp_file_path
        )
            
        return "Diff applied successfully"
    except GitCommandError as gce:
        return f"Error applying diff: {gce.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
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
        return f"Error reading file {file_path}: {e}"

def git_stage_all(repo: git.Repo) -> str:
    try:
        repo.git.add(A=True)
        return "All files staged successfully."
    except git.GitCommandError as e:
        return f"Error staging all files: {e.stderr}"
async def _generate_diff_output(original_content: str, new_content: str, file_path: str) -> str:
    diff_lines = list(difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="" # Avoid adding extra newlines
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

        # Attempt literal search first
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
            # Read original content for diff generation
            original_content = "".join(lines)
            with open(full_file_path, 'w') as f:
                f.writelines(modified_lines_literal)
            
            result_message = f"Successfully replaced '{search_string}' with '{replace_string}' in {file_path} using literal search. Total changes: {changes_made_literal}."
            result_message += await _generate_diff_output(original_content, "".join(modified_lines_literal), file_path)
            result_message += await _run_tsc_if_applicable(repo_path, file_path)
            return result_message
        else:
            # If literal search yields no changes, attempt regex search
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
                # Read original content for diff generation
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
        return f"An unexpected error occurred: {e}"

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

    # --- Attempt using sed first ---
    sed_command_parts = ["flatpak-spawn", "--host", "sed", "-i"]

    # Use '#' as a delimiter for sed to avoid issues with '/' in search/replace strings
    # Escape '#' in search_string and replace_string if they exist
    sed_pattern = search_string.replace('#', '\#')
    sed_replacement = replace_string.replace('#', '\#').replace('&', '\&').replace('\\', '\\\\')

    sed_flags = "g"
    if ignore_case:
        sed_flags += "i"

    sed_sub_command = f"s#{sed_pattern}#{sed_replacement}#{sed_flags}"

    # Add line range if specified
    if start_line is not None and end_line is not None:
        sed_sub_command = f"{start_line},{end_line}{sed_sub_command}"
    elif start_line is not None:
        sed_sub_command = f"{start_line},${sed_sub_command}"
    elif end_line is not None:
        sed_sub_command = f"1,{end_line}{sed_sub_command}"

    # Enclose the sed substitution command in single quotes for the shell
    # and quote the file path
    sed_full_command = f"{' '.join(sed_command_parts)} '{sed_sub_command}' {shlex.quote(str(full_file_path))}"

    try:
        # Read original content to check for changes after sed
        with open(full_file_path, 'r') as f:
            original_content = f.read()

        sed_result = await execute_custom_command(repo_path, sed_full_command)
        logging.info(f"Sed command result: {sed_result}")

        # Check if sed command itself failed (e.g., sed not found, syntax error)
        if "Command failed with exit code" in sed_result or "Error executing command" in sed_result:
            logging.warning(f"Sed command failed: {sed_result}. Falling back to Python logic.")
            # Fallback to Python logic
            return await _search_and_replace_python_logic(repo_path, search_string, replace_string, file_path, ignore_case, start_line, end_line)
        
        # Read content after sed to check if changes were made
        with open(full_file_path, 'r') as f:
            modified_content_sed = f.read()

        if original_content != modified_content_sed:
            result_message = f"Successfully replaced '{search_string}' with '{replace_string}' in {file_path} using sed."
            result_message += await _generate_diff_output(original_content, modified_content_sed, file_path)
            result_message += await _run_tsc_if_applicable(repo_path, file_path)
            return result_message
        else:
            logging.info(f"Sed command executed but made no changes. Falling back to Python logic.")
            # Sed made no changes, fall back to Python logic to try literal/regex
            return await _search_and_replace_python_logic(repo_path, search_string, replace_string, file_path, ignore_case, start_line, end_line)

    except FileNotFoundError:
        return f"Error: File not found at {full_file_path}"
    except Exception as e:
        logging.error(f"An unexpected error occurred during sed attempt: {e}. Falling back to Python logic.")
        # Fallback to Python logic for any other unexpected errors
        return await _search_and_replace_python_logic(repo_path, search_string, replace_string, file_path, ignore_case, start_line, end_line)

async def write_to_file_content(repo_path: str, file_path: str, content: str) -> str:
    try:
        full_file_path = Path(repo_path) / file_path
        
        # Read original content if file exists
        original_content = ""
        file_existed = full_file_path.exists() # Check if file existed before writing
        if file_existed:
            with open(full_file_path, 'r') as f:
                original_content = f.read()

        full_file_path.parent.mkdir(parents=True, exist_ok=True) # Create parent directories if they don't exist
        with open(full_file_path, 'w', encoding='utf-8') as f: # Explicitly set encoding to utf-8
            f.write(content)
        
        # --- Debugging: Read back raw bytes and compare ---
        with open(full_file_path, 'rb') as f_read_back:
            written_bytes = f_read_back.read()
        
        logging.debug(f"Content input to write_to_file (repr): {content!r}")
        logging.debug(f"Raw bytes written to file: {written_bytes!r}")
        logging.debug(f"Input content encoded (UTF-8): {content.encode('utf-8')!r}")

        if written_bytes != content.encode('utf-8'):
            logging.error("Mismatch between input content and written bytes! File corruption detected during write.")
        # --- End Debugging ---

        result_message = ""
        if not file_existed: # If it's a new file
            result_message = f"Successfully created new file: {file_path}."
        else: # If file existed, generate diff
            # Generate diff
            result_message += await _generate_diff_output(original_content, content, file_path)

        result_message += await _run_tsc_if_applicable(repo_path, file_path)

        return result_message
    except Exception as e:
        return f"Error writing to file {file_path}: {e}"

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
        return f"Error executing command: {e}"

# Global MCP Server instance
mcp_server = Server("mcp-git")

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name=GitTools.STATUS,
            description="Shows the working tree status",
            inputSchema=GitStatus.schema(),
        ),
        Tool(
            name=GitTools.DIFF_UNSTAGED,
            description="Shows changes in the working directory that are not yet staged",
            inputSchema=GitDiffUnstaged.schema(),
        ),
        Tool(
            name=GitTools.DIFF_STAGED,
            description="Shows changes that are staged for commit",
            inputSchema=GitDiffStaged.schema(),
        ),
        Tool(
            name=GitTools.DIFF,
            description="Shows differences between branches or commits",
            inputSchema=GitDiff.schema(),
        ),
        Tool(
            name=GitTools.COMMIT,
            description="Records changes to the repository",
            inputSchema=GitCommit.schema(),
        ),
        Tool(
            name=GitTools.ADD,
            description="Adds file contents to the staging area",
            inputSchema=GitAdd.schema(),
        ),
        Tool(
            name=GitTools.RESET,
            description="Unstages all staged changes",
            inputSchema=GitReset.schema(),
        ),
        Tool(
            name=GitTools.LOG,
            description="Shows the commit logs",
            inputSchema=GitLog.schema(),
        ),
        Tool(
            name=GitTools.CREATE_BRANCH,
            description="Creates a new branch from an optional base branch",
            inputSchema=GitCreateBranch.schema(),
        ),
        Tool(
            name=GitTools.CHECKOUT,
            description="Switches branches",
            inputSchema=GitCheckout.schema(),
        ),
        Tool(
            name=GitTools.SHOW,
            description="Shows the contents of a commit",
            inputSchema=GitShow.schema(),
        ),
        Tool(
            name=GitTools.APPLY_DIFF,
            description="Applies a diff to the working directory",
            inputSchema=GitApplyDiff.schema(),
        ),
        Tool(
            name=GitTools.READ_FILE,
            description="Reads the content of a file in the repository",
            inputSchema=GitReadFile.schema(),
        ),
        Tool(
            name=GitTools.STAGE_ALL,
            description="Stages all changes in the working directory",
            inputSchema=GitStageAll.schema(),
        ),
        Tool(
            name=GitTools.SEARCH_AND_REPLACE,
            description="Searches for a string or regex pattern in a file and replaces it with another string.",
            inputSchema=SearchAndReplace.schema(),
        ),
        Tool(
            name=GitTools.WRITE_TO_FILE,
            description="Writes content to a specified file, creating it if it doesn't exist or overwriting it if it does.",
            inputSchema=WriteToFile.schema(),
        ),
        Tool(
            name=GitTools.EXECUTE_COMMAND,
            description="Executes a custom shell command within the specified repository path.",
            inputSchema=ExecuteCommand.schema(),
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
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    repo_path = Path(arguments.get("repo_path", ".")) # Default to current directory if repo_path is not provided
    
    repo = None # Initialize repo to None
    
    match name:
        case GitTools.STATUS:
            repo = git.Repo(repo_path)
            status = git_status(repo)
            return [TextContent(
                type="text",
                text=f"Repository status:\n{status}"
            )]

        case GitTools.DIFF_UNSTAGED:
            repo = git.Repo(repo_path)
            diff = git_diff_unstaged(repo)
            return [TextContent(
                type="text",
                text=f"Unstaged changes:\n{diff}"
            )]

        case GitTools.DIFF_STAGED:
            repo = git.Repo(repo_path)
            diff = git_diff_staged(repo)
            return [TextContent(
                type="text",
                text=f"Staged changes:\n{diff}"
            )]

        case GitTools.DIFF:
            repo = git.Repo(repo_path)
            diff = git_diff(repo, arguments["target"])
            return [TextContent(
                type="text",
                text=f"Diff with {arguments['target']}:\n{diff}"
            )]

        case GitTools.COMMIT:
            repo = git.Repo(repo_path)
            result = git_commit(repo, arguments["message"])
            return [TextContent(
                type="text",
                text=result
            )]

        case GitTools.ADD:
            repo = git.Repo(repo_path)
            result = git_add(repo, arguments["files"])
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
            result = git_apply_diff(repo, arguments["diff_content"])
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
        case GitTools.STAGE_ALL:
            repo = git.Repo(repo_path)
            result = git_stage_all(repo)
            return [TextContent(
                type="text",
                text=result
            )]
        
        case GitTools.SEARCH_AND_REPLACE:
            result = await search_and_replace_in_file( # Add await here
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
            result = await write_to_file_content( # Await this call
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

        case _:
            raise ValueError(f"Unknown tool: {name}")

# Define the endpoint for POST messages
POST_MESSAGE_ENDPOINT = "/messages/"

# Create an SSE transport instance
sse_transport = SseServerTransport(POST_MESSAGE_ENDPOINT)

# Define handler for SSE GET requests
async def handle_sse(request):
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        options = mcp_server.create_initialization_options()
        await mcp_server.run(read_stream, write_stream, options, raise_exceptions=True)
    return Response() # Return empty response to avoid NoneType error

# Define handler for client POST messages
async def handle_post_message(scope, receive, send):
    await sse_transport.handle_post_message(scope, receive, send)

# Create Starlette routes
routes = [
    Route("/sse", endpoint=handle_sse, methods=["GET"]),
    Mount(POST_MESSAGE_ENDPOINT, app=handle_post_message),
]

# Create the Starlette application
app = Starlette(routes=routes)

if __name__ == "__main__":
    # This block will be executed when the script is run directly.
    # Uvicorn will typically run the 'app' object.
    # For local testing, you might run uvicorn directly:
    # uvicorn server:app --host 127.0.0.1 --port 8000 --reload
    # However, the server.sh script will handle this.
    pass # Uvicorn will run the 'app'
