import logging
from pathlib import Path
from typing import Sequence
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
    try:
        # Dry-run validation before actual application
        repo.git.apply(_input=diff_content, check=True)
        # Apply with three-way merge for conflict resolution
        repo.git.apply(_input=diff_content, threeway=True)
        return "Diff applied successfully."
    except GitCommandError as gce:
        return f"Error applying diff: {gce.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def git_read_file(repo: git.Repo, file_path: str) -> str:
    try:
        full_path = Path(repo.working_dir) / file_path
        with open(full_path, 'r') as f:
            content = f.read()
        return f"Content of {file_path}:\n{content}"
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

def git_stage_all(repo: git.Repo) -> str:
    try:
        repo.git.add(A=True)
        return "All files staged successfully."
    except git.GitCommandError as e:
        return f"Error staging all files: {e.stderr}"

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
    repo_path = Path(arguments["repo_path"])
    repo = git.Repo(repo_path)

    match name:
        case GitTools.STATUS:
            status = git_status(repo)
            return [TextContent(
                type="text",
                text=f"Repository status:\n{status}"
            )]

        case GitTools.DIFF_UNSTAGED:
            diff = git_diff_unstaged(repo)
            return [TextContent(
                type="text",
                text=f"Unstaged changes:\n{diff}"
            )]

        case GitTools.DIFF_STAGED:
            diff = git_diff_staged(repo)
            return [TextContent(
                type="text",
                text=f"Staged changes:\n{diff}"
            )]

        case GitTools.DIFF:
            diff = git_diff(repo, arguments["target"])
            return [TextContent(
                type="text",
                text=f"Diff with {arguments['target']}:\n{diff}"
            )]

        case GitTools.COMMIT:
            result = git_commit(repo, arguments["message"])
            return [TextContent(
                type="text",
                text=result
            )]

        case GitTools.ADD:
            result = git_add(repo, arguments["files"])
            return [TextContent(
                type="text",
                text=result
            )]

        case GitTools.RESET:
            result = git_reset(repo)
            return [TextContent(
                type="text",
                text=result
            )]

        case GitTools.LOG:
            log = git_log(repo, arguments.get("max_count", 10))
            return [TextContent(
                type="text",
                text="Commit history:\n" + "\n".join(log)
            )]

        case GitTools.CREATE_BRANCH:
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
            result = git_checkout(repo, arguments["branch_name"])
            return [TextContent(
                type="text",
                text=result
            )]

        case GitTools.SHOW:
            result = git_show(repo, arguments["revision"])
            return [TextContent(
                type="text",
                text=result
            )]

        case GitTools.APPLY_DIFF:
            result = git_apply_diff(repo, arguments["diff_content"])
            return [TextContent(
                type="text",
                text=result
            )]

        case GitTools.READ_FILE:
            result = git_read_file(repo, arguments["file_path"])
            return [TextContent(
                type="text",
                text=result
            )]

        case GitTools.STAGE_ALL:
            result = git_stage_all(repo)
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
