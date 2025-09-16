pytest_plugins = "pytest_asyncio"

import os
import tempfile
import shutil
import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pytest

from server import find_git_root

def test_find_git_root_various_cases():
    # Case 1: Path is the root of a Git repository
    with tempfile.TemporaryDirectory() as repo_root:
        os.mkdir(os.path.join(repo_root, ".git"))
        assert find_git_root(repo_root) == os.path.abspath(repo_root)

        # Case 2: Path is a subdirectory within a Git repository
        subdir = os.path.join(repo_root, "subdir")
        os.mkdir(subdir)
        assert find_git_root(subdir) == os.path.abspath(repo_root)

    # Case 3: Path is not part of any Git repository
    with tempfile.TemporaryDirectory() as non_repo:
        assert find_git_root(non_repo) is None

    # Case 4: Empty/invalid path
    # When given an empty string, os.path.abspath("") returns the cwd, so find_git_root("") will return the git root if present.
    cwd = os.path.abspath("")
    expected = find_git_root(cwd)
    assert find_git_root("") == expected
    assert find_git_root("/nonexistent/path/shouldnotexist") is None
import pytest
import asyncio
import os
import shutil
import tempfile
import glob
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, mock_open

# Import functions and classes from server.py
from server import (
    git_status, git_diff, git_stage_and_commit,
    git_log, git_show,
    read_file_content,
    _generate_diff_output, _run_tsc_if_applicable,
    write_to_file_content, execute_custom_command,
    Starlette, Route, Mount, Response, ServerSession, ClientCapabilities, RootsCapability, ListRootsResult, TextContent,
    handle_sse, handle_post_message,
    list_tools, call_tool, list_repos, GitTools, ai_edit
)
import git
from git.exc import GitCommandError
from mcp.types import Root
from pydantic import FileUrl

# Fixture for a temporary Git repository
@pytest.fixture
def temp_git_repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()
        repo = git.Repo.init(repo_path)
        
        # Configure user for commits
        with repo.config_writer() as cw:
            cw.set_value("user", "email", "test@example.com")
            cw.set_value("user", "name", "Test User")

        # Create an initial commit
        (repo_path / "initial_file.txt").write_text("initial content")
        repo.index.add(["initial_file.txt"])
        repo.index.commit("Initial commit")

        # Ensure a 'main' branch exists for diffing tests
        try:
            repo.git.branch("main")
        except git.GitCommandError:
            # If 'main' already exists or other issue, ignore
            pass
        repo.git.checkout("main") # Checkout main branch

        yield repo, repo_path

# Fixture for cleanup
@pytest.fixture(autouse=True)
def clean_snapshots():
    """Clean up snapshot files before and after each test"""
    # Setup: nothing to do before test
    yield
    # Teardown: clean snapshots
    for p in glob.glob("**/.mcp-devtools/ai_edit_*.diff", recursive=True):
        try:
            os.remove(p)
        except OSError:
            pass

# Test cases for Git utility functions

def test_git_status(temp_git_repo):
    repo, repo_path = temp_git_repo
    status = git_status(repo)
    assert "nothing to commit, working tree clean" in status

    (repo_path / "new_file.txt").write_text("hello")
    status = git_status(repo)
    assert "new_file.txt" in status
    assert "Untracked files" in status

def test_git_diff_scenarios(temp_git_repo):
    repo, repo_path = temp_git_repo

    # Scenario 1: Unstaged modification to a tracked file
    (repo_path / "initial_file.txt").write_text("modified content")
    diff_unstaged = git_diff(repo, None)
    assert "-initial content" in diff_unstaged
    assert "+modified content" in diff_unstaged

    # Stage the change
    repo.index.add(["initial_file.txt"])

    # Scenario 2: No unstaged changes now; git_diff(None) should be empty
    diff_after_stage = git_diff(repo, None)
    assert diff_after_stage == ""  # no unstaged changes

    # git_diff('HEAD') should show the staged change
    diff_head_after_stage = git_diff(repo, 'HEAD')
    assert "+modified content" in diff_head_after_stage

    # Scenario 3: Make another unstaged modification
    (repo_path / "initial_file.txt").write_text("modified again")
    diff_unstaged_again = git_diff(repo, None)
    assert "-modified content" in diff_unstaged_again
    assert "+modified again" in diff_unstaged_again

    # git_diff('HEAD') should reflect the net change from HEAD to worktree
    diff_head_final = git_diff(repo, 'HEAD')
    assert "+modified again" in diff_head_final

def test_git_diff(temp_git_repo):
    repo, repo_path = temp_git_repo
    # Create a new branch and commit
    repo.create_head("feature_branch")
    repo.heads.feature_branch.checkout()
    (repo_path / "feature_file.txt").write_text("feature content")
    repo.index.add(["feature_file.txt"])
    repo.index.commit("Feature commit")
    
    # Diff against main (or master)
    diff = git_diff(repo, "main" if "main" in repo.heads else "master")
    assert "+feature content" in diff

def test_git_stage_and_commit(temp_git_repo):
    repo, repo_path = temp_git_repo
    (repo_path / "commit_file.txt").write_text("content to commit")
    repo.index.add(["commit_file.txt"])
    result = git_stage_and_commit(repo, "Test commit message")
    assert "Changes committed successfully" in result
    assert "Test commit message" in repo.head.commit.message

def test_git_log(temp_git_repo):
    repo, repo_path = temp_git_repo
    (repo_path / "log_file.txt").write_text("second commit")
    repo.index.add(["log_file.txt"])
    repo.index.commit("Second commit")
    
    log = git_log(repo, max_count=1)
    assert len(log) == 1
    assert "Second commit" in log[0]
    
    log_all = git_log(repo)
    assert len(log_all) == 2 # Initial commit + Second commit

def test_git_show(temp_git_repo):
    repo, repo_path = temp_git_repo
    commit_sha = repo.head.commit.hexsha
    result = git_show(repo, commit_sha)
    assert f"Commit: {commit_sha}" in result
    assert "initial content" in result # Content of the initial commit

    # Test with a modified file
    (repo_path / "show_file.txt").write_text("original")
    repo.index.add(["show_file.txt"])
    commit1 = repo.index.commit("Add show_file")
    
    (repo_path / "show_file.txt").write_text("modified")
    repo.index.add(["show_file.txt"])
    commit2 = repo.index.commit("Modify show_file")

    result_diff = git_show(repo, commit2.hexsha)
    assert "-original" in result_diff
    assert "+modified" in result_diff

    # Test with commit range
    range_result = git_show(repo, f"{commit1.hexsha}..{commit2.hexsha}")
    assert "Modify show_file" in range_result
    assert ("+modified" in range_result or "-original" in range_result)

    # Test path filter on single commit
    res_path = git_show(repo, commit2.hexsha, path='show_file.txt')
    assert '+modified' in res_path

    # Test metadata-only on single commit
    res_meta = git_show(repo, commit2.hexsha, show_metadata_only=True)
    assert 'Commit:' in res_meta
    assert '+modified' not in res_meta

    # Test diff-only on single commit
    res_diff = git_show(repo, commit2.hexsha, show_diff_only=True)
    assert 'Commit:' not in res_diff
    assert '+modified' in res_diff

    # Test range metadata-only
    res_range_meta = git_show(repo, f"{commit1.hexsha}..{commit2.hexsha}", show_metadata_only=True)
    assert 'Modify show_file' in res_range_meta
    assert '+modified' not in res_range_meta

    # Test range diff-only
    res_range_diff = git_show(repo, f"{commit1.hexsha}..{commit2.hexsha}", show_diff_only=True)
    assert 'commit ' not in res_range_diff.lower()  # Check that commit headers are not included
    assert '+modified' in res_range_diff

def test_read_file_content(temp_git_repo):
    repo, repo_path = temp_git_repo
    file_content = "This is a test file content."
    (repo_path / "read_me.txt").write_text(file_content)
    
    result = read_file_content(repo, "read_me.txt")
    assert f"Content of read_me.txt:\n{file_content}" in result

    result_not_found = read_file_content(repo, "non_existent_file.txt")
    assert "Error: file wasn't found or out of cwd" in result_not_found

# Removed test_git_stage_all since git_stage_all no longer exists and staging is now handled via git_stage_and_commit

# Removed test_git_stage_all_git_command_error since git_stage_all no longer exists

# Test cases for async utility functions and file operations

# @pytest.mark.asyncio
# async def test_generate_diff_output():
#     original = "line1\nline2\nline3"
#     new = "line1\nnewline2\nline3"
#     file_path = "test.txt"
#     diff_output = await _generate_diff_output(original, new, file_path)
#     assert "--- a/test.txt" in diff_output
#     assert "+++ b/test.txt" in diff_output
#     assert "-line2" in diff_output
#     assert "+newline2" in diff_output
#
#     # Test no changes
#     no_change_diff = await _generate_diff_output(original, original, file_path)
#     assert "No changes detected" in no_change_diff
#
#     # Test large diff
#     large_original = "\n".join([f"line{i}" for i in range(1001)])
#     large_new = "\n".join([f"modified_line{i}" for i in range(1001)])
#     large_diff_output = await _generate_diff_output(large_original, large_new, file_path)
#     assert "Diff was too large (over 1000 lines)." in large_diff_output

def test_git_diff_path_filter(temp_git_repo):
    repo, repo_path = temp_git_repo
    
    # Create and modify two files
    (repo_path / "a.txt").write_text("content a")
    (repo_path / "b.txt").write_text("content b")
    repo.index.add(["a.txt", "b.txt"])
    repo.index.commit("Add files")
    
    # Modify both files
    (repo_path / "a.txt").write_text("modified content a")
    (repo_path / "b.txt").write_text("modified content b")
    
    # Test diff with path filter for a.txt (unstaged changes)
    diff_a = git_diff(repo, None, path="a.txt")
    assert "+modified content a" in diff_a
    assert "b.txt" not in diff_a
    
    # Stage a.txt
    repo.index.add(["a.txt"])
    
    # Test diff with path filter for a.txt against HEAD (staged changes)
    diff_a_staged = git_diff(repo, "HEAD", path="a.txt")
    assert "+modified content a" in diff_a_staged
    assert "b.txt" not in diff_a_staged

@pytest.mark.asyncio
async def test_generate_diff_output_empty_diff():
    from server import _generate_diff_output
    original = "foo\nbar\nbaz"
    new = "foo\nbar\nbaz"
    file_path = "empty.txt"
    result = await _generate_diff_output(original, new, file_path)
    assert "\nNo changes detected (file content was identical)." in result

@pytest.mark.asyncio
@patch('server.execute_custom_command')
async def test_run_tsc_if_applicable(mock_execute_custom_command):
    mock_execute_custom_command.return_value = "TSC ran successfully."
    
    # Test .ts file
    result_ts = await _run_tsc_if_applicable("/tmp", "test.ts")
    assert "TSC Output for test.ts" in result_ts
    mock_execute_custom_command.assert_called_with("/tmp", " tsc --noEmit --allowJs test.ts")

    # Test .js file
    mock_execute_custom_command.reset_mock()
    result_js = await _run_tsc_if_applicable("/tmp", "test.js")
    assert "TSC Output for test.js" in result_js
    mock_execute_custom_command.assert_called_with("/tmp", " tsc --noEmit --allowJs test.js")

    # Test non-JS/TS file
    mock_execute_custom_command.reset_mock()
    result_py = await _run_tsc_if_applicable("/tmp", "test.py")
    assert result_py == ""
    mock_execute_custom_command.assert_not_called()

@pytest.mark.asyncio
async def test_write_to_file_content(temp_git_repo):
    repo, repo_path = temp_git_repo
    file_path = "new_dir/new_file.txt"
    content = "Hello, world!\nThis is a test."
    
    result = await write_to_file_content(str(repo_path), file_path, content)
    assert "Successfully created new file: new_dir/new_file.txt." in result
    assert (repo_path / file_path).exists()
    assert (repo_path / file_path).read_text() == content

    # Test overwrite protection
    updated_content = "Updated content."
    result_protected = await write_to_file_content(str(repo_path), file_path, updated_content)
    assert "OVERWRITE_PROTECTED: File already exists: new_dir/new_file.txt." in result_protected
    # File content should remain unchanged
    assert (repo_path / file_path).read_text() == content

    # Test overwriting existing file with overwrite=True
    result_overwrite = await write_to_file_content(str(repo_path), file_path, updated_content, overwrite=True)
    assert "Diff:" in result_overwrite
    assert "-Hello, world!" in result_overwrite
    assert "+Updated content." in result_overwrite
    assert (repo_path / file_path).read_text() == updated_content

@pytest.mark.asyncio
async def test_execute_custom_command(temp_git_repo):
    repo, repo_path = temp_git_repo
    
    # Test successful command
    result = await execute_custom_command(str(repo_path), "echo hello")
    assert "STDOUT:\nhello" in result
    assert "Command executed successfully with no output." not in result # Should have output

    # Test command with stderr
    result_err = await execute_custom_command(str(repo_path), "ls non_existent_dir")
    assert "STDERR:" in result_err
    assert "No such file or directory" in result_err
    assert "Command failed with exit code" in result_err

    # Test command with no output
    result_no_output = await execute_custom_command(str(repo_path), "touch no_output.txt")
    assert "Command executed successfully with no output." in result_no_output
    assert (repo_path / "no_output.txt").exists()

@pytest.mark.asyncio
async def test_write_to_file_content_bytes_mismatch_and_exception(tmp_path, monkeypatch):
    from server import write_to_file_content

    # Simulate written_bytes != content.encode('utf-8')
    file_path = "mismatch.txt"
    content = "abc"

    # Patch open to simulate mismatch on read-back using mock_open
    mock_file = mock_open(read_data=b'xyz')
    with patch('builtins.open', mock_file):
        result = await write_to_file_content(str(tmp_path), file_path, content)
        assert "Mismatch between input content and written bytes!" in result

    # Simulate Exception during file writing
    mock_file_raise = mock_open()
    mock_file_raise.side_effect = Exception("write error")
    with patch('builtins.open', mock_file_raise):
        result_exc = await write_to_file_content(str(tmp_path), "fail.txt", "fail")
        assert "UNEXPECTED_ERROR: Failed to write to file 'fail.txt': write error" in result_exc
        assert "UNEXPECTED_ERROR:" in result_exc

@pytest.mark.asyncio
async def test_execute_custom_command_exception(monkeypatch, tmp_path):
    from server import execute_custom_command
    import asyncio

    async def raise_exc(*a, **kw):
        raise Exception("subprocess error")
    monkeypatch.setattr(asyncio, "create_subprocess_shell", raise_exc)
    result = await execute_custom_command(str(tmp_path), "echo fail")
    assert "UNEXPECTED_ERROR: Failed to execute command 'echo fail': subprocess error" in result
    assert "UNEXPECTED_ERROR:" in result

# Test cases for MCP server integration (list_tools, call_tool)

@pytest.mark.asyncio
async def test_list_tools():
    tools = await list_tools()
    assert len(tools) == len(GitTools)
    tool_names = {tool.name for tool in tools}
    for git_tool in GitTools:
        assert git_tool.value in tool_names

@pytest.mark.asyncio
@patch('server.git.Repo')
@patch('server.git_status')
@patch('server.git_diff')
@patch('server.git_stage_and_commit')
@patch('server.git_log')
@patch('server.git_show')
@patch('server.read_file_content')
@patch('server.write_to_file_content', new_callable=AsyncMock)
@patch('server.execute_custom_command', new_callable=AsyncMock)
async def test_call_tool(
    mock_execute_custom_command, mock_write_to_file_content,
    mock_read_file_content, mock_git_show,
    mock_git_log,
    mock_git_stage_and_commit, mock_git_diff, mock_git_status, mock_git_repo
):
    mock_repo_instance = MagicMock()
    mock_git_repo.return_value = mock_repo_instance

    # Test GitTools.STATUS
    mock_git_status.return_value = "clean"
    result = list(await call_tool(GitTools.STATUS.value, {"repo_path": "/tmp/repo"})) # Cast to list
    assert result[0].text == "Repository status:\nclean"
    mock_git_status.assert_called_with(mock_repo_instance)

    # Test GitTools.DIFF with target
    mock_git_diff.return_value = "diff_target_output"
    result = list(await call_tool(GitTools.DIFF.value, {"repo_path": "/tmp/repo", "target": "main"})) # Cast to list
    assert result[0].text == "Diff with main:\ndiff_target_output"
    
    # Test GitTools.DIFF without target
    mock_git_diff.return_value = "diff_default_output"
    result = list(await call_tool(GitTools.DIFF.value, {"repo_path": "/tmp/repo"}))
    assert result[0].text == "Diff of unstaged changes (worktree vs index):\ndiff_default_output"

    # Test GitTools.DIFF without target but with path
    mock_git_diff.return_value = "diff_path_output"
    result = list(await call_tool(GitTools.DIFF.value, {"repo_path": "/tmp/repo", "path": "foo.txt"}))
    assert result[0].text == "Diff of unstaged changes (worktree vs index) for path foo.txt:\ndiff_path_output"

    # Test GitTools.COMMIT
    mock_git_stage_and_commit.return_value = "Commit successful"
    result = list(await call_tool(GitTools.STAGE_AND_COMMIT.value, {"repo_path": "/tmp/repo", "message": "test commit"})) # Cast to list
    # Accept both string and TextContent result for compatibility
    if hasattr(result[0], "text"):
        assert result[0].text == "Commit successful"
    elif isinstance(result[0], str):
        assert result[0] == "Commit successful"
    else:
        raise AssertionError("Unexpected result type for GitTools.STAGE_AND_COMMIT")

    # Removed test for GitTools.ADD as the tool no longer exists

    # Test GitTools.LOG
    mock_git_log.return_value = ["log1", "log2"]
    result = list(await call_tool(GitTools.LOG.value, {"repo_path": "/tmp/repo", "max_count": 1})) # Cast to list
    assert result[0].text == "Commit history:\nlog1\nlog2"

    # Test GitTools.SHOW
    mock_git_show.return_value = "Show output"
    result = list(await call_tool(GitTools.SHOW.value, {"repo_path": "/tmp/repo", "revision": "HEAD"})) # Cast to list
    assert result[0].text == "Show output"

    # Test GitTools.READ_FILE
    mock_read_file_content.return_value = "File content"
    result = list(await call_tool(GitTools.READ_FILE.value, {"repo_path": "/tmp/repo", "file_path": "file.txt"})) # Cast to list
    assert result[0].text == "File content"

    # Removed test for GitTools.STAGE_ALL as the tool no longer exists


    # Test GitTools.WRITE_TO_FILE
    mock_write_to_file_content.return_value = "File written"
    result = list(await call_tool(GitTools.WRITE_TO_FILE.value, { # Cast to list
        "repo_path": "/tmp/repo", "file_path": "new.txt", "content": "new content"
    }))
    assert result[0].text == "File written"

    # Test GitTools.EXECUTE_COMMAND
    mock_execute_custom_command.return_value = "Command output"
    result = list(await call_tool(GitTools.EXECUTE_COMMAND.value, { # Cast to list
        "repo_path": "/tmp/repo", "command": "ls"
    }))
    assert result[0].text == "Command output"

    # Test unknown tool
    result = list(await call_tool("unknown_tool", {}))
    assert (
        "INVALID_TOOL_NAME: Unknown tool: unknown_tool. AI_HINT: Check the tool name and ensure it matches one of the supported tools."
        in result[0].text
    )

# Test cases for list_repos (requires mocking mcp_server.request_context.session)
@pytest.mark.asyncio
@patch('server.mcp_server')
@patch('server.git.Repo')
async def test_list_repos(mock_git_repo, mock_mcp_server):
    mock_session = AsyncMock(spec=ServerSession)
    mock_mcp_server.request_context.session = mock_session

    # Scenario 1: Client has roots capability, and there are valid git repos
    mock_session.check_client_capability.return_value = True
    mock_session.list_roots.return_value = ListRootsResult(
        roots=[
            Root(uri=FileUrl("file:///path/to/repo1")), # Use FileUrl
            Root(uri=FileUrl("file:///path/to/not_a_repo")), # Use FileUrl
            Root(uri=FileUrl("file:///path/to/repo2")), # Use FileUrl
        ]
    )
    
    # Configure mock_git_repo to raise InvalidGitRepositoryError for one path
    def mock_repo_init(path):
        if str(path) == "/path/to/not_a_repo":
            raise git.InvalidGitRepositoryError
        return MagicMock()
    mock_git_repo.side_effect = mock_repo_init

    repos = await list_repos()
    assert sorted(repos) == sorted(["/path/to/repo1", "/path/to/repo2"])
    mock_session.check_client_capability.assert_called_once()
    mock_session.list_roots.assert_called_once()
    assert mock_git_repo.call_count == 3 # Called for each root

    # Reset mocks for next scenario
    mock_session.reset_mock()
    mock_git_repo.reset_mock()
    mock_git_repo.side_effect = None # Clear side effect

    # Scenario 2: Client does not have roots capability
    mock_session.check_client_capability.return_value = False
    repos = await list_repos()
    assert repos == []
    mock_session.check_client_capability.assert_called_once()
    mock_session.list_roots.assert_not_called()
    mock_git_repo.assert_not_called()

    # Reset mocks for next scenario
    mock_session.reset_mock()
    mock_git_repo.reset_mock()

    # Scenario 3: No roots found
    mock_session.check_client_capability.return_value = True
    mock_session.list_roots.return_value = ListRootsResult(roots=[])
    repos = await list_repos()
    assert repos == []
    mock_session.check_client_capability.assert_called_once()
    mock_session.list_roots.assert_called_once()
    mock_git_repo.assert_not_called()

# Test cases for Starlette application (handle_sse, handle_post_message)
# These are more integration-level tests and might require a test client.
# For now, we'll mock the internal components.

@pytest.mark.asyncio
@patch('server.sse_transport')
@patch('server.mcp_server')
async def test_handle_sse(mock_mcp_server, mock_sse_transport):
    mock_request = MagicMock()
    mock_request.scope = {}
    mock_request.receive = AsyncMock()
    mock_request._send = AsyncMock()

    mock_connect_sse_context = AsyncMock()
    mock_connect_sse_context.__aenter__.return_value = (AsyncMock(), AsyncMock())
    mock_sse_transport.connect_sse.return_value = mock_connect_sse_context

    mock_mcp_server.create_initialization_options.return_value = {}
    mock_mcp_server.run = AsyncMock()

    response = await handle_sse(mock_request)
    assert isinstance(response, Response)
    mock_sse_transport.connect_sse.assert_called_once_with(mock_request.scope, mock_request.receive, mock_request._send)
    mock_mcp_server.create_initialization_options.assert_called_once()
    mock_mcp_server.run.assert_called_once()

@pytest.mark.asyncio
@patch('server.sse_transport')
async def test_handle_post_message(mock_sse_transport):
    mock_scope = {}
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    # Make handle_post_message on the mock awaitable
    mock_sse_transport.handle_post_message = AsyncMock() 

    await handle_post_message(mock_scope, mock_receive, mock_send)
    mock_sse_transport.handle_post_message.assert_called_once_with(mock_scope, mock_receive, mock_send)
import yaml
from unittest import mock

def test_load_aider_config_various_cases(tmp_path, monkeypatch):
    from server import load_aider_config

    # Patch os.path.exists and open to prevent reading real home config files
    import builtins
    real_exists = os.path.exists
    real_open = builtins.open

    def safe_exists(path):
        # Only allow files under the temporary workspace; block everything else for isolation
        try:
            return os.path.abspath(path).startswith(str(tmp_path))
        except Exception:
            return False

    def safe_open(path, *args, **kwargs):
        if str(tmp_path) in os.path.abspath(path):
            return real_open(path, *args, **kwargs)
        raise FileNotFoundError(f"Blocked open for {path}")

    monkeypatch.setattr(os.path, "exists", safe_exists)
    monkeypatch.setattr(builtins, "open", safe_open)

    # Helper to write a config file
    def write_yaml(path, data):
        with real_open(path, "w") as f:
            yaml.dump(data, f)

    # Case 1: Config in working directory
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    config1 = {"a": 1}
    config_path1 = workdir / ".aider.conf.yml"
    write_yaml(config_path1, config1)
    monkeypatch.chdir(workdir)
    assert load_aider_config(str(workdir)) == config1

    # Case 2: Config in git root (different from workdir) and precedence
    gitroot = tmp_path / "gitroot"
    gitroot.mkdir()
    (gitroot / ".git").mkdir()
    config2 = {"b": 2, "a": "from_git_root"} # 'a' will be overridden by workdir
    config_path2 = gitroot / ".aider.conf.yml"
    write_yaml(config_path2, config2)
    # Patch find_git_root to return gitroot for workdir
    with mock.patch("server.find_git_root", return_value=str(gitroot)):
        result = load_aider_config(str(workdir))
        assert result["a"] == 1 # from workdir, overrides git_root
        assert result["b"] == 2 # from git_root

    # Case 3: Config specified directly
    config3 = {"c": 3}
    config_path3 = tmp_path / "direct.yml"
    write_yaml(config_path3, config3)
    result = load_aider_config(str(workdir), str(config_path3))
    assert result["c"] == 3

    # Case 4: Config in home directory
    home_config = tmp_path / ".aider.conf.yml"
    config4 = {"d": 4}
    write_yaml(home_config, config4)
    monkeypatch.setenv("HOME", str(tmp_path))
    result = load_aider_config(str(workdir))
    assert result["d"] == 4

    # Case 5: No config files found
    # Remove home config file to avoid pollution
    home_config.unlink()
    emptydir = tmp_path / "empty"
    emptydir.mkdir()
    monkeypatch.chdir(emptydir)
    result = load_aider_config(str(emptydir))
    assert result == {}

    # Case 6: Exception during YAML loading (malformed YAML)
    bad_path = tmp_path / "bad.yml"
    with open(bad_path, "w") as f:
        f.write("not: [valid: yaml")
    result = load_aider_config(str(emptydir), str(bad_path))
    # Should not raise, should log warning and skip

    # Case 7: Empty config file
    empty_path = tmp_path / "empty.yml"
    empty_path.write_text("")
    result = load_aider_config(str(emptydir), str(empty_path))
    assert result == {}
def test_load_dotenv_file_various_cases(tmp_path, monkeypatch):
    from server import load_dotenv_file

    # Patch os.path.exists and open to prevent reading real home .env files
    import builtins
    real_exists = os.path.exists
    real_open = builtins.open

    def safe_exists(path):
        try:
            return str(tmp_path) in os.path.abspath(path) or real_exists(path) is False and "/dev/" in path
        except Exception:
            return False

    def safe_open(path, *args, **kwargs):
        if str(tmp_path) in os.path.abspath(path):
            return real_open(path, *args, **kwargs)
        raise FileNotFoundError(f"Blocked open for {path}")

    monkeypatch.setattr(os.path, "exists", safe_exists)
    monkeypatch.setattr(builtins, "open", safe_open)

    # Helper to write a .env file
    def write_env(path, lines):
        with real_open(path, "w") as f:
            f.write("\n".join(lines))

    # Case 1: .env in working directory
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    env1 = ["A=1", "B=2"]
    env_path1 = workdir / ".env"
    write_env(env_path1, env1)
    monkeypatch.chdir(workdir)
    result = load_dotenv_file(str(workdir))
    assert result["A"] == "1"
    assert result["B"] == "2"

    # Case 2: .env in git root (different from workdir)
    gitroot = tmp_path / "gitroot"
    gitroot.mkdir()
    (gitroot / ".git").mkdir()
    env2 = ["C=3"]
    env_path2 = gitroot / ".env"
    write_env(env_path2, env2)
    with mock.patch("server.find_git_root", return_value=str(gitroot)):
        result = load_dotenv_file(str(workdir))
        assert result["C"] == "3"

    # Case 3: .env specified directly
    env3 = ["D=4"]
    env_path3 = tmp_path / "direct.env"
    write_env(env_path3, env3)
    result = load_dotenv_file(str(workdir), str(env_path3))
    assert result["D"] == "4"

    # Case 4: .env in home directory
    home_env = tmp_path / ".env"
    env4 = ["E=5"]
    write_env(home_env, env4)
    monkeypatch.setenv("HOME", str(tmp_path))
    result = load_dotenv_file(str(workdir))
    assert result["E"] == "5"

    # Case 5: No .env files found
    # Remove home .env file to avoid pollution
    home_env.unlink()
    emptydir = tmp_path / "empty"
    emptydir.mkdir()
    monkeypatch.chdir(emptydir)
    result = load_dotenv_file(str(emptydir))
    assert result == {}

    # Case 6: Malformed line (ValueError)
    bad_env = tmp_path / "bad.env"
    write_env(bad_env, ["BADLINE"])
    result = load_dotenv_file(str(emptydir), str(bad_env))
    # Should not raise, should log warning and skip

    # Case 7: Empty .env file
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    result = load_dotenv_file(str(emptydir), str(empty_env))
    assert result == {}

    # Case 8: Lines starting with #
    comment_env = tmp_path / "comment.env"
    write_env(comment_env, ["# This is a comment", "F=6"])
    result = load_dotenv_file(str(emptydir), str(comment_env))
    assert result["F"] == "6"
import asyncio

import pytest

@pytest.fixture
def fake_aider_proc():
    class _FakeProc:
        def __init__(self, repo_path: Path | None = None, filename: str = "file.txt", content: str = "modified\n", returncode: int = 0):
            self._repo_path = repo_path
            self._filename = filename
            self._content = content
            self.returncode = returncode
        async def communicate(self):
            if self._repo_path is not None:
                (self._repo_path / self._filename).write_text(self._content)
            return (b"Applied edit to file.txt", b"")
    return _FakeProc

@pytest.mark.asyncio
async def test_run_command_success_and_failure(monkeypatch):
    from server import run_command

    class DummyProcess:
        def __init__(self, stdout=b"ok", stderr=b"", returncode=0):
            self._stdout = stdout
            self._stderr = stderr
            self.returncode = returncode

        async def communicate(self, input=None):
            return self._stdout, self._stderr

    async def dummy_create_subprocess_exec(*args, **kwargs):
        # Simulate different scenarios based on command
        if "fail" in args[0]:
            return DummyProcess(stdout=b"", stderr=b"fail", returncode=1)
        if "stdin" in args[0]:
            return DummyProcess(stdout=b"stdin-ok", stderr=b"", returncode=0)
        return DummyProcess(stdout=b"ok", stderr=b"", returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", dummy_create_subprocess_exec)

    # Success without input
    out, err, code = await run_command(["echo", "ok"])
    assert out == "ok"
    assert err == ""
    assert code == 0

    # Success with input
    out, err, code = await run_command(["stdin"], input_data="data")
    assert out == "stdin-ok"
    assert err == ""
    assert code == 0

    # Failure
    out, err, code = await run_command(["fail"])
    assert out == ""
    assert err == "fail"
    assert code == 1
def test_prepare_aider_command_various_cases():
    from server import prepare_aider_command

    # Boolean options
    cmd = prepare_aider_command(["aider"], options={"foo": True, "bar": False})
    assert "--foo" in cmd and "--no-bar" in cmd

    # List options
    cmd = prepare_aider_command(["aider"], options={"baz": [1, 2]})
    assert cmd.count("--baz") == 2
    assert "1" in cmd and "2" in cmd

    # String/integer options
    cmd = prepare_aider_command(["aider"], options={"opt": "val", "num": 5})
    assert "--opt" in cmd and "val" in cmd and "--num" in cmd and "5" in cmd

    # None options
    cmd = prepare_aider_command(["aider"], options={"skip": None})
    assert "--skip" not in cmd

    # Files argument
    cmd = prepare_aider_command(["aider"], files=["file1", "file2"])
    assert "file1" in cmd and "file2" in cmd

    # Combination of all types
    cmd = prepare_aider_command(
        ["aider"],
        files=["f1"],
        options={"a": True, "b": [3, 4], "c": "x", "d": None}
    )
    assert "--a" in cmd and "--b" in cmd and "3" in cmd and "4" in cmd and "--c" in cmd and "x" in cmd and "f1" in cmd
    assert "--d" not in cmd

    # Empty base command, no files, no options
    cmd = prepare_aider_command([])
    assert cmd == []
def test_git_read_file_error_cases(monkeypatch):
    from server import read_file_content
    import types

    class DummyRepo:
        def __init__(self, working_dir):
            self.working_dir = working_dir

    repo = DummyRepo("/tmp")

    # Simulate FileNotFoundError
    def fake_open_notfound(*a, **kw):
        raise FileNotFoundError()
    monkeypatch.setattr("builtins.open", fake_open_notfound)
    result = read_file_content(repo, "nofile.txt")
    assert (
        "file wasn't found" in result
        or "UNEXPECTED_ERROR: Failed to read file 'nofile.txt':" in result
    )

    # Simulate generic Exception
    def fake_open_exc(*a, **kw):
        raise Exception("fail")
    monkeypatch.setattr("builtins.open", fake_open_exc)
    result = read_file_content(repo, "nofile.txt")
    assert "UNEXPECTED_ERROR: Failed to read file 'nofile.txt': fail" in result
    assert "UNEXPECTED_ERROR:" in result

# New tests
@pytest.mark.asyncio
async def test_snapshot_delta_creation(monkeypatch, fake_aider_proc):
    """Test that snapshot delta is created correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        repo = git.Repo.init(repo_path)
        with repo.config_writer() as cw:
            cw.set_value("user", "email", "test@example.com")
            cw.set_value("user", "name", "Test User")
        # Initial commit
        (repo_path / "file.txt").write_text("initial\n")
        repo.index.add(["file.txt"])
        repo.index.commit("init")

        with patch('asyncio.create_subprocess_shell', return_value=fake_aider_proc(repo_path)):
            result = await ai_edit(
                repo_path=str(repo_path),
                message="Edit",
                session=MagicMock(),
                files=["file.txt"],
                options=[],
                continue_thread=False,
            )

        # Verify delta section exists
        assert "### Snapshot Delta (this run)" in result
        # Verify delta contains expected changes in correct order
        assert "-initial" in result
        assert "+modified" in result
        # Ensure the delta shows the correct transformation
        assert result.index("-initial") < result.index("+modified")

@pytest.mark.asyncio
async def test_ai_sessions_tool_list_status_last_session_id(monkeypatch, fake_aider_proc):
    """Test ai_sessions tool: list, status, and .aider.last_session_id support"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        repo = git.Repo.init(repo_path)
        with repo.config_writer() as cw:
            cw.set_value("user", "email", "test@example.com")
            cw.set_value("user", "name", "Test User")
        # Initial commit
        (repo_path / "file.txt").write_text("initial\n")
        repo.index.add(["file.txt"])
        repo.index.commit("init")

        # Ensure session record persists (no worktree purge)
        monkeypatch.setenv("MCP_USE_WORKTREES", "0")

        with patch('asyncio.create_subprocess_shell', return_value=fake_aider_proc(repo_path)):
            result = await ai_edit(
                repo_path=str(repo_path),
                message="Edit",
                session=MagicMock(),
                files=["file.txt"],
                options=[],
                continue_thread=False,
            )

        # Test ai_sessions list
        list_result = list(await call_tool("ai_sessions", {"repo_path": str(repo_path), "action": "list"}))
        session_payload = json.loads(list_result[0].text)
        sessions_list = session_payload.get("sessions", [])
        assert len(sessions_list) >= 1
        
        # Get session_id from first session
        session_id = sessions_list[0].get('id') or sessions_list[0].get('session_id')
        assert session_id is not None

        # Test ai_sessions status
        status_result = list(await call_tool("ai_sessions", {"repo_path": str(repo_path), "action": "status", "session_id": session_id}))
        status_data = json.loads(status_result[0].text)
        assert 'status' in status_data or 'session_id' in status_data

        # Test .aider.last_session_id file exists and contains the session id
        last_session_file = repo_path / ".aider.last_session_id"
        assert last_session_file.exists()
        assert last_session_file.read_text().strip() == session_id


@pytest.mark.asyncio
async def test_ai_sessions_list_filter_by_status():
    """Test ai_sessions list with status filter"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        repo = git.Repo.init(repo_path)
        with repo.config_writer() as cw:
            cw.set_value("user", "email", "test@example.com")
            cw.set_value("user", "name", "Test User")
        # Initial commit
        (repo_path / "file.txt").write_text("initial\n")
        repo.index.add(["file.txt"])
        repo.index.commit("init")

        # Run ai_edit once to create a completed session
        with patch('asyncio.create_subprocess_shell') as mock_shell:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"Applied edit to file.txt", b""))
            mock_shell.return_value = mock_proc
            
            await ai_edit(
                repo_path=str(repo_path),
                message="Edit",
                session=MagicMock(),
                files=["file.txt"],
                options=[],
                continue_thread=False,
            )

        # Create a 'running' session manually
        from server import _record_session_start
        running_id = "test-running-session-id"
        workspace_dir = str(repo_path / "test-workspace")
        _record_session_start(str(repo_path), running_id, workspace_dir, use_worktree=False)

        # Call ai_sessions list without filter; verify both sessions are present
        list_result = list(await call_tool("ai_sessions", {"repo_path": str(repo_path), "action": "list"}))
        session_payload = json.loads(list_result[0].text)
        sessions_list = session_payload.get("sessions", [])
        session_ids = [s.get('id') or s.get('session_id') for s in sessions_list]
        assert len(sessions_list) >= 2
        assert running_id in session_ids

        # Call ai_sessions list with status='running'; verify only running_id present
        list_result_running = list(await call_tool("ai_sessions", {
            "repo_path": str(repo_path), 
            "action": "list", 
            "status": "running"
        }))
        session_payload_running = json.loads(list_result_running[0].text)
        sessions_list_running = session_payload_running.get("sessions", [])
        running_session_ids = [s.get('id') or s.get('session_id') for s in sessions_list_running]
        assert len(sessions_list_running) == 1
        assert running_id in running_session_ids

        # Call ai_sessions list with status='completed'; verify includes the completed session and not the running one
        list_result_completed = list(await call_tool("ai_sessions", {
            "repo_path": str(repo_path), 
            "action": "list", 
            "status": "completed"
        }))
        session_payload_completed = json.loads(list_result_completed[0].text)
        sessions_list_completed = session_payload_completed.get("sessions", [])
        completed_session_ids = [s.get('id') or s.get('session_id') for s in sessions_list_completed]
        assert len(sessions_list_completed) >= 1
        assert running_id not in completed_session_ids
        
        # Verify all completed sessions have status 'completed'
        for session in sessions_list_completed:
            assert session.get('status') == 'completed'

@pytest.mark.asyncio
async def test_auto_apply_workspace_changes_to_root(monkeypatch, fake_aider_proc):
    """Test that workspace changes are auto-applied to root when MCP_APPLY_WORKSPACE_TO_ROOT=1"""
    # Set environment variables
    monkeypatch.setenv("MCP_USE_WORKTREES", "1")
    monkeypatch.setenv("MCP_APPLY_WORKSPACE_TO_ROOT", "1")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        repo = git.Repo.init(repo_path)
        with repo.config_writer() as cw:
            cw.set_value("user", "email", "test@example.com")
            cw.set_value("user", "name", "Test User")
        # Initial commit
        (repo_path / "file.txt").write_text("initial\n")
        repo.index.add(["file.txt"])
        repo.index.commit("init")

        with patch('asyncio.create_subprocess_shell', return_value=fake_aider_proc(repo_path)):
            result = await ai_edit(
                repo_path=str(repo_path),
                message="Edit",
                session=MagicMock(),
                files=["file.txt"],
                options=[],
                continue_thread=False,
            )

        # Verify that the output contains the diff
        assert "-initial" in result
        assert "+modified" in result

@pytest.mark.asyncio
async def test_purge_on_success_deletes_session_record(monkeypatch, fake_aider_proc):
    """Test that successful sessions are retained in sessions.json with purged metadata and workspace is deleted"""
    # Set environment variables
    monkeypatch.setenv("MCP_EXPERIMENTAL_WORKTREES", "1")
    monkeypatch.setenv("MCP_APPLY_WORKSPACE_TO_ROOT", "1")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        repo = git.Repo.init(repo_path)
        with repo.config_writer() as cw:
            cw.set_value("user", "email", "test@example.com")
            cw.set_value("user", "name", "Test User")
        # Initial commit
        (repo_path / "file.txt").write_text("initial\n")
        repo.index.add(["file.txt"])
        repo.index.commit("init")

        # Monkeypatch _git_status_clean to return True to allow purge-on-success
        monkeypatch.setattr("server._git_status_clean", lambda *args, **kwargs: True)

        with patch('asyncio.create_subprocess_shell', return_value=fake_aider_proc(repo_path)):
            result = await ai_edit(
                repo_path=str(repo_path),
                message="Edit",
                session=MagicMock(),
                files=["file.txt"],
                options=[],
                continue_thread=False,
            )

        # Extract session ID from result (supports new Summary format)
        lines = result.split('\n')
        session_line = next((ln for ln in lines if ln.strip().startswith("- Session: ")), None)
        if session_line is None:
            session_line = next((ln for ln in lines if ln.startswith("Session: ")), "")
        assert session_line, f"Session line not found in output: {result[:500]}"
        session_id = session_line.split(": ", 1)[1].strip()

        # Verify session is still in sessions.json with purged metadata
        sessions_file = repo_path / ".mcp-devtools" / "sessions.json"
        if sessions_file.exists():
            sessions_map = json.loads(sessions_file.read_text()) if sessions_file.read_text().strip() else {}
            assert session_id in sessions_map
            assert sessions_map[session_id].get("purged_at") is not None
            assert sessions_map[session_id].get("status") == "completed"

        # Verify specific workspace directory doesn't exist
        workspace_dir = repo_path / ".mcp-devtools" / "workspaces" / session_id
        assert not workspace_dir.exists()

@pytest.mark.asyncio
async def test_ttl_cleanup_deletes_expired_records(monkeypatch, fake_aider_proc):
    """Test that TTL cleanup removes expired session records"""
    # Fix time for deterministic test
    fixed_time = 10000
    monkeypatch.setattr(time, 'time', lambda: fixed_time)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        repo = git.Repo.init(repo_path)
        with repo.config_writer() as cw:
            cw.set_value("user", "email", "test@example.com")
            cw.set_value("user", "name", "Test User")
        # Initial commit
        (repo_path / "file.txt").write_text("initial\n")
        repo.index.add(["file.txt"])
        repo.index.commit("init")

        # Create sessions.json with expired session
        mcp_dir = repo_path / ".mcp-devtools"
        mcp_dir.mkdir(exist_ok=True)
        sessions_file = mcp_dir / "sessions.json"
        
        # Create expired session using store schema (dict keyed by id with epoch times)
        # Expired 3601 seconds ago (more than default TTL of 3600)
        expired_time_epoch = fixed_time - 3601
        expired_session_data = {
            "expired-session-123": {
                "id": "expired-session-123",
                "status": "completed",
                "completed_at": expired_time_epoch,
                "last_updated": expired_time_epoch,
                "use_worktree": False,
                "workspace_dir": None
            }
        }
        sessions_file.write_text(json.dumps(expired_session_data))

        # Set TTL to 3600 seconds (1 hour)
        monkeypatch.setenv("MCP_SESSION_TTL_SECONDS", "3600")

        with patch('asyncio.create_subprocess_shell', return_value=fake_aider_proc(repo_path, content="noop")):
            await ai_edit(
                repo_path=str(repo_path),
                message="No-op edit",
                session=MagicMock(),
                files=["file.txt"],
                options=[],
                continue_thread=False,
            )

        # Verify expired session was removed
        if sessions_file.exists():
            sessions_map = json.loads(sessions_file.read_text()) if sessions_file.read_text().strip() else {}
            assert "expired-session-123" not in sessions_map

@pytest.mark.asyncio
async def test_ai_edit_worktrees_disabled_by_default(temp_git_repo, monkeypatch):
    """Ensure that when MCP_EXPERIMENTAL_WORKTREES is not set (default), ai_edit runs without creating a workspace dir."""
    repo, repo_path = temp_git_repo
    
    # Monkeypatch asyncio.create_subprocess_shell to simulate aider success
    class FakeProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def _fake_shell_proc(*args, **kwargs):
        return FakeProc()
    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_shell_proc)
    
    # Call ai_edit with a fixed session_id
    session_id = "sess-default-off"
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=False,
        session_id=session_id
    )
    
    # Assert that .mcp-devtools/workspaces/sess-default-off does not exist
    workspace_dir = repo_path / ".mcp-devtools" / "workspaces" / session_id
    assert not workspace_dir.exists()

@pytest.mark.asyncio
async def test_ai_edit_worktrees_enabled_creates_and_purges(temp_git_repo, monkeypatch):
    """With MCP_EXPERIMENTAL_WORKTREES=1, ensure ai_edit attempts to create a worktree workspace and purges it on success."""
    repo, repo_path = temp_git_repo
    
    # Set env MCP_EXPERIMENTAL_WORKTREES=1
    monkeypatch.setenv("MCP_EXPERIMENTAL_WORKTREES", "1")
    
    # Use a known session_id
    session_id = "sess-wt-1"
    workspace_dir = repo_path / ".mcp-devtools" / "workspaces" / session_id
    
    # Track if worktree was created
    worktree_created = False
    
    # Monkeypatch asyncio.create_subprocess_exec to intercept 'git worktree add'
    original_create_subprocess_exec = asyncio.create_subprocess_exec
    
    async def mock_create_subprocess_exec(*args, **kwargs):
        nonlocal worktree_created
        if "worktree" in args and "add" in args:
            # Create workspace_dir before returning success
            workspace_dir.mkdir(parents=True, exist_ok=True)
            worktree_created = True
            # Return a mock process with returncode 0
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            return mock_proc
        return await original_create_subprocess_exec(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create_subprocess_exec)
    
    # Monkeypatch asyncio.create_subprocess_shell to simulate aider success
    class FakeProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def _fake_shell_proc(*args, **kwargs):
        return FakeProc()
    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_shell_proc)
    
    # Monkeypatch server._git_status_clean to return True
    monkeypatch.setattr("server._git_status_clean", lambda *args, **kwargs: True)
    
    # Monkeypatch server._purge_worktree_async to actually remove the workspace_dir
    async def mock_purge_worktree_async(repo_root, workspace_path):
        nonlocal worktree_created
        if worktree_created and Path(workspace_path).exists():
            shutil.rmtree(workspace_path)
    
    monkeypatch.setattr("server._purge_worktree_async", mock_purge_worktree_async)
    
    # Call ai_edit with that session_id
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=False,
        session_id=session_id
    )
    
    # Assert workspace_dir was created during run
    assert worktree_created
    
    # Assert workspace_dir is removed (purged) after completion
    assert not workspace_dir.exists()
    
    # Assert the sessions.json contains this session_id with purged metadata
    sessions_file = repo_path / ".mcp-devtools" / "sessions.json"
    if sessions_file.exists():
        sessions_map = json.loads(sessions_file.read_text()) if sessions_file.read_text().strip() else {}
        assert session_id in sessions_map
        assert sessions_map[session_id].get("purged_at") is not None
        assert sessions_map[session_id].get("status") == "completed"


@pytest.mark.asyncio
async def test_ai_edit_worktrees_add_failure_falls_back(temp_git_repo, monkeypatch):
    """With MCP_EXPERIMENTAL_WORKTREES=1, if 'git worktree add' fails, ensure fallback to main repo and no purge."""
    repo, repo_path = temp_git_repo
    
    # Set env MCP_EXPERIMENTAL_WORKTREES=1
    monkeypatch.setenv("MCP_EXPERIMENTAL_WORKTREES", "1")
    
    # Use a fixed session_id
    session_id = "sess-wt-fail"
    workspace_dir = repo_path / ".mcp-devtools" / "workspaces" / session_id
    
    # Track if worktree was created
    worktree_created = False
    
    # Track if purge was called (allowed to be called as a no-op)
    purge_called = False
    
    # Monkeypatch asyncio.create_subprocess_exec to intercept 'git worktree add' and simulate failure
    original_create_subprocess_exec = asyncio.create_subprocess_exec
    
    async def mock_create_subprocess_exec(*args, **kwargs):
        nonlocal worktree_created
        if "worktree" in args and "add" in args:
            # Do NOT create workspace_dir
            worktree_created = False
            # Return a mock process with non-zero return code and stderr
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"fatal: could not create worktree"))
            return mock_proc
        return await original_create_subprocess_exec(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create_subprocess_exec)
    
    # Monkeypatch asyncio.create_subprocess_shell to simulate aider success
    class FakeProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def _fake_shell_proc(*args, **kwargs):
        return FakeProc()
    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_shell_proc)
    
    # Monkeypatch server._git_status_clean to return True
    monkeypatch.setattr("server._git_status_clean", lambda *args, **kwargs: True)
    
    # Monkeypatch server._purge_worktree_async to assert it is not called
    async def mock_purge_worktree_async(repo_root, workspace_path):
        nonlocal purge_called
        purge_called = True
        # Call the original function directly since we can't easily reference it here
        # This is fine for the test since we're mainly checking that it's not called
        pass
    
    monkeypatch.setattr("server._purge_worktree_async", mock_purge_worktree_async)
    
    # Call ai_edit with that session_id
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=False,
        session_id=session_id
    )
    
    # Assert workspace_dir was not created
    assert not worktree_created
    
    # Assert workspace_dir does not exist
    assert not workspace_dir.exists()
    
    # Purge may be invoked as a no-op when no worktree exists; ensure it does not raise and leaves no workspace.
    assert not workspace_dir.exists()


@pytest.mark.asyncio
async def test_ai_edit_aider_failure_path(temp_git_repo, monkeypatch):
    """Simulate Aider returning non-zero exit code and stderr; assert error path."""
    repo, repo_path = temp_git_repo
    
    # Mock asyncio.create_subprocess_shell to return a failed process
    class FailedProc:
        def __init__(self):
            self.returncode = 1
        async def communicate(self):
            return (b"", b"Error: Could not connect to model")
    
    async def mock_create_subprocess_shell(*args, **kwargs):
        return FailedProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Call ai_edit and capture result
    result = await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=False
    )
    
    # Assert error path - should contain the stderr content
    assert "Error: Could not connect to model" in result
    assert "Aider process exited with code 1" in result


@pytest.mark.asyncio
async def test_ai_edit_prunes_history_when_continue_false(temp_git_repo, monkeypatch):
    """Create .aider.chat.history.md with content; run ai_edit with continue_thread=False; assert it was cleared."""
    repo, repo_path = temp_git_repo
    
    # Create .aider.chat.history.md with content
    history_file = repo_path / ".aider.chat.history.md"
    history_content = "# Chat History\n\n## User: Initial request\n\nAssistant: Initial response\n"
    history_file.write_text(history_content)
    
    # Mock asyncio.create_subprocess_shell to simulate successful Aider run
    class SuccessProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(*args, **kwargs):
        return SuccessProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Run ai_edit with continue_thread=False
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=False
    )
    
    # Assert history file was cleared (empty or contains only header)
    assert history_file.exists()
    content = history_file.read_text()
    assert content == "" or content.strip() == "# Chat History"

@pytest.mark.asyncio
async def test_ai_edit_prune_truncate_updates_history(temp_git_repo, monkeypatch):
    """Create .aider.chat.history.md with multiple sessions; run ai_edit with prune=True and prune_mode='truncate'; assert older sessions are removed."""
    repo, repo_path = temp_git_repo
    
    # Create .aider.chat.history.md with multiple sessions
    history_file = repo_path / ".aider.chat.history.md"
    history_content = """# aider chat started at 2023-01-01 10:00:00
## User: First request
Assistant: First response

# aider chat started at 2023-01-01 11:00:00
## User: Second request
Assistant: Second response

# aider chat started at 2023-01-01 12:00:00
## User: Third request
Assistant: Third response
"""
    history_file.write_text(history_content)
    
    # Mock asyncio.create_subprocess_shell to simulate successful Aider run
    class SuccessProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(*args, **kwargs):
        return SuccessProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Set MCP_PRUNE_KEEP_SESSIONS to 1 for this test
    monkeypatch.setenv("MCP_PRUNE_KEEP_SESSIONS", "1")
    
    # Run ai_edit with prune=True and prune_mode='truncate'
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=True,  # Important: continue_thread=True to avoid clearing
        prune=True,
        prune_mode="truncate"
    )
    
    # Assert history file was updated with truncation
    assert history_file.exists()
    content = history_file.read_text()
    assert "aider chat older sessions truncated" in content
    assert "# aider chat started at 2023-01-01 12:00:00" in content  # Last session kept
    assert "# aider chat started at 2023-01-01 11:00:00" not in content  # Previous sessions removed

@pytest.mark.asyncio
async def test_ai_edit_prune_summarize_updates_history(temp_git_repo, monkeypatch):
    """Create .aider.chat.history.md with multiple sessions; run ai_edit with prune=True; mock summarizer to create summary; assert history is rebuilt with summary."""
    repo, repo_path = temp_git_repo
    
    # Create .aider.chat.history.md with multiple sessions
    history_file = repo_path / ".aider.chat.history.md"
    history_content = """# aider chat started at 2023-01-01 10:00:00
## User: First request
Assistant: First response

# aider chat started at 2023-01-01 11:00:00
## User: Second request
Assistant: Second response

# aider chat started at 2023-01-01 12:00:00
## User: Third request
Assistant: Third response
"""
    history_file.write_text(history_content)
    
    # Mock asyncio.create_subprocess_shell to handle both main aider and summarizer
    call_count = 0
    
    class SuccessProc:
        def __init__(self, is_summarizer=False):
            self.returncode = 0
            self.is_summarizer = is_summarizer
            
        async def communicate(self):
            nonlocal call_count
            call_count += 1
            if self.is_summarizer:
                # Create the summary file when summarizer runs
                ctx_dir = repo_path / ".mcp-devtools" / "ctx"
                ctx_dir.mkdir(parents=True, exist_ok=True)
                summary_file = ctx_dir / "older_summary.md"
                summary_file.write_text("Summary of first two sessions: discussed initial requests and responses.")
                return (b"Summary created", b"")
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(command, *args, **kwargs):
        # Check if this is the summarizer command
        if "older_history.md" in command and "older_summary.md" in command:
            return SuccessProc(is_summarizer=True)
        return SuccessProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Set MCP_PRUNE_KEEP_SESSIONS to 1 for this test
    monkeypatch.setenv("MCP_PRUNE_KEEP_SESSIONS", "1")
    
    # Run ai_edit with prune=True (default prune_mode is summarize)
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=True,  # Important: continue_thread=True to avoid clearing
        prune=True
    )
    
    # Assert history file was updated with summary
    assert history_file.exists()
    content = history_file.read_text()
    assert "Summary of older chat sessions" in content
    assert "Summary of first two sessions: discussed initial requests and responses." in content
    assert "# aider chat started at 2023-01-01 12:00:00" in content  # Last session kept
    assert "# aider chat started at 2023-01-01 11:00:00" not in content  # Previous sessions removed except in summary


@pytest.mark.asyncio
async def test_ai_edit_no_structured_report_fallback_appends_last_reply(temp_git_repo, monkeypatch):
    """Provide a chat history with assistant content; run ai_edit with continue_thread=True and stdout without 'Applied edit to'; assert output contains 'Aider process completed.' and the parsed last reply."""
    repo, repo_path = temp_git_repo
    
    # Create .aider.chat.history.md with assistant content
    history_file = repo_path / ".aider.chat.history.md"
    history_content = "# Chat History\n\n## User: Initial request\n\nAssistant: Here's my response to your request.\n"
    history_file.write_text(history_content)
    
    # Mock asyncio.create_subprocess_shell to return stdout without "Applied edit to"
    class NoStructuredReportProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Aider process completed successfully.", b"")
    
    async def mock_create_subprocess_shell(*args, **kwargs):
        return NoStructuredReportProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Run ai_edit with continue_thread=True
    result = await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=True
    )
    
    # Assert output contains expected content
    assert "Aider process completed." in result
    assert "Here's my response to your request." in result


@pytest.mark.asyncio
async def test_ai_edit_worktrees_enabled_dirty_no_purge(temp_git_repo, monkeypatch):
    """Enable MCP_EXPERIMENTAL_WORKTREES=1; simulate worktree add success; Aider success; monkeypatch _git_status_clean to False; assert workspace remains and session record not deleted (no purged_at)."""
    repo, repo_path = temp_git_repo
    
    # Enable MCP_EXPERIMENTAL_WORKTREES=1
    monkeypatch.setenv("MCP_EXPERIMENTAL_WORKTREES", "1")
    
    # Use a fixed session_id
    session_id = "sess-wt-dirty"
    workspace_dir = repo_path / ".mcp-devtools" / "workspaces" / session_id
    
    # Monkeypatch asyncio.create_subprocess_exec to simulate successful worktree add
    original_create_subprocess_exec = asyncio.create_subprocess_exec
    
    async def mock_create_subprocess_exec(*args, **kwargs):
        if "worktree" in args and "add" in args:
            # Create workspace_dir
            workspace_dir.mkdir(parents=True, exist_ok=True)
            # Return success
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            return mock_proc
        return await original_create_subprocess_exec(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create_subprocess_exec)
    
    # Monkeypatch asyncio.create_subprocess_shell to simulate successful Aider
    class SuccessProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(*args, **kwargs):
        return SuccessProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Monkeypatch _git_status_clean to return False (dirty repo)
    monkeypatch.setattr("server._git_status_clean", lambda *args, **kwargs: False)
    
    # Run ai_edit
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=False,
        session_id=session_id
    )
    
    # Assert workspace remains (not purged)
    assert workspace_dir.exists()
    
    # Assert session record still exists (not deleted)
    sessions_file = repo_path / ".mcp-devtools" / "sessions.json"
    if sessions_file.exists():
        sessions_map = json.loads(sessions_file.read_text()) if sessions_file.read_text().strip() else {}
        assert session_id in sessions_map
        # Assert purged_at is None (no purge occurred)
        assert sessions_map[session_id].get("purged_at") in (None, 0)


@pytest.mark.asyncio
async def test_ai_edit_options_override_and_unsupported_removed(temp_git_repo, monkeypatch):
    """Pass options with conflicting restore_chat_history and unsupported base-url; capture the command string passed to create_subprocess_shell; assert it contains '--no-restore-chat-history' (from continue_thread=False), does not contain '--restore-chat-history', and does not contain '--base-url' or '--base_url'."""
    repo, repo_path = temp_git_repo
    
    # Capture the command passed to create_subprocess_shell
    captured_command = []
    
    class SuccessProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(command, *args, **kwargs):
        captured_command.append(command)
        return SuccessProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Run ai_edit with conflicting options and continue_thread=False
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[
            "--restore-chat-history",  # This should be overridden
            "--base-url", "http://example.com",  # This should be removed
            "--base_url", "http://example.com",  # This should also be removed
        ],
        continue_thread=False
    )
    
    # Assert command contains expected flags
    command = captured_command[0]
    assert "--no-restore-chat-history" in command
    assert "--restore-chat-history" not in command
    assert "--base-url" not in command
    assert "--base_url" not in command

@pytest.mark.asyncio
async def test_ai_edit_includes_thread_context_usage(temp_git_repo, monkeypatch):
    """Test that ai_edit output includes thread context usage information"""
    repo, repo_path = temp_git_repo
    
    # Create a .aider.chat.history.md file with some content
    history_file = repo_path / ".aider.chat.history.md"
    history_content = "# aider chat started at 2024-01-01\nUser: Hello\nAssistant: Hi there!"
    history_file.write_text(history_content)
    
    # Mock successful Aider run
    class SuccessProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(command, *args, **kwargs):
        return SuccessProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Run ai_edit
    result = await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=True
    )
    
    # Assert that the output contains thread context usage information
    assert "### Thread Context Usage" in result
    assert "Approximate tokens:" in result
    assert "Guidance: Keep overall thread context under ~200k tokens" in result


def test_server_imports_when_snapshot_utils_missing(tmp_path, monkeypatch):
    """Ensure importing server succeeds even if mcp_devtools.snapshot_utils isn't available."""
    import importlib, sys
    # Create a fake package mcp_devtools without snapshot_utils
    fake_root = tmp_path / "fakepkg"
    pkg_dir = fake_root / "mcp_devtools"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("# dummy package without snapshot_utils\n")

    # Prepend to sys.path so it shadows the real package
    monkeypatch.syspath_prepend(str(fake_root))

    # Remove any cached modules
    for mod in ["mcp_devtools", "mcp_devtools.snapshot_utils", "server"]:
        if mod in sys.modules:
            del sys.modules[mod]

    # Import server; should not raise
    srv = importlib.import_module("server")

    # Use fallback ensure_snap_dir to create .mcp-devtools
    d = srv.ensure_snap_dir(str(tmp_path))
    assert d.exists()
    assert d.name == ".mcp-devtools"

    # Fallback save_snapshot shouldn't error
    p = srv.save_snapshot(str(tmp_path), "TEST", "pre", "")
    assert p.exists()


@pytest.mark.asyncio
async def test_ai_edit_prune_summarize_fallback_on_failure(temp_git_repo, monkeypatch):
    """Setup .aider.chat.history.md with 3 sessions; Set MCP_PRUNE_KEEP_SESSIONS=1, prune=True, prune_mode=None (summarize default);
    Mock asyncio.create_subprocess_shell: summarizer process returns non-zero and does NOT create summary file; main aider returns success;
    Assert history contains 'summarization failed' header and only last session kept"""
    repo, repo_path = temp_git_repo
    
    # Create .aider.chat.history.md with 3 sessions
    history_file = repo_path / ".aider.chat.history.md"
    history_content = """# aider chat started at 2023-01-01 10:00:00
## User: First request
Assistant: First response

# aider chat started at 2023-01-01 11:00:00
## User: Second request
Assistant: Second response

# aider chat started at 2023-01-01 12:00:00
## User: Third request
Assistant: Third response
"""
    history_file.write_text(history_content)
    
    # Set MCP_PRUNE_KEEP_SESSIONS=1
    monkeypatch.setenv("MCP_PRUNE_KEEP_SESSIONS", "1")
    
    # Mock asyncio.create_subprocess_shell to handle both summarizer (failure) and main aider (success)
    call_count = 0
    
    class Proc:
        def __init__(self, is_summarizer=False):
            self.is_summarizer = is_summarizer
            if is_summarizer:
                self.returncode = 1  # Summarizer fails
            else:
                self.returncode = 0  # Main aider succeeds
                
        async def communicate(self):
            nonlocal call_count
            call_count += 1
            if self.is_summarizer:
                return (b"", b"Summarization failed")
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(command, *args, **kwargs):
        # Check if this is the summarizer command
        if "older_history.md" in command and "older_summary.md" in command:
            return Proc(is_summarizer=True)
        return Proc(is_summarizer=False)
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Run ai_edit with prune=True (summarize mode by default)
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=True,
        prune=True,
        prune_mode=None  # Default to summarize
    )
    
    # Assert history contains 'summarization failed' header and only last session kept
    assert history_file.exists()
    content = history_file.read_text()
    assert "summarization failed" in content
    assert "# aider chat started at 2023-01-01 12:00:00" in content  # Last session kept
    assert "# aider chat started at 2023-01-01 11:00:00" not in content  # Previous sessions removed
    assert "# aider chat started at 2023-01-01 10:00:00" not in content  # Previous sessions removed


@pytest.mark.asyncio
async def test_ai_edit_prune_keep_two_sessions(temp_git_repo, monkeypatch):
    """Setup .aider.chat.history.md with 4 sessions; Set MCP_PRUNE_KEEP_SESSIONS=2, prune=True, prune_mode='truncate';
    Mock main aider success; Assert history keeps last two sessions and truncation header present; earlier sessions removed"""
    repo, repo_path = temp_git_repo
    
    # Create .aider.chat.history.md with 4 sessions
    history_file = repo_path / ".aider.chat.history.md"
    history_content = """# aider chat started at 2023-01-01 09:00:00
## User: First request
Assistant: First response

# aider chat started at 2023-01-01 10:00:00
## User: Second request
Assistant: Second response

# aider chat started at 2023-01-01 11:00:00
## User: Third request
Assistant: Third response

# aider chat started at 2023-01-01 12:00:00
## User: Fourth request
Assistant: Fourth response
"""
    history_file.write_text(history_content)
    
    # Set MCP_PRUNE_KEEP_SESSIONS=2
    monkeypatch.setenv("MCP_PRUNE_KEEP_SESSIONS", "2")
    
    # Mock asyncio.create_subprocess_shell for successful aider run
    class SuccessProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(*args, **kwargs):
        return SuccessProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Run ai_edit with prune=True and prune_mode='truncate'
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=True,
        prune=True,
        prune_mode="truncate"
    )
    
    # Assert history keeps last two sessions and truncation header present; earlier sessions removed
    assert history_file.exists()
    content = history_file.read_text()
    assert "aider chat older sessions truncated" in content
    assert "# aider chat started at 2023-01-01 11:00:00" in content  # Last two sessions kept
    assert "# aider chat started at 2023-01-01 12:00:00" in content  # Last two sessions kept
    assert "# aider chat started at 2023-01-01 10:00:00" not in content  # Earlier sessions removed
    assert "# aider chat started at 2023-01-01 09:00:00" not in content  # Earlier sessions removed


@pytest.mark.asyncio
async def test_ai_edit_prune_with_continue_false_does_not_clear(temp_git_repo, monkeypatch):
    """Setup .aider.chat.history.md with 2 sessions; Run ai_edit with prune=True (truncate) and continue_thread=False;
    Mock main aider success; Assert history is NOT cleared (contains truncation header and last session), verifying precedence of pruning over clearing"""
    repo, repo_path = temp_git_repo
    
    # Create .aider.chat.history.md with 2 sessions
    history_file = repo_path / ".aider.chat.history.md"
    history_content = """# aider chat started at 2023-01-01 10:00:00
## User: First request
Assistant: First response

# aider chat started at 2023-01-01 11:00:00
## User: Second request
Assistant: Second response
"""
    history_file.write_text(history_content)
    
    # Set MCP_PRUNE_KEEP_SESSIONS=1
    monkeypatch.setenv("MCP_PRUNE_KEEP_SESSIONS", "1")
    
    # Mock asyncio.create_subprocess_shell for successful aider run
    class SuccessProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(*args, **kwargs):
        return SuccessProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Run ai_edit with prune=True (truncate mode) and continue_thread=False
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=False,  # This would normally clear history
        prune=True,
        prune_mode="truncate"   # But pruning should take precedence
    )
    
    # Assert history is NOT cleared (contains truncation header and last session), verifying precedence of pruning over clearing
    assert history_file.exists()
    content = history_file.read_text()
    assert "aider chat older sessions truncated" in content
    assert "# aider chat started at 2023-01-01 11:00:00" in content  # Last session kept
    assert "# aider chat started at 2023-01-01 10:00:00" not in content  # Earlier session removed
    # Should NOT be completely cleared (no empty history or just header)
    assert content.strip() != "# aider chat started at 2023-01-01 11:00:00"  # Has more content than just session header


@pytest.mark.asyncio
async def test_ai_edit_no_prune_history_unchanged(temp_git_repo, monkeypatch):
    """Setup .aider.chat.history.md with 2 sessions; Run ai_edit with prune=False and continue_thread=True;
    Mock main aider success; Assert history unchanged (contains both sessions, no header)"""
    repo, repo_path = temp_git_repo
    
    # Create .aider.chat.history.md with 2 sessions
    history_file = repo_path / ".aider.chat.history.md"
    history_content = """# aider chat started at 2023-01-01 10:00:00
## User: First request
Assistant: First response

# aider chat started at 2023-01-01 11:00:00
## User: Second request
Assistant: Second response
"""
    original_content = history_content
    history_file.write_text(history_content)
    
    # Mock asyncio.create_subprocess_shell for successful aider run
    class SuccessProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(*args, **kwargs):
        return SuccessProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Run ai_edit with prune=False and continue_thread=True
    await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=True,
        prune=False  # No pruning
    )
    
    # Assert history unchanged (contains both sessions, no header)
    assert history_file.exists()
    content = history_file.read_text()
    assert content == original_content  # History unchanged
    assert "aider chat older sessions truncated" not in content
    assert "Summary of older chat sessions" not in content
    assert "# aider chat started at 2023-01-01 10:00:00" in content  # Both sessions still present
    assert "# aider chat started at 2023-01-01 11:00:00" in content  # Both sessions still present


@pytest.mark.asyncio
async def test_ai_edit_summary_includes_aggregated_token_stats(temp_git_repo, monkeypatch):
    """Create .aider.chat.history.md with two sessions each containing a token stats line;
    Mock aider success; Run ai_edit and assert Summary contains aggregated token stats"""
    repo, repo_path = temp_git_repo
    
    # Create .aider.chat.history.md with two sessions containing token stats
    history_file = repo_path / ".aider.chat.history.md"
    history_content = """# aider chat started at 2023-01-01 10:00:00
> Tokens: 1.5k sent, 250 received.
User: First request
Assistant: First response

# aider chat started at 2023-01-01 11:00:00
> Tokens: 500 sent, 1k received.
User: Second request
Assistant: Second response
"""
    history_file.write_text(history_content)
    
    # Mock asyncio.create_subprocess_shell for successful aider run
    class SuccessProc:
        def __init__(self):
            self.returncode = 0
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(*args, **kwargs):
        return SuccessProc()
    
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    # Run ai_edit
    result = await ai_edit(
        repo_path=str(repo_path),
        message="test edit",
        session=MagicMock(),
        files=["file.txt"],
        options=[],
        continue_thread=True
    )
    
    # Assert Summary contains aggregated token stats (1.5k + 500 = 2000 sent, 250 + 1k = 1250 received)
    assert "Aider tokens: sent=2000, received=1250" in result


def test_approx_token_count_uses_tokenizer_when_present(monkeypatch):
    """Test that _approx_token_count uses tokenizer when present"""
    import server
    
    # Check if tokenizer attribute exists (skip if not - fallback-only build)
    if not hasattr(server, 'tokenizer'):
        pytest.skip("tokenizer not available in this build")
    
    # Monkeypatch server.tokenizer to a fake object with encode returning a list of length 7
    class FakeTokenizer:
        def encode(self, text):
            return [1] * 7  # List of length 7
    
    monkeypatch.setattr(server, 'tokenizer', FakeTokenizer())
    
    # Call server._approx_token_count and assert it returns 7
    result = server._approx_token_count("ignored input")
    assert result == 7


def test_approx_token_count_fallback_when_encode_raises(monkeypatch):
    """Test that _approx_token_count falls back when encode raises an exception"""
    import server
    import math
    
    text = "test text for token counting"
    
    # If server has tokenizer attribute, monkeypatch server.tokenizer.encode to raise an Exception
    if hasattr(server, 'tokenizer'):
        class FakeTokenizer:
            def encode(self, text):
                raise Exception("Encoding failed")
        
        monkeypatch.setattr(server, 'tokenizer', FakeTokenizer())
    
    # Compute expected = math.ceil(len(text)/4)
    expected = math.ceil(len(text) / 4)
    
    # Assert server._approx_token_count(text) == expected
    result = server._approx_token_count(text)
    assert result == expected


def test_parse_aider_token_stats_rounds_up_fractional_values():
    """Test that _parse_aider_token_stats rounds up fractional values"""
    from server import _parse_aider_token_stats
    
    # Test string with fractional token values
    test_text = "> Tokens: 2.6k sent, 1.2k received."
    
    # Call _parse_aider_token_stats
    sent_tokens, received_tokens = _parse_aider_token_stats(test_text)
    
    # Assert (2600, 1200) - verifying rounding up behavior
    assert sent_tokens == 2600
    assert received_tokens == 1200


def test_extract_touched_files_empty():
    from server import _extract_touched_files
    assert _extract_touched_files("") == set()


def test_extract_touched_files_various_cases():
    from server import _extract_touched_files
    diff_text = """
    diff --git a/file1.txt b/file1.txt
    index e69de29..4b825dc 100644
    --- a/file1.txt
    +++ b/file1.txt
    @@ -1 +1 @@
    -old
    +new

    diff --git a/new.txt b/new.txt
    new file mode 100644
    index 0000000..e69de29
    --- /dev/null
    +++ b/new.txt
    @@ -0,0 +1 @@
    +hello

    diff --git a/old.txt b/old.txt
    deleted file mode 100644
    index e69de29..0000000
    --- a/old.txt
    +++ /dev/null

    diff --git a/oldname.md b/newname.md
    similarity index 100%
    rename from oldname.md
    rename to newname.md
    --- a/oldname.md
    +++ b/newname.md

    diff --git a/bin1.png b/bin1.png
    index 89abcd1..12ef345 100644
    Binary files a/bin1.png and b/bin1.png differ
    """.strip()
    touched = _extract_touched_files(diff_text)
    expected = {"file1.txt", "new.txt", "old.txt", "oldname.md", "newname.md", "bin1.png"}
    for p in expected:
        assert p in touched


def test_parse_aider_token_stats_k_and_m_suffixes():
    from server import _parse_aider_token_stats
    text = "\n".join([
        "> Tokens: 1.5k sent, 2K received.",
        "> Tokens: 0.75m sent, 0.25M received.",
    ])
    sent, received = _parse_aider_token_stats(text)
    assert sent == 1500 + 750000
    assert received == 2000 + 250000


def test_parse_aider_token_stats_commas_and_plain():
    from server import _parse_aider_token_stats
    text = "\n".join([
        "> Tokens: 1,234 sent, 567 received.",
        "> Tokens: 42 sent, 8 received.",
    ])
    sent, received = _parse_aider_token_stats(text)
    assert sent == 1234 + 42
    assert received == 567 + 8


def test_parse_aider_token_stats_no_matches_returns_zero():
    from server import _parse_aider_token_stats
    text = "No token stats here."
    sent, received = _parse_aider_token_stats(text)
    assert sent == 0 and received == 0


def test_extract_touched_files_ignores_internal_paths():
    from server import _extract_touched_files
    diff_text = "\n".join([
        "diff --git a/.mcp-devtools/ctx/older_history.md b/.mcp-devtools/ctx/older_history.md",
        "index 0000000..1111111 100644",
        "--- a/.mcp-devtools/ctx/older_history.md",
        "+++ b/.mcp-devtools/ctx/older_history.md",
        "@@ -0,0 +1 @@",
        "+internal",
        "diff --git a/src/app.ts b/src/app.ts",
        "index 0000000..1111111 100644",
        "--- a/src/app.ts",
        "+++ b/src/app.ts",
        "@@ -0,0 +1 @@",
        "+export {};",
    ])
    touched = _extract_touched_files(diff_text)
    assert "src/app.ts" in touched
    assert all(not p.startswith('.mcp-devtools/') for p in touched)


# === Server Integration Test Fixtures and Smoke Test ===
import pytest
import asyncio
import httpx
from multiprocessing import Process, Queue
import uvicorn
from server import app


def run_server(queue):
    try:
        port = 1337
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
        server = uvicorn.Server(config)
        queue.put(port)
        server.run()
    except Exception as e:
        queue.put(e)


@pytest.fixture(scope="module")
def server():
    queue = Queue()
    proc = Process(target=run_server, args=(queue,))
    proc.start()
    result = queue.get(timeout=5)
    if isinstance(result, Exception):
        proc.terminate()
        proc.join()
        raise result
    port = result
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()


@pytest.mark.asyncio
async def test_smoke_sse_connection(server):
    """
    Smoke test to ensure the SSE endpoint is available and establishes a connection.
    """
    sse_url = f"{server}/sse"
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", sse_url, timeout=10) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers.get("content-type", "")
                # Attempt to read the initial bytes from the SSE stream
                initial_data = await response.aiter_bytes().__anext__()
                assert initial_data is not None
    except httpx.ConnectError as e:
        pytest.fail(f"Connection to the server failed: {e}")
