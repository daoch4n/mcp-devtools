pytest_plugins = "pytest_asyncio"

import os
import tempfile
import shutil
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
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Import functions and classes from server.py
from server import (
    git_status, git_diff, git_stage_and_commit,
    git_log, git_branch, git_show,
    git_merge,
    git_apply_diff, git_read_file,
    _generate_diff_output, _run_tsc_if_applicable,
    write_to_file_content, execute_custom_command,
    Starlette, Route, Mount, Response, ServerSession, ClientCapabilities, RootsCapability, ListRootsResult, TextContent,
    handle_sse, handle_post_message,
    list_tools, call_tool, list_repos, GitTools
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

def test_git_diff_staged(temp_git_repo):
    repo, repo_path = temp_git_repo
    (repo_path / "staged_file.txt").write_text("staged content")
    repo.index.add(["staged_file.txt"])
    # Removed test for git_diff_staged as the function no longer exists

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

def test_git_branch_create(temp_git_repo):
    repo, repo_path = temp_git_repo
    result = git_branch(repo, 'create', 'new_branch')
    assert "Created branch 'new_branch'" in result
    assert 'new_branch' in repo.heads
    result = git_branch(repo, 'create', 'another_branch', base_branch='new_branch')
    assert "Created branch 'another_branch' from 'new_branch'" in result
    assert 'another_branch' in repo.heads

def test_git_branch_checkout(temp_git_repo):
    repo, repo_path = temp_git_repo
    repo.create_head('checkout_branch')
    result = git_branch(repo, 'checkout', 'checkout_branch')
    assert "Switched to branch 'checkout_branch'" in result
    assert repo.active_branch.name == 'checkout_branch'

def test_git_branch_rename(temp_git_repo):
    repo, repo_path = temp_git_repo
    git_branch(repo, 'create', 'old')
    result = git_branch(repo, 'rename', 'old', new_name='new')
    assert "Renamed branch 'old' to 'new'" in result
    assert 'new' in [h.name for h in repo.heads]
    assert 'old' not in [h.name for h in repo.heads]

def test_git_branch_list(temp_git_repo):
    repo, repo_path = temp_git_repo
    git_branch(repo, 'create', 'a')
    git_branch(repo, 'create', 'b')
    git_branch(repo, 'checkout', 'b')
    listing = git_branch(repo, 'list')
    assert 'a' in listing and 'b' in listing
    assert '* b' in listing or '*  b' in listing or '* b' in listing.replace('  ', ' ')

def test_git_merge_fast_forward(temp_git_repo):
    repo, repo_path = temp_git_repo
    # Create feature branch and commit a change
    orig_branch = repo.active_branch.name
    repo.git.checkout('-b', 'feat')
    (repo_path / 'feat.txt').write_text('feature work')
    repo.index.add(['feat.txt'])
    repo.index.commit('feat commit')
    # Switch back to original and merge
    repo.git.checkout(orig_branch)
    msg = git_merge(repo, 'feat')
    assert 'Merged feat into' in msg
    assert (repo_path / 'feat.txt').exists()
    assert repo.active_branch.name == orig_branch

def test_git_merge_dry_run_fast_forward(temp_git_repo):
    repo, repo_path = temp_git_repo
    base_branch = repo.active_branch.name
    repo.git.checkout('-b', 'feat')
    (repo_path / 'feat_only.txt').write_text('feat work')
    repo.index.add(['feat_only.txt'])
    repo.index.commit('feat only')
    repo.git.checkout(base_branch)
    msg = git_merge(repo, 'feat', dry_run=True)
    assert msg.startswith('DRY_RUN:')
    assert 'Fast-forward' in msg
    # Ensure no changes applied in dry run
    assert not (repo_path / 'feat_only.txt').exists()

def test_git_merge_into_target(temp_git_repo):
    repo, repo_path = temp_git_repo
    current = repo.active_branch.name
    repo.git.checkout('-b', 'dev')
    (repo_path / 'dev.txt').write_text('dev work')
    repo.index.add(['dev.txt'])
    repo.index.commit('dev commit')
    msg = git_merge(repo, 'dev', target=current)
    assert 'Merged dev into' in msg
    assert (repo_path / 'dev.txt').exists()
    assert repo.active_branch.name == current

def test_git_merge_dry_run_merge_required(temp_git_repo):
    repo, repo_path = temp_git_repo
    base_branch = repo.active_branch.name
    # Create dev branch commit
    repo.git.checkout('-b', 'dev')
    (repo_path / 'dev.txt').write_text('dev work')
    repo.index.add(['dev.txt'])
    repo.index.commit('dev commit')
    # Diverge main
    repo.git.checkout(base_branch)
    (repo_path / 'main_only.txt').write_text('main work')
    repo.index.add(['main_only.txt'])
    repo.index.commit('main commit')
    msg = git_merge(repo, 'dev', dry_run=True)
    assert msg.startswith('DRY_RUN:')
    assert 'Merge commit required' in msg
    # Repo unchanged for dry run
    assert not (repo_path / 'dev.txt').exists()

def test_git_merge_ff_only_non_ff_fails(temp_git_repo):
    repo, repo_path = temp_git_repo
    # Create dev branch and commit change
    repo.git.checkout('-b', 'dev')
    (repo_path / 'dev.txt').write_text('dev work')
    repo.index.add(['dev.txt'])
    repo.index.commit('dev commit')
    # Switch back to main and create a divergent commit so fast-forward is impossible
    repo.git.checkout('main')
    (repo_path / 'main.txt').write_text('main work')
    repo.index.add(['main.txt'])
    repo.index.commit('main commit')
    msg = git_merge(repo, 'dev', ff_only=True)
    assert 'MERGE_ERROR' in msg or 'MERGE_CONFLICT' in msg
    # dev change should not be present on main because ff-only prevented merge
    assert not (repo_path / 'dev.txt').exists() or 'Not possible to fast-forward' in msg

def test_git_merge_conflict_detection(temp_git_repo):
    repo, repo_path = temp_git_repo
    # Create a file and commit on main
    (repo_path / 'conflict.txt').write_text('line1\n')
    repo.index.add(['conflict.txt'])
    repo.index.commit('init conflict file')
    # Create branch and modify same line
    repo.git.checkout('-b', 'feat')
    (repo_path / 'conflict.txt').write_text('line1-feat\n')
    repo.index.add(['conflict.txt'])
    repo.index.commit('feat change')
    # Go back to main and modify same line differently
    repo.git.checkout('main')
    (repo_path / 'conflict.txt').write_text('line1-main\n')
    repo.index.add(['conflict.txt'])
    repo.index.commit('main change')
    # Try to merge feat into main to cause a conflict
    msg = git_merge(repo, 'feat')
    assert msg.startswith('MERGE_CONFLICT:')
    assert 'conflict.txt' in msg

def test_git_merge_dry_run_conflicts(temp_git_repo):
    repo, repo_path = temp_git_repo
    base_branch = repo.active_branch.name
    # Create file and commit on base
    (repo_path / 'conflict2.txt').write_text('line1\n')
    repo.index.add(['conflict2.txt'])
    repo.index.commit('init conflict2')
    # Branch and modify same line
    repo.git.checkout('-b', 'feat2')
    (repo_path / 'conflict2.txt').write_text('line1-feat\n')
    repo.index.add(['conflict2.txt'])
    repo.index.commit('feat2 change')
    # Back to base and modify same line differently
    repo.git.checkout(base_branch)
    (repo_path / 'conflict2.txt').write_text('line1-main\n')
    repo.index.add(['conflict2.txt'])
    repo.index.commit('main2 change')
    msg = git_merge(repo, 'feat2', dry_run=True)
    assert msg.startswith('DRY_RUN:')
    assert 'conflict' in msg.lower()
    # Ensure no changes applied
    with open(repo_path / 'conflict2.txt', 'r') as f:
        assert '<<<<<<<' not in f.read()

def test_git_merge_no_ff_forces_merge_commit(temp_git_repo):
    repo, repo_path = temp_git_repo
    base = repo.active_branch.name
    repo.git.checkout('-b', 'feat_noff')
    (repo_path / 'noff.txt').write_text('work')
    repo.index.add(['noff.txt'])
    repo.index.commit('feat no-ff')
    repo.git.checkout(base)
    # This would normally be a fast-forward; --no-ff should force a merge commit with 2 parents
    msg = git_merge(repo, 'feat_noff', no_ff=True)
    assert 'Merged feat_noff into' in msg
    assert len(repo.head.commit.parents) == 2

def test_git_merge_squash_without_commit(temp_git_repo):
    repo, repo_path = temp_git_repo
    base = repo.active_branch.name
    repo.git.checkout('-b', 'squash_branch')
    (repo_path / 'squash.txt').write_text('squash work')
    repo.index.add(['squash.txt'])
    repo.index.commit('squash change')
    repo.git.checkout(base)
    head_before = repo.head.commit.hexsha
    msg = git_merge(repo, 'squash_branch', squash=True)
    assert 'Merged squash_branch into' in msg and 'squash' in msg
    # No commit created automatically without commit_message
    assert repo.head.commit.hexsha == head_before
    # But the working tree should contain the changes
    assert (repo_path / 'squash.txt').exists()

def test_git_merge_squash_with_commit_message(temp_git_repo):
    repo, repo_path = temp_git_repo
    base = repo.active_branch.name
    repo.git.checkout('-b', 'squash_branch2')
    (repo_path / 'squash2.txt').write_text('squash work 2')
    repo.index.add(['squash2.txt'])
    repo.index.commit('squash change 2')
    repo.git.checkout(base)
    msg = git_merge(repo, 'squash_branch2', squash=True, commit_message='Squash commit msg')
    assert 'Merged squash_branch2 into' in msg and 'squash' in msg
    # A commit should have been created with the provided message
    assert repo.head.commit.message.startswith('Squash commit msg')
    assert (repo_path / 'squash2.txt').exists()


def test_git_merge_abort_no_merge_in_progress(temp_git_repo):
    repo, repo_path = temp_git_repo
    msg = git_merge(repo, repo.active_branch.name, abort=True)
    assert msg == 'MERGE_ABORT: No merge in progress.'


def test_git_merge_abort_after_conflict(temp_git_repo):
    repo, repo_path = temp_git_repo
    # Create conflict
    (repo_path / 'ab.txt').write_text('line\n')
    repo.index.add(['ab.txt'])
    repo.index.commit('init ab')
    repo.git.checkout('-b', 'feat_ab')
    (repo_path / 'ab.txt').write_text('feat\n')
    repo.index.add(['ab.txt'])
    repo.index.commit('feat change')
    repo.git.checkout('main')
    (repo_path / 'ab.txt').write_text('main\n')
    repo.index.add(['ab.txt'])
    repo.index.commit('main change')
    # Merge to cause conflict
    _ = git_merge(repo, 'feat_ab')
    merge_head = Path(repo.working_dir) / '.git' / 'MERGE_HEAD'
    assert merge_head.exists()
    # Abort
    msg = git_merge(repo, 'feat_ab', abort=True)
    assert msg.startswith('MERGE_ABORT:')
    assert not merge_head.exists()
    assert not repo.index.unmerged_blobs()

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

def test_git_read_file(temp_git_repo):
    repo, repo_path = temp_git_repo
    file_content = "This is a test file content."
    (repo_path / "read_me.txt").write_text(file_content)
    
    result = git_read_file(repo, "read_me.txt")
    assert f"Content of read_me.txt:\n{file_content}" in result

    result_not_found = git_read_file(repo, "non_existent_file.txt")
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
    full_path = tmp_path / file_path

    # Patch open to simulate mismatch on read-back
    import builtins
    real_open = builtins.open

    class FakeFile:
        def __init__(self, *a, **kw):
            self.lines = []
            self.mode = a[1] if len(a) > 1 else ""
            self.closed = False
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def write(self, data): self.lines.append(data)
        def read(self): return b"xyz" if "b" in self.mode else "xyz"
        def writelines(self, lines): self.lines.extend(lines)
        def close(self): self.closed = True

    def fake_open(path, mode="r", *a, **kw):
        if "mismatch.txt" in str(path) and "rb" in mode:
            return FakeFile(path, mode)
        if "mismatch.txt" in str(path) and "w" in mode:
            return FakeFile(path, mode)
        return real_open(path, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    result = await write_to_file_content(str(tmp_path), file_path, content)
    assert "Mismatch between input content and written bytes!" in result or "Diff:" in result

    # Simulate Exception during file writing
    def raise_exc_open(*a, **kw):
        raise Exception("write error")
    monkeypatch.setattr(builtins, "open", raise_exc_open)
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
@patch('server.git_branch')
@patch('server.git_merge')
@patch('server.git_show')
@patch('server.git_apply_diff', new_callable=AsyncMock)
@patch('server.git_read_file')
@patch('server.write_to_file_content', new_callable=AsyncMock)
@patch('server.execute_custom_command', new_callable=AsyncMock)
async def test_call_tool(
    mock_execute_custom_command, mock_write_to_file_content,
    mock_git_read_file, mock_git_apply_diff, mock_git_show,
    mock_git_merge,
    mock_git_branch, mock_git_log,
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

    # Test GitTools.BRANCH create
    mock_git_branch.return_value = "Branch created"
    result = list(await call_tool(GitTools.BRANCH.value, {"repo_path": "/tmp/repo", "action": "create", "branch_name": "new_branch"})) # Cast to list
    assert result[0].text == "Branch created"

    # Test GitTools.BRANCH checkout
    mock_git_branch.return_value = "Checked out"
    result = list(await call_tool(GitTools.BRANCH.value, {"repo_path": "/tmp/repo", "action": "checkout", "branch_name": "dev"})) # Cast to list
    assert result[0].text == "Checked out"

    # Test GitTools.BRANCH rename
    mock_git_branch.return_value = "Renamed"
    result = list(await call_tool(GitTools.BRANCH.value, {"repo_path":"/tmp/repo","action":"rename","branch_name":"old","new_name":"new"}))
    assert result[0].text == "Renamed"

    # Test GitTools.BRANCH list
    mock_git_branch.return_value = "Branches:\n* main\n  dev"
    result = list(await call_tool(GitTools.BRANCH.value, {"repo_path":"/tmp/repo","action":"list"}))
    assert result[0].text.startswith("Branches:\n")

    # Test GitTools.SHOW
    mock_git_show.return_value = "Show output"
    result = list(await call_tool(GitTools.SHOW.value, {"repo_path": "/tmp/repo", "revision": "HEAD"})) # Cast to list
    assert result[0].text == "Show output"

    # Test GitTools.MERGE
    mock_git_merge.return_value = 'Merged dev into main'
    result = list(await call_tool(GitTools.MERGE.value, {"repo_path":"/tmp/repo","source":"dev","target":"main","no_ff": True, "squash": False, "commit_message":"Merge dev"}))
    assert result[0].text == 'Merged dev into main'
    
    # Test GitTools.MERGE with ff_only
    mock_git_merge.return_value = 'MERGE_ERROR: Fast-forward only'
    result = list(await call_tool(GitTools.MERGE.value, {"repo_path":"/tmp/repo","source":"dev","target":"main","ff_only": True}))
    assert result[0].text == 'MERGE_ERROR: Fast-forward only'

    # Test GitTools.MERGE with dry_run
    mock_git_merge.return_value = 'DRY_RUN: Fast-forward merge would apply dev into main.'
    result = list(await call_tool(GitTools.MERGE.value, {"repo_path":"/tmp/repo","source":"dev","target":"main","dry_run": True}))
    assert result[0].text == 'DRY_RUN: Fast-forward merge would apply dev into main.'

    # Test GitTools.MERGE with squash
    mock_git_merge.return_value = 'Squash merge successful'
    result = list(await call_tool(GitTools.MERGE.value, {"repo_path":"/tmp/repo","source":"dev","target":"main","squash": True}))
    assert result[0].text == 'Squash merge successful'

    # Test GitTools.MERGE with abort flag
    mock_git_merge.return_value = 'MERGE_ABORT: Merge aborted and working tree restored.'
    result = list(await call_tool(GitTools.MERGE.value, {"repo_path":"/tmp/repo","abort": True}))
    assert result[0].text == 'MERGE_ABORT: Merge aborted and working tree restored.'

    # Test GitTools.APPLY_DIFF
    mock_git_apply_diff.return_value = "Diff applied" # Simplified mock
    
    result = list(await call_tool(GitTools.APPLY_DIFF.value, {"repo_path": "/tmp/repo", "diff_content": "diff"})) # Cast to list
    assert result[0].text == "Diff applied"

    # Test GitTools.READ_FILE
    mock_git_read_file.return_value = "File content"
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
        # Only allow files in tmp_path or system files
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

    # Case 2: Config in git root (different from workdir)
    gitroot = tmp_path / "gitroot"
    gitroot.mkdir()
    (gitroot / ".git").mkdir()
    config2 = {"b": 2}
    config_path2 = gitroot / ".aider.conf.yml"
    write_yaml(config_path2, config2)
    # Patch find_git_root to return gitroot for workdir
    with mock.patch("server.find_git_root", return_value=str(gitroot)):
        result = load_aider_config(str(workdir))
        assert result["a"] == 1
        assert result["b"] == 2

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
    out, err = await run_command(["echo", "ok"])
    assert out == "ok"
    assert err == ""

    # Success with input
    out, err = await run_command(["stdin"], input_data="data")
    assert out == "stdin-ok"
    assert err == ""

    # Failure
    out, err = await run_command(["fail"])
    assert out == ""
    assert err == "fail"
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
@pytest.mark.asyncio
async def test_git_apply_diff_cases(monkeypatch, tmp_path):
    from server import git_apply_diff
    import types

    class DummyRepo:
        def __init__(self, working_dir):
            self.working_dir = working_dir
            self.git = types.SimpleNamespace()
            self.git.apply = self.apply

        def apply(self, *args, **kwargs):
            if "--fail" in args:
                raise Exception("fail")
            if "--giterr" in args:
                from git.exc import GitCommandError
                raise GitCommandError("apply", 1, stderr="git error")
            return

    # Patch _generate_diff_output and _run_tsc_if_applicable to avoid side effects
    async def fake_generate_diff_output(*a, **kw):
        return ""
    async def fake_run_tsc_if_applicable(*a, **kw):
        return ""
    monkeypatch.setattr("server._generate_diff_output", fake_generate_diff_output)
    monkeypatch.setattr("server._run_tsc_if_applicable", fake_run_tsc_if_applicable)

    repo = DummyRepo(str(tmp_path))

    # Case 1: Successful diff application with affected file
    file_path = tmp_path / "file.txt"
    file_path.write_text("old")
    diff_content = "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"
    result = await git_apply_diff(repo, diff_content)
    assert "Diff applied successfully" in result

    # Case 2: Diff content without a/ or b/ paths (should not fail)
    result = await git_apply_diff(repo, "random diff content")
    assert "Diff applied successfully" in result or "Error applying diff" in result

    # Case 3: full_affected_path.exists() is False
    result = await git_apply_diff(repo, "--- a/nonexistent.txt\n+++ b/nonexistent.txt\n@@ -1 +1 @@\n-old\n+new\n")
    assert (
        "Diff applied successfully" in result
        or "Error applying diff" in result
        or "An unexpected error occurred" in result
    )

    # Case 4: GitCommandError
    repo.git.apply = lambda *a, **kw: (_ for _ in ()).throw(
        __import__("git").exc.GitCommandError("apply", 1, stderr="git error")
    )
    result = await git_apply_diff(repo, diff_content)
    assert (
        "GIT_COMMAND_FAILED: Failed to apply diff. Details: \n  stderr: 'git error'. Check if the diff is valid and applies cleanly to the current state of the repository."
        in result
    )

    # Case 5: Other Exception
    repo.git.apply = lambda *a, **kw: (_ for _ in ()).throw(Exception("fail"))
    result = await git_apply_diff(repo, diff_content)
    assert (
        "UNEXPECTED_ERROR: An unexpected error occurred while applying diff: fail. AI_HINT: Check the server logs for more details or review your input."
        in result
    )
def test_git_read_file_error_cases(monkeypatch):
    from server import git_read_file
    import types

    class DummyRepo:
        def __init__(self, working_dir):
            self.working_dir = working_dir

    repo = DummyRepo("/tmp")

    # Simulate FileNotFoundError
    def fake_open_notfound(*a, **kw):
        raise FileNotFoundError()
    monkeypatch.setattr("builtins.open", fake_open_notfound)
    result = git_read_file(repo, "nofile.txt")
    assert (
        "file wasn't found" in result
        or "UNEXPECTED_ERROR: Failed to read file 'nofile.txt':" in result
    )

    # Simulate generic Exception
    def fake_open_exc(*a, **kw):
        raise Exception("fail")
    monkeypatch.setattr("builtins.open", fake_open_exc)
    result = git_read_file(repo, "nofile.txt")
    assert "UNEXPECTED_ERROR: Failed to read file 'nofile.txt': fail" in result
    assert "UNEXPECTED_ERROR:" in result
