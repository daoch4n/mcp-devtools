"""Tests for snapshot_utils.py"""

from pathlib import Path
import os
import re
import tempfile
from unittest.mock import patch, MagicMock
from typing import Any
import pytest
import git

from mcp_devtools import snapshot_utils as su


class FakeRepo:
    """Mock git.Repo for testing"""
    def __init__(self, untracked_files=None, diff_result="", head_exception=None, diff_exception=None):
        self.untracked_files = untracked_files or []
        self._diff_result = diff_result
        self._head_exception = head_exception
        self._diff_exception = diff_exception
        self.head = MagicMock()
        self.head.commit = MagicMock()
        if head_exception:
            self.head.commit.side_effect = head_exception
        self.git = MagicMock()
        if diff_exception:
            self.git.diff.side_effect = diff_exception
        else:
            self.git.diff.return_value = diff_result

def test_looks_binary_bytes_variants():
    """Test binary detection with various inputs"""
    # Test NUL byte detection
    assert su.looks_binary_bytes(b"hello\x00world") is True
    
    # Test valid UTF-8 text
    assert su.looks_binary_bytes(b"hello world") is False
    assert su.looks_binary_bytes("hello world 🌍".encode("utf-8")) is False
    
    # Test non-UTF-8 bytes
    assert su.looks_binary_bytes(b"\xff\xfe") is True

def test_sanitize_binary_markers():
    """Test sanitization of binary diff markers"""
    # Test GIT binary patch block
    diff_with_patch = """diff --git a/file.bin b/file.bin
index e69de29..d00491f 100644
GIT binary patch
literal 123
zcmeAS@N?(olHy`lMa ...

literal 0
HcmV?d00001
--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old line
+new line
Binary files a/old.bin and b/new.bin differ"""
    
    result = su.sanitize_binary_markers(diff_with_patch)
    assert "[Binary file omitted from diff]" in result
    assert "GIT binary patch" not in result
    assert "literal 123" not in result
    assert "Binary files a/old.bin and b/new.bin differ" not in result
    assert "@@ -1 +1 @@" in result  # Regular diff content preserved

def test_snapshot_worktree_untracked_and_exclusions(tmp_path):
    """Test untracked file handling with exclusions"""
    # Create a real git repo for this test
    repo = git.Repo.init(tmp_path)
    
    # Create various files
    (tmp_path / "text.txt").write_text("hello world")
    (tmp_path / "binary.bin").write_bytes(b"hello\x00world")
    (tmp_path / "excluded.txt").write_text("should be excluded")
    (tmp_path / ".mcp-devtools").mkdir()
    (tmp_path / ".mcp-devtools" / "internal.txt").write_text("internal")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "included.txt").write_text("included in subdir")
    
    # Test with exclusions
    exclude_set = {"excluded.txt"}
    result = su.snapshot_worktree(str(tmp_path), exclude_untracked=exclude_set)
    
    # Should include binary.bin and subdir/included.txt (untracked files)
    # Should exclude .mcp-devtools/internal.txt and excluded.txt
    assert "binary.bin" in result or "subdir/included.txt" in result
    assert ".mcp-devtools/internal.txt" not in result
    assert "excluded.txt" not in result

def test_snapshot_worktree_tracked_diff_error():
    """Test snapshot_worktree when tracked diff fails"""
    with patch.object(su.git, 'Repo', 
               return_value=FakeRepo(
                   untracked_files=[],
                   head_exception=ValueError("No HEAD"),
                   diff_exception=git.exc.GitCommandError('diff', 1, b'fail')
               )):
        result = su.snapshot_worktree("/fake/repo")
        assert result == ""

def test_snapshot_worktree_invalid_repo_during_untracked():
    """Test snapshot_worktree when repo becomes invalid during untracked access"""
    class FakeRepoInvalid:
        """Fake repo that raises InvalidGitRepositoryError when accessing untracked_files"""
        def __init__(self):
            self.head = MagicMock()
            self.head.commit = "dummy"
            self.git = MagicMock()
            self.git.diff.return_value = ""
        
        @property
        def untracked_files(self):
            raise git.exc.InvalidGitRepositoryError("Invalid repo")
    
    with patch.object(su.git, 'Repo', return_value=FakeRepoInvalid()):
        result = su.snapshot_worktree("/fake/repo")
        assert result == ""

def test_snapshot_worktree_skip_non_file_and_oserror(tmp_path):
    """Test skipping non-files and handling OSError"""
    # Create test structure
    (tmp_path / "adir").mkdir()
    (tmp_path / "ok.txt").write_text("ok content")
    (tmp_path / "err.txt").write_text("error content")
    
    fake_repo = FakeRepo(
        untracked_files=["adir", "ok.txt", "err.txt"],
        diff_result=""
    )
    
    # Mock Path.read_bytes to raise OSError for err.txt
    original_read_bytes = Path.read_bytes
    
    def mock_read_bytes(self):
        if self.name == "err.txt":
            raise OSError("Permission denied")
        return original_read_bytes(self)
    
    with patch.object(su.git, 'Repo', return_value=fake_repo), \
         patch.object(Path, 'read_bytes', side_effect=mock_read_bytes, autospec=True):
        result = su.snapshot_worktree(str(tmp_path))
        
        # Should include ok.txt but not adir (directory) or err.txt (OSError)
        assert "ok.txt" in result
        assert "adir" not in result
        assert "err.txt" not in result

def test_ensure_and_save_snapshot(tmp_path):
    """Test directory creation and snapshot saving"""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    # Test ensure_snap_dir creates directory
    snap_dir = su.ensure_snap_dir(str(repo_dir))
    assert snap_dir.exists()
    assert snap_dir.name == ".mcp-devtools"
    
    # Test save_snapshot
    content = "test diff content\nwith multiple lines"
    ts = "20231201T103045"
    kind = "before"
    
    saved_path = su.save_snapshot(str(repo_dir), ts, kind, content)
    
    # Verify file was created with correct name and content
    assert saved_path.exists()
    assert saved_path.name == f"ai_edit_{ts}_{kind}.diff"
    assert saved_path.read_text() == content

def test_now_ts_format():
    """Test timestamp format"""
    ts = su.now_ts()
    
    # Should match YYYYMMDDTHHMMSS format
    assert len(ts) == 15  # YYYY(4) + MM(2) + DD(2) + T(1) + HH(2) + MM(2) + SS(2)
    assert ts[8] == 'T'
    assert re.match(r'^\d{8}T\d{6}$', ts) is not None
