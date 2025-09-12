import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from pathlib import Path


@pytest.fixture
def fake_aider_success(monkeypatch):
    """Monkeypatch asyncio.create_subprocess_shell to return rc=0 and stdout 'Applied edit to file.txt'."""
    class FakeProc:
        def __init__(self):
            self.returncode = 0
            
        async def communicate(self):
            return (b"Applied edit to file.txt", b"")
    
    async def mock_create_subprocess_shell(*args, **kwargs):
        return FakeProc()
        
    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)


@pytest.fixture
def fake_worktree_add_exec(monkeypatch):
    """Monkeypatch asyncio.create_subprocess_exec to detect ['git','worktree','add',...] and return rc=0."""
    original_create_subprocess_exec = asyncio.create_subprocess_exec
    
    async def mock_create_subprocess_exec(*args, **kwargs):
        if "worktree" in args and "add" in args:
            # Return a mock process with returncode 0
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            return mock_proc
        return await original_create_subprocess_exec(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create_subprocess_exec)


@pytest.fixture
def git_status_clean_true(monkeypatch):
    """Monkeypatch server._git_status_clean to return True."""
    monkeypatch.setattr("server._git_status_clean", lambda *args, **kwargs: True)


@pytest.fixture
def git_status_clean_false(monkeypatch):
    """Monkeypatch server._git_status_clean to return False."""
    monkeypatch.setattr("server._git_status_clean", lambda *args, **kwargs: False)
