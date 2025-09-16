import asyncio
import os
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from pathlib import Path
import yaml


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


@pytest.fixture
def patch_create_subprocess_shell(monkeypatch):
    """Fixture to patch asyncio.create_subprocess_shell with a custom factory."""
    def _patch(factory):
        async def mock_create_subprocess_shell(command, *args, **kwargs):
            return factory(command)
        monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    
    return _patch


@pytest.fixture
def fs_guard(monkeypatch, tmp_path):
    """Fixture to guard filesystem access and provide helpers for tests."""
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
    
    def write_yaml(path, data):
        with real_open(path, "w") as f:
            yaml.dump(data, f)
            
    return {"write_yaml": write_yaml, "tmp_path": tmp_path}
