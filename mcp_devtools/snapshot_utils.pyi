"""Type stubs for snapshot_utils"""

from pathlib import Path
from typing import Optional, Set


def now_ts() -> str: ...

def ensure_snap_dir(repo_dir: str) -> Path: ...

def looks_binary_bytes(data: bytes) -> bool: ...

def sanitize_binary_markers(diff_text: str) -> str: ...

def snapshot_worktree(
    repo_dir: str, 
    exclude_untracked: Optional[Set[str]] = None
) -> str: ...

def save_snapshot(repo_dir: str, ts: str, kind: str, content: str) -> Path: ...
