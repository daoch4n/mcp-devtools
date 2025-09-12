"""Utility functions for worktree snapshots"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

import git


def now_ts() -> str:
    """Return current timestamp in YYYYMMDDTHHMMSS format"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def ensure_snap_dir(repo_dir: str) -> Path:
    """Ensure snapshot directory exists"""
    p = Path(repo_dir) / ".mcp-devtools"
    p.mkdir(parents=True, exist_ok=True)
    return p


def looks_binary_bytes(data: bytes) -> bool:
    """Check if bytes look like binary data"""
    if b"\x00" in data:
        return True
    try:
        data.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def sanitize_binary_markers(diff_text: str) -> str:
    """Replace binary diff markers with placeholders"""
    placeholder = "[Binary file omitted from diff]"
    lines = diff_text.splitlines()
    out: list[str] = []
    skip_binary_block = False
    
    for ln in lines:
        if ln.startswith("GIT binary patch"):
            out.append(placeholder)
            skip_binary_block = True
            continue
            
        if skip_binary_block:
            if re.match(r"(diff --git|\+\+\+|---|^$)", ln):
                skip_binary_block = False
            else:
                continue
                
        if ln.startswith("Binary files ") and ln.endswith(" differ"):
            out.append(placeholder)
        else:
            out.append(ln)
            
    return "\n".join(out)


def snapshot_worktree(
    repo_dir: str,
    exclude_untracked: Optional[Set[str]] = None
) -> str:
    """Capture worktree snapshot.

    Raises exceptions to caller for unexpected failures so they can be surfaced/logged appropriately.
    """
    repo_local = git.Repo(repo_dir)
    exclude_untracked = exclude_untracked or set()

    # Determine target
    try:
        _ = repo_local.head.commit
        target = "HEAD"
    except (ValueError, git.exc.BadName):
        target = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # Empty tree SHA

    # Tracked changes
    try:
        tracked = repo_local.git.diff(target)
    except git.exc.GitCommandError:
        tracked = ""

    # Untracked files
    untracked_parts: list[str] = []
    try:
        # Sort for deterministic output
        for uf in sorted(repo_local.untracked_files):
            if uf.startswith(".mcp-devtools/") or uf in exclude_untracked:
                continue

            full = Path(repo_dir) / uf
            if not full.is_file():
                continue

            try:
                raw = full.read_bytes()
                if looks_binary_bytes(raw):
                    body = "+[Binary file omitted from diff]\n"
                else:
                    text = raw.decode("utf-8", errors="ignore")
                    body = "".join(f"+{line}\n" for line in text.splitlines())

                hdr = f"--- /dev/null\n+++ b/{uf}\n"
                untracked_parts.append(hdr + body)
            except OSError:
                # Skip file if we can't read it
                continue
    except git.exc.InvalidGitRepositoryError:
        # If repo is invalid, return empty snapshot rather than raising here
        return ""

    combined = "\n".join([p for p in [tracked] + untracked_parts if p]).strip()
    return sanitize_binary_markers(combined)


def save_snapshot(repo_dir: str, ts: str, kind: str, content: str) -> Path:
    """Save snapshot to file.

    Raises OSError to caller on failure so it can be handled explicitly.
    """
    snap_dir = ensure_snap_dir(repo_dir)
    path = snap_dir / f"ai_edit_{ts}_{kind}.diff"
    path.write_text(content, encoding="utf-8")
    return path
