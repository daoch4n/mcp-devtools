"""
MCP Git Server

This module implements a server for the MCP (Multi-Agent Collaboration Platform)
that provides a set of Git-related tools and AI-powered code editing capabilities
using Aider. It allows clients to interact with Git repositories, perform file
operations, execute commands, and initiate AI-driven code modifications.

Key Components:
- Git Operations: Functions for common Git commands like status, diff, commit,
  reset, log, checkout, and applying diffs.
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
import asyncio
import uuid
import shutil
import time
import math
from datetime import datetime
from pathlib import Path, PurePath
from typing import Sequence, Optional, TypeAlias, Any, Dict, List, Tuple, Union, cast, Literal, Set
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
import git
from git.exc import GitCommandError
from pydantic import BaseModel, Field
import asyncio
import tempfile
import os
import re
import difflib
import shlex
import json
import yaml
import subprocess

# Git constant for the empty tree SHA, used for diffing initial commits.
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

# Session management constants
MCP_SESSION_TTL_SECONDS = int(os.getenv("MCP_SESSION_TTL_SECONDS", "3600"))

def _env_truthy(val: Optional[str]) -> bool:
    return (val or "").lower() in ("1", "true", "yes")
MCP_EXPERIMENTAL_WORKTREES = _env_truthy(os.getenv("MCP_EXPERIMENTAL_WORKTREES"))

logging.basicConfig(level=logging.DEBUG)

from starlette.applications import Starlette
try:
    from mcp_devtools.snapshot_utils import (
        ensure_snap_dir,
        looks_binary_bytes,
        now_ts,
        sanitize_binary_markers,
        save_snapshot,
        snapshot_worktree,
    )
except (ImportError, ModuleNotFoundError):
    # Define minimal fallbacks so server can start even if package missing
    from pathlib import Path
    import os
    from datetime import datetime, timezone
    def ensure_snap_dir(repo_dir: str) -> Path:
        p = Path(repo_dir) / ".mcp-devtools"
        p.mkdir(parents=True, exist_ok=True)
        return p
    def now_ts() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    def looks_binary_bytes(data: bytes) -> bool:
        try:
            data.decode("utf-8")
            return False
        except Exception:
            return True
    def sanitize_binary_markers(diff_text: str) -> str:
        return diff_text
    def snapshot_worktree(repo_dir: str, exclude_untracked: Optional[Set[str]] = None) -> str:
        # Fallback: no snapshot available
        return ""
    def save_snapshot(repo_dir: str, ts: str, kind: str, content: str) -> Path:
        d = ensure_snap_dir(repo_dir)
        path = d / f"ai_edit_{ts}_{kind}.diff"
        try:
            path.write_text(content, encoding="utf-8")
        except Exception:
            logging.getLogger(__name__).debug(f"Failed to save snapshot to {path}")
        return path
from starlette.routing import Route, Mount
from starlette.responses import Response
from starlette.requests import Request
from starlette.types import Scope, Receive, Send

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


# === Session Store Helpers ===
def _devtools_dir(repo_root: str) -> Path:
    """Get the .mcp-devtools directory path."""
    return Path(repo_root) / ".mcp-devtools"

def _sessions_file(repo_root: str) -> Path:
    """Get the path to the sessions.json file."""
    return _devtools_dir(repo_root) / "sessions.json"

def _workspaces_dir(repo_root: str) -> Path:
    """Get the workspaces directory path."""
    return _devtools_dir(repo_root) / "workspaces"

# Async lock for session file operations
_sessions_lock = asyncio.Lock()

def _load_sessions(repo_root: str) -> dict[str, Any]:
    """Load sessions from the JSON file, returning empty dict if file doesn't exist or is invalid."""
    path = _sessions_file(repo_root)
    if not path.exists():
        return {}
    
    try:
        with open(path, 'r') as f:
            return cast(dict[str, Any], json.load(f))
    except (json.JSONDecodeError, IOError) as e:
        logger.debug(f"Failed to load sessions from {path}: {e}")
        return {}

def _save_sessions(repo_root: str, data: dict[str, Any]) -> None:
    """Save sessions to the JSON file."""
    path = _sessions_file(repo_root)
    try:
        # Ensure the directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to temporary file first for atomic operation
        tmp_path = path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Atomically replace the old file
        try:
            os.replace(tmp_path, path)
        except OSError as e:
            # Best-effort cleanup of tmp file
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise
    except IOError as e:
        logger.debug(f"Failed to save sessions to {path}: {e}")

def _generate_session_id() -> str:
    """Generate a new session ID."""
    return str(uuid.uuid4())

def _record_session_start(repo_root: str, session_id: str, workspace_dir: str, use_worktree: bool) -> None:
    """Record the start of a session."""
    session_record = {
        "session_id": session_id,
        "repo_root": repo_root,
        "workspace_dir": workspace_dir,
        "use_worktree": use_worktree,
        "status": "running",
        "created_at": time.time(),
        "last_updated": time.time(),
        "completed_at": None,
        "success": None,
        "purged_at": None
    }
    
    sessions = _load_sessions(repo_root)
    sessions[session_id] = session_record
    _save_sessions(repo_root, sessions)

async def _record_session_start_async(repo_root: str, session_id: str, workspace_dir: str, use_worktree: bool) -> None:
    async with _sessions_lock:
        await asyncio.to_thread(_record_session_start, repo_root, session_id, workspace_dir, use_worktree)

async def _load_sessions_async(repo_root: str) -> dict[str, Any]:
    """Load sessions from the JSON file asynchronously with locking."""
    async with _sessions_lock:
        return await asyncio.to_thread(_load_sessions, repo_root)

def _record_session_update(repo_root: str, session_id: str, **fields: Any) -> None:
    """Update session record with additional fields."""
    sessions = _load_sessions(repo_root)
    if session_id in sessions:
        sessions[session_id].update(fields)
        sessions[session_id]["last_updated"] = time.time()
        _save_sessions(repo_root, sessions)

async def _record_session_update_async(repo_root: str, session_id: str, **fields: Any) -> None:
    async with _sessions_lock:
        await asyncio.to_thread(_record_session_update, repo_root, session_id, **fields)

def _record_session_complete(repo_root: str, session_id: str, success: bool) -> None:
    """Record the completion of a session."""
    _record_session_update(repo_root, session_id, status="completed", completed_at=time.time(), success=success)

async def _record_session_complete_async(repo_root: str, session_id: str, success: bool) -> None:
    async with _sessions_lock:
        await asyncio.to_thread(_record_session_complete, repo_root, session_id, success)

def _delete_session_record(repo_root: str, session_id: str) -> None:
    """Delete a session record from the sessions file."""
    sessions = _load_sessions(repo_root)
    if session_id in sessions:
        del sessions[session_id]
        _save_sessions(repo_root, sessions)

async def _delete_session_record_async(repo_root: str, session_id: str) -> None:
    async with _sessions_lock:
        await asyncio.to_thread(_delete_session_record, repo_root, session_id)


def _get_ttl_seconds() -> int:
    """Get the session TTL in seconds."""
    return MCP_SESSION_TTL_SECONDS

def _is_session_expired(session: dict[str, Any]) -> bool:
    """Check if a session is expired based on TTL."""
    ttl = _get_ttl_seconds()
    last_active = max(
        session.get("completed_at", 0) or 0,
        session.get("last_updated", 0) or 0
    )
    return (time.time() - last_active) > ttl

def _git_status_clean(path: str) -> bool:
    """Check if the git repository status is clean (no changes)."""
    try:
        repo = git.Repo(path)
        return not bool(repo.git.status("--porcelain"))
    except Exception as e:
        logger.debug(f"Failed to check git status for {path}: {e}")
        return False

def _purge_worktree(repo_root: str, workspace_dir: str) -> None:
    """Purge a worktree directory, trying git worktree remove first, then rm -rf."""
    try:
        # Try to remove via git worktree command first
        subprocess.run(
            ["git", "worktree", "remove", "--force", workspace_dir],
            cwd=repo_root,
            check=True,
            capture_output=True
        )
        logger.debug(f"Successfully removed worktree {workspace_dir} via git command")
    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to remove worktree via git command: {e}. Trying rm -rf.")
        # Fallback to rm -rf
        try:
            if os.path.exists(workspace_dir):
                shutil.rmtree(workspace_dir)
                logger.debug(f"Successfully removed worktree {workspace_dir} via rm -rf")
        except Exception as e2:
            logger.debug(f"Failed to remove worktree {workspace_dir} via rm -rf: {e2}")

async def _purge_worktree_async(repo_root: str, workspace_dir: str) -> None:
    """Purge a worktree directory asynchronously, trying git worktree remove first, then rm -rf."""
    try:
        # If already gone, nothing to do
        if not os.path.exists(workspace_dir):
            return
        # Try to remove via git worktree command first
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "remove", "--force", workspace_dir,
            cwd=repo_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0:
            logger.debug(f"Successfully removed worktree {workspace_dir} via git command")
        else:
            logger.debug(f"Failed to remove worktree via git command: {stderr.decode()}. Trying rm -rf.")
            # Fallback to rm -rf
            if os.path.exists(workspace_dir):
                await asyncio.to_thread(shutil.rmtree, workspace_dir)
                logger.debug(f"Successfully removed worktree {workspace_dir} via rm -rf")
    except Exception as e:
        logger.debug(f"Failed to remove worktree {workspace_dir} via git command: {e}. Trying rm -rf.")
        # Fallback to rm -rf
        try:
            if os.path.exists(workspace_dir):
                await asyncio.to_thread(shutil.rmtree, workspace_dir)
                logger.debug(f"Successfully removed worktree {workspace_dir} via rm -rf")
        except Exception as e2:
            logger.debug(f"Failed to remove worktree {workspace_dir} via rm -rf: {e2}")

def _apply_workspace_changes_to_root(workspace_dir: str, repo_path: str, files: list[str]) -> None:
    """Apply changes from workspace to root repository for specified files."""
    for rel in files:
        try:
            src = Path(workspace_dir) / rel
            dst = Path(repo_path) / rel
            if src.is_file():
                # Avoid clobbering root-side edits: only copy if workspace file is newer
                try:
                    src_mtime = src.stat().st_mtime
                    dst_mtime = dst.stat().st_mtime if dst.exists() else 0
                except Exception:
                    src_mtime = time.time()
                    dst_mtime = 0
                if src_mtime > dst_mtime:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        except Exception as e:
            logger.debug(f"[ai_edit] Failed to apply workspace change for {rel}: {e}")

def cleanup_expired_sessions(repo_root: str) -> None:
    """Clean up expired sessions and their worktrees."""
    try:
        sessions = _load_sessions(repo_root)
        changed = False
        
        for session_id, session in list(sessions.items()):
            if session.get("status") == "completed" and _is_session_expired(session):
                # Clean up worktree if it exists and wasn't already purged
                if session.get("use_worktree") and session.get("workspace_dir") and not session.get("purged_at"):
                    _purge_worktree(repo_root, session["workspace_dir"])
                # Delete the session record
                del sessions[session_id]
                changed = True
        
        if changed:
            _save_sessions(repo_root, sessions)
    except Exception as e:
        logger.debug(f"Error during session cleanup: {e}")

async def cleanup_expired_sessions_async(repo_root: str) -> None:
    async with _sessions_lock:
        await asyncio.to_thread(cleanup_expired_sessions, repo_root)


def _read_last_aider_session_text(directory_path: str) -> str:
    """
    Read the Aider chat history and return the content of the last session.
    Returns the content from the last '# aider chat started at' anchor to the end,
    or the full content if no anchor is found, or empty string if file is missing.
    """
    try:
        history_path = Path(directory_path) / ".aider.chat.history.md"
        if not history_path.exists():
            return ""
        history_content = history_path.read_text(encoding="utf-8", errors="ignore")

        # Find the last session
        anchor = "# aider chat started at"
        last_anchor_pos = history_content.rfind(anchor)
        if last_anchor_pos != -1:
            return history_content[last_anchor_pos:]
        else:
            return history_content
    except Exception as e:
        logger.debug(f"Failed to read Aider chat history: {e}")
        return ""

# Try to import tiktoken for better token counting
try:
    import tiktoken  # type: ignore[import-not-found]
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def _approx_token_count(text: str) -> int:
        """
        Count tokens using tiktoken when available, otherwise fall back to character-based approximation.
        """
        try:
            return len(tokenizer.encode(text))
        except Exception:
            # Fallback to character-based approximation if tiktoken fails
            return math.ceil(len(text) / 4)
except ImportError:
    def _approx_token_count(text: str) -> int:
        """
        Approximate token count using the rule of thumb: tokens ≈ characters / 4
        """
        return math.ceil(len(text) / 4)

def _parse_aider_token_stats(text: str) -> tuple[int, int]:
    """
    Parse Aider token statistics from chat history text.
    Looks for patterns like "> Tokens: 21k sent, 2.6k received."
    
    Args:
        text: The Aider chat history text
        
    Returns:
        A tuple of (sent_tokens, received_tokens) as integers
    """
    total_sent = 0
    total_received = 0
    
    # Pattern to match token stats lines
    pattern = r"> Tokens: ([\d\.km,]+) sent, ([\d\.km,]+) received"
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    for sent_str, received_str in matches:
        def parse_token_value(value_str: str) -> int:
            # Remove commas and convert to lowercase
            value_str = value_str.replace(',', '').lower()
            
            # Handle k/m suffixes
            if value_str.endswith('k'):
                return math.ceil(float(value_str[:-1]) * 1000)
            elif value_str.endswith('m'):
                return math.ceil(float(value_str[:-1]) * 1000000)
            else:
                return math.ceil(float(value_str))
        
        try:
            sent_tokens = parse_token_value(sent_str)
            received_tokens = parse_token_value(received_str)
            total_sent += sent_tokens
            total_received += received_tokens
        except ValueError:
            # Skip invalid token values
            continue
    
    return (total_sent, total_received)

def _split_aider_sessions(text: str) -> list[str]:
    """
    Split Aider chat history into sessions by the '# aider chat started at' anchor.
    Each chunk will include the anchor line for clarity.
    
    Args:
        text: The full chat history text
        
    Returns:
        A list of session chunks, each starting with the anchor
    """
    anchor = "# aider chat started at"
    chunks = []
    lines = text.split('\n')
    current_chunk: list[str] = []
    
    for line in lines:
        if line.startswith(anchor) and current_chunk:
            # Found a new session, save the previous one
            chunks.append('\n'.join(current_chunk) + '\n')
            current_chunk = [line]
        else:
            current_chunk.append(line)
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk) + '\n')
        
    return chunks

def _rebuild_history_with_summary(summary: str, kept_sessions: list[str]) -> str:
    """
    Construct a new chat history with a summary block and kept sessions.
    
    Args:
        summary: The summary of older sessions
        kept_sessions: List of recent session texts to keep
        
    Returns:
        The reconstructed chat history
    """
    # Create summary block
    summary_block = f"# Summary of older chat sessions\n\n{summary}\n\n"
    
    # Combine summary with kept sessions
    return summary_block + ''.join(kept_sessions)

def _get_last_aider_reply(directory_path: str) -> Optional[str]:
    """
    Read the Aider chat history, extract the last session, clean it,
    and return the last assistant reply.
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

        # In architect mode, the reply is a mix of prose and large code blocks.
        # The goal is to extract all text content while discarding only the
        # fenced code blocks. A line-by-line parser is more robust than a
        # single large regex for this.
        
        lines = session_content.split('\n')
        
        # Isolate the assistant's reply by finding the start of the last message
        # that is NOT part of the user's command prompt.
        
        # Find the last occurrence of '####' which indicates the start of an assistant message.
        last_message_start = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith('####'):
                last_message_start = i
                break
        
        # If no '####' is found, fallback to the first non-prompt line.
        if last_message_start == -1:
            for i, line in enumerate(lines):
                if not line.startswith('> '):
                    last_message_start = i
                    break
        
        relevant_lines = lines[last_message_start:]
        
        final_text_lines: List[str] = []
        in_code_block = False
        
        for line in relevant_lines:
            # Toggle state when a code fence is encountered.
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue # Exclude the fence line itself from the output.
            
            # Ignore lines that are inside a code block.
            if in_code_block:
                continue
                
            stripped = line.strip()
            
            # Filter out metadata and leftover artifacts.
            if stripped.startswith('> Tokens:'):
                continue
            if re.fullmatch(r'[\w\-\./]+\.[\w\-\./]+', stripped):
                continue
            
            # Append the cleaned, relevant line.
            final_text_lines.append(line)
            
        # Post-process the collected lines to remove any trailing blank lines
        # that were preserved.
        while final_text_lines and not final_text_lines[-1].strip():
            final_text_lines.pop()
        
        # Hotfix: If the first line is part of a numbered or bulleted list
        # (e.g., "1. " or "- "), it's likely part of the prompt, so skip it.
        if final_text_lines:
            first_line_stripped = final_text_lines[0].strip()
            if re.match(r'^\s*(\d+\.|\-|\*)\s', first_line_stripped):
                final_text_lines.pop(0)

        # Join the collected lines and perform final cleanup.
        full_reply = '\n'.join(final_text_lines)
        
        # Remove any remaining "####" markdown headers, keeping the text.
        full_reply = re.sub(r'####\s?', '', full_reply)
        
        return full_reply.strip()

    except Exception as e:
        logger.debug(f"Failed to get Aider chat history: {e}")
        return None


def _extract_touched_files(diff_text: str) -> set[str]:
    """Extract touched file paths from a unified git diff text.
    Handles modified, added, deleted, renamed, and binary blocks."""
    touched: set[str] = set()
    
    def normalize_path(path: str) -> str:
        """Strip quotes, leading a/ or b/, and handle /dev/null"""
        # Strip surrounding quotes if present
        if (path.startswith('"') and path.endswith('"')) or \
           (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        
        # Strip leading a/ or b/ prefix
        if path.startswith('a/') or path.startswith('b/'):
            path = path[2:]
            
        return path
    
    # Handle diff --git lines
    diff_git_pattern = re.compile(r'^\s*diff --git\s+(?:"?a/(.+?)"?)\s+(?:"?b/(.+?)"?)\s*$', re.MULTILINE)
    for match in diff_git_pattern.finditer(diff_text):
        a_path, b_path = match.groups()
        a_path = normalize_path(a_path)
        b_path = normalize_path(b_path)
        
        if a_path != '/dev/null':
            touched.add(a_path)
        if b_path != '/dev/null':
            touched.add(b_path)
    
    # Handle +++ and --- lines
    hunk_pattern = re.compile(r'^\s*[\+\-]{3}\s+(?:(?:a|b)/)?(.*)$', re.MULTILINE)
    for match in hunk_pattern.finditer(diff_text):
        path = normalize_path(match.group(1))
        if path != '/dev/null':
            touched.add(path)
    
    # Handle rename from/to lines
    rename_from_pattern = re.compile(r'^\s*rename from\s+(?:(?:a|b)/)?(.*)$', re.MULTILINE)
    rename_to_pattern = re.compile(r'^\s*rename to\s+(?:(?:a|b)/)?(.*)$', re.MULTILINE)
    
    for match in rename_from_pattern.finditer(diff_text):
        path = normalize_path(match.group(1))
        if path != '/dev/null':
            touched.add(path)
            
    for match in rename_to_pattern.finditer(diff_text):
        path = normalize_path(match.group(1))
        if path != '/dev/null':
            touched.add(path)
    
    # Handle Binary files lines
    binary_pattern = re.compile(r'^\s*Binary files\s+(?:(?:a|b)/)?(.*?)\s+and\s+(?:(?:a|b)/)?(.*?)\s+differ$', re.MULTILINE)
    for match in binary_pattern.finditer(diff_text):
        a_path, b_path = match.groups()
        a_path = normalize_path(a_path)
        b_path = normalize_path(b_path)
        
        if a_path != '/dev/null':
            touched.add(a_path)
        if b_path != '/dev/null':
            touched.add(b_path)
    
    # Exclude internal snapshot artifacts just in case
    touched = {p for p in touched if not p.startswith('.mcp-devtools/')}
    return touched


def _collect_touched_files(repo: git.Repo, diff_target: str, final_diff_text: str, new_untracked: list[str]) -> set[str]:
    """
    Collect touched files using git plumbing commands with fallback to regex parsing.
    
    Args:
        repo: Git repository object
        diff_target: The target to diff against (e.g., "HEAD")
        final_diff_text: The diff text for fallback parsing
        new_untracked: List of new untracked files to include
        
    Returns:
        Set of touched file paths
    """
    touched_files: set[str] = set()
    
    try:
        # Try to use git plumbing command for reliable parsing
        diff_output = repo.git.diff('--name-status', '-z', diff_target)
        
        # Parse NUL-separated records
        records = diff_output.split('\x00')
        i = 0
        while i < len(records) - 1:  # -1 because last element is usually empty
            if not records[i]:  # Skip empty records
                i += 1
                continue
                
            rec = records[i]
            # Split at the first TAB to separate status and path
            if '\t' in rec:
                status_code, first_path = rec.split('\t', 1)
            else:
                status_code = rec
                first_path = ''
                
            # Handle rename/copy operations (status starts with R or C)
            if status_code.startswith(('R', 'C')):
                # For renames/copies, the next record is the new path
                if i + 1 < len(records):
                    old_path = first_path
                    new_path = records[i + 1]
                    # Add both paths, filtering out /dev/null and .mcp-devtools/
                    for path in [old_path, new_path]:
                        if path and path != '/dev/null' and not path.startswith('.mcp-devtools/'):
                            touched_files.add(path)
                    i += 2
                else:
                    # Malformed record, skip
                    i += 1
            else:
                # Regular operations (M, A, D, etc.)
                path = first_path
                if path and path != '/dev/null' and not path.startswith('.mcp-devtools/'):
                    touched_files.add(path)
                i += 1
    except Exception as e:
        # Fallback to regex parsing if git command fails
        logger.debug(f"Git plumbing command failed, falling back to regex parsing: {e}")
        touched_files = _extract_touched_files(final_diff_text)
    
    # Always add new untracked files
    for untracked_file in new_untracked:
        if not untracked_file.startswith('.mcp-devtools/'):
            touched_files.add(untracked_file)
            
    return touched_files

# === AI_HINT helper builders (keep terse, agent-friendly) ===

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


def _ensure_gitignore_has_devtools(repo_root: str) -> None:
    """
    Ensure .gitignore contains an entry for .mcp-devtools/ directory.
    
    Args:
        repo_root: The Git repository root path
    """
    try:
        gitignore_path = Path(repo_root) / ".gitignore"
        
        # Pattern to match .mcp-devtools entries (with or without trailing slash)
        pattern = re.compile(r"^\s*\.mcp-devtools\/?\s*$")
        
        # If .gitignore doesn't exist, create it with the entry
        if not gitignore_path.exists():
            gitignore_path.write_text(".mcp-devtools/\n", encoding="utf-8")
            logger.debug(f"Created .gitignore with .mcp-devtools/ entry at {gitignore_path}")
            return
        
        # Read existing content
        content = gitignore_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        
        # Check if entry already exists
        for line in lines:
            if pattern.match(line):
                logger.debug(f".mcp-devtools/ already in .gitignore at {gitignore_path}")
                return
        
        # Re-read and append to reduce race window
        try:
            current = gitignore_path.read_text(encoding="utf-8")
        except Exception:
            current = content
            
        # Check again if entry already exists to avoid duplicates
        for line in current.splitlines():
            if pattern.match(line):
                logger.debug(f".mcp-devtools/ already in .gitignore at {gitignore_path} (confirmed after re-read)")
                return
            
        with open(gitignore_path, "a", encoding="utf-8") as f:
            # Add newline if file doesn't end with one
            if current and not current.endswith("\n"):
                f.write("\n")
            f.write(".mcp-devtools/\n")
        
        logger.debug(f"Added .mcp-devtools/ to .gitignore at {gitignore_path}")
        
    except Exception as e:
        logger.debug(f"Failed to ensure .mcp-devtools/ in .gitignore: {e}")

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
                for item in value:
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

class AiEdit(BaseModel):
    """
    Represents the input schema for the `ai_edit` tool.
    """
    repo_path: str = Field(description="The absolute path to the Git repository's working directory where the AI edit should be performed.")
    message: str = Field(description="A detailed natural language message describing the code changes to make. Be specific about files, desired behavior, and any constraints.")
    files: List[str] = Field(description="A list of file paths (relative to the repository root) that Aider should operate on. This argument is mandatory.")
    continue_thread: bool = Field(description="Required. Whether to continue the Aider thread by restoring chat history. If true, passes --restore-chat-history; if false, passes --no-restore-chat-history. Clients must explicitly choose.")
    options: list[str] | None = Field(
        None,
        description="Optional. A list of additional command-line options to pass directly to Aider. Each option should be a string."
    )
    session_id: str | None = Field(
        None,
        description="Optional. A session ID to associate with this edit operation. If not provided, a new UUID will be generated."
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
    SHOW = "git_show"
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
    return str(repo.git.status())


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
    return str(repo.git.diff(*args))

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
            
        return str(repo.git.show(*args))
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
    options: list[str] | None,
    continue_thread: bool,
    aider_path: Optional[str] = None,
    config_file: Optional[str] = None,
    env_file: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    AI pair programming tool for making targeted code changes using Aider.
    This function encapsulates the logic from aider_mcp/server.py's edit_files tool.

    Note:
    - The server does not modify chat history directly.
      Chat history usage is controlled solely by Aider via the
      `--restore-chat-history` or `--no-restore-chat-history` flags, which we set
      based on `continue_thread`.
    """
    start_time = time.time()
    touched_files: Set[str] = set()
    aider_stderr = ""

    aider_path = aider_path or "aider"

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

    aider_options: Dict[str, Any] = {
        "yes_always": True,
        "auto_commit": False,
    }

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
    # We rely solely on Aider's built-in chat history handling; the server does not
    # prune or clear `.aider.chat.history.md` files anymore.
    aider_options["restore_chat_history"] = continue_thread

    # Ensure .mcp-devtools is in .gitignore before capturing untracked files
    try:
        repo_root_ig = find_git_root(directory_path)
        if repo_root_ig:
            _ensure_gitignore_has_devtools(repo_root_ig)
    except Exception as e:
        logger.debug(f"Failed to ensure .mcp-devtools in .gitignore: {e}")

    for fname in files:
        fpath = os.path.join(directory_path, fname)
        if not os.path.isfile(fpath):
            logger.error(f"[ai_edit] Provided file not found in repo: {fname}. Aider may fail.")

    # === Worktree Snapshot helpers (User Story 1) ===
    # Imports were moved to module top to avoid shadowing and mypy issues.

    # Capture pre-existing untracked files BEFORE aider runs
    try:
        repo = git.Repo(repo_path)
        pre_existing_untracked_files = set(repo.untracked_files)
    except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
        pre_existing_untracked_files = set()

    # Take pre-execution snapshot from workspace but save under root repo
    pre_snapshot = ""
    pre_snapshot_path = Path(ensure_snap_dir(repo_path)) / f"ai_edit_{now_ts()}_pre.diff"
    if os.getenv("MCP_DISABLE_SNAPSHOTS") != "1":
        try:
            pre_snapshot = snapshot_worktree(repo_path, exclude_untracked=pre_existing_untracked_files)
            pre_ts = now_ts()
            pre_snapshot_path = save_snapshot(repo_path, pre_ts, "pre", pre_snapshot)
        except Exception as e:
            logger.debug(f"[ai_edit] pre-snapshot failed: {e}")
    else:
        logger.debug("[ai_edit] Snapshots disabled via MCP_DISABLE_SNAPSHOTS")

    # ... run aider ...

    # Take post-execution snapshot from workspace but save under root repo
    post_snapshot = ""
    post_snapshot_path = Path(ensure_snap_dir(repo_path)) / f"ai_edit_{now_ts()}_post.diff"
    if os.getenv("MCP_DISABLE_SNAPSHOTS") != "1":
        try:
            post_snapshot = snapshot_worktree(repo_path, exclude_untracked=pre_existing_untracked_files)
            post_ts = now_ts()
            post_snapshot_path = save_snapshot(repo_path, post_ts, "post", post_snapshot)
        except Exception as e:
            logger.debug(f"[ai_edit] post-snapshot failed: {e}")

        # Compute delta between pre and post snapshots
        try:
            # Use difflib to compute unified diff
            import difflib
            delta_lines = list(difflib.unified_diff(
                pre_snapshot.splitlines(keepends=True),
                post_snapshot.splitlines(keepends=True),
                fromfile=str(pre_snapshot_path.name),
                tofile=str(post_snapshot_path.name),
            ))
            delta_section = "### Snapshot Delta (this run)\n\n" + "".join(delta_lines)
            # Ensure this delta is included in the final result
            snapshot_delta_section = delta_section
        except Exception as e:
            delta_section = f"\n\nError generating delta: {str(e)}"
    else:
        logger.debug("[ai_edit] Snapshots disabled via MCP_DISABLE_SNAPSHOTS")
        snapshot_delta_section = "### Snapshot Delta (this run)\n\n<snapshots disabled>"

    # === Session Management ===
    # Determine effective session ID
    if session_id:
        effective_session_id = session_id
    else:
        # Try to read from .aider.last_session_id file
        last_session_file = Path(directory_path) / ".aider.last_session_id"
        try:
            if last_session_file.exists():
                last_session_id = last_session_file.read_text().strip()
                if last_session_id:
                    effective_session_id = last_session_id
                else:
                    effective_session_id = _generate_session_id()
            else:
                effective_session_id = _generate_session_id()
        except Exception:
            effective_session_id = _generate_session_id()
    
    # Write the effective session ID to .aider.last_session_id
    try:
        last_session_file = Path(directory_path) / ".aider.last_session_id"
        last_session_file.write_text(effective_session_id)
    except Exception as e:
        logger.debug(f"Failed to write .aider.last_session_id: {e}")
    
    # Find the git root or use the directory path
    repo_root = find_git_root(directory_path) or directory_path
    
    # Determine if we should use worktrees
    use_worktree = _env_truthy(os.getenv("MCP_EXPERIMENTAL_WORKTREES"))
    # Set up workspace directory path (but don't create the worktree yet)
    workspace_dir = str(Path(_workspaces_dir(repo_root)) / effective_session_id) if use_worktree else directory_path
    # Record session start (async, with lock)
    await _record_session_start_async(repo_root, effective_session_id, workspace_dir, use_worktree=use_worktree)

    # === Isolated Workspace Setup (US2 Step 3) ===
    if use_worktree:
        try:
            # Ensure workspaces directory exists
            workspaces_dir_path = _workspaces_dir(repo_root)
            workspaces_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create worktree
            proc = await asyncio.create_subprocess_exec(
                "git", "worktree", "add", "--force", workspace_dir, "HEAD",
                cwd=repo_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
            
            if proc.returncode == 0:
                Path(workspace_dir).mkdir(parents=True, exist_ok=True)
                directory_path = workspace_dir
                logger.debug(f"[ai_edit] Using workspace directory: {directory_path}")
            else:
                logger.warning(f"[ai_edit] Failed to create worktree: {stderr_bytes.decode()}. Falling back to repo_path.")
                directory_path = repo_path
        except Exception as e:
            logger.warning(f"[ai_edit] Exception during worktree creation: {e}. Falling back to repo_path.")
            directory_path = repo_path
    else:
        directory_path = repo_path

    # ... rest of ai_edit ...

    original_dir = os.getcwd()
    structured_report_built = False
    result_message = ""
    success = False
    try:
        repo_pre = git.Repo(directory_path)
        pre_existing_untracked_files_set = set(repo_pre.untracked_files)
        logger.debug(f"[ai_edit] Pre-existing untracked files: {sorted(pre_existing_untracked_files_set)}")
    except git.InvalidGitRepositoryError:
        logger.debug("[ai_edit] Not a git repository when capturing pre-existing untracked files.")
    except Exception as e:
        logger.debug(f"[ai_edit] Failed to capture pre-existing untracked files: {e}")
    try:
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
        aider_stderr = stderr_bytes.decode('utf-8')

        # Post-snapshot for this ai_edit run (User Story 1)
        try:
            # Compute delta between pre and post snapshots directly
            delta_lines = list(difflib.unified_diff(
                pre_snapshot.splitlines(keepends=True),
                post_snapshot.splitlines(keepends=True),
                fromfile=str(pre_snapshot_path.name),
                tofile=str(post_snapshot_path.name),
            ))
            delta_text = "".join(delta_lines).strip()
            if delta_text:
                snapshot_delta_section = f"### Snapshot Delta (this run)\n```diff\n{delta_text}\n```"
        except Exception as e:
            logger.debug(f"[ai_edit] Failed to compute snapshot delta: {e}")
            snapshot_delta_section = "### Snapshot Delta (this run)\n<snapshot delta unavailable due to error>"

        return_code = process.returncode
        success = (return_code == 0)
        
        if return_code != 0:
            logger.error(f"Aider process exited with code {return_code}")
            result_message = f"Error: Aider process exited with code {return_code}.\nSTDERR:\n{aider_stderr}"
        else:
            # Apply workspace changes to root repo if using worktrees and feature is enabled
            apply_to_root = os.getenv("MCP_APPLY_WORKSPACE_TO_ROOT", "1") != "0"
            if success and use_worktree and directory_path != repo_path and apply_to_root:
                try:
                    _apply_workspace_changes_to_root(directory_path, repo_path, files)
                except Exception as e:
                    logger.debug(f"[ai_edit] Failed to apply workspace changes to root: {e}")
            
            if "Applied edit to" in stdout:
                # Build a structured report with Aider's plan and the diff
                applied_changes = ""
                try:
                    # Re-initialize Repo to avoid stale caches after external process (Aider) runs
                    logger.debug("[ai_edit] Re-initializing git.Repo to refresh state after Aider run.")
                    repo = git.Repo(repo_path)

                    # Log git status to understand staged vs unstaged changes
                    status_output = ""
                    try:
                        status_output = repo.git.status()
                        logger.debug(f"[ai_edit] git status after Aider run:\n{status_output}")
                    except Exception as se:
                        logger.debug(f"[ai_edit] Failed to get git status: {se}")

                    # Determine if the repo has any commits to select appropriate diff target
                    has_commits = True
                    try:
                        _ = repo.head.commit  # accessing commit will raise in empty repo
                    except Exception:
                        has_commits = False
                    diff_target = "HEAD" if has_commits else EMPTY_TREE_SHA
                    logger.debug(f"[ai_edit] Diff target determined: {'HEAD' if has_commits else 'EMPTY_TREE_SHA'}")

                    # Smart diff logic:
                    # 1) Get diff for all tracked files (staged and unstaged).
                    final_diff = ""
                    try:
                        # This diff will capture both worktree changes and staged changes against HEAD.
                        final_diff = git_diff(repo, target=diff_target)
                    except GitCommandError as ge:
                        logger.debug(f"[ai_edit] git diff against {diff_target} failed: {ge}. This is expected in a new repo.")
                        # In a new repo with no commits, diff against HEAD fails. Fallback to empty tree.
                        try:
                            final_diff = git_diff(repo, target=EMPTY_TREE_SHA)
                        except Exception as e:
                            logger.error(f"[ai_edit] Fallback diff against EMPTY_TREE_SHA failed: {e}")
                            final_diff = ""
                    except Exception as e:
                        logger.error(f"[ai_edit] An unexpected error occurred during git diff: {e}")
                        final_diff = ""
                    
                    # 2) Handle untracked files, which are not included in standard diffs.
                    # 2) Handle untracked files, which are not included in standard diffs.
                    untracked_diffs = []
                    try:
                        current_untracked = list(repo.untracked_files)
                    except Exception:
                        current_untracked = []
                    # Only include files that became untracked during this Aider run
                    new_untracked = [f for f in current_untracked if f not in pre_existing_untracked_files]
                    # Exclude internal snapshot artifacts from being reported in Applied Changes
                    new_untracked = [f for f in new_untracked if not f.startswith(".mcp-devtools/")]
                    if new_untracked:
                        logger.debug(f"[ai_edit] New untracked files created by Aider: {new_untracked}")
                        for untracked_file in new_untracked:
                            try:
                                untracked_file_path = Path(directory_path) / untracked_file
                                if untracked_file_path.is_file():
                                    with open(untracked_file_path, 'r') as f:
                                        content = f.read()
                                    diff_header = f"--- /dev/null\n+++ b/{untracked_file}\n"
                                    diff_body = "".join([f"+{line}\n" for line in content.splitlines()])
                                    untracked_diffs.append(diff_header + diff_body)
                            except Exception as e:
                                logger.error(f"[ai_edit] Error reading untracked file {untracked_file}: {e}")
                    
                    all_diff_parts = []
                    if final_diff:
                        all_diff_parts.append(final_diff)
                    
                    if untracked_diffs:
                        all_diff_parts.extend(untracked_diffs)
                    
                    # Join all parts with a newline. This ensures separation between diff blocks.
                    final_diff_combined = "\n".join(all_diff_parts).strip()

                    # Extract touched files from the diff
                    touched_files = _collect_touched_files(repo, diff_target, final_diff_combined, new_untracked)
                    
                    # Log the final diff length and a preview for diagnostics
                    preview = (final_diff_combined[:1000] + "…") if len(final_diff_combined) > 1000 else final_diff_combined
                    logger.debug(f"[ai_edit] Final diff length: {len(final_diff_combined)}; preview:\n{preview}")

                    if final_diff_combined:
                        applied_changes = f"```diff\n{final_diff_combined}\n```"
                    else:
                        applied_changes = "No changes detected in the working directory or index."
                except git.InvalidGitRepositoryError:
                    applied_changes = "Could not access Git repository to get diff after Aider run."
                except Exception as e:
                    applied_changes = f"Error generating diff for Aider changes: {e}"

                last_reply = _get_last_aider_reply(directory_path) or ""
                result_message = (
                    f"### Aider's Plan\n"
                    f"{last_reply}\n\n"
                    f"### Applied Changes (Diff)\n"
                    f"{applied_changes}\n\n"
                    f"### Verification Result\n"
                    f"⏳ Not yet implemented.\n\n"
                    f"### Next Steps\n"
                    f"- Please review the changes above. If they are correct, please stage and commit them.\n"
                    f"- Consider summarizing what was done and starting a fresh thread with continue_thread=false; pass the summary in the next ai_edit message to give the agent a fresh perspective.\n"
                    f"- Remember the server will not prune Aider history; to shorten context, start a new thread and feed summarized content via input.\n"
                    f"- Whether continuing or restarting, include key decisions, constraints, relevant files, and acceptance criteria to maintain context continuity.\n\n"
                    f"{snapshot_delta_section if snapshot_delta_section else ''}"
                )
                structured_report_built = True
            else:
                result_message = "Aider process completed."

    except Exception as e:
        logger.error(f"An unexpected error occurred during ai_edit: {e}")
        result_message = ai_hint_ai_edit_unexpected(e)
    finally:
        # Always restore working directory first to avoid purging current CWD
        try:
            if os.getcwd() != original_dir:
                os.chdir(original_dir)
                logger.debug(f"Restored working directory to: {original_dir}")
        except Exception as e:
            logger.debug(f"Failed to restore working directory: {e}")

        # Record session completion
        try:
            await _record_session_complete_async(repo_root, effective_session_id, success)
            
            # If using worktrees, check if we should purge the worktree
            if use_worktree and workspace_dir != repo_path and _git_status_clean(directory_path):
                try:
                    await _purge_worktree_async(repo_root, workspace_dir)
                    await _record_session_update_async(repo_root, effective_session_id, purged_at=time.time())
                except Exception as e:
                    logger.debug(f"Failed to purge worktree: {e}")
        except Exception as e:
            logger.debug(f"Failed to update session metadata: {e}")
        
        # Calculate duration
        duration_s = round(time.time() - start_time, 2)
        
        # Read full Aider chat history for aggregated token stats
        try:
            history_path = Path(directory_path) / ".aider.chat.history.md"
            if history_path.exists():
                full_history_text = history_path.read_text(encoding="utf-8", errors="ignore")
                sent_tokens, received_tokens = _parse_aider_token_stats(full_history_text)
            else:
                sent_tokens, received_tokens = 0, 0
        except Exception as e:
            logger.warning(f"Failed to parse token stats from chat history: {e}")
            sent_tokens, received_tokens = 0, 0
        
        total_thread_tokens = sent_tokens + received_tokens
        
        # For backward compatibility, still calculate tokens for last session
        thread_text = _read_last_aider_session_text(directory_path)
        s, r = _parse_aider_token_stats(thread_text)
        tokens = s + r
        if tokens == 0:
            tokens = _approx_token_count(thread_text)
        
        # Build summary section
        session_status = "completed/success" if success else "error"
        files_touched_count = len(touched_files)
        
        # Format touched files list (first 10 files)
        if touched_files:
            sorted_touched_files = sorted(list(touched_files))
            if len(sorted_touched_files) > 10:
                files_list = ", ".join(sorted_touched_files[:10]) + f" (+{len(sorted_touched_files) - 10} more)"
            else:
                files_list = ", ".join(sorted_touched_files)
        else:
            files_list = "None"
        
        files_touched_line = (
            f"- Files touched: {files_touched_count} ({files_list})\n" if files_touched_count > 0 else ""
        )
        
        summary_section = (
            f"### Summary\n"
            f"- Status: {session_status}\n"
            f"- Session: {effective_session_id}\n"
            f"- Duration: {duration_s}s\n"
            + files_touched_line +
            f"- Aider tokens: sent={sent_tokens}, received={received_tokens}\n"
            f"- Total thread tokens: {total_thread_tokens}\n"
            f"- Last session tokens: {tokens}\n"
        )
        
        # Add warnings section if there were any stderr messages
        warnings_section = ""
        if aider_stderr.strip():
            warnings_section = f"\n### Warnings\n{aider_stderr.strip()}\n"
        
        # Add summary and warnings to the result message
        result_message = summary_section + result_message + warnings_section
        
        # Cleanup expired sessions
        try:
            await cleanup_expired_sessions_async(repo_root)
        except Exception as e:
            logger.debug(f"Failed to cleanup expired sessions: {e}")

        last_reply = _get_last_aider_reply(directory_path) or ""
        if not structured_report_built and last_reply:
            # Legacy append when structured report isn't built
            result_message += f"\n\nAider's last reply:\n{last_reply}"
        
        # Add thread context usage information
        result_message += (
            f"\n\n### Thread Context Usage\n"
            f"Last session tokens: {tokens}\n"
            f"Total thread tokens: {total_thread_tokens}\n"
            f"Guidance: Long threads increase context cost and latency. To shorten context, start a fresh session with continue_thread=false and include a concise summary of previous work. Remember to carry forward important details like key decisions, constraints, and relevant files to maintain continuity."
        )
        
        return result_message

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


mcp_server: Server[ServerSession] = Server("mcp-git")  # Server's generic signature is not compatible with the expected type due to type system limitations.

@mcp_server.list_tools()  # type: ignore[misc, no-untyped-call]
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
            name=GitTools.SHOW,
            description="Shows the metadata (author, date, message) and the diff of a specific commit. This allows inspection of changes introduced by a particular commit. Supports commit ranges like 'A..B' or 'A...B' as well. Optional path filter and metadata/diff-only options available.",
            inputSchema=GitShow.model_json_schema(),
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
            description=(
                "AI pair programming tool for making targeted code changes using Aider. "
                "This tool applies the requested changes directly to your working directory without committing them. "
                "After the tool runs, it returns a structured report containing:\n\n"
                "1.  **Aider's Plan:** The approach Aider decided to take.\n"
                "2.  **Applied Changes (Diff):** A diff of the modifications made to your files.\n"
                "3.  **Next Steps:** Guidance on how to manually review, stage, and commit the changes.\n"
                "4.  **Thread Context Usage:** Information about the approximate token count of the conversation history and guidance on keeping it under ~200k tokens.\n\n"
                "Use this tool to:\n"
                "- Implement new features or functionality in existing code\n"
                "- Add tests to an existing codebase\n"
                "- Fix bugs in code\n"
                "- Refactor or improve existing code\n\n"
                "**IMPORTANT:** This tool does NOT automatically commit changes. You are responsible for reviewing and committing the work."
            ),
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
        ),
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

@mcp_server.call_tool()  # type: ignore[misc, no-untyped-call]
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
                    session_id = arguments.get("session_id")
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
                        session_id=session_id,
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

async def handle_sse(request: Request) -> Response:
    """
    Handles Server-Sent Events (SSE) connections from MCP clients.
    Establishes a communication channel for the MCP server to send events.

    Args:
        request: The Starlette Request object.

    Returns:
        A Starlette Response object for the SSE connection.
    """
    # The `_send` attribute is marked as protected, but accessing it is the
    # intended way to integrate with the SseServerTransport according to the
    # mcp library's design for Starlette integration. Suppressing the warning
    # here is safe and necessary for the SSE transport to function correctly.
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        options = mcp_server.create_initialization_options()
        await mcp_server.run(read_stream, write_stream, options, raise_exceptions=True)
    return Response()

async def handle_post_message(scope: Scope, receive: Receive, send: Send) -> None:
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
