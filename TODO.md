## User Story 1: Worktree Snapshots
**As a** developer using `ai_edit`  
**I want** to see only the changes made by the current tool execution  
**So that** I can focus on relevant modifications without unrelated changes  

**Acceptance Criteria:**  
- Automatically create a worktree diff before each `ai_edit` execution  
- Generate diff showing only changes made during the current operation relative to the previous diff

## User Story 2: Parallel Execution Support  
**As a** developer working on same feature
**I want** to run multiple `ai_edit` operations concurrently  
**So that** I can optimize my development workflow  

**Acceptance Criteria:**  
- Implement isolated workspaces for concurrent operations  
- Add session IDs to track parallel executions and manage separate `.aider.chat.history.md` files for each session with session resume support by adding the optional session ID argument to the `ai_edit` tool, if not specified, last session ID will be used, so we can resume the last session. (we also need to track the last session ID in a temporary file named .aider.last_session_id in the repo_path)
- The session ID will be generated automatically.
- Track and show current ai_edit operation status in next ai_edit tool output.
- Add session ID to the output of the ai_edit tool so LLM will be aware of which session to resume.

## User Story 3: Automatic Worktree Management (Experimental)  
**As a** developer  
**I want** automatic Git worktree creation for each operation  
**So that** I have a clean environment for each task  

**Acceptance Criteria:**  
- Git worktree creation is EXPERIMENTAL and disabled by default
- Opt-in via `MCP_EXPERIMENTAL_WORKTREES=1`
- When enabled, create temporary worktrees for each `ai_edit` execution
- Automatically clean up worktrees after successful operations  

## User Story 4: Change Context Awareness  
**As a** code reviewer  
**I want** to see the exact context of changes  
**So that** I understand how modifications fit into existing code  

**Acceptance Criteria:**  
- Show 3 lines of context before/after each change  
- Highlight changed lines in diff view  
- Provide file navigation in change preview  

## User Story 6: Performance Metrics  
**As a** developer  
**I want** performance tracking for `ai_edit` operations  
**So that** I can optimize my development workflow  

**Acceptance Criteria:**  
- Track execution time per operation  
- Show execution time in the output of the ai_edit tool
- TBD
