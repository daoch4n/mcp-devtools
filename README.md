# multi-functional MCP-devtools server over SSE <br> [🌸 リードミー](https://github.com/daoch4n/mcp-devtools/blob/main/%E3%83%AA%E3%83%BC%E3%83%89%E3%83%9F%E3%83%BC.MD) [🏮 读我](https://github.com/daoch4n/mcp-devtools/blob/main/%E8%AF%BB%E6%88%91.MD)

[![GitHub repository](https://img.shields.io/badge/GitHub-repo-blue?logo=github)](https://github.com/daoch4n/mcp-devtools)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/daoch4n/mcp-devtools/python-package.yml?branch=main)](https://github.com/daoch4n/mcp-devtools/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/mcp-devtools)](https://pypi.org/project/mcp-devtools)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/mcp-devtools)](https://clickpy.clickhouse.com/dashboard/mcp-devtools)
[![Hit Counter](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fdaoch4n%2Fmcp-devtools&label=views&icon=eye&color=%230d6efd&message=&style=flat&tz=UTC)](https://hitscounter.dev/history?url=https://github.com/daoch4n/mcp-devtools)

- 🔧 `mcp-devtools` server offers a comprehensive suite of software development tools:
  -  🤖 Agentic editing (`ai_edit`)
  -  📁 File management (`read_file`, `write_to_file`)
  -  🎋 Git management (`git_diff`, `git_show`, `git_stage_and_commit`, `git_status`, `git_log`)
  -  🖥️ Terminal integration (`execute_command`)

## 1️⃣ Install

- **Python 3.12**, **[uv](https://github.com/astral-sh/uv)**
  - **Installation:** (🐧 🍎 Linux/macOS )
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
  - **Installation:** ( 🪟 Windows )
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
- **[Aider](https://github.com/Aider-AI/aider)** (only needed for the `ai_edit` tool)
  - **Installation:**
    ```bash
    python -m pip install aider-install --break-system-packages && aider-install
    ```
    
    <details>
    <summary> <h4> Configuration: </h4> </summary>
    
    - Create or copy provided [.aider.conf.yml](./.aider.conf.yml) into your home directory (🐧`~/.aider.conf.yml`) or your project repo root (make sure to double check if its `.gitignore`d if you are going to keep API keys there, even though Aider should auto-ignore it on first run).  
    - Adjust Aider options to your needs (model, provider, base url, API keys, etc.).
    - Follow [📄 official Aider documentation](https://aider.chat/docs/config.html) for detailed descriptions of each available option.
    
    </details>
    
    <details>
    <summary> <h4> Usage: </h4> </summary>
    
    The `ai_edit` tool provides a powerful way to make code changes using natural language. It no longer automatically commits changes. Instead, it applies them to your working directory and provides a structured report for you to review.
    
    ##### How it Works
    
    1.  **Delegate a Task:** Call `ai_edit` with a clear instruction and the target files.
    2.  **Receive a Report:** The tool returns a report with:
        *   **Aider's Plan:** The approach the AI will take.
        *   **Applied Changes (Diff):** The exact changes made to your files.
        *   **Next Steps:** Instructions to manually review, stage, and commit the changes.
        *   **Thread Context Usage:** Information about the approximate token count of the conversation history and guidance on keeping it under ~200k tokens.
    3.  **Review and Commit:** You are in full control. Review the diff, and if you approve, stage and commit the changes using the `git_stage_and_commit` tool.
    
    Aider reads .aider.conf.yml itself. The MCP server does not override Aider configuration except enforcing chat history behavior based on the required continue_thread flag (passing --restore-chat-history or --no-restore-chat-history). Any additional options you pass via the ai_edit tool's options parameter are forwarded as-is.

    **Note on workspaces and Git worktrees (Experimental)**:
      - Git worktrees are EXPERIMENTAL and disabled by default.
      - Enable per-session worktrees by setting `MCP_EXPERIMENTAL_WORKTREES=1` (also accepts `true`/`yes`).
      - When enabled, `ai_edit` may create per-session worktrees under `.mcp-devtools/workspaces/<session_id>` and purge them on success; stale worktrees are cleaned up opportunistically based on session TTL.
      - Diffs and snapshot artifacts are still computed from and stored under the root repository (e.g., `.mcp-devtools/`), so user-facing behavior remains unchanged when disabled.

    </details>

## 2️⃣ Run

### 🐍 PyPi:

```bash
uvx mcp-devtools -p 1337
```

### 🐈‍⬛ GitHub:

#### 🐧 🍎 Linux/macOS

```bash
git clone "https://github.com/daoch4n/mcp-devtools/"
cd mcp-devtools
./server.sh -p 1337
```

#### 🪟 Windows

```powershell
git clone "https://github.com/daoch4n/mcp-devtools/"
cd mcp-devtools
.\server.ps1 -p 1337
```

## 3️⃣ Use

To integrate `mcp-devtools` with your AI agent, add the following configuration to your MCP settings file:

```json
{
  "mcpServers": {
    "devtools": {
      "url": "http://localhost:1337/sse"
    }
  }
}
```

https://github.com/user-attachments/assets/d0b7b41b-c420-4b84-8807-d8a00300bd3e

<div align=right><sup>ℹ️ This demo was recorded using mcp-devtools@1.2.6, current version behaviour: ai_edit tool no longer autocommits, instead it provides structured report of the changes and prompts caller to consciously review and commit changes manually.</sup></div>
<details>
<summary> <h4> 👾 Show System Prompt </h4> </summary>
  
```
# ROLE AND DIRECTIVE

**You are a Senior Software Architect.** Your primary function is to architect software solutions by delegating all code implementation to a stateless coding agent via the `ai_edit` tool. Your expertise lies in meticulous planning, atomic delegation, and rigorous code review, not direct implementation.

---

# STANDARD OPERATING PROCEDURE (SOP)

You must adhere to the following five-step, iterative workflow:

1.  **Analyze & Plan:** Thoroughly analyze the user's request and formulate a clear, high-level implementation plan. Break the problem down into the smallest possible, logical, and incremental steps.
2.  **Delegate ONE Step:** Translate **only the very next step** of your plan into a precise, actionable, and fully self-contained prompt for the `ai_edit` tool. **Never bundle multiple steps into a single delegation.** Default to continue_thread = false. Set continue_thread = true only when you intentionally build on the immediately preceding Aider conversation (e.g., iterative refinement of the same change).
3.  **Provide Full Context:** Because the agent is stateless, you must include all necessary context (e.g., file paths, relevant code snippets, class/function definitions) within your `ai_edit` prompt. (See "Agent Memory & Context Protocol" below). Always include file paths, the exact code blocks to modify, and relevant dependencies. This applies whether continue_thread is true or false.
4.  **Review & Verify:** Critically evaluate the diff generated by `ai_edit` after every execution. This is a **mandatory code review**.
    * Does the code correctly implement the single step you delegated?
    * Is the code quality acceptable?
    * Are there any errors or edge cases missed?
5.  **Iterate & Guide:**
    * **If Approved:** The step is complete. Proceed to delegate the *next* incremental step in your plan.
    * **If Revision is Needed:** The implementation is flawed. Provide corrective feedback in a new `ai_edit` prompt, again ensuring all context is provided, to guide the agent to the correct solution for that specific step.

---

# AGENT MEMORY MODEL (CONDITIONAL STATELESSNESS)

- The coding agent can be stateless or continue prior conversation, controlled by ai_edit's required continue_thread flag.
- If continue_thread = false:
  - Aider does not restore prior chat. Treat every call as a fresh agent with no memory.
  - Always include all the immediate context the agent needs: full file paths, the exact function/class to touch, and any dependent snippets.
- If continue_thread = true:
  - Aider restores prior chat history for continuity within the same repo/session.
  - Still include critical context to make the agent robust. Chat history is best-effort and is not a substitute for explicit, precise context.

## Choosing continue_thread
- Set false:
  - Switching features or tasks
  - After significant repository changes
  - When you want clean isolation between prompts
- Set true:
  - Iterating immediately on the same feature or fix
  - Correcting the previous Aider change
  - Resuming a short-lived session in the same repo

---

# CONSTRAINTS & TOOL PROTOCOL

**Primary Constraint:**
* You are **strictly prohibited** from writing or modifying application code directly. All code implementation must be delegated.
* **Forbidden Tools for Coding:** `apply_diff`, `write_to_file`, and `{your_native_tool_slug}` must NOT be used to modify code.

**Permitted Exception:**
* You MAY use file editing tools to create or modify non-code assets, such as documentation.

**`ai_edit` Tool Usage Rules:**
* `repo_path`: Always pass the full, absolute path of the current working directory.

```

</details>

💬 *But I'm too lazy to copy paste prompts myself!*

<img width="1408" height="1502" alt="meme-imageonline co-merged" src="https://github.com/user-attachments/assets/b6251406-e120-47bc-bf75-3e844919ea7c" />

### 😻 Prompt-Driven Dev Flow: inspired by [pure vibes](https://github.com/RooCodeInc/Roo-Code) 🦘, optimized for Vibing human-out-of-loop

<details>
<summary> <h4> 🪪 Show Description </h4> </summary>

- Just connect Roo to `mcp-devtools` server and code as usual but use `❓ Ask` mode instead of `💻 Code`, AI will automatically use the `ai_edit` tool if available to apply all changes. 

</details>

### 🙀 Spec-Driven Dev Flow: inspired by [spooky vibes](https://kiro.dev) 👻, optimized for Agile human-in-the-loop
<details>
<summary> <h4> 🪪 Show Description </h4> </summary>

-  To experience agile spec-driven flow, place the [.kiroomodes](https://github.com/daoch4n/mcp-devtools/blob/main/.kiroomodes) file and [.kiroo/](https://github.com/daoch4n/mcp-devtools/tree/main/.kiroo) folder into your repo root and rename them to `.roomodes` and `.roo/`:
   -  Start writing Epic Specs and User Stories with `✒️ Agile Writer`
   -  After your confirmation, it will auto-switch to `✏️ Agile Architect` and write Epic Design
   -  After next confirmation, it will auto-switch to `🖊️ Agile Planner` and write Epic Tasks
   -  After final confirmation, it will auto-switch to `🖋️ Agile Dev` and orchestrate Epic Code writing, followed by Epic Review of each commit.

</details>

### 😼 Plan-Driven Dev Flow: inspired by [minimal vibes](https://github.com/marv1nnnnn/rooroo) ♾️, optimized for Waterfall human-out-of-loop

<details>
<summary> <h4> 🪪 Show Description </h4> </summary>

 -  To experience structured waterfall flow, place the [.rooroomodes](https://github.com/daoch4n/mcp-devtools/blob/main/.rooroomodes) file and [.rooroo/](https://github.com/daoch4n/mcp-devtools/tree/main/.rooroo) folder into your repo root and rename them to `.roomodes` and `.roo/`:
    - `🧭 Rooroo Navigator` co-pilot is your project manager. Responsible for agent coordination and task orchestration, lifecycles, delegation. Provides `context.md` instruction file links to other agents, either the ones generated by `🗓️ Rooroo Planner`, or self-generated ones if Planner wasn't deemed neccessary for the task. <!-- TODO: Fix brittle task files passing on weaker LLMs -->
    - `👩🏻‍💻 Rooroo Developer` agent receives instructions, delegates all code changes to subagent via `ai_edit`, then reviews the changes and commits them, or just codes itself if `ai_edit` tool unavailable.
    - `📊 Rooroo Analyzer` agent receives instructions and analyzes the code, then provides structured report to caller. <!-- TODO: align with `Project Research` from markerplace -->
    - `🗓️ Rooroo Planner` agent decomposes complex goals requiring multi-expert coordination into clear, actionable sub-tasks for other agents to do. It is also the main supplier of `context.md` instructions for other agents.
    - `💡 Rooroo Idea Sparker` co-pilot is your brainstorming and innovation catalyst, talk to it if you'd like some creative thinking and assumption challenging done, or just explore something new with it.

</details>

## 🙈 Security Considerations

<details>
<summary> <h4> ⚠️ Show Disclaimer </h4> </summary>

- 🛡️ For automated workflows, always run MCP Servers in isolated environments:
  - Containers: 🐧 [daoch4n/systemd-nspawn-wayland](https://github.com/daoch4n/systemd-nspawn-wayland) 🍎 🪟 [Docker](https://github.com/docker/cli)
  - Sandboxes: 🐧 [Firejail](https://github.com/netblue30/firejail) 🪟 [Sandboxie](https://github.com/sandboxie-plus/Sandboxie)
- 🗃️ Filesystem access boundaries are maintained via passing `repo_path` to every tool call, so AI agent only has read/write access to files in the current workspace (relative to any path AI decides to pass as `repo_path` , make sure system prompt is solid on cwd use).
- ⚠️ `execute_command` doesn't have strict access boundaries defined, while it does execute all commands with cwd set to `repo_path` (relative to it), nothing is there to stop AI from passing full paths to other places it seems fit; reading, altering or deleting unintended data on your whole computer, so execise extreme caution with auto-allowing `execute_command` tool or at least don't leave AI agent unattended while doing so. MCP server is not responsible for your AI agent executing rm -rf * in your home folder.

</details>

## ❔ FAQ

<details>
<summary> <h4> 💾 Direct Code Editing vs 🤖 AI-assisted Editing </h4> </summary>

**Issue:**

*    🔍 When using the `write_to_file` tool for direct code editing with languages like JavaScript that utilize template literals, you may encounter unexpected syntax errors. This issue stems from how the AI agent generates the `content` string, where backticks and dollar signs within template literals might be incorrectly escaped with extra backslashes (`\`).

**Mitigation:** 

*    🔨 The `write_to_file` tool is dynamically integrated with `tsc` (TypeScript compiler) for conditional type checking of `.js`, `.mjs`, and `.ts` files on edit. The output of `tsc --noEmit --allowJs` is provided as part of the tool's response. AI agent should parse this output to detect any compiler errors and should not proceed with further actions if errors are reported, indicating a problem with the written code.

**Workarounds:**

*    🤖 Instruct your AI agent to delegate editing files to the `ai_edit` tool. It's more suitable for direct code manipulation than `write_to_file`. `ai_edit` will apply the changes and return a diff for review. Your assistant can then orchestrate the review and commit process.

</details>

## ℹ️ Available Tools

<details>
<summary> <h3> 📄 Show Descriptions and JSON Schemas </h3> </summary>

### `git_status`
- **Description:** Shows the current status of the Git working tree, including untracked, modified, and staged files.
- **Input Schema:**

  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory."
      }
    },
    "required": [
      "repo_path"
    ]
  }
  ```


### `git_diff`
- **Description:** Shows differences in the working directory. By default (without target), shows worktree vs index like `git diff`. Pass target='HEAD' for previous 'all changes vs HEAD' behavior.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory."
      },
      "target": {
        "type": "string",
        "description": "Optional. If omitted, behaves like `git diff` (worktree vs index). Pass 'HEAD' or another ref to compare against a commit or branch."
      }
    },
    "required": [
      "repo_path"
    ]
  }
  ```

### `git_stage_and_commit`
- **Description:** Stages specified files (or all changes if no files are specified) and then commits them to the repository with a given message. This creates a new commit in the Git history.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory."
      },
      "message": {
        "type": "string",
        "description": "The commit message for the changes."
      },
      "files": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "An optional list of specific file paths (relative to the repository root) to stage before committing. If not provided, all changes will be staged."
      }
    },
    "required": [
      "repo_path",
      "message"
    ]
  }
  ```

### `git_log`
- **Description:** Shows the commit history for the repository, listing recent commits with their hash, author, date, and message. The number of commits can be limited.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory."
      },
      "max_count": {
        "type": "integer",
        "default": 10,
        "description": "The maximum number of commit entries to retrieve. Defaults to 10."
      }
    },
    "required": [
      "repo_path"
    ]
  }
  ```

### `git_show`
- **Description:** Shows the metadata (author, date, message) and the diff of a specific commit or commit range (A..B or A...B). This allows inspection of changes introduced by a particular commit or range of commits. Optionally filter by path or show only metadata/diff.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory."
      },
      "revision": {
        "type": "string",
        "description": "The commit hash, reference (e.g., 'HEAD', 'main', 'abc1234'), or range (A..B or A...B) to show details for."
      },
      "path": {
        "type": "string",
        "description": "Optional. Filter the output to show only changes for the specified file path."
      },
      "show_metadata_only": {
        "type": "boolean",
        "description": "Optional. If true, shows only the commit metadata (author, date, message) without the diff."
      },
      "show_diff_only": {
        "type": "boolean",
        "description": "Optional. If true, shows only the diff without the commit metadata."
      }
    },
    "required": [
      "repo_path",
      "revision"
    ]
  }
  ```

### `git_read_file`
- **Description:** Reads and returns the entire content of a specified file within the Git repository's working directory. The file path must be relative to the repository root.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory."
      },
      "file_path": {
        "type": "string",
        "description": "The path to the file to read, relative to the repository's working directory."
      }
    },
    "required": [
      "repo_path",
      "file_path"
    ]
  }
  ```


### `write_to_file`
- **Description:** Writes the provided content to a specified file within the repository. If the file does not exist, it will be created. If it exists, its content will be completely overwritten. Includes a check to ensure content was written correctly and generates a diff.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory."
      },
      "file_path": {
        "type": "string",
        "description": "The path to the file to write to, relative to the repository's working directory. The file will be created if it doesn't exist, or overwritten if it does."
      },
      "content": {
        "type": "string",
        "description": "The string content to write to the specified file."
      }
    },
    "required": [
      "repo_path",
      "file_path",
      "content"
    ]
  }
  ```

### `execute_command`
- **Description:** Executes an arbitrary shell command within the context of the specified repository's working directory. This tool can be used for tasks not covered by other specific Git tools, such as running build scripts, linters, or other system commands.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the directory where the command should be executed."
      },
      "command": {
        "type": "string",
        "description": "The shell command string to execute (e.g., 'ls -l', 'npm install')."
      }
    },
    "required": [
      "repo_path",
      "command"
    ]
  }
  ```

### `ai_edit`
- **Description:** AI pair programming tool for making targeted code changes using Aider. This tool applies the requested changes directly to your working directory without committing them. After the tool runs, it returns a structured report containing:

  1.  **Aider's Plan:** The approach Aider decided to take.
  2.  **Applied Changes (Diff):** A diff of the modifications made to your files.
  3.  **Next Steps:** Guidance on how to manually review, stage, and commit the changes.
  4.  **Thread Context Usage:** Information about the approximate token count of the conversation history and guidance on keeping it under ~200k tokens.

  Use this tool to:
  - Implement new features or functionality in existing code
  - Add tests to an existing codebase
  - Fix bugs in code
  - Refactor or improve existing code

  **IMPORTANT:** This tool does NOT automatically commit changes. You are responsible for reviewing and committing the work.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory where the AI edit should be performed."
      },
      "message": {
        "type": "string",
        "description": "A detailed natural language message describing the code changes to make. Be specific about files, desired behavior, and any constraints."
      },
      "files": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "A list of file paths (relative to the repository root) that Aider should operate on. This argument is mandatory."
      },
      "continue_thread": {
        "type": "boolean",
        "description": "Required. Whether to continue the Aider thread by restoring chat history. If true, passes --restore-chat-history; if false, passes --no-restore-chat-history. Clients must explicitly choose."
      },
      "options": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Optional. A list of additional command-line options to pass directly to Aider. Each option should be a string."
      },
      "edit_format": {
        "type": "string",
        "enum": [
          "diff",
          "diff-fenced",
          "udiff",
          "whole"
        ],
        "default": "diff",
        "description": "Optional. The format Aider should use for edits. Defaults to 'diff'. Options: 'diff', 'diff-fenced', 'udiff', 'whole'."
      },
    },
    "required": [
      "repo_path",
      "message",
      "files",
      "continue_thread"
    ]
  }
  ```

## Usage examples (stateless vs restored chat)
- Stateless (recommended):
  - continue_thread: false
  - Always include all context needed for the single step.
- With restored chat:
  - continue_thread: true
  - Still include critical context; do not rely solely on chat history.
  - Use this to refine a change made in the immediately previous run.

> Note: The server no longer modifies or prunes `.aider.chat.history.md`. Chat history usage is controlled solely by Aider via `--restore-chat-history` (when `continue_thread` is true) or `--no-restore-chat-history` (when false).

> Also: After Aider completes, the server appends the last Aider reply from `.aider.chat.history.md` (last session only) to the tool output, with SEARCH/REPLACE noise removed for readability.

### `aider_status`
- **Description:** Check the status of Aider and its environment. Use this to:
  1. Verify Aider is correctly installed
  2. Check API keys
  3. View the current configuration
  4. Diagnose connection or setup issues
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository or working directory to check Aider's status within."
      },
      "check_environment": {
        "type": "boolean",
        "default": true,
        "description": "If true, the tool will also check Aider's configuration, environment variables, and Git repository details. Defaults to true."
      }
    },
    "required": [
      "repo_path"
    ]
  }

</details>

💬 *But I'm too lazy to even read this all!*

https://github.com/user-attachments/assets/05670a7a-72c5-4276-925c-dbd1ed617d99
