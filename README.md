# ⚙️ mcp-devtools: multi-functional development tools MCP server over SSE [🌸 リードミー](https://github.com/daoch4n/mcp-devtools/blob/main/%E3%83%AA%E3%83%BC%E3%83%89%E3%83%9F%E3%83%BC.MD) [🏮 读我](https://github.com/daoch4n/mcp-devtools/blob/main/%E8%AF%BB%E6%88%91.MD)

https://github.com/user-attachments/assets/d0b7b41b-c420-4b84-8807-d8a00300bd3e

[![GitHub repository](https://img.shields.io/badge/GitHub-repo-blue?logo=github)](https://github.com/daoch4n/mcp-devtools)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/daoch4n/mcp-devtools/python-package.yml?branch=main)](https://github.com/daoch4n/mcp-devtools/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/mcp-devtools)](https://pypi.org/project/mcp-devtools)

- 🔧 `mcp-devtools` offers a comprehensive suite of software development tools: [ℹ️ Available Tools](#%E2%84%B9%EF%B8%8F-available-tools)
  -  🤖 AI-assisted file operations (`ai_edit`(uses [Aider](https://github.com/Aider-AI/aider)) [ℹ️ Aider Configuration](docs/aider_config.md)
  -  📁 Git-assisted file operations (`git_read_file`, `git_apply_diff`)
  -  📂 Direct file operations (`search_and_replace`(uses `sed` with fallback to direct mode), `write_to_file`) [ℹ️ Direct vs AI-assisted](#-direct-code-editing-vs--ai-assisted-editing)
  -  🎋 Git management operations (`git_diff_all`(compares to HEAD, `git_diff`(compares to specific commits/branches), `git_show`, `git_stage_and_commit`, `git_status`, `git_log`, `git_create_branch`, `git_checkout`, `git_reset`)
  -  🖥️ Terminal commands execution (`execute_command`) [⚠️ Automation-Related Security](#-automation-related-security-considerations)

## 🎧 Audio Overview

https://github.com/user-attachments/assets/05670a7a-72c5-4276-925c-dbd1ed617d99

### [⬇️ Skip to Downloads](#1%EF%B8%8F%E2%83%A3-prerequisites)

## ⛎ Use Cases

- 🌐 Use it to extend AI assistants with agentic coding, access to local files, git and terminal.
- 👩🏻‍💻 Use it to boost code editors with agentic coding capabilities and say goodbye to those pesky diff application failures wasting your time. Aider seems to get diffs right! (if it still doesn't, try to find perfect editing way by passing different `edit_format` [parameters](#ai_edit) to `ai_edit` tool):
  - `unidiff` seems to work better with GPT
  - `diff-fenced`  performs best with Gemini
  - `diff` provides balanced quick results on all models (default)
  - `whole` is the slowest but most reliable option as it simply overwrites file

## 🦘 [Agentic-Driven Workflows](https://github.com/daoch4n/research/tree/ai/agentic-driven-workflows) with [Roo](https://github.com/RooCodeInc/Roo-Code)

https://github.com/user-attachments/assets/4d218e5e-906c-4d24-abc3-09ab0acdc1d0

### 😻 Vibe Flow
- Just connect Roo to `mcp-devtools` server and code as usual but use `❓ Ask` mode instead of `💻 Code`, AI will automatically use the `ai_edit` tool to apply all code changes instead of editing files directly. 
  
### 😼 Advanced Flow ([如如](https://github.com/marv1nnnnn/rooroo) 🧭 optimized for autonomous operation) [ℹ️](#%E2%84%B9%EF%B8%8F-advanced-flow-tutorial)
 -  To experience automated two-levels deep [nested agents execution](https://www.perplexity.ai/search/nested-agent-execution-BsD4hcqjTdKlEUKJLnv9.g), place the [.roomodes](https://github.com/daoch4n/mcp-devtools/blob/main/.roomodes) file and [.roo/](https://github.com/daoch4n/mcp-devtools/tree/main/.roo) folder into your repo root:
    ℹ️ To enforce strict adherence to following rules, enable `experimental power steering` mode in Roo Code settings.  
    - `🧭 Rooroo Navigator` agent is your Advanced Flow manager, so talk to it if you prefer to keep your agent management in orderly manner. Responsible for overall project coordination and task orchestration, task lifecycles, delegation to Planner, Developer, Analyzer, Idea Sparker, processes your commands, and oversees your workflows. Provides `context.md` files to tasks, either the ones generated by `🗓️ Rooroo Planner`, or generates new one if Planner wasn't deemed neccessary for the task. <br> Optimized deep reasoning models like [Gemini 2.5 Pro](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro) with resoning effort (thinking budget) set to `low` (`8192`) or `medium` (`16384`) (100 req/day free (250K tokens/min) via [Gemini API](https://aistudio.google.com/apikey)) or [DeepSeek R1T2](https://huggingface.co/tngtech/DeepSeek-TNG-R1T2-Chimera) recommended.
    - `👩🏻‍💻 Rooroo Developer` agent: <br> Delegates all code changes to Aider subagent via ai_edit tool then reviews Aider work results <br> ℹ️ Or applies diffs directly if you modify `.roomodes` and remove directive to use `ai_edit` tool, `rules.md` provided in `.roo/` folder are optimized for both ways <br> Outputs task result in strict JSON schema. Choose model based on task complexity: optimized deep reasoning models mentioned above for more difficult tasks, or faster models like [Gemini Flash 2.5](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash) with resoning effort (thinking budget) set to `high` (`32768`) (500 req/day (250K tokens/min) free via Gemini API) for daily routine tasks.
    - `📊 Rooroo Analyzer` agent acts as a researcher and analyzes the code. Full Deep Reasoning models like Gemini 2.5 Pro with resoning effort (thinking budget) set to `high` (`32768`) or [DeepSeek R1 0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) recommended. Adjust temperature based on creativity needed.
    - `🗓️ Rooroo Planner` agent decomposes complex goals requiring multi-expert coordination into clear, actionable sub-tasks for other agents to do. It is also the main supplier of `context.md` files for them. Optimized deep reasoning model recommended.
    - `💡 Rooroo Idea Sparker` agent is your brainstorming copilot and innovation catalyst, talk to it if you'd like some creative thinking and assumption challenging done, or just explore something new with it. Deep reasoning models with higher temperature set or internet-connected reasoning models like Perplexity Sonar recommended.
    #### ℹ️ Advanced Flow Tutorial 
    - **Initiate:** Select `🧭 Rooroo Navigator` agent and state your goal.
    -  **Navigator Triage:** The Navigator assesses your request:
    -  *   For *complex/uncertain tasks*, it engages the `🗓️ Rooroo Planner` agent to break it down into sub-tasks with `context.md` briefings. These go into the `.rooroo/queue.jsonl`.
    -  *   For *simple, clear single-expert tasks*, it prepares `context.md` and may execute directly or queue the task.
    *   If *ambiguous*, it asks `🗓️ Rooroo Planner` for clarification.
    - **Execution:** The Navigator dispatches tasks from the queue to the assigned Rooroo expert. The expert uses its `context.md` and stores outputs in `.rooroo/tasks/TASK_ID/`.
    - **Reporting:** The expert returns a JSON **Output Envelope** (status, message, artifacts) to the Navigator.
    - **Processing & Iteration:** The Navigator parses the envelope:
    - *   `NeedsClarification`: Relays question to relevant expert.
    - *   `Done`/`Failed`: Updates queue, informs you. Auto-proceeds with plans if queue is not empty.

## 1️⃣ Prerequisites

- Python 3.12, [uv](https://github.com/astral-sh/uv)

### 🐧 Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 🪟 Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2️⃣ Usage

### 🐍 Running from PyPi

```bash
uvx mcp-devtools@1.2.6 -p 1337
```

### 🐈‍⬛ Running from GitHub

#### 🐧 Linux/macOS

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

## 3️⃣ MCP Server Configuration

To integrate `mcp-devtools` with your AI assistant, add the following configuration to your MCP settings file:

```json
{
  "mcpServers": {
    "devtools": {
      "url": "http://127.0.0.1:1337/sse",
      "disabled": false,
      "alwaysAllow": [],
      "timeout": 900
    }
  }
}
```

## 4️⃣ [AI System Prompt](https://github.com/daoch4n/research/tree/ai/prompt-engineering/google-whitepaper) Example
<details>
<summary> <h3> ℹ️ Show Prompt </h3> </summary>
  
```
⚠️IMPORTANT⚠️ 
IN THIS MODE, YOU ARE OPERATING A STATELESS AGENT USING ai_edit TOOL. EMBRACE CRITICAL THINKING! CAREFULLY ANALYZE ai_edit TOOL OUTPUT EVERY TIME IT COMPLETES! DO A CODE REVIEW AND GUIDE THE AGENT TOWARDS TASK COMPLETION! YOU ARE STRICTLY PROHIBITED TO USE apply_diff , search_and_replace and write_to_file TOOLS. ALL CODE EDITING MUST BE DELEGATED TO AGENT VIA ai_edit TOOL.
```

</details>

## 🙈 Automation-Related Security Considerations

- 🛡️ For automated workflows, always run MCP Servers in isolated environments (🐧[Firejail](https://github.com/netblue30/firejail) or 🪟[Sandboxie](https://github.com/sandboxie-plus/Sandboxie))
- 🗃️ Filesystem access boundaries are maintained via passing `repo_path` to every tool call, so AI assistant only has read/write access to files in the current workspace (relative to any path AI decides to pass as `repo_path` , make sure system prompt is solid on cwd use).
- ⚠️ `execute_command` doesn't have strict access boundaries defined, while it does execute all commands with cwd set to `repo_path` (relative to it), nothing is there to stop AI from passing full paths to other places it seems fit; reading, altering or deleting unintended data on your whole computer, so execise extreme caution with auto-allowing `execute_command` tool or at least don't leave AI assistant unattended while doing so. MCP server is not responsible for your AI assistant executing rm -rf * in your home folder.

## ⁉️ Known Issues and Workarounds

### 💾 Direct Code Editing vs 🤖 AI-assisted Editing

**Issue:**

*    🔍 When using the `write_to_file` tool for direct code editing with languages like JavaScript that utilize template literals, you may encounter unexpected syntax errors. This issue stems from how the AI assistant generates the `content` string, where backticks and dollar signs within template literals might be incorrectly escaped with extra backslashes (`\`).

**Mitigation:** 

*    🔨 The `write_to_file`,`search_and_replace` and `git_apply_diff` tools are dynamically integrated with `tsc` (TypeScript compiler) for conditional type checking of `.js`, `.mjs`, and `.ts` files on edit. The output of `tsc --noEmit --allowJs` is provided as part of the tool's response. AI assistants should parse this output to detect any compiler errors and *should not proceed with further actions* if errors are reported, indicating a problem with the written code.

**Workarounds:**

*    🤖 Instruct your AI assistant to delegate editing files to MCP-compatible coding agent by using `ai_edit` tool instead, as it is more suitable for direct code manipulation, automatically commits changes and produces resulting diff as tool output, and let AI assistant act as task orchestrator that will write down plans and docs with `write_to_file` tool then delegate actual coding to specialized agent, get its report (diff) as tool call result, use `git_read_file` tool to double check agent's work, and manage commits and branches (`ai_edit` tool basically integrates `Aider` via some logic ported from [its MCP bridge](https://github.com/sengokudaikon/aider-mcp-server)).

### ❔ Aider limitations due to its commit-first nature

**Issue:**

*    🔍 When using `ai_edit` tool in a dirty repo state, e.g. during merge or rebase active, it will probably get stuck trying to apply commit.
  
**Workarounds:**

*    ⚙️ Temporarily disable auto commiting functions in your `.aider.conf.yml` configuration file.

## ℹ️ Available Tools

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


### `git_diff_all`
- **Description:** Shows all changes in the working directory, including both staged and unstaged modifications, compared to the HEAD commit. This provides a comprehensive view of all local changes.
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
- **Description:** Shows differences between the current working directory and a specified Git target (e.g., another branch, a specific commit hash, or a tag).
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
        "description": "The target (e.g., branch name, commit hash, tag) to diff against. For example, 'main', 'HEAD~1', or a full commit SHA."
      }
    },
    "required": [
      "repo_path",
      "target"
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

### `git_reset`
- **Description:** Unstages all currently staged changes in the repository, moving them back to the working directory without discarding modifications. This is equivalent to `git reset` without arguments.
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

### `git_create_branch`
- **Description:** Creates a new Git branch with the specified name. Optionally, you can base the new branch on an existing branch or commit, otherwise it defaults to the current active branch.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory."
      },
      "branch_name": {
        "type": "string",
        "description": "The name of the new branch to create."
      },
      "base_branch": {
        "type": "string",
        "nullable": true,
        "description": "Optional. The name of the branch or commit hash to base the new branch on. If not provided, the new branch will be based on the current active branch."
      }
    },
    "required": [
      "repo_path",
      "branch_name"
    ]
  }
  ```

### `git_checkout`
- **Description:** Switches the current active branch to the specified branch name. This updates the working directory to reflect the state of the target branch.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory."
      },
      "branch_name": {
        "type": "string",
        "description": "The name of the branch to checkout."
      }
    },
    "required": [
      "repo_path",
      "branch_name"
    ]
  }
  ```

### `git_show`
- **Description:** Shows the metadata (author, date, message) and the diff of a specific commit. This allows inspection of changes introduced by a particular commit.
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
        "description": "The commit hash or reference (e.g., 'HEAD', 'main', 'abc1234') to show details for."
      }
    },
    "required": [
      "repo_path",
      "revision"
    ]
  }
  ```

### `git_apply_diff`
- **Description:** Applies a given diff content (in unified diff format) to the working directory of the repository. This can be used to programmatically apply patches or changes.
- **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "The absolute path to the Git repository's working directory."
      },
      "diff_content": {
        "type": "string",
        "description": "The diff content string to apply to the repository. This should be in a unified diff format."
      }
    },
    "required": [
      "repo_path",
      "diff_content"
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


### `search_and_replace`
- **Description:** Searches for a specified string or regex pattern within a file and replaces all occurrences with a new string. Supports case-insensitive search and line-range restrictions. It attempts to use `sed` for efficiency, falling back to Python logic if `sed` fails or makes no changes.
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
        "description": "The path to the file to modify, relative to the repository's working directory."
      },
      "search_string": {
        "type": "string",
        "description": "The string or regex pattern to search for within the file."
      },
      "replace_string": {
        "type": "string",
        "description": "The string to replace all matches of the search string with."
      },
      "ignore_case": {
        "type": "boolean",
        "default": false,
        "description": "If true, the search will be case-insensitive. Defaults to false."
      },
      "start_line": {
        "type": "integer",
        "nullable": true,
        "description": "Optional. The 1-based starting line number for the search and replace operation (inclusive). If not provided, search starts from the beginning of the file."
      },
      "end_line": {
        "type": "integer",
        "nullable": true,
        "description": "Optional. The 1-based ending line number for the search and replace operation (inclusive). If not provided, search continues to the end of the file."
      }
    },
    "required": [
      "repo_path",
      "file_path",
      "search_string",
      "replace_string"
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
- **Description:** AI pair programming tool for making targeted code changes using Aider. Use this tool to:
  1. Implement new features or functionality in existing code
  2. Add tests to an existing codebase
  3. Fix bugs in code
  4. Refactor or improve existing code
  5. Make structural changes across multiple files

  The tool requires:
  - A repository path where the code exists
  - A detailed message describing what changes to make. Please only describe one change per message. If you need to make multiple changes, please submit multiple requests.

  **Edit Format Selection:**
  If the `edit_format` option is not explicitly provided, the default is selected based on the model name:
  - If the model includes `gemini`, defaults to `diff-fenced`
  - If the model includes `gpt`, defaults to `udiff`
  - Otherwise, defaults to `diff`

  Best practices for messages:
  - Be specific about what files or components to modify
  - Describe the desired behavior or functionality clearly
  - Provide context about the existing codebase structure
  - Include any constraints or requirements to follow

  Examples of good messages:
  - "Add unit tests for the Customer class in src/models/customer.rb testing the validation logic"
  - "Implement pagination for the user listing API in the controllers/users_controller.js file"
  - "Fix the bug in utils/date_formatter.py where dates before 1970 aren't handled correctly"
  - "Refactor the authentication middleware in middleware/auth.js to use async/await instead of callbacks"
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
      "options": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Optional. A list of additional command-line options to pass directly to Aider (e.g., ['--model=gpt-4o', '--dirty-diff']). Each option should be a string."
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
      }
    },
    "required": [
      "repo_path",
      "message",
      "files"
    ]
  }
  ```

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
