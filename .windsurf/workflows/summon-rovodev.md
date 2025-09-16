---
description: Summon Rovo Dev coding agent
auto_execution_mode: 3
---

### Workflow Steps

1. **Formulate detailed prompt** - Create a clear task description in `.agent.local.md`
2. **Execute command** based on context:

   ````bash
   # First time working with agent
   acli rovodev run --yolo "lets do it"

   # Continuing existing task
   acli rovodev --yolo --restore "continue"  ```
   ````

### Answering Rovo Dev questions or continuing conversation:
If you got any questions from Rovo Dev during your interaction or want to continue your conversation with the agent, please *overwrite* the .agent.local.md file with new info. --continue flag transforms agent into stateful pair programmer. Make best use of its abilities to assist you. Use your best judgment to answer its questions clearly. Please also ignore all Rovo's suggestions about pull requests, tickets and pages. Focus on the task at hand, but do encourage the agent to make optional changes if you think they fall under scope of current task.

### Task Template (.agent.local.md)

````markdown
# [Task Title]

## Objective

[Clearly describe what you want to achieve]

## Files to Modify

- [File path 1]
- [File path 2]

## Code Changes

```[language]
[Specific code changes or examples]
```
````

## Additional Context

[Any relevant information or constraints]

````

### Best Practices
- **Start small** - Limit changes to 10-20 lines at a time
- **Break down tasks** - Divide complex changes into sequential steps
- **Review and iterate** - Provide feedback on generated code
- **Commit frequently** - Save progress regularly

### Example Task
```markdown
# Add greeting function

## Objective
Create a greeting function in utils.py

## Files to Modify
- /src/utils.py

## Code Changes
```python
def greet(name):
    return f"Hello, {name}!"
````
