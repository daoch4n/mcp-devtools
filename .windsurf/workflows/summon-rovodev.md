---
description: Summon Rovo Dev coding agent
auto_execution_mode: 3
---

## Windsurf Workflow: Summon Rovo Dev

### Workflow Steps
1. **Formulate detailed prompt** - Create a clear task description in `.agent.local.md`
2. **Execute command** based on context:
   ```bash
   # First time working with agent
   acli rovodev --yolo
   
   # Continuing existing task
   acli rovodev --yolo --restore   ```

### Task Template (.agent.local.md)
```markdown
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

## Additional Context
[Any relevant information or constraints]
```

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
```