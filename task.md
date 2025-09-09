# Add version command to CLI

## Objective
Add a `--version` option to the CLI tool that prints the version and exits.

## Files to Modify
- /home/vi/mcp-devtools/mcp_devtools_cli.py

## Code Changes
We will use the `argparse` module to add an option.

In `mcp_devtools_cli.py`, we will:

  1. Add an argument for `--version`
  2. When the option is present, print the version (e.g., "mcp-devtools v0.1.0") and exit.

Example code:

```python
import argparse

# ... existing code ...

parser = argparse.ArgumentParser(description='MCP DevTools CLI')
# Add the version argument
parser.add_argument('--version', action='version', version='mcp-devtools v0.1.0')
