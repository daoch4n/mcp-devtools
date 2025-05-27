#!/bin/sh
# Install required packages using UV
uv venv
uv pip install fastapi uvicorn "git-mcp-server" httpx python-dotenv

# Create and configure environment file
echo "MCP_HOST=127.0.0.1" > .env
echo "MCP_PORT=5757" >> .env
echo "GITHUB_TOKEN=your_token_here" >> .env  # Optional for GitHub integration

uv run server.py

# Run server using UV's enhanced execution
# uv run uvicorn git_mcp_server:app \
#    --host $MCP_HOST \
#    --port $MCP_PORT \
#    --reload \
#    --log-level info
