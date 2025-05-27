#!/bin/sh
# Install required packages using UV
uv venv
uv pip install fastapi uvicorn "mcp-server-git" httpx python-dotenv

# Create and configure environment file
echo "MCP_HOST=127.0.0.1" > .env
echo "MCP_PORT=5757" >> .env

# Run server using UV's enhanced execution
echo "Git MCP server listening at http://localhost:4754/sse"
uv run uvicorn server:app \
   --host 127.0.0.1 \
   --port 4754 \
   --reload \
   --log-level info
