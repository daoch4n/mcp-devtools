#!/bin/sh

# Default port
PORT=4754

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --port|-p)
            PORT="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Install required packages using UV
uv venv
uv pip install fastapi uvicorn "mcp-server-git" httpx python-dotenv

# Create and configure environment file
echo "MCP_HOST=127.0.0.1" > .env
echo "MCP_PORT=${PORT}" >> .env

# Run server using UV's enhanced execution
echo "Git MCP server listening at http://localhost:${PORT}/sse"
uv run uvicorn server:app \
   --host 127.0.0.1 \
   --port "${PORT}" \
   --reload \
   --log-level info
