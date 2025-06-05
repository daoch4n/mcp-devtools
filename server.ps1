# Default port
$PORT = 1337

# Parse command-line arguments
$args | ForEach-Object {
    if ($_ -eq "--port" -or $_ -eq "-p") {
        $PORT = $args[($args.IndexOf($_) + 1)]
    }
}

# Install required packages using UV
uv venv
uv pip install fastapi uvicorn "mcp-server-git" httpx python-dotenv

# Create and configure environment file
"MCP_HOST=127.0.0.1" | Out-File -FilePath .env -Encoding utf8
"MCP_PORT=$PORT" | Out-File -FilePath .env -Encoding utf8 -Append

# Run server using UV's enhanced execution
# Run server using the Python entrypoint, with reload explicitly enabled
$env:MCP_DEVTOOLS_RELOAD="true"
uv run python -m mcp_devtools_cli --port "$PORT"
