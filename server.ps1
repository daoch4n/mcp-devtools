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
Write-Host "Git MCP server listening at http://localhost:$PORT/sse"
uv run uvicorn server:app `
   --host 127.0.0.1 `
   --port "$PORT" `
   --reload `
   --log-level info
