import argparse
import os
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Run the Git MCP server.")
    parser.add_argument("--port", "-p", type=int, default=4754, help="Port to run the server on.")
    args = parser.parse_args()

    # Create and configure environment file
    # This part is typically handled by the uv venv and uv pip install,
    # but for direct execution via uvx, we ensure the env vars are set.
    # For uvx, the environment is usually managed by uv itself.
    # However, if .env is explicitly needed by the application, we can create it.
    # For now, let's rely on passing arguments directly to uvicorn.
    # If the application itself reads from .env, we might need to adjust.

    # Set environment variables for MCP_HOST and MCP_PORT if the application relies on them
    # rather than uvicorn arguments.
    os.environ["MCP_HOST"] = "127.0.0.1"
    os.environ["MCP_PORT"] = str(args.port)

    print(f"Git MCP server listening at http://localhost:{args.port}/sse")
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=args.port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()