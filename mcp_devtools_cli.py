import uvicorn
import os
import argparse

def main():
    """
    CLI entrypoint for the MCP DevTools server.
    Conditionally enables reload based on an environment variable.
    """
    parser = argparse.ArgumentParser(description="Run the MCP DevTools Server.")
    parser.add_argument('-p', '--port', type=int, help='Port to run the server on.')
    args = parser.parse_args()

    # Use port from args, then .env, then default
    # The server scripts create a .env file.
    from dotenv import load_dotenv
    load_dotenv()
    port = args.port or int(os.getenv("MCP_PORT", 1337))
    host = os.getenv("MCP_HOST", "127.0.0.1")

    # Check for the reload environment variable.
    # This will be 'false' by default when running via 'uvx'.
    reload_enabled = os.getenv('MCP_DEVTOOLS_RELOAD', 'false').lower() in ('true', '1', 't')

    print(f"Git MCP server listening at http://{host}:{port}/sse")
    if reload_enabled:
        print("Auto-reloading is enabled.")
    else:
        print("Auto-reloading is disabled.")

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload_enabled, # Set reload conditionally
        log_level="info"
    )

if __name__ == "__main__":
    main()