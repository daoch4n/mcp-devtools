from mcp.server.sse import SseServerTransport
from git_mcp_server import GitMCPServer

# Initialize the Git MCP server
server = GitMCPServer()

# Configure SSE transport with HTTP endpoints
sse_transport = SseServerTransport(
    sse_endpoint="/sse",
    post_endpoint="/messages"
)

# Start the server with SSE transport
server.connect(sse_transport)
server.run()
