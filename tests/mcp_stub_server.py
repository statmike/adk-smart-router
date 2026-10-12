"""A three-tool MCP server over stdio, for `test_tool_origins.py` to prune.

Not a test and not collected as one. It is run as a subprocess by `McpToolset`, which speaks
the protocol to it over stdin and stdout, so it has to be a real server rather than a stub
object: the thing under test is whether a tool that arrived over MCP looks like any other tool
by the time the router sees the request, and a hand-built `BaseTool` would beg that question.

The tools do nothing interesting on purpose. What matters is that they have names, schemas and
docstrings that ADK will turn into `FunctionDeclaration`s.
"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("weather")


@server.tool()
def get_forecast(city: str) -> str:
    """Tomorrow's forecast for a city."""
    return f"{city}: sunny, 22C"


@server.tool()
def get_alerts(state: str) -> str:
    """Active weather alerts for a state."""
    return f"{state}: no alerts"


@server.tool()
def get_air_quality(city: str) -> str:
    """The current air quality index for a city."""
    return f"{city}: AQI 31"


if __name__ == "__main__":
    server.run(transport="stdio")
