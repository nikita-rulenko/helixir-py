"""
Helixir MCP Server Package

MCP server for instant memory operations:
- Add memories with LLM extraction
- Search memories with hybrid search
- Update/delete memories
"""

from helixir.mcp.server import mcp, run_server

__all__ = [
    "mcp",
    "run_server",
]
