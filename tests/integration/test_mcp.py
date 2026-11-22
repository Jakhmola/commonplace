"""The MCP server over its real transport: a subprocess speaking stdio."""

import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.integration.conftest import INNGEST_UP

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "commonplace.mcp.server"])


async def test_tools_are_listed_over_stdio() -> None:
    async with stdio_client(SERVER) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = await session.list_tools()
    assert len(listed.tools) == 9
    assert {"cp_search", "cp_ask", "cp_sync"} <= {tool.name for tool in listed.tools}


async def test_health_can_be_called_over_stdio() -> None:
    if not INNGEST_UP:
        pytest.skip("The Inngest dev server is not running (make up)")
    async with stdio_client(SERVER) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("cp_health", {})
    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["database"] == "ok"
