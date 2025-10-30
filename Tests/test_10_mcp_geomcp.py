# tests/test_10_mcp_geomcp.py

import asyncio
import importlib
import pytest


@pytest.mark.asyncio
async def test_mcp_health():
    mcp_mod = importlib.import_module("mcp_geomcp")
    # FastMCP registers tools on import, they live on mcp_mod.mcp
    tools = mcp_mod.mcp._tools  # yes, private, but good enough for test
    assert "health" in tools

    # call the tool like the LLM would
    health_tool = tools["health"]
    result = await health_tool.func()
    assert "ok" in result


@pytest.mark.asyncio
async def test_mcp_ndvi_preview_or_png():
    mcp_mod = importlib.import_module("mcp_geomcp")
    tools = mcp_mod.mcp._tools

    # accept either name, depending on which version you end up with
    tool_name = None
    for cand in ["get_ndvi_preview", "get_ndvi_png"]:
        if cand in tools:
            tool_name = cand
            break

    assert tool_name is not None, "MCP must expose get_ndvi_preview or get_ndvi_png"

    f = tools[tool_name].func

    res = await f(
        bbox=[15.2, -0.2, 15.25, -0.15],
        from_date="2025-01-01",
        to_date="2025-01-05",
        width=128,
        height=128,
    )

    assert "data_base64" in res, "MCP NDVI tool must return base64 image"
    assert res["data_base64"].strip() != "", "base64 image must not be empty"