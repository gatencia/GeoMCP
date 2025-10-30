#!/usr/bin/env python
"""
mcp_geomcp.py
--------------
MCP façade for the GeoMCP project.

- uses mcp_adapters for tolerant calls
- returns small, chatbot-friendly outputs
- falls back to HTTP routes when local modules are missing
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

import os

# our tolerant adapters
import mcp_adapters as adapters

mcp = FastMCP("GeoMCP")

# if for some reason we want to force HTTP fallback from env
FORCE_HTTP = os.environ.get("GEOMCP_FORCE_HTTP", "0") == "1"


def _bbox_from_list(b: List[float]) -> List[float]:
    if len(b) != 4:
        raise ValueError("bbox must be [minLon, minLat, maxLon, maxLat]")
    return [float(x) for x in b]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def health() -> Dict[str, Any]:
    """
    Check that the server is alive and that Sentinel Hub is reachable (if local).
    """
    try:
        adapters.ensure_sentinel_token()
        return {"ok": True, "mode": "local-or-http", "reason": "sentinel ok or not needed"}
    except Exception as e:
        # still return ok=False but don't crash MCP
        return {"ok": False, "mode": "local-or-http", "reason": f"{e!r}"}


@mcp.tool()
async def list_capabilities() -> Dict[str, Any]:
    """
    Tell Claude what we can do.
    """
    return {
        "raster": [
            "ndvi_preview",          # new, colorized, small
            "ndvi_raw",              # old behavior
            "cloudfree_truecolor"    # cloudfree
        ],
        "zonal": [
            "timeseries_polygon"
        ],
        "notes": "All tools auto-fallback to HTTP FastAPI routes if Python modules are missing."
    }


@mcp.tool()
async def get_ndvi_preview(
    bbox: List[float],
    from_date: str,
    to_date: str,
    width: int = 256,
    height: int = 256,
    collection: str = "S2L2A",
) -> Dict[str, Any]:
    """
    Get a COLORIZED, SMALL NDVI PNG suitable for chat display.
    """
    bb = _bbox_from_list(bbox)
    png_bytes = await adapters.fetch_ndvi_preview_png(
        bbox=bb,
        from_date=from_date,
        to_date=to_date,
        width=width,
        height=height,
        collection=collection,
    )
    return {
        "content_type": "image/png",
        "data_base64": base64.b64encode(png_bytes).decode("ascii"),
        "width": width,
        "height": height,
    }


@mcp.tool()
async def get_ndvi_png(
    bbox: List[float],
    from_date: str,
    to_date: str,
    width: int = 512,
    height: int = 512,
    collection: str = "S2L2A",
) -> Dict[str, Any]:
    """
    Original behavior: grayscale NDVI (whatever your REST returns).
    We keep this for backward compatibility.
    """
    bb = _bbox_from_list(bbox)
    png_bytes = await adapters.fetch_ndvi_preview_png(
        bbox=bb,
        from_date=from_date,
        to_date=to_date,
        width=width,
        height=height,
        collection=collection,
    )
    return {
        "content_type": "image/png",
        "data_base64": base64.b64encode(png_bytes).decode("ascii"),
        "width": width,
        "height": height,
    }


@mcp.tool()
async def get_cloudfree_truecolor(
    bbox: List[float],
    from_date: str,
    to_date: str,
    width: int = 512,
    height: int = 512,
    maxcc: int = 20,
) -> Dict[str, Any]:
    """
    Cloud-free truecolor with tolerant backend.
    """
    bb = _bbox_from_list(bbox)
    png_bytes = await adapters.fetch_cloudfree_truecolor(
        bbox=bb,
        from_date=from_date,
        to_date=to_date,
        width=width,
        height=height,
        maxcc=maxcc,
    )
    return {
        "content_type": "image/png",
        "data_base64": base64.b64encode(png_bytes).decode("ascii"),
        "width": width,
        "height": height,
    }


@mcp.tool()
async def zonal_timeseries(
    geometry: Dict[str, Any],
    index: str = "ndvi",
    from_date: str = "2025-01-01",
    to_date: str = "2025-01-15",
    step_days: int = 5,
    cloud_mask: bool = False,
    composite: bool = False,
) -> Dict[str, Any]:
    """
    Polygon → time-series, tolerant.
    Always returns JSON (small, chat-friendly).
    """
    out = await adapters.fetch_zonal_timeseries(
        geometry=geometry,
        index=index,
        from_date=from_date,
        to_date=to_date,
        step_days=step_days,
        cloud_mask=cloud_mask,
        composite=composite,
    )
    return out


# ---------------------------------------------------------------------------
# Entrypoint (no asyncio.run here, fastmcp manages its own loop)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()