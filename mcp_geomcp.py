#!/usr/bin/env python
"""
GeoMCP → MCP façade for the existing FastAPI geo service.

This tries **local imports first** (best for you, because your logic lives in
`server.py` and `modules/`), and if that fails it will fall back to calling
the running FastAPI app over HTTP at http://127.0.0.1:8000.

Run (local, stdio):
    python mcp_geomcp.py
or:
    python mcp_geomcp.py stdio
"""

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# optional .env — so you can keep Sentinel Hub creds in config/api_keys.env
try:
    from dotenv import load_dotenv
    load_dotenv("config/api_keys.env")
except Exception:
    pass

mcp = FastMCP("GeoMCP")

# ---------------------------------------------------------------------------
# Try to use your real Python modules (best path)
# ---------------------------------------------------------------------------
LOCAL_MODE = True
geoapp = None
sh = None
cloudfree = None
zonal = None

try:
    import server as geoapp          # your existing FastAPI app
    from modules import sentinel_hub as sh
    from modules import cloudfree
    from modules import zonal
except Exception:
    # if we can't import, we will call the HTTP app
    LOCAL_MODE = False

# base URL for HTTP fallback (when we can’t import)
GEOMCP_BASE = os.environ.get("GEOMCP_BASE", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_from_list(b: List[float]) -> List[float]:
    if len(b) != 4:
        raise ValueError("bbox must be [minLon, minLat, maxLon, maxLat]")
    return [float(x) for x in b]


def _ensure_token():
    """
    Make sure Sentinel Hub auth works. Only relevant in LOCAL_MODE.
    """
    if not LOCAL_MODE:
        return
    # your sentinel_hub.py exposes something like _get_token()
    if hasattr(sh, "_get_token"):
        sh._get_token()
    else:
        # older module name? just do nothing
        pass


async def _http_post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import httpx
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{GEOMCP_BASE}{path}", json=payload)
        r.raise_for_status()
        return r.json()


async def _http_get_bytes(path: str, params: Dict[str, Any]) -> bytes:
    import httpx
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"{GEOMCP_BASE}{path}", params=params)
        r.raise_for_status()
        return r.content

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def health() -> Dict[str, Any]:
    """
    Check that the GeoMCP server can reach Sentinel (local mode) or HTTP app.
    """
    if LOCAL_MODE:
        try:
            _ensure_token()
            return {"ok": True, "mode": "local", "reason": "sentinel ok"}
        except Exception as e:
            return {"ok": False, "mode": "local", "reason": f"sentinel auth failed: {e!r}"}
    else:
        # HTTP fallback: just try /health on the FastAPI app
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{GEOMCP_BASE}/health")
                r.raise_for_status()
            return {"ok": True, "mode": "http", "reason": "fastapi /health ok"}
        except Exception as e:
            return {"ok": False, "mode": "http", "reason": f"http /health failed: {e!r}"}


@mcp.tool()
async def list_capabilities() -> Dict[str, Any]:
    """
    Quick discovery for Claude: what can this GeoMCP do?
    """
    return {
        "raster": ["ndvi", "ndwi", "ndbi", "truecolor", "cloudfree_truecolor", "cloudfree_ndvi", "cloudfree_ndwi"],
        "zonal": ["timeseries_polygon", "timeseries_bbox"],
        "charts": ["point_series_png"],
        "notes": "Most raster ops need valid Sentinel Hub creds in config/api_keys.env",
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
    Get NDVI PNG for a bbox/time window. Returns base64 PNG.
    Mirrors your GET /ndvi.png endpoint.
    """
    bb = _bbox_from_list(bbox)

    if LOCAL_MODE:
        _ensure_token()
        png_bytes = sh.process_png(
            bbox=bb,
            from_iso=from_date,
            to_iso=to_date,
            width=width,
            height=height,
            evalscript=sh.NDVI_PNG_EVALSCRIPT,
            collection=collection,
        )
    else:
        # call your HTTP endpoint
        params = dict(bbox=",".join(map(str, bb)), from_date=from_date, to_date=to_date, width=width, height=height)
        png_bytes = await _http_get_bytes("/ndvi.png", params)

    return {
        "content_type": "image/png",
        "data_base64": base64.b64encode(png_bytes).decode("ascii"),
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
    Cloud-free truecolor composite. Mirrors /cloudfree.truecolor.png
    """
    bb = _bbox_from_list(bbox)

    if LOCAL_MODE:
        _ensure_token()
        png_bytes = cloudfree.truecolor_png(
            bbox=bb,
            from_iso=from_date,
            to_iso=to_date,
            width=width,
            height=height,
            maxcc=maxcc,
        )
    else:
        params = dict(
            bbox=",".join(map(str, bb)),
            from_date=from_date,
            to_date=to_date,
            width=width,
            height=height,
            maxcc=maxcc,
        )
        png_bytes = await _http_get_bytes("/cloudfree.truecolor.png", params)

    return {
        "content_type": "image/png",
        "data_base64": base64.b64encode(png_bytes).decode("ascii"),
    }


@mcp.tool()
async def zonal_timeseries(
    geometry: Dict[str, Any],
    index: str = "ndvi",
    from_date: str = "2025-01-01",
    to_date: str = "2025-01-31",
    step_days: int = 7,
    cloud_mask: bool = False,
    composite: bool = False,
) -> Dict[str, Any]:
    """
    Get NDVI/NDWI/etc time-series over a polygon. Mirrors POST /zonal.timeseries.json
    """
    payload = {
        "geometry": geometry,
        "index": index,
        "from_date": from_date,
        "to_date": to_date,
        "step_days": step_days,
        "cloud_mask": cloud_mask,
        "composite": composite,
    }

    if LOCAL_MODE:
        _ensure_token()
        # call the real Python function from modules/zonal.py
        out = zonal.zonal_timeseries_index(
            index=index,
            geometry=geometry,
            from_iso=from_date,
            to_iso=to_date,
            step_days=step_days,
            cloud_mask=cloud_mask,
            composite=composite,
        )
        return out
    else:
        out = await _http_post_json("/zonal.timeseries.json", payload)
        return out


@mcp.tool()
async def point_series_png(
    lat: float,
    lon: float,
    index: str = "ndvi",
    from_date: str = "2025-01-01",
    to_date: str = "2025-01-31",
    step_days: int = 7,
    buffer_m: float = 0.0,
) -> Dict[str, Any]:
    """
    Return the chart PNG (base64) for a single point. Mirrors /series/point.png.
    """
    if LOCAL_MODE and geoapp is not None:
        _ensure_token()
        body = geoapp.PointSeriesQuery(
            lat=lat,
            lon=lon,
            index=index,
            from_date=from_date,
            to_date=to_date,
            step_days=step_days,
            buffer_m=buffer_m,
            cloud_mask=False,
            composite=False,
        )
        # in your FastAPI app this returns a Response, here we want raw bytes
        series_json = geoapp.post_series_point_json(body)
        png_bytes = geoapp._render_series_chart(
            series=series_json["series"],
            value_key="value",
            title=f"{index.upper()} at ({lat:.5f},{lon:.5f})",
            size=(640, 360),
        )
    else:
        params = dict(
            lat=lat,
            lon=lon,
            index=index,
            from_date=from_date,
            to_date=to_date,
            step_days=step_days,
            buffer_m=buffer_m,
        )
        png_bytes = await _http_get_bytes("/series/point.png", params)

    return {
        "content_type": "image/png",
        "data_base64": base64.b64encode(png_bytes).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    # stdio is what Claude Desktop expects
    await mcp.run()


if __name__ == "__main__":
    asyncio.run(main())