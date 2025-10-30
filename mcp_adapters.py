"""
mcp_adapters.py
----------------
Thin compatibility layer for the GeoMCP project.

Goal:
- hide naming mismatches between the MCP tools and the actual project modules
- provide HTTP fallbacks to the existing FastAPI app (server.py routes)
- offer a colorized NDVI evalscript so chatbots don't get flat gray PNGs
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

# HTTP client only used for fallback
import httpx

# try local imports, but don't crash if missing
try:
    import server as geoapp
except Exception:
    geoapp = None  # type: ignore

try:
    from modules import sentinel_hub as sh
except Exception:
    sh = None  # type: ignore

try:
    from modules import cloudfree as cloudfree_mod
except Exception:
    cloudfree_mod = None  # type: ignore

try:
    from modules import zonal as zonal_mod
except Exception:
    zonal_mod = None  # type: ignore


# base URL for HTTP fallback (your FastAPI app)
GEOMCP_BASE = os.environ.get("GEOMCP_BASE", "http://127.0.0.1:8000")


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------

def ensure_sentinel_token() -> None:
    """
    Make sure Sentinel Hub auth is ready.
    If the local module isn't present, we do nothing (HTTP routes will handle it).
    """
    if sh is not None and hasattr(sh, "_get_token"):
        sh._get_token()


# ---------------------------------------------------------------------------
# Colorized NDVI evalscript (sentinel-hub style)
# ---------------------------------------------------------------------------

NDVI_COLOR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "dataMask"],
    output: { bands: 4 }
  };
}

function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  // clamp
  if (!isFinite(ndvi)) {
    ndvi = 0.0;
  }

  // simple palette
  // <0.0  -> dark gray
  // 0.0-0.2 -> light gray
  // 0.2-0.4 -> yellow-ish
  // 0.4-0.6 -> green
  // >0.6 -> dark green
  let r, g, b;
  if (ndvi < 0.0) {
    r = 0.05; g = 0.05; b = 0.05;
  } else if (ndvi < 0.2) {
    r = 0.85; g = 0.85; b = 0.85;
  } else if (ndvi < 0.4) {
    r = 0.9; g = 0.9; b = 0.2;
  } else if (ndvi < 0.6) {
    r = 0.2; g = 0.8; b = 0.2;
  } else {
    r = 0.0; g = 0.4; b = 0.0;
  }

  return [r, g, b, sample.dataMask];
}
"""


# ---------------------------------------------------------------------------
# HTTP helpers (fallback)
# ---------------------------------------------------------------------------

async def http_get_bytes(path: str, params: Dict[str, Any]) -> bytes:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"{GEOMCP_BASE}{path}", params=params)
        r.raise_for_status()
        return r.content


async def http_post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{GEOMCP_BASE}{path}", json=payload)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Cloud-free adapter
# ---------------------------------------------------------------------------

async def fetch_cloudfree_truecolor(
    bbox: List[float],
    from_date: str,
    to_date: str,
    width: int,
    height: int,
    maxcc: int,
) -> bytes:
    """
    Try local python first (modules.cloudfree.truecolor_png),
    else hit /cloudfree.truecolor.png
    """
    # local mode
    if cloudfree_mod is not None and hasattr(cloudfree_mod, "truecolor_png"):
        ensure_sentinel_token()
        return cloudfree_mod.truecolor_png(
            bbox=bbox,
            from_iso=from_date,
            to_iso=to_date,
            width=width,
            height=height,
            maxcc=maxcc,
        )

    # fallback to HTTP
    params = dict(
        bbox=",".join(map(str, bbox)),
        from_date=from_date,
        to_date=to_date,
        width=width,
        height=height,
        maxcc=maxcc,
    )
    return await http_get_bytes("/cloudfree.truecolor.png", params)


# ---------------------------------------------------------------------------
# Zonal adapter
# ---------------------------------------------------------------------------

async def fetch_zonal_timeseries(
    geometry: Dict[str, Any],
    index: str,
    from_date: str,
    to_date: str,
    step_days: int,
    cloud_mask: bool,
    composite: bool,
) -> Dict[str, Any]:
    """
    Try modules.zonal.zonal_timeseries_index(...), else POST to /zonal.timeseries.json
    """
    if zonal_mod is not None and hasattr(zonal_mod, "zonal_timeseries_index"):
        ensure_sentinel_token()
        out = zonal_mod.zonal_timeseries_index(
            index=index,
            geometry=geometry,
            from_iso=from_date,
            to_iso=to_date,
            step_days=step_days,
            cloud_mask=cloud_mask,
            composite=composite,
        )
        return out

    # fallback to HTTP
    payload = {
        "geometry": geometry,
        "index": index,
        "from_date": from_date,
        "to_date": to_date,
        "step_days": step_days,
        "cloud_mask": cloud_mask,
        "composite": composite,
    }
    return await http_post_json("/zonal.timeseries.json", payload)


# ---------------------------------------------------------------------------
# NDVI creators
# ---------------------------------------------------------------------------

async def fetch_ndvi_preview_png(
    bbox: List[float],
    from_date: str,
    to_date: str,
    width: int = 256,
    height: int = 256,
    collection: str = "S2L2A",
) -> bytes:
    """
    Small, colorized NDVI that chatbots can show easily.
    """
    if sh is not None and hasattr(sh, "process_png"):
        ensure_sentinel_token()
        return sh.process_png(
            bbox=bbox,
            from_iso=from_date,
            to_iso=to_date,
            width=width,
            height=height,
            evalscript=NDVI_COLOR_EVALSCRIPT,
            collection=collection,
        )

    # fallback to HTTP: we assume your REST API has /ndvi.png but that one is grayscale.
    # so we hit it and just return whatever we got.
    params = dict(
        bbox=",".join(map(str, bbox)),
        from_date=from_date,
        to_date=to_date,
        width=width,
        height=height,
    )
    return await http_get_bytes("/ndvi.png", params)