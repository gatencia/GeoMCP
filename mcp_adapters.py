"""
mcp_adapters.py
----------------
Thin compatibility layer for the GeoMCP project.

- Hides naming mismatches between MCP tools and project modules.
- Provides a hybrid execution model: tries local Python functions first,
  then falls back to HTTP requests to the FastAPI backend.
- Offers a colorized NDVI evalscript for better chat client previews.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

import httpx

# --- Try to import local modules, but don't fail if they are not available ---
try:
    from modules import sentinel_hub as sh
except ImportError:
    sh = None

try:
    from modules import cloudfree as cloudfree_mod
except ImportError:
    cloudfree_mod = None

try:
    from modules import zonal as zonal_mod
except ImportError:
    zonal_mod = None

try:
    from modules import elevation as elevation_mod
except ImportError:
    elevation_mod = None

try:
    from modules import ndwi as ndwi_mod
except ImportError:
    ndwi_mod = None

try:
    from modules import ndbi as ndbi_mod
except ImportError:
    ndbi_mod = None

try:
    from modules import classification as classification_mod
except ImportError:
    classification_mod = None

# --- Constants and Configuration ---
GEOMCP_BASE = os.environ.get("GEOMCP_BASE", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Sentinel Hub Helpers
# ---------------------------------------------------------------------------

def ensure_sentinel_token() -> None:
    """
    Ensures Sentinel Hub authentication is ready by calling the local module's
    token function. Does nothing if the module isn't present.
    """
    if sh and hasattr(sh, "_get_token"):
        sh._get_token()

def check_sentinel_token_local() -> Dict[str, Any]:
    """
    Checks the status of the local Sentinel Hub token without making HTTP calls.
    """
    if sh and hasattr(sh, "_token_cache"):
        if sh.TOKEN:
            return {"status": "ok", "method": "static_token"}
        if sh._token_cache.get("access_token"):
            expires_in = sh._token_cache.get("expires_at", 0) - sh.time.time()
            if expires_in > 0:
                return {"status": "ok", "method": "oauth", "expires_in_seconds": round(expires_in)}
            return {"status": "expired", "method": "oauth"}
    return {"status": "unavailable", "reason": "sentinel_hub module not found or configured"}

# ---------------------------------------------------------------------------
# Generic HTTP Helpers (Fallback Mechanism)
# ---------------------------------------------------------------------------

async def http_get_bytes(
    path: str, params: Dict[str, Any], base_url: str, force_http: bool
) -> bytes:
    """Generic GET request that returns raw bytes."""
    url = f"{base_url}{path}"
    if force_http:
        url = url.replace("https://", "http://")
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.content
        except httpx.RequestError as e:
            raise ConnectionError(f"HTTP GET failed for {url}: {e}") from e

async def http_post_json(
    path: str, payload: Dict[str, Any], base_url: str, force_http: bool
) -> Dict[str, Any]:
    """Generic POST request that sends and receives JSON."""
    url = f"{base_url}{path}"
    if force_http:
        url = url.replace("https://", "http://")
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            raise ConnectionError(f"HTTP POST failed for {url}: {e}") from e

# ---------------------------------------------------------------------------
# Health & Status Adapter
# ---------------------------------------------------------------------------

async def fetch_http_status(base_url: str, force_http: bool) -> Dict[str, Any]:
    """Hits the /health endpoint of the FastAPI app."""
    url = f"{base_url}/health"
    if force_http:
        url = url.replace("https://", "http://")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"status": "error", "reason": str(e)}

# ---------------------------------------------------------------------------
# Indices Adapters (NDVI, NDWI, NDBI)
# ---------------------------------------------------------------------------

async def fetch_ndvi_png(
    *, bbox: List[float], from_date: str, to_date: str, width: int, height: int,
    collection: str, base_url: str, force_http: bool
) -> bytes:
    """Adapter for NDVI PNG, using local `process_png` or falling back to HTTP."""
    if sh and hasattr(sh, "process_png") and not force_http:
        ensure_sentinel_token()
        return sh.process_png(
            bbox=bbox, from_iso=from_date, to_iso=to_date,
            width=width, height=height, evalscript=sh.NDVI_PNG_EVALSCRIPT,
            collection=collection
        )
    params = {
        "bbox": ",".join(map(str, bbox)), "from": from_date, "to": to_date,
        "width": width, "height": height, "collection": collection
    }
    return await http_get_bytes("/ndvi.png", params, base_url, force_http)

async def fetch_ndwi_png(
    *, bbox: List[float], from_date: str, to_date: str, width: int, height: int,
    base_url: str, force_http: bool
) -> bytes:
    if ndwi_mod and hasattr(ndwi_mod, "get_ndwi") and not force_http:
        ensure_sentinel_token()
        # Assuming get_ndwi takes from_date/to_date, which it should.
        # If not, this local call needs adjustment.
        return ndwi_mod.get_ndwi(bbox=bbox, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "from": from_date, "to": to_date, "width": width, "height": height}
    return await http_get_bytes("/ndwi.png", params, base_url, force_http)

async def fetch_ndwi_tiff(*, bbox: List[float], from_date: str, to_date: str, width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if ndwi_mod and hasattr(ndwi_mod, "get_ndwi_raw") and not force_http:
        ensure_sentinel_token()
        return ndwi_mod.get_ndwi_raw(bbox=bbox, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "from": from_date, "to": to_date, "width": width, "height": height}
    return await http_get_bytes("/ndwi.tif", params, base_url, force_http) # Assumes /ndwi.tif route

async def fetch_ndwi_matrix(*, bbox: List[float], from_date: str, to_date: str, width: int, height: int, base_url: str, force_http: bool) -> Dict[str, Any]:
    if ndwi_mod and hasattr(ndwi_mod, "get_ndwi_matrix") and not force_http:
        ensure_sentinel_token()
        return ndwi_mod.get_ndwi_matrix(bbox=bbox, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "from": from_date, "to": to_date, "width": width, "height": height}
    # This should be a POST or GET returning JSON, guessing GET returning JSON
    url = f"{base_url}/ndwi.matrix"
    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def fetch_ndbi_png(*, bbox: List[float], from_date: str, to_date: str, width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if ndbi_mod and hasattr(ndbi_mod, "get_ndbi_png") and not force_http:
        ensure_sentinel_token()
        return ndbi_mod.get_ndbi_png(bbox=bbox, from_iso=from_date, to_iso=to_date, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "from_date": from_date, "to_date": to_date, "width": width, "height": height}
    return await http_get_bytes("/ndbi.png", params, base_url, force_http)

async def fetch_ndbi_tiff(*, bbox: List[float], from_date: str, to_date: str, width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if ndbi_mod and hasattr(ndbi_mod, "get_ndbi_tiff") and not force_http:
        ensure_sentinel_token()
        return ndbi_mod.get_ndbi_tiff(bbox=bbox, from_iso=from_date, to_iso=to_date, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "from_date": from_date, "to_date": to_date, "width": width, "height": height}
    return await http_get_bytes("/ndbi.tif", params, base_url, force_http)

async def fetch_ndbi_matrix(*, bbox: List[float], from_date: str, to_date: str, width: int, height: int, base_url: str, force_http: bool) -> Dict[str, Any]:
    if ndbi_mod and hasattr(ndbi_mod, "get_ndbi_matrix") and not force_http:
        ensure_sentinel_token()
        return ndbi_mod.get_ndbi_matrix(bbox=bbox, from_iso=from_date, to_iso=to_date, width=width, height=height)
    payload = {"bbox": bbox, "from_date": from_date, "to_date": to_date, "width": width, "height": height}
    return await http_post_json("/ndbi.matrix", payload, base_url, force_http)

# ---------------------------------------------------------------------------
# Cloud-free Composite Adapters
# ---------------------------------------------------------------------------

async def fetch_cloudfree_truecolor_png(
    *, bbox: List[float], from_date: str, to_date: str, maxcc: int,
    width: int, height: int, base_url: str, force_http: bool
) -> bytes:
    if cloudfree_mod and hasattr(cloudfree_mod, "get_cloudfree_truecolor_png") and not force_http:
        ensure_sentinel_token()
        return cloudfree_mod.get_cloudfree_truecolor_png(
            bbox=bbox, from_iso=from_date, to_iso=to_date,
            maxcc=maxcc, width=width, height=height
        )
    params = {
        "bbox": ",".join(map(str, bbox)), "from_date": from_date, "to_date": to_date,
        "maxcc": maxcc, "width": width, "height": height
    }
    return await http_get_bytes("/cloudfree/truecolor.png", params, base_url, force_http)

async def fetch_cloudfree_ndvi_png(*, bbox: List[float], from_date: str, to_date: str, maxcc: int, width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if cloudfree_mod and hasattr(cloudfree_mod, "get_cloudfree_ndvi_png") and not force_http:
        ensure_sentinel_token()
        return cloudfree_mod.get_cloudfree_ndvi_png(bbox=bbox, from_iso=from_date, to_iso=to_date, maxcc=maxcc, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "from_date": from_date, "to_date": to_date, "maxcc": maxcc, "width": width, "height": height}
    return await http_get_bytes("/cloudfree/ndvi.png", params, base_url, force_http) # Route guess

async def fetch_cloudfree_ndwi_png(*, bbox: List[float], from_date: str, to_date: str, maxcc: int, width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if cloudfree_mod and hasattr(cloudfree_mod, "get_cloudfree_ndwi_png") and not force_http:
        ensure_sentinel_token()
        return cloudfree_mod.get_cloudfree_ndwi_png(bbox=bbox, from_iso=from_date, to_iso=to_date, maxcc=maxcc, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "from_date": from_date, "to_date": to_date, "maxcc": maxcc, "width": width, "height": height}
    return await http_get_bytes("/cloudfree/ndwi.png", params, base_url, force_http) # Route guess

# ---------------------------------------------------------------------------
# Elevation & Terrain Adapters
# ---------------------------------------------------------------------------

async def _fetch_dem_first(bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> bytes:
    """Helper to get raw DEM tiff, required by many terrain functions."""
    if elevation_mod and hasattr(elevation_mod, "get_elevation_raw") and not force_http:
        ensure_sentinel_token()
        return elevation_mod.get_elevation_raw(bbox=bbox, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "width": width, "height": height}
    return await http_get_bytes("/elevation/raw", params, base_url, force_http) # Assumes /elevation/raw route

async def fetch_elevation_png(*, bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if elevation_mod and hasattr(elevation_mod, "get_elevation") and not force_http:
        ensure_sentinel_token()
        return elevation_mod.get_elevation(bbox=bbox, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "width": width, "height": height}
    return await http_get_bytes("/elevation.png", params, base_url, force_http)

async def fetch_elevation_tiff(*, bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> bytes:
    return await _fetch_dem_first(bbox, width, height, base_url, force_http)

async def fetch_elevation_matrix(*, bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> Dict[str, Any]:
    if elevation_mod and hasattr(elevation_mod, "get_elevation_matrix") and not force_http:
        ensure_sentinel_token()
        return elevation_mod.get_elevation_matrix(bbox=bbox, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "width": width, "height": height}
    url = f"{base_url}/elevation.matrix"
    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def fetch_slope_png(*, bbox: List[float], width: int, height: int, vmax: float, base_url: str, force_http: bool) -> bytes:
    if elevation_mod and hasattr(elevation_mod, "slope_png_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.slope_png_from_dem_tiff(dem_tiff, bbox=bbox, width=width, height=height, vmax=vmax)
    params = {"bbox": ",".join(map(str, bbox)), "width": width, "height": height, "vmax": vmax}
    return await http_get_bytes("/elevation/slope.png", params, base_url, force_http)

async def fetch_slope_tiff(*, bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if elevation_mod and hasattr(elevation_mod, "slope_tiff_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.slope_tiff_from_dem_tiff(dem_tiff, bbox=bbox, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "width": width, "height": height}
    return await http_get_bytes("/elevation/slope.tif", params, base_url, force_http)

async def fetch_slope_matrix(*, bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> Dict[str, Any]:
    if elevation_mod and hasattr(elevation_mod, "slope_matrix_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.slope_matrix_from_dem_tiff(dem_tiff, bbox=bbox, width=width, height=height)
    payload = {"bbox": bbox, "width": width, "height": height}
    return await http_post_json("/elevation/slope.matrix", payload, base_url, force_http)

async def fetch_aspect_tiff(*, bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if elevation_mod and hasattr(elevation_mod, "aspect_tiff_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.aspect_tiff_from_dem_tiff(dem_tiff, bbox=bbox, width=width, height=height)
    params = {"bbox": ",".join(map(str, bbox)), "width": width, "height": height}
    return await http_get_bytes("/elevation/aspect.tif", params, base_url, force_http)

async def fetch_aspect_matrix(*, bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> Dict[str, Any]:
    if elevation_mod and hasattr(elevation_mod, "aspect_matrix_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.aspect_matrix_from_dem_tiff(dem_tiff, bbox=bbox, width=width, height=height)
    payload = {"bbox": bbox, "width": width, "height": height}
    return await http_post_json("/elevation/aspect.matrix", payload, base_url, force_http)

async def fetch_hillshade_png(*, bbox: List[float], width: int, height: int, azimuth_deg: float, altitude_deg: float, base_url: str, force_http: bool) -> bytes:
    if elevation_mod and hasattr(elevation_mod, "hillshade_png_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.hillshade_png_from_dem_tiff(dem_tiff, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg)
    params = {"bbox": ",".join(map(str, bbox)), "width": width, "height": height, "azimuth_deg": azimuth_deg, "altitude_deg": altitude_deg}
    return await http_get_bytes("/elevation/hillshade.png", params, base_url, force_http)

async def fetch_hillshade_tiff(*, bbox: List[float], width: int, height: int, azimuth_deg: float, altitude_deg: float, base_url: str, force_http: bool) -> bytes:
    if elevation_mod and hasattr(elevation_mod, "hillshade_tiff_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.hillshade_tiff_from_dem_tiff(dem_tiff, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg)
    params = {"bbox": ",".join(map(str, bbox)), "width": width, "height": height, "azimuth_deg": azimuth_deg, "altitude_deg": altitude_deg}
    return await http_get_bytes("/elevation/hillshade.tif", params, base_url, force_http)

async def fetch_hillshade_matrix(*, bbox: List[float], width: int, height: int, azimuth_deg: float, altitude_deg: float, base_url: str, force_http: bool) -> Dict[str, Any]:
    if elevation_mod and hasattr(elevation_mod, "hillshade_matrix_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.hillshade_matrix_from_dem_tiff(dem_tiff, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg)
    payload = {"bbox": bbox, "width": width, "height": height, "azimuth_deg": azimuth_deg, "altitude_deg": altitude_deg}
    return await http_post_json("/elevation/hillshade.matrix", payload, base_url, force_http)

async def fetch_slope_vector_field_png(*, bbox: List[float], width: int, height: int, step: int, base_url: str, force_http: bool) -> bytes:
    if elevation_mod and hasattr(elevation_mod, "compute_slope_vector_field") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.compute_slope_vector_field(dem_tiff, step=step)
    # This is a complex visualization, assuming no simple HTTP endpoint exists.
    # A proper implementation would require a dedicated server route.
    raise NotImplementedError("Slope vector field PNG via HTTP is not supported. Use local mode.")

# ---------------------------------------------------------------------------
# Hydrology Adapters
# ---------------------------------------------------------------------------

async def fetch_flow_accumulation_png(*, bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if elevation_mod and hasattr(elevation_mod, "flow_accum_png_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.flow_accum_png_from_dem_tiff(dem_tiff)
    params = {"bbox": ",".join(map(str, bbox)), "width": width, "height": height}
    return await http_get_bytes("/flow/accumulation.png", params, base_url, force_http)

async def fetch_flow_accumulation_tiff(*, bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if elevation_mod and hasattr(elevation_mod, "flow_accum_tiff_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.flow_accum_tiff_from_dem_tiff(dem_tiff)
    params = {"bbox": ",".join(map(str, bbox)), "width": width, "height": height}
    return await http_get_bytes("/flow/accumulation.tif", params, base_url, force_http)

async def fetch_flow_accumulation_matrix(*, bbox: List[float], width: int, height: int, base_url: str, force_http: bool) -> Dict[str, Any]:
    if elevation_mod and hasattr(elevation_mod, "flow_accum_matrix_from_dem_tiff") and not force_http:
        dem_tiff = await _fetch_dem_first(bbox, width, height, base_url, force_http)
        return elevation_mod.flow_accum_matrix_from_dem_tiff(dem_tiff)
    payload = {"bbox": bbox, "width": width, "height": height}
    return await http_post_json("/flow/accumulation.matrix", payload, base_url, force_http)

# ---------------------------------------------------------------------------
# Classification Adapters
# ---------------------------------------------------------------------------

async def fetch_unsupervised_classification_png(*, bbox: List[float], from_date: str, to_date: str, k: int, width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if classification_mod and hasattr(classification_mod, "get_unsupervised_png") and not force_http:
        ensure_sentinel_token()
        return classification_mod.get_unsupervised_png(bbox=bbox, from_iso=from_date, to_iso=to_date, k=k, width=width, height=height)
    payload = {"bbox": bbox, "from_date": from_date, "to_date": to_date, "k": k, "width": width, "height": height}
    # Custom endpoint, requires POST
    return await http_post_json("/custom/unsupervised_classification.png_bytes", payload, base_url, force_http)

async def fetch_supervised_classification_png(*, bbox: List[float], from_date: str, to_date: str, training_points: List[Dict[str, Any]], width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if classification_mod and hasattr(classification_mod, "get_supervised_png") and not force_http:
        ensure_sentinel_token()
        return classification_mod.get_supervised_png(bbox=bbox, from_iso=from_date, to_iso=to_date, training_points=training_points, width=width, height=height)
    payload = {"bbox": bbox, "from_date": from_date, "to_date": to_date, "training_points": training_points, "width": width, "height": height}
    return await http_post_json("/custom/supervised_classification.png_bytes", payload, base_url, force_http)

async def fetch_rule_based_classification_png(*, bbox: List[float], from_date: str, to_date: str, width: int, height: int, base_url: str, force_http: bool) -> bytes:
    if classification_mod and hasattr(classification_mod, "get_rule_based_png") and not force_http:
        ensure_sentinel_token()
        return classification_mod.get_rule_based_png(bbox=bbox, from_iso=from_date, to_iso=to_date, width=width, height=height)
    payload = {"bbox": bbox, "from_date": from_date, "to_date": to_date, "width": width, "height": height}
    return await http_post_json("/custom/rule_based_classification.png_bytes", payload, base_url, force_http)

# ---------------------------------------------------------------------------
# Zonal Analysis Adapters
# ---------------------------------------------------------------------------

async def fetch_zonal_timeseries(
    *, index: str, from_date: str, to_date: str, step_days: int, cloud_mask: bool,
    composite: bool, base_url: str, force_http: bool,
    geometry: Optional[Dict[str, Any]] = None, bbox: Optional[List[float]] = None
) -> Dict[str, Any]:
    if zonal_mod and hasattr(zonal_mod, "area_series_index") and not force_http:
        ensure_sentinel_token()
        return zonal_mod.area_series_index(
            index=index, geometry=geometry, bbox=bbox, from_iso=from_date, to_iso=to_date,
            step_days=step_days, cloud_mask=cloud_mask, composite=composite
        )
    payload = {
        "geometry": geometry, "bbox": bbox, "index": index, "from_date": from_date,
        "to_date": to_date, "step_days": step_days, "cloud_mask": cloud_mask, "composite": composite
    }
    return await http_post_json("/zonal_timeseries.json", payload, base_url, force_http)

async def fetch_point_timeseries(
    *, index: str, lat: float, lon: float, from_date: str, to_date: str,
    step_days: int, cloud_mask: bool, base_url: str, force_http: bool
) -> Dict[str, Any]:
    if zonal_mod and hasattr(zonal_mod, "point_series_index") and not force_http:
        ensure_sentinel_token()
        return zonal_mod.point_series_index(
            index=index, lat=lat, lon=lon, from_iso=from_date, to_iso=to_date,
            step_days=step_days, cloud_mask=cloud_mask
        )
    payload = {
        "index": index, "lat": lat, "lon": lon, "from_date": from_date,
        "to_date": to_date, "step_days": step_days, "cloud_mask": cloud_mask
    }
    return await http_post_json("/point_timeseries.json", payload, base_url, force_http)