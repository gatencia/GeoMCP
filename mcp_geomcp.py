"""
mcp_geomcp.py
-------------
MCP tool facade for the GeoMCP project.

Exposes all geospatial functions from the `modules/` directory as
discoverable and callable tools for LLM agents like Claude.
"""

import base64
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeAlias
from uuid import uuid4

import numpy as np
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import File

from mcp_adapters import (
    GEOMCP_BASE,
    fetch_http_status,
    check_sentinel_token_local,
    fetch_ndvi_png,
    fetch_ndvi_matrix,
    fetch_ndwi_png,
    fetch_ndwi_tiff,
    fetch_ndwi_matrix,
    fetch_ndbi_png,
    fetch_ndbi_tiff,
    fetch_ndbi_matrix,
    fetch_ndre_matrix,
    fetch_evi_matrix,
    fetch_msavi_matrix,
    fetch_nbr_matrix,
    fetch_cloudfree_truecolor_png,
    fetch_cloudfree_ndvi_png,
    fetch_cloudfree_ndwi_png,
    fetch_elevation_png,
    fetch_elevation_tiff,
    fetch_elevation_matrix,
    fetch_slope_png,
    fetch_slope_tiff,
    fetch_slope_matrix,
    fetch_aspect_tiff,
    fetch_aspect_matrix,
    fetch_hillshade_png,
    fetch_hillshade_tiff,
    fetch_hillshade_matrix,
    fetch_flow_accumulation_png,
    fetch_flow_accumulation_tiff,
    fetch_flow_accumulation_matrix,
    fetch_unsupervised_classification_png,
    fetch_supervised_classification_png,
    fetch_rule_based_classification_png,
    fetch_zonal_timeseries,
    fetch_point_timeseries,
)

# --- MCP Setup ---
mcp = FastMCP("GeoMCP")

# FastMCP 2.x no longer exposes `_tools`; add back for test-compatibility.
if not hasattr(mcp, "_tools") and hasattr(mcp, "_tool_manager"):
    tool_manager = getattr(mcp, "_tool_manager")
    if hasattr(tool_manager, "_tools"):
        mcp._tools = tool_manager._tools  # type: ignore[attr-defined]

_DEFAULT_MATRIX_DIR = Path(__file__).resolve().parent / "tmp" / "matrix_cache"
MATRIX_STORAGE_DIR = Path(os.environ.get("GEOMCP_MATRIX_DIR", _DEFAULT_MATRIX_DIR))

BBoxInput: TypeAlias = Sequence[float] | Sequence[Sequence[float]]


def _normalize_bbox(bbox: BBoxInput) -> list[float]:
    """Accept [minLon, minLat, maxLon, maxLat] or [(minLon, minLat), (maxLon, maxLat)]."""

    if len(bbox) == 4 and all(isinstance(value, (int, float)) for value in bbox):
        return [float(value) for value in bbox]

    if len(bbox) == 2:
        first, second = bbox
        if (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) == len(second) == 2
        ):
            min_lon, min_lat = first
            max_lon, max_lat = second
            return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]

    raise ValueError(
        "bbox must be [minLon, minLat, maxLon, maxLat] or [(minLon, minLat), (maxLon, maxLat)]"
    )


def _normalize_optional_bbox(bbox: Optional[BBoxInput]) -> Optional[list[float]]:
    return _normalize_bbox(bbox) if bbox is not None else None

# ---------------------------------------------------------------------------
# Helper for Image-Returning Tools
# ---------------------------------------------------------------------------

def _format_png_response(png_bytes: bytes, width: int, height: int) -> Dict[str, Any]:
    """Encodes PNG bytes into a base64 JSON response for chat clients."""
    return {
        "content_type": "image/png",
        "data_base64": base64.b64encode(png_bytes).decode("ascii"),
        "width": width,
        "height": height,
    }

def _format_tiff_response(tiff_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Encodes TIFF bytes into a base64 JSON response for download."""
    return {
        "content_type": "image/tiff",
        "data_base64": base64.b64encode(tiff_bytes).decode("ascii"),
        "filename": filename,
        "description": "GeoTIFF file with float32 data.",
    }


def _detect_matrix_field(payload: Dict[str, Any]) -> tuple[str, list]:
    """Identify the key that contains the primary matrix payload."""
    candidate_keys = (
        "matrix",
        "values",
        "elevation",
        "slope",
        "aspect",
        "hillshade",
        "accumulation_log",
    )
    for key in candidate_keys:
        matrix = payload.get(key)
        if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
            return key, matrix
    raise ValueError("Matrix payload missing expected array field")


def _matrix_to_file_response(
    payload: Dict[str, Any],
    *,
    as_file: bool,
    stem: str,
    description: str,
) -> ToolResult | Dict[str, Any]:
    """Return either the JSON payload or a file-backed ToolResult."""
    if not as_file:
        return payload

    key, matrix_values = _detect_matrix_field(payload)
    matrix_array = np.asarray(matrix_values, dtype=np.float32)

    MATRIX_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{stem}_{uuid4().hex}.npy"
    path = MATRIX_STORAGE_DIR / filename
    np.save(path, matrix_array)

    stats = {
        "shape": list(matrix_array.shape),
        "dtype": str(matrix_array.dtype),
        "min": float(np.min(matrix_array)) if matrix_array.size else 0.0,
        "max": float(np.max(matrix_array)) if matrix_array.size else 0.0,
        "mean": float(np.mean(matrix_array)) if matrix_array.size else 0.0,
    }

    metadata = {k: v for k, v in payload.items() if k != key}

    return ToolResult(
        content=[File(path=path)],
        structured_content={
            "matrix_key": key,
            "matrix_stats": stats,
            "metadata": metadata,
            "local_path": str(path),
            "description": description,
        },
    )


async def _collect_tool_names() -> List[str]:
    """Return sorted tool names from the FastMCP registry."""
    registered = await mcp.get_tools()
    names: List[str] = []

    if isinstance(registered, Mapping):
        names = [str(name) for name in registered.keys()]
    elif isinstance(registered, str):
        names = [registered]
    elif isinstance(registered, Sequence):
        for entry in registered:
            if isinstance(entry, str):
                names.append(entry)
            elif hasattr(entry, "key"):
                names.append(str(entry.key))
            elif hasattr(entry, "name"):
                names.append(str(entry.name))

    return sorted(set(names))


def _categorize_tools(tool_names: List[str]) -> Dict[str, List[str]]:
    """Group tool names into high-level capability buckets."""
    lowered = {name: name.lower() for name in tool_names}

    def _filter_by_keywords(keywords: Sequence[str]) -> List[str]:
        return sorted(
            [
                name
                for name, lowered_name in lowered.items()
                if any(keyword in lowered_name for keyword in keywords)
            ]
        )

    capabilities: Dict[str, List[str]] = {
    "indices": _filter_by_keywords(("ndvi", "ndwi", "ndbi", "ndre", "evi", "msavi", "nbr")),
        "composites": _filter_by_keywords(("cloudfree",)),
        "terrain": _filter_by_keywords(("elevation", "slope", "aspect", "hillshade")),
        "hydrology": _filter_by_keywords(("flow",)),
        "classification": _filter_by_keywords(("classification",)),
        "zonal_analysis": _filter_by_keywords(("zonal", "point_timeseries")),
    }

    core_tools = sorted(name for name in tool_names if name in {"health", "list_capabilities"})
    capabilities["core"] = core_tools
    return capabilities


async def _build_capability_summary() -> Dict[str, List[str]]:
    """Generate the capability summary used by status tools."""
    tool_names = await _collect_tool_names()
    return _categorize_tools(tool_names)

# ---------------------------------------------------------------------------
# Core & Status Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def health(force_http: bool = False) -> dict:
    """
    Checks the operational status of the GeoMCP system.

    This tool verifies both the local Python environment (e.g., Sentinel Hub token)
    and the connectivity to the backend FastAPI server.

    :param force_http: If True, forces the check to use the HTTP backend, bypassing local modules.
    :return: A dictionary with status information.
    """
    http_status = await fetch_http_status(base_url=GEOMCP_BASE, force_http=force_http)
    local_sentinel_status = check_sentinel_token_local()

    capabilities = await _build_capability_summary()
    status_ok = http_status.get("status") == "ok"

    return {
        "status": "ok" if status_ok else "degraded",
        "ok": status_ok,
        "local_sentinel_hub_auth": local_sentinel_status,
        "http_backend": http_status,
        "mcp_capabilities": capabilities,
    }


@mcp.tool()
async def list_capabilities() -> dict:
    """
    Lists all available geospatial tools grouped by category.

    :return: A dictionary containing all tool names organized by function.
    """
    return await _build_capability_summary()

# ---------------------------------------------------------------------------
# Sentinel / Indices Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_ndvi_png(
    bbox: BBoxInput, from_date: str, to_date: str,
    width: int = 512, height: int = 512, collection: str = "S2L2A",
    force_http: bool = False
) -> dict:
    """
    Generates a PNG image of the Normalized Difference Vegetation Index (NDVI).

    :param bbox: Bounding box [minLon, minLat, maxLon, maxLat].
    :param from_date: Start date in YYYY-MM-DD format.
    :param to_date: End date in YYYY-MM-DD format.
    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :param collection: Sentinel Hub collection (e.g., 'S2L2A').
    :param force_http: Force use of HTTP backend.
    :return: A dictionary containing the base64-encoded PNG image.
    """
    bbox = _normalize_bbox(bbox)

    png_bytes = await fetch_ndvi_png(
        bbox=bbox, from_date=from_date, to_date=to_date, width=width, height=height,
        collection=collection, base_url=GEOMCP_BASE, force_http=force_http
    )
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def get_ndvi_matrix(
    bbox: BBoxInput,
    from_date: str,
    to_date: str,
    width: int = 256,
    height: int = 256,
    maxcc: int = 20,
    force_http: bool = False,
    as_file: bool = False,
) -> ToolResult | dict:
    """Gets NDVI as JSON matrix or NumPy file when `as_file` is true (cloud-filtered)."""
    bbox = _normalize_bbox(bbox)

    payload = await fetch_ndvi_matrix(
        bbox=bbox,
        from_date=from_date,
        to_date=to_date,
        width=width,
        height=height,
        maxcc=maxcc,
        base_url=GEOMCP_BASE,
        force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="ndvi",
        description="NDVI float32 matrix (cloud filtered) saved as NumPy .npy file.",
    )

@mcp.tool()
async def get_ndwi_png(
    bbox: BBoxInput, from_date: str, to_date: str,
    width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Generates a PNG of the Normalized Difference Water Index (NDWI)."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_ndwi_png(bbox=bbox, from_date=from_date, to_date=to_date, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http)
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def get_ndwi_tiff(
    bbox: BBoxInput, from_date: str, to_date: str,
    width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Gets a GeoTIFF of the Normalized Difference Water Index (NDWI)."""
    bbox = _normalize_bbox(bbox)
    tiff_bytes = await fetch_ndwi_tiff(bbox=bbox, from_date=from_date, to_date=to_date, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http)
    return _format_tiff_response(tiff_bytes, "ndwi.tif")

@mcp.tool()
async def get_ndwi_matrix(
    bbox: BBoxInput, from_date: str, to_date: str,
    width: int = 256, height: int = 256, maxcc: int = 20,
    force_http: bool = False, as_file: bool = False,
) -> ToolResult | dict:
    """Gets NDWI as JSON matrix or NumPy file when `as_file` is true (cloud filtered)."""
    bbox = _normalize_bbox(bbox)

    payload = await fetch_ndwi_matrix(
        bbox=bbox, from_date=from_date, to_date=to_date,
        width=width, height=height, maxcc=maxcc,
        base_url=GEOMCP_BASE, force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="ndwi",
        description="NDWI float32 matrix saved as NumPy .npy file.",
    )

@mcp.tool()
async def get_ndbi_png(
    bbox: BBoxInput, from_date: str, to_date: str,
    width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Generates a PNG of the Normalized Difference Built-up Index (NDBI)."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_ndbi_png(bbox=bbox, from_date=from_date, to_date=to_date, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http)
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def get_ndbi_tiff(
    bbox: BBoxInput, from_date: str, to_date: str,
    width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Gets a GeoTIFF of the Normalized Difference Built-up Index (NDBI)."""
    bbox = _normalize_bbox(bbox)
    tiff_bytes = await fetch_ndbi_tiff(bbox=bbox, from_date=from_date, to_date=to_date, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http)
    return _format_tiff_response(tiff_bytes, "ndbi.tif")

@mcp.tool()
async def get_ndbi_matrix(
    bbox: BBoxInput, from_date: str, to_date: str,
    width: int = 256, height: int = 256, maxcc: int = 20,
    force_http: bool = False, as_file: bool = False,
) -> ToolResult | dict:
    """Gets NDBI as JSON matrix or NumPy file when `as_file` is true (cloud filtered)."""
    bbox = _normalize_bbox(bbox)

    payload = await fetch_ndbi_matrix(
        bbox=bbox, from_date=from_date, to_date=to_date,
        width=width, height=height, maxcc=maxcc,
        base_url=GEOMCP_BASE, force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="ndbi",
        description="NDBI float32 matrix saved as NumPy .npy file.",
    )


@mcp.tool()
async def get_ndre_matrix(
    bbox: BBoxInput,
    from_date: str,
    to_date: str,
    width: int = 256,
    height: int = 256,
    maxcc: int = 20,
    force_http: bool = False,
    as_file: bool = False,
) -> ToolResult | dict:
    """Gets NDRE as JSON matrix or NumPy file when `as_file` is true (cloud filtered)."""
    bbox = _normalize_bbox(bbox)

    payload = await fetch_ndre_matrix(
        bbox=bbox,
        from_date=from_date,
        to_date=to_date,
        width=width,
        height=height,
        maxcc=maxcc,
        base_url=GEOMCP_BASE,
        force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="ndre",
        description="NDRE float32 matrix (cloud filtered) saved as NumPy .npy file.",
    )


@mcp.tool()
async def get_evi_matrix(
    bbox: BBoxInput,
    from_date: str,
    to_date: str,
    width: int = 256,
    height: int = 256,
    maxcc: int = 20,
    force_http: bool = False,
    as_file: bool = False,
) -> ToolResult | dict:
    """Gets EVI as JSON matrix or NumPy file when `as_file` is true (cloud filtered)."""
    bbox = _normalize_bbox(bbox)

    payload = await fetch_evi_matrix(
        bbox=bbox,
        from_date=from_date,
        to_date=to_date,
        width=width,
        height=height,
        maxcc=maxcc,
        base_url=GEOMCP_BASE,
        force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="evi",
        description="EVI float32 matrix (cloud filtered) saved as NumPy .npy file.",
    )


@mcp.tool()
async def get_msavi_matrix(
    bbox: BBoxInput,
    from_date: str,
    to_date: str,
    width: int = 256,
    height: int = 256,
    maxcc: int = 20,
    force_http: bool = False,
    as_file: bool = False,
) -> ToolResult | dict:
    """Gets MSAVI as JSON matrix or NumPy file when `as_file` is true (cloud filtered)."""
    bbox = _normalize_bbox(bbox)

    payload = await fetch_msavi_matrix(
        bbox=bbox,
        from_date=from_date,
        to_date=to_date,
        width=width,
        height=height,
        maxcc=maxcc,
        base_url=GEOMCP_BASE,
        force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="msavi",
        description="MSAVI float32 matrix (cloud filtered) saved as NumPy .npy file.",
    )


@mcp.tool()
async def get_nbr_matrix(
    bbox: BBoxInput,
    from_date: str,
    to_date: str,
    width: int = 256,
    height: int = 256,
    maxcc: int = 20,
    force_http: bool = False,
    as_file: bool = False,
) -> ToolResult | dict:
    """Gets NBR as JSON matrix or NumPy file when `as_file` is true (cloud filtered)."""
    bbox = _normalize_bbox(bbox)

    payload = await fetch_nbr_matrix(
        bbox=bbox,
        from_date=from_date,
        to_date=to_date,
        width=width,
        height=height,
        maxcc=maxcc,
        base_url=GEOMCP_BASE,
        force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="nbr",
        description="NBR float32 matrix (cloud filtered) saved as NumPy .npy file.",
    )

# ---------------------------------------------------------------------------
# Cloud-free / Composites Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_cloudfree_truecolor_png(
    bbox: BBoxInput, from_date: str, to_date: str, maxcc: int = 20,
    width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Generates a cloud-free true-color composite PNG."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_cloudfree_truecolor_png(
        bbox=bbox, from_date=from_date, to_date=to_date, maxcc=maxcc,
        width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http
    )
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def get_cloudfree_ndvi_png(
    bbox: BBoxInput, from_date: str, to_date: str, maxcc: int = 20,
    width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Generates a cloud-free NDVI composite PNG."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_cloudfree_ndvi_png(
        bbox=bbox, from_date=from_date, to_date=to_date, maxcc=maxcc,
        width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http
    )
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def get_cloudfree_ndwi_png(
    bbox: BBoxInput, from_date: str, to_date: str, maxcc: int = 20,
    width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Generates a cloud-free NDWI composite PNG."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_cloudfree_ndwi_png(
        bbox=bbox, from_date=from_date, to_date=to_date, maxcc=maxcc,
        width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http
    )
    return _format_png_response(png_bytes, width, height)

# ---------------------------------------------------------------------------
# Elevation / Terrain Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_elevation_png(
    bbox: BBoxInput, width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Generates a grayscale PNG of a Digital Elevation Model (DEM)."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_elevation_png(
        bbox=bbox, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http
    )
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def get_elevation_tiff(
    bbox: BBoxInput, width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Gets a raw float32 GeoTIFF of a Digital Elevation Model (DEM)."""
    bbox = _normalize_bbox(bbox)
    tiff_bytes = await fetch_elevation_tiff(
        bbox=bbox, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http
    )
    return _format_tiff_response(tiff_bytes, "elevation.tif")

@mcp.tool()
async def get_elevation_matrix(
    bbox: BBoxInput, width: int = 256, height: int = 256,
    force_http: bool = False, as_file: bool = False,
) -> ToolResult | dict:
    """Gets elevation as JSON matrix or NumPy file when `as_file` is true."""
    bbox = _normalize_bbox(bbox)
    payload = await fetch_elevation_matrix(
        bbox=bbox, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="elevation",
        description="Elevation (meters) matrix saved as NumPy .npy file.",
    )

@mcp.tool()
async def get_slope_png(
    bbox: BBoxInput, width: int = 512, height: int = 512, vmax: float = 45.0, force_http: bool = False
) -> dict:
    """Generates a PNG visualizing terrain slope in degrees."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_slope_png(
        bbox=bbox, width=width, height=height, vmax=vmax, base_url=GEOMCP_BASE, force_http=force_http
    )
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def get_slope_tiff(
    bbox: BBoxInput, width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Gets a GeoTIFF of terrain slope in degrees."""
    bbox = _normalize_bbox(bbox)
    tiff_bytes = await fetch_slope_tiff(
        bbox=bbox, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http
    )
    return _format_tiff_response(tiff_bytes, "slope.tif")

@mcp.tool()
async def get_slope_matrix(
    bbox: BBoxInput, width: int = 256, height: int = 256,
    force_http: bool = False, as_file: bool = False,
) -> ToolResult | dict:
    """Gets slope as JSON matrix or NumPy file when `as_file` is true."""
    bbox = _normalize_bbox(bbox)
    payload = await fetch_slope_matrix(
        bbox=bbox, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="slope",
        description="Slope (degrees) matrix saved as NumPy .npy file.",
    )

@mcp.tool()
async def get_aspect_tiff(
    bbox: BBoxInput, width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Gets a GeoTIFF of terrain aspect (direction) in degrees from North."""
    bbox = _normalize_bbox(bbox)
    tiff_bytes = await fetch_aspect_tiff(
        bbox=bbox, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http
    )
    return _format_tiff_response(tiff_bytes, "aspect.tif")

@mcp.tool()
async def get_aspect_matrix(
    bbox: BBoxInput, width: int = 256, height: int = 256,
    force_http: bool = False, as_file: bool = False,
) -> ToolResult | dict:
    """Gets aspect as JSON matrix or NumPy file when `as_file` is true."""
    bbox = _normalize_bbox(bbox)
    payload = await fetch_aspect_matrix(
        bbox=bbox, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="aspect",
        description="Aspect (degrees clockwise from north) matrix saved as NumPy .npy file.",
    )

@mcp.tool()
async def get_hillshade_png(
    bbox: BBoxInput, width: int = 512, height: int = 512,
    azimuth_deg: float = 315.0, altitude_deg: float = 45.0, force_http: bool = False
) -> dict:
    """Generates a hillshade visualization of the terrain."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_hillshade_png(
        bbox=bbox, width=width, height=height, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg,
        base_url=GEOMCP_BASE, force_http=force_http,
    )
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def get_hillshade_tiff(
    bbox: BBoxInput, width: int = 512, height: int = 512,
    azimuth_deg: float = 315.0, altitude_deg: float = 45.0, force_http: bool = False
) -> dict:
    """Gets a GeoTIFF of the hillshade calculation."""
    bbox = _normalize_bbox(bbox)
    tiff_bytes = await fetch_hillshade_tiff(
        bbox=bbox, width=width, height=height, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg,
        base_url=GEOMCP_BASE, force_http=force_http,
    )
    return _format_tiff_response(tiff_bytes, "hillshade.tif")

@mcp.tool()
async def get_hillshade_matrix(
    bbox: BBoxInput, width: int = 256, height: int = 256,
    azimuth_deg: float = 315.0, altitude_deg: float = 45.0,
    force_http: bool = False, as_file: bool = False,
) -> ToolResult | dict:
    """Gets hillshade as JSON matrix or NumPy file when `as_file` is true."""
    bbox = _normalize_bbox(bbox)
    payload = await fetch_hillshade_matrix(
        bbox=bbox, width=width, height=height,
        azimuth_deg=azimuth_deg, altitude_deg=altitude_deg,
        base_url=GEOMCP_BASE, force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="hillshade",
        description="Hillshade (0-1) matrix saved as NumPy .npy file.",
    )

# ---------------------------------------------------------------------------
# Hydrology Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_flow_accumulation_png(
    bbox: BBoxInput, width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Generates a PNG visualizing hydrological flow accumulation (log-scaled)."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_flow_accumulation_png(bbox=bbox, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http)
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def get_flow_accumulation_tiff(
    bbox: BBoxInput, width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Gets a GeoTIFF of hydrological flow accumulation."""
    bbox = _normalize_bbox(bbox)
    tiff_bytes = await fetch_flow_accumulation_tiff(bbox=bbox, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http)
    return _format_tiff_response(tiff_bytes, "flow_accumulation.tif")

@mcp.tool()
async def get_flow_accumulation_matrix(
    bbox: BBoxInput, width: int = 256, height: int = 256,
    force_http: bool = False, as_file: bool = False,
) -> ToolResult | dict:
    """Gets flow accumulation as JSON matrix or NumPy file when `as_file` is true."""
    bbox = _normalize_bbox(bbox)
    payload = await fetch_flow_accumulation_matrix(
        bbox=bbox, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http,
    )
    return _matrix_to_file_response(
        payload,
        as_file=as_file,
        stem="flow_accumulation",
        description="Log-scaled flow accumulation matrix saved as NumPy .npy file.",
    )

# ---------------------------------------------------------------------------
# Classification Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def run_unsupervised_classification_png(
    bbox: BBoxInput, from_date: str, to_date: str, k: int = 5,
    width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Performs unsupervised classification (k-means) on satellite imagery and returns a PNG."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_unsupervised_classification_png(bbox=bbox, from_date=from_date, to_date=to_date, k=k, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http)
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def run_supervised_classification_png(
    bbox: BBoxInput, from_date: str, to_date: str,
    training_points: List[Dict[str, Any]],
    width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """
    Performs supervised classification (Random Forest) on satellite imagery.

    :param training_points: A list of dictionaries, e.g.,
           [{"class": "water", "lat": 40.71, "lon": -74.0}, {"class": "urban", "lat": 40.72, "lon": -74.01}]
    """
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_supervised_classification_png(bbox=bbox, from_date=from_date, to_date=to_date, training_points=training_points, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http)
    return _format_png_response(png_bytes, width, height)

@mcp.tool()
async def run_rule_based_classification_png(
    bbox: BBoxInput, from_date: str, to_date: str,
    width: int = 512, height: int = 512, force_http: bool = False
) -> dict:
    """Performs rule-based land cover classification (Water, Vegetation, Barren) and returns a PNG."""
    bbox = _normalize_bbox(bbox)
    png_bytes = await fetch_rule_based_classification_png(bbox=bbox, from_date=from_date, to_date=to_date, width=width, height=height, base_url=GEOMCP_BASE, force_http=force_http)
    return _format_png_response(png_bytes, width, height)

# ---------------------------------------------------------------------------
# Zonal / Analysis Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_zonal_timeseries_json(
    index: str, from_date: str, to_date: str, step_days: int = 10,
    geometry: Optional[dict] = None, bbox: Optional[BBoxInput] = None,
    cloud_mask: bool = True, composite: bool = False, force_http: bool = False
) -> dict:
    """
    Calculates a time series of a spectral index over a specified area (geometry or bbox).

    :param index: The spectral index to compute (e.g., 'ndvi', 'ndwi').
    :param from_date: Start date (YYYY-MM-DD).
    :param to_date: End date (YYYY-MM-DD).
    :param step_days: The interval in days for each time series data point.
    :param geometry: A GeoJSON-like dictionary defining the area of interest.
    :param bbox: A bounding box (list or tuple of [minLon, minLat, maxLon, maxLat]) as an alternative to geometry.
    :param cloud_mask: Whether to apply a cloud mask.
    :param composite: Whether to create a composite image for each step.
    :param force_http: Force use of HTTP backend.
    :return: A JSON object containing the time series data.
    """
    normalized_bbox = _normalize_optional_bbox(bbox)

    if not geometry and not normalized_bbox:
        raise ValueError("Either 'geometry' or 'bbox' must be provided.")
    return await fetch_zonal_timeseries(
        index=index, from_date=from_date, to_date=to_date, step_days=step_days,
        geometry=geometry, bbox=normalized_bbox, cloud_mask=cloud_mask, composite=composite,
        base_url=GEOMCP_BASE, force_http=force_http
    )


def _ensure_tool_func_aliases() -> None:
    """Provide backwards-compatible `.func` attribute exposed in tests."""
    tools = getattr(mcp, "_tools", {})
    for tool in tools.values():
        if hasattr(tool, "fn") and not hasattr(tool, "func"):
            object.__setattr__(tool, "func", tool.fn)  # type: ignore[attr-defined]


_ensure_tool_func_aliases()

@mcp.tool()
async def get_point_timeseries_json(
    index: str, lat: float, lon: float, from_date: str, to_date: str,
    step_days: int = 10, cloud_mask: bool = True, force_http: bool = False
) -> dict:
    """
    Calculates a time series of a spectral index for a single point (lat/lon).

    :param index: The spectral index to compute (e.g., 'ndvi', 'ndwi').
    :param lat: Latitude of the point.
    :param lon: Longitude of the point.
    :param from_date: Start date (YYYY-MM-DD).
    :param to_date: End date (YYYY-MM-DD).
    :param step_days: The interval in days for each time series data point.
    :param cloud_mask: Whether to apply a cloud mask.
    :param force_http: Force use of HTTP backend.
    :return: A JSON object containing the time series data for the point.
    """
    return await fetch_point_timeseries(
        index=index, lat=lat, lon=lon, from_date=from_date, to_date=to_date,
        step_days=step_days, cloud_mask=cloud_mask, base_url=GEOMCP_BASE, force_http=force_http
    )