# modules/ndwi.py
from __future__ import annotations

import io
from typing import Dict, List, Optional

import numpy as np
import rasterio
import requests
from fastapi import HTTPException

from modules.sentinel_hub import _get_token, PROCESS_URL


# ───────────────────────────────────────────────────────────────────────────────
# EvalScripts for Sentinel-2 L2A (Green = B03, NIR = B08)
# ───────────────────────────────────────────────────────────────────────────────

NDWI_PNG_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B03", "B08"],
    output: { bands: 3 }
  };
}
function evaluatePixel(sample) {
  let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
  ndwi = (ndwi + 1.0) / 2.0; // normalize to [0,1] for PNG
  return [ndwi, ndwi, ndwi];
}
"""

NDWI_TIFF_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B03", "B08"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [(sample.B03 - sample.B08) / (sample.B03 + sample.B08)];
}
"""


# ───────────────────────────────────────────────────────────────────────────────
# NDWI EXPORTS (PNG, GeoTIFF float32, JSON matrix)
# ───────────────────────────────────────────────────────────────────────────────

def _normalize_iso(date_str: Optional[str], *, is_end: bool = False) -> Optional[str]:
    if not date_str:
        return None
    if "T" in date_str:
        return date_str
    return f"{date_str}T23:59:59Z" if is_end else f"{date_str}T00:00:00Z"


def _build_process_body(
    *,
    bbox: List[float],
    width: int,
    height: int,
    evalscript: str,
    out_type: str,
    from_iso: Optional[str],
    to_iso: Optional[str],
    maxcc: int,
) -> Dict[str, object]:
    data_filter: Dict[str, object] = {
        "mosaickingOrder": "leastCC",
        "maxCloudCoverage": int(maxcc),
    }

    start = _normalize_iso(from_iso, is_end=False)
    end = _normalize_iso(to_iso, is_end=True)
    if start and end:
        data_filter["timeRange"] = {"from": start, "to": end}

    return {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "S2L2A", "dataFilter": data_filter}],
        },
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [{"identifier": "default", "format": {"type": out_type}}],
        },
        "evalscript": evalscript,
    }


def _post_process(body: Dict[str, object]) -> bytes:
    headers = {"Authorization": f"Bearer {_get_token()}"}
    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=180)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:300])
    return r.content


def get_ndwi(
    bbox: List[float],
    *,
    width: int = 512,
    height: int = 512,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    maxcc: int = 20,
):
    """
    Return NDWI as PNG (grayscale, 0..1 after normalization).
    Uses the same SH auth flow as elevation.
    """
    body = _build_process_body(
        bbox=bbox,
        width=width,
        height=height,
        evalscript=NDWI_PNG_EVALSCRIPT,
        out_type="image/png",
        from_iso=from_iso,
        to_iso=to_iso,
        maxcc=maxcc,
    )
    return _post_process(body)


def get_ndwi_raw(
    bbox: List[float],
    *,
    width: int = 512,
    height: int = 512,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    maxcc: int = 20,
):
    """
    Return NDWI as raw float32 GeoTIFF (range typically ~[-1, 1]).
    """
    body = _build_process_body(
        bbox=bbox,
        width=width,
        height=height,
        evalscript=NDWI_TIFF_EVALSCRIPT,
        out_type="image/tiff",
        from_iso=from_iso,
        to_iso=to_iso,
        maxcc=maxcc,
    )
    return _post_process(body)


def get_ndwi_matrix(
    bbox: List[float],
    *,
    width: int = 256,
    height: int = 256,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    maxcc: int = 20,
) -> Dict[str, object]:
    """
    Return NDWI as JSON float matrix (LLM / numeric analysis).
    """
    tiff_bytes = get_ndwi_raw(
        bbox,
        width=width,
        height=height,
        from_iso=from_iso,
        to_iso=to_iso,
        maxcc=maxcc,
    )
    with rasterio.open(io.BytesIO(tiff_bytes)) as src:
        arr = src.read(1).astype(float)
    arr = np.nan_to_num(arr).tolist()
    return {
        "bbox": bbox,
        "width": width,
        "height": height,
        "range": [-1.0, 1.0],
        "index": "NDWI",
        "values": arr,
        "max_cloud_coverage": int(maxcc),
        "time_range": {"from": from_iso, "to": to_iso},
    }