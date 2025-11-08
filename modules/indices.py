"""Sentinel-derived vegetation indices with cloud-aware matrix outputs."""

from __future__ import annotations

import io
from typing import Dict, List, Optional

import numpy as np
import requests
import tifffile
from fastapi import HTTPException

from modules.sentinel_hub import _get_token, PROCESS_URL


NDRE_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B05", "B8A"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  let denom = s.B8A + s.B05;
  if (Math.abs(denom) < 1e-6) {
    return [0.0];
  }
  return [(s.B8A - s.B05) / denom];
}
"""

EVI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B04", "B08"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  let numerator = 2.5 * (s.B08 - s.B04);
  let denom = (s.B08 + 6.0 * s.B04 - 7.5 * s.B02 + 1.0);
  if (Math.abs(denom) < 1e-6) {
    return [0.0];
  }
  return [numerator / denom];
}
"""

MSAVI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  let term = 2.0 * s.B08 + 1.0;
  let inner = term * term - 8.0 * (s.B08 - s.B04);
  inner = inner < 0.0 ? 0.0 : inner;
  return [0.5 * (term - Math.sqrt(inner))];
}
"""

NBR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B08", "B12"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  let denom = s.B08 + s.B12;
  if (Math.abs(denom) < 1e-6) {
    return [0.0];
  }
  return [(s.B08 - s.B12) / denom];
}
"""


def _normalize_iso(date_str: Optional[str], *, is_end: bool = False) -> Optional[str]:
    if not date_str:
        return None
    if "T" in date_str:
        return date_str
    return f"{date_str}T23:59:59Z" if is_end else f"{date_str}T00:00:00Z"


def _build_body(
    *,
    bbox: List[float],
    width: int,
    height: int,
    evalscript: str,
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
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{"type": "S2L2A", "dataFilter": data_filter}],
        },
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }


def _request_matrix(body: Dict[str, object]) -> np.ndarray:
    headers = {"Authorization": f"Bearer {_get_token()}"}
    resp = requests.post(PROCESS_URL, headers=headers, json=body, timeout=180)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:300])
    arr = tifffile.imread(io.BytesIO(resp.content)).astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _wrap_response(
    *,
    index: str,
    bbox: List[float],
    matrix: np.ndarray,
    from_iso: Optional[str],
    to_iso: Optional[str],
    maxcc: int,
) -> Dict[str, object]:
    return {
        "bbox": bbox,
        "width": int(matrix.shape[1]),
        "height": int(matrix.shape[0]),
        "matrix": matrix.tolist(),
        "index": index,
        "range": [-1.0, 1.0],
        "max_cloud_coverage": int(maxcc),
        "time_range": {"from": from_iso, "to": to_iso},
    }


def _compute_index(
    *,
    bbox: List[float],
    width: int,
    height: int,
    from_iso: Optional[str],
    to_iso: Optional[str],
    maxcc: int,
    evalscript: str,
    index_name: str,
) -> Dict[str, object]:
    body = _build_body(
        bbox=bbox,
        width=width,
        height=height,
        evalscript=evalscript,
        from_iso=from_iso,
        to_iso=to_iso,
        maxcc=maxcc,
    )
    matrix = _request_matrix(body)
    return _wrap_response(index=index_name, bbox=bbox, matrix=matrix, from_iso=from_iso, to_iso=to_iso, maxcc=maxcc)


def get_ndre_matrix(
    *,
    bbox: List[float],
    width: int = 256,
    height: int = 256,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    maxcc: int = 20,
) -> Dict[str, object]:
    return _compute_index(
        bbox=bbox,
        width=width,
        height=height,
        from_iso=from_iso,
        to_iso=to_iso,
        maxcc=maxcc,
        evalscript=NDRE_EVALSCRIPT,
        index_name="NDRE",
    )


def get_evi_matrix(
    *,
    bbox: List[float],
    width: int = 256,
    height: int = 256,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    maxcc: int = 20,
) -> Dict[str, object]:
    return _compute_index(
        bbox=bbox,
        width=width,
        height=height,
        from_iso=from_iso,
        to_iso=to_iso,
        maxcc=maxcc,
        evalscript=EVI_EVALSCRIPT,
        index_name="EVI",
    )


def get_msavi_matrix(
    *,
    bbox: List[float],
    width: int = 256,
    height: int = 256,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    maxcc: int = 20,
) -> Dict[str, object]:
    return _compute_index(
        bbox=bbox,
        width=width,
        height=height,
        from_iso=from_iso,
        to_iso=to_iso,
        maxcc=maxcc,
        evalscript=MSAVI_EVALSCRIPT,
        index_name="MSAVI",
    )


def get_nbr_matrix(
    *,
    bbox: List[float],
    width: int = 256,
    height: int = 256,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    maxcc: int = 20,
) -> Dict[str, object]:
    return _compute_index(
        bbox=bbox,
        width=width,
        height=height,
        from_iso=from_iso,
        to_iso=to_iso,
        maxcc=maxcc,
        evalscript=NBR_EVALSCRIPT,
        index_name="NBR",
    )
