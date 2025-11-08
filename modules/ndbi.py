# modules/ndbi.py
import io
import requests
import tifffile
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import HTTPException
from modules.sentinel_hub import _get_token, PROCESS_URL

# NDBI = (SWIR - NIR) / (SWIR + NIR)
# Sentinel-2 bands: SWIR = B11, NIR = B08

# --- Evalscript helpers ---
NDBI_EVALSCRIPT_TIFF = """
//VERSION=3
function setup() {
  return {
    input: ["B11", "B08"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  let swir = sample.B11;
  let nir  = sample.B08;
  let denom = swir + nir;
  if (Math.abs(denom) < 1e-6) return [0.0];
  let ndbi = (swir - nir) / denom;  // raw NDBI, ~[-1, 1]
  return [ndbi];
}
"""

NDBI_EVALSCRIPT_PNG = """
//VERSION=3
function setup() {
  return {
    input: ["B11", "B08"],
    output: { bands: 3 }
  };
}
function evaluatePixel(sample) {
  let swir = sample.B11;
  let nir  = sample.B08;
  let denom = swir + nir;
  let ndbi = 0.0;
  if (Math.abs(denom) >= 1e-6) {
    ndbi = (swir - nir) / denom;  // [-1, 1]
  }
  // visualize to [0,1]
  let v = (ndbi + 1.0) * 0.5;
  v = Math.min(1.0, Math.max(0.0, v));
  return [v, v, v];
}
"""

def _normalize_iso(date_str: Optional[str], *, is_end: bool = False) -> Optional[str]:
    """
    Accepts:
      - None  -> None
      - 'YYYY-MM-DD' -> 'YYYY-MM-DDT00:00:00Z' (or T23:59:59Z if is_end)
      - full ISO strings are left as-is
    """
    if not date_str:
        return None
    if "T" not in date_str:
        return f"{date_str}T23:59:59Z" if is_end else f"{date_str}T00:00:00Z"
    return date_str

def _build_input_data(
    from_iso: Optional[str],
    to_iso: Optional[str],
    maxcc: Optional[int],
) -> List[Dict[str, Any]]:
    """Construct Sentinel Hub data filter with cloud-masking preferences."""
    data_filter: Dict[str, Any] = {"mosaickingOrder": "leastCC"}

    if maxcc is not None:
        data_filter["maxCloudCoverage"] = int(maxcc)

    f = _normalize_iso(from_iso, is_end=False)
    t = _normalize_iso(to_iso, is_end=True)
    if f and t:
        data_filter["timeRange"] = {"from": f, "to": t}

    return [{"type": "S2L2A", "dataFilter": data_filter}]

def _process(
    bbox: List[float],
    width: int,
    height: int,
    evalscript: str,
    from_iso: Optional[str],
    to_iso: Optional[str],
    out_type: str,  # "image/tiff" or "image/png"
    maxcc: Optional[int],
) -> bytes:
    headers = {"Authorization": f"Bearer {_get_token()}"}
    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": _build_input_data(from_iso, to_iso, maxcc)
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": out_type}}]
        },
        "evalscript": evalscript
    }
    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=180)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:500])
    return r.content

# ---------- Public API ----------

def get_ndbi_png(
    bbox: List[float],
    width: int = 512,
    height: int = 512,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    maxcc: Optional[int] = 20,
) -> bytes:
    """PNG visualization of NDBI (scaled to [0,1])."""
    return _process(
        bbox=bbox,
        width=width,
        height=height,
        evalscript=NDBI_EVALSCRIPT_PNG,
        from_iso=from_iso,
        to_iso=to_iso,
        out_type="image/png",
        maxcc=maxcc,
    )

def get_ndbi_tiff(
    bbox: List[float],
    width: int = 512,
    height: int = 512,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    maxcc: Optional[int] = 20,
) -> bytes:
    """Float32 GeoTIFF of raw NDBI values."""
    return _process(
        bbox=bbox,
        width=width,
        height=height,
        evalscript=NDBI_EVALSCRIPT_TIFF,
        from_iso=from_iso,
        to_iso=to_iso,
        out_type="image/tiff",
        maxcc=maxcc,
    )

def get_ndbi_matrix(
    bbox: List[float],
    width: int = 256,
    height: int = 256,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    maxcc: Optional[int] = 20,
) -> Dict[str, Any]:
    """
    JSON float matrix of NDBI values:
      {
        "bbox": [...],
        "width": W,
        "height": H,
        "matrix": [[...], [...], ...]   // H x W float list
      }
    """
    tiff_bytes = get_ndbi_tiff(
        bbox,
        width=width,
        height=height,
        from_iso=from_iso,
        to_iso=to_iso,
        maxcc=maxcc,
    )
    arr = tifffile.imread(io.BytesIO(tiff_bytes)).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "bbox": bbox,
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "matrix": arr.tolist(),
        "max_cloud_coverage": int(maxcc) if maxcc is not None else None,
        "time_range": {"from": from_iso, "to": to_iso},
    }