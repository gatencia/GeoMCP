# modules/cloudfree.py
# Cloud-free composites for TrueColor, NDVI, NDWI using Sentinel Hub PROCESS API
# - Uses the same auth as other modules via modules.sentinel_hub
# - Chooses least-cloudy scenes within the given date window
# - Returns opaque PNG bytes (no alpha)

from __future__ import annotations
import requests
from typing import List

from modules.sentinel_hub import _get_token, PROCESS_URL


def _ensure_iso8601(s: str, end: bool = False) -> str:
    """Normalize YYYY-MM-DD into full ISO-8601 (UTC) if needed."""
    if "T" in s:
        return s
    return f"{s}T23:59:59Z" if end else f"{s}T00:00:00Z"


def _process_png_cloudfree(
    bbox: List[float],
    from_iso: str,
    to_iso: str,
    width: int,
    height: int,
    maxcc: int,
    evalscript: str,
) -> bytes:
    """
    Generic cloud-free PNG processor: uses mosaickingOrder='leastCC' and maxCloudCoverage.
    """
    headers = {"Authorization": f"Bearer {_get_token()}"}

    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "S2L2A",
                "dataFilter": {
                    "timeRange": {
                        "from": _ensure_iso8601(from_iso, end=False),
                        "to":   _ensure_iso8601(to_iso,   end=True),
                    },
                    "maxCloudCoverage": int(maxcc),   # percent 0..100
                    "mosaickingOrder": "leastCC",     # choose least-cloudy scenes
                },
            }],
        },
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/png"}
            }],
        },
        "evalscript": evalscript,
    }

    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Cloud-free PROCESS error {r.status_code}: {r.text[:400]}")
    return r.content


# ----------------------------- Evalscripts -----------------------------------

# True color (B04,B03,B02), simple gain to brighten, no alpha channel.
TRUECOLOR_EVAL = """
//VERSION=3
function setup() {
  return {
    input: ["B02","B03","B04"],
    output: { bands: 3 }  // PNG RGB, opaque
  };
}
function evaluatePixel(s) {
  // simple stretch
  let r = Math.min(1, s.B04 * 2.5);
  let g = Math.min(1, s.B03 * 2.5);
  let b = Math.min(1, s.B02 * 2.5);
  return [r,g,b];
}
"""

# NDVI mapped to 0..1 for PNG grayscale (opaque)
NDVI_EVAL_PNG = """
//VERSION=3
function setup() {
  return {
    input: ["B04","B08"],
    output: { bands: 1 } // PNG L (no alpha)
  };
}
function evaluatePixel(s) {
  let n = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
  // Map [-1,1] -> [0,1]
  n = (n + 1.0) * 0.5;
  n = Math.max(0, Math.min(1, n));
  return [n];
}
"""

# NDWI mapped to 0..1 for PNG grayscale (opaque)
NDWI_EVAL_PNG = """
//VERSION=3
function setup() {
  return {
    input: ["B03","B08"],
    output: { bands: 1 } // PNG L (no alpha)
  };
}
function evaluatePixel(s) {
  let w = (s.B03 - s.B08) / (s.B03 + s.B08 + 1e-6);
  // Map [-1,1] -> [0,1]
  w = (w + 1.0) * 0.5;
  w = Math.max(0, Math.min(1, w));
  return [w];
}
"""


# ----------------------------- Public API ------------------------------------

def get_cloudfree_truecolor_png(
    bbox: List[float],
    from_iso: str,
    to_iso: str,
    maxcc: int = 30,
    width: int = 512,
    height: int = 512,
) -> bytes:
    return _process_png_cloudfree(
        bbox=bbox,
        from_iso=from_iso,
        to_iso=to_iso,
        width=width,
        height=height,
        maxcc=maxcc,
        evalscript=TRUECOLOR_EVAL,
    )


def get_cloudfree_ndvi_png(
    bbox: List[float],
    from_iso: str,
    to_iso: str,
    maxcc: int = 30,
    width: int = 512,
    height: int = 512,
) -> bytes:
    return _process_png_cloudfree(
        bbox=bbox,
        from_iso=from_iso,
        to_iso=to_iso,
        width=width,
        height=height,
        maxcc=maxcc,
        evalscript=NDVI_EVAL_PNG,
    )


def get_cloudfree_ndwi_png(
    bbox: List[float],
    from_iso: str,
    to_iso: str,
    maxcc: int = 30,
    width: int = 512,
    height: int = 512,
) -> bytes:
    return _process_png_cloudfree(
        bbox=bbox,
        from_iso=from_iso,
        to_iso=to_iso,
        width=width,
        height=height,
        maxcc=maxcc,
        evalscript=NDWI_EVAL_PNG,
    )