# modules/ndwi.py
import io
import numpy as np
import rasterio
import requests
import tifffile
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

def get_ndwi(bbox, width=512, height=512):
    """
    Return NDWI as PNG (grayscale, 0..1 after normalization).
    Uses the same SH auth flow as elevation.
    """
    headers = {"Authorization": f"Bearer {_get_token()}"}
    body = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "S2L2A"}]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": NDWI_PNG_EVALSCRIPT
    }
    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:300])
    return r.content


def get_ndwi_raw(bbox, width=512, height=512):
    """
    Return NDWI as raw float32 GeoTIFF (range typically ~[-1, 1]).
    """
    headers = {"Authorization": f"Bearer {_get_token()}"}
    body = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "S2L2A"}]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": NDWI_TIFF_EVALSCRIPT
    }
    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:300])
    return r.content


def get_ndwi_matrix(bbox, width=256, height=256):
    """
    Return NDWI as JSON float matrix (LLM / numeric analysis).
    """
    tiff_bytes = get_ndwi_raw(bbox, width, height)
    with rasterio.open(io.BytesIO(tiff_bytes)) as src:
        arr = src.read(1).astype(float)
    arr = np.nan_to_num(arr).tolist()
    return {
        "bbox": bbox,
        "width": width,
        "height": height,
        "range": [-1.0, 1.0],
        "index": "NDWI",
        "values": arr
    }