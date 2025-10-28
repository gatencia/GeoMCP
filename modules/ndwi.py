# modules/ndwi.py
import requests
import io
from fastapi import HTTPException
from modules.sentinel_hub import _get_token, PROCESS_URL

# NDWI formula: (Green - NIR) / (Green + NIR)
NDWI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B03", "B08"],
    output: { bands: 3 }
  };
}
function evaluatePixel(sample) {
  let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
  ndwi = (ndwi + 1.0) / 2.0;  // normalize to [0,1] for PNG visualization
  return [ndwi, ndwi, ndwi];
}
"""

def get_ndwi(bbox, width=512, height=512):
    """
    Returns NDWI PNG for a given bounding box using Sentinel-2 Level-2A data.
    Uses the same auth system as the elevation endpoints.
    """
    headers = {"Authorization": f"Bearer {_get_token()}"}
    body = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "S2L2A"}],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": NDWI_EVALSCRIPT,
    }

    r = requests.post(PROCESS_URL, headers=headers, json=body)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:300])
    return r.content