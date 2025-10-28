# modules/sentinel_hub.py
import io
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv("config/api_keys.env")

BASE_URL = os.getenv("SENTINELHUB_BASE_URL", "https://services.sentinel-hub.com")
OAUTH_URL = f"{BASE_URL}/oauth/token"
PROCESS_URL = f"{BASE_URL}/api/v1/process"

CLIENT_ID = os.getenv("SENTINELHUB_CLIENT_ID")
CLIENT_SECRET = os.getenv("SENTINELHUB_CLIENT_SECRET")
TOKEN = os.getenv("SENTINELHUB_TOKEN")

_token_cache = {"access_token": None, "expires_at": 0.0}


def _get_token() -> str:
    """Fetch or reuse OAuth token or fallback to SENTINELHUB_TOKEN."""
    now = time.time()
    if TOKEN:
        return TOKEN  # direct static token fallback
    if _token_cache["access_token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["access_token"]

    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("Missing SENTINELHUB_CLIENT_ID/SECRET or TOKEN")

    resp = requests.post(
        OAUTH_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    resp.raise_for_status()
    j = resp.json()
    _token_cache["access_token"] = j["access_token"]
    _token_cache["expires_at"] = now + int(j.get("expires_in", 3600))
    return _token_cache["access_token"]


def process_png(
    bbox,
    from_iso,
    to_iso,
    width,
    height,
    evalscript,
    collection="S2L2A",
    mosaicking="mostRecent",
):
    """Call Sentinel Hub Process API and return image bytes (PNG)."""
    headers = {"Authorization": f"Bearer {_get_token()}"}
    body = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [
                {
                    "type": collection,
                    "dataFilter": {
                        "timeRange": {"from": f"{from_iso}T00:00:00Z", "to": f"{to_iso}T23:59:59Z"},
                        "mosaickingOrder": mosaicking,
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": evalscript,
    }

    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Process API error {r.status_code}: {r.text[:200]}")
    return r.content


NDVI_PNG_EVALSCRIPT = """
//VERSION=3
function setup() {
  return { input: ["B04","B08"], output: { bands: 1, sampleType: "UINT8" } };
}
function evaluatePixel(s) {
  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
  let v = Math.round(((ndvi + 1.0) / 2.0) * 255.0);
  v = isFinite(v) ? Math.max(0, Math.min(255, v)) : 0;
  return [v];
}
"""