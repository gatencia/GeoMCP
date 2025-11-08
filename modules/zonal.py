# modules/zonal.py
# -----------------------------------------------------------------------------
# Zonal statistics & time-series utilities over Sentinel-2 indices
# - Uses the same auth/token + PROCESS API as other modules.
# - Supports either bbox OR GeoJSON geometry (the latter is preferred for AOIs).
# - Cloud masking via SCL by default (removes clouds & shadows).
# - Optional temporal composite per bin ("first valid" across scenes).
# - Returns clean JSON stats; server can expose PNG/TIFF elsewhere as needed.
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import tifffile

from modules.sentinel_hub import _get_token, PROCESS_URL


# ----------------------------- Helpers ----------------------------------------

def _ensure_iso8601(s: str, end: bool = False) -> str:
    """Normalize a 'YYYY-MM-DD' string into full ISO-8601 (Z)."""
    if "T" in s:
        return s
    return f"{s}T23:59:59Z" if end else f"{s}T00:00:00Z"


def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """Return (m/deg_lon, m/deg_lat) at a given latitude."""
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(lat)
    return m_per_deg_lon, m_per_deg_lat


def _point_bbox(lat: float, lon: float, buffer_m: float = 20.0) -> List[float]:
    """Tiny bbox around a point, sized by buffer in meters."""
    m_per_deg_lon, m_per_deg_lat = _meters_per_degree(lat)
    dlon = max(buffer_m / max(m_per_deg_lon, 1e-9), 1e-7)
    dlat = max(buffer_m / max(m_per_deg_lat, 1e-9), 1e-7)
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


# ----------------------------- Index builder ----------------------------------

SUPPORTED = {
    "NDVI": "ndvi",
    "NDWI": "ndwi",
    "NDBI": "ndbi",
    "NDMI": "ndmi",
    "EVI":  "evi",
    # RAW:* will be handled dynamically (e.g., RAW:B08)
}

def _evalscript_for_index(index_name: str, cloud_mask: bool = True, composite: bool = False) -> str:
    """
    Build an evalscript that outputs 2 bands:
      [0] value (FLOAT32) – the index or raw band
      [1] mask  (FLOAT32) – 1 for valid+unclouded, 0 otherwise

    If composite=True, we use per-pixel temporal compositing with mosaicking "ORBIT"
    and iterate over samples[] to pick the first unclouded pixel in the bin.
    """
    # Input bands we may need
    inputs = ['"B02"', '"B03"', '"B04"', '"B08"', '"B8A"', '"B11"', '"B12"', '"SCL"', '"dataMask"']

    # Index expression
    name = index_name.upper().strip()
    if name == "NDVI":
        idx_expr = "(s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6)"
    elif name == "NDWI":
        idx_expr = "(s.B03 - s.B08) / (s.B03 + s.B08 + 1e-6)"
    elif name == "NDBI":
        idx_expr = "(s.B11 - s.B08) / (s.B11 + s.B08 + 1e-6)"
    elif name == "NDMI":
        idx_expr = "(s.B8A - s.B11) / (s.B8A + s.B11 + 1e-6)"
    elif name == "EVI":
        idx_expr = "2.5 * (s.B08 - s.B04) / (s.B08 + 6.0*s.B04 - 7.5*s.B02 + 1.0 + 1e-6)"
    elif name.startswith("RAW:"):
        band = name.split(":", 1)[1].strip().upper()  # e.g., RAW:B08
        if f"\"{band}\"" not in inputs:
            inputs.insert(0, f"\"{band}\"")
        idx_expr = f"s.{band}"
    else:
        raise ValueError(f"Unsupported index '{index_name}'")

    # Cloud mask logic (SCL: 3 shadow, 8/9/10 clouds)
    cloud_logic = "true"
    if cloud_mask:
        cloud_logic = "!(s.SCL == 8 || s.SCL == 9 || s.SCL == 10)"

    if composite:
        # Per-pixel temporal composite: iterate samples[] (ORBIT mosaicking)
        evalscript = """
//VERSION=3
function setup() {{
  return {{
    input: [{inputs}],
    output: {{ bands: 2, sampleType: "FLOAT32" }},
    mosaicking: "ORBIT"
  }};
}}
function evaluatePixel(samples, scenes) {{
  for (var i = 0; i < samples.length; i++) {{
    var s = samples[i];
    var ok = (s.dataMask > 0) && ({cloud_logic});
    if (ok) {{
      var val = {idx_expr};
      return [val, 1.0];
    }}
  }}
  return [0.0, 0.0];
}}
""".format(inputs=", ".join(inputs), cloud_logic=cloud_logic, idx_expr=idx_expr)
    else:
        # Single-scene (SIMPLE) with pixel-level cloud mask
        evalscript = """
//VERSION=3
function setup() {{
  return {{
    input: [{inputs}],
    output: {{ bands: 2, sampleType: "FLOAT32" }},
    mosaicking: "SIMPLE"
  }};
}}
function evaluatePixel(s) {{
  var ok  = (s.dataMask > 0) && ({cloud_logic});
  var val = {idx_expr};
  return [val, ok ? 1.0 : 0.0];
}}
""".format(inputs=", ".join(inputs), cloud_logic=cloud_logic, idx_expr=idx_expr)
    print(f"[zonal] index={index_name} cloud_mask={cloud_mask} composite={composite}")
    return evalscript


# ----------------------------- Core fetch -------------------------------------

def _fetch_index_and_mask(
    *,
    index: str,
    from_iso: str,
    to_iso: str,
    width: int,
    height: int,
    bbox: Optional[List[float]] = None,
    geometry: Optional[Dict] = None,
    cloud_mask: bool = True,
    composite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call PROCESS API and return (val, mask) arrays as float32 with shape (H, W).
    If composite=True, the evalscript performs a temporal 'first valid' composite
    inside the requested time bin.
    """
    headers = {"Authorization": f"Bearer {_get_token()}"}

    from_iso = _ensure_iso8601(from_iso, end=False)
    to_iso   = _ensure_iso8601(to_iso, end=True)

    bounds_obj: Dict
    if geometry is not None:
        bounds_obj = {"geometry": geometry, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}
    elif bbox is not None:
        bounds_obj = {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}
    else:
        raise ValueError("Either 'bbox' or 'geometry' must be provided.")

    evalscript = _evalscript_for_index(index, cloud_mask=cloud_mask, composite=composite)

    body = {
        "input": {
            "bounds": bounds_obj,
            "data": [{
                "type": "S2L2A",
                "dataFilter": {"timeRange": {"from": from_iso, "to": to_iso}}
            }]
        },
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": evalscript
    }

    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"PROCESS error {r.status_code}: {r.text[:400]}")

    arr = tifffile.imread(io.BytesIO(r.content)).astype(np.float32)

    # Expected (H,W,2). Handle edge cases gracefully.
    if arr.ndim == 2:
        h, w = arr.shape
        return arr, np.ones((h, w), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 2:
        return arr[..., 0], arr[..., 1]

    h = arr.shape[0]
    w = arr.shape[1]
    val = arr[..., 0] if arr.ndim == 3 else arr
    msk = np.ones((h, w), dtype=np.float32)
    return val.astype(np.float32), msk


# ----------------------------- Zonal stats ------------------------------------

def compute_stats(values: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute robust stats on valid pixels (mask>0.5 & finite)."""
    valid = (mask > 0.5) & np.isfinite(values)
    if not np.any(valid):
        return {
            "count": 0, "valid_pct": 0.0,
            "mean": float("nan"), "median": float("nan"),
            "p25": float("nan"), "p75": float("nan"),
            "std": float("nan"), "min": float("nan"), "max": float("nan"),
        }
    v = values[valid]
    return {
        "count": int(v.size),
        "valid_pct": float(v.size) / float(values.size),
        "mean": float(np.nanmean(v)),
        "median": float(np.nanmedian(v)),
        "p25": float(np.nanpercentile(v, 25)),
        "p75": float(np.nanpercentile(v, 75)),
        "std": float(np.nanstd(v)),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
    }


def zonal_stats_index(
    *,
    index: str,
    from_iso: str,
    to_iso: str,
    width: int,
    height: int,
    bbox: Optional[List[float]] = None,
    geometry: Optional[Dict] = None,
    cloud_mask: bool = True,
    composite: bool = False,
) -> Dict:
    """
    Fetch index+mask and return zonal stats over the AOI (bbox/geometry).
    If composite=True, each bin uses a temporal 'first valid' across scenes.
    """
    val, msk = _fetch_index_and_mask(
        index=index, from_iso=from_iso, to_iso=to_iso,
        width=width, height=height, bbox=bbox, geometry=geometry,
        cloud_mask=cloud_mask, composite=composite,
    )
    stats = compute_stats(val, msk)
    return {
        "index": index,
        "from": _ensure_iso8601(from_iso, end=False),
        "to":   _ensure_iso8601(to_iso, end=True),
        "width": int(width),
        "height": int(height),
        "stats": stats
    }


# ----------------------------- Time series (point) ----------------------------

def point_series_index(
    *,
    index: str,
    lat: float,
    lon: float,
    from_iso: str,
    to_iso: str,
    step_days: int = 10,
    buffer_m: float = 20.0,
    cloud_mask: bool = True,
    composite: bool = False,
) -> Dict:
    """
    Build a time series at a point by splitting the date window into bins
    and sampling the nearest pixel in a tiny bbox around (lat,lon).
    If composite=True, each bin uses a temporal 'first valid' composite.
    """
    # Compute sub-intervals
    import datetime as dt
    t0 = dt.datetime.fromisoformat(_ensure_iso8601(from_iso, end=False).replace("Z", "+00:00"))
    t1 = dt.datetime.fromisoformat(_ensure_iso8601(to_iso, end=True).replace("Z", "+00:00"))
    step = dt.timedelta(days=max(1, int(step_days)))

    out: List[Dict] = []
    cur = t0
    while cur <= t1:
        end = min(cur + step, t1)
        bbox = _point_bbox(lat, lon, buffer_m=buffer_m)
        # Keep resolution consistent but small (we only need 3x3 to be safe)
        width = height = 3

        try:
            val, msk = _fetch_index_and_mask(
                index=index,
                from_iso=cur.isoformat().replace("+00:00", "Z"),
                to_iso=end.isoformat().replace("+00:00", "Z"),
                width=width,
                height=height,
                bbox=bbox,
                geometry=None,
                cloud_mask=cloud_mask,
                composite=composite,
            )
            # Take center pixel
            r = height // 2
            c = width // 2
            v = float(val[r, c]) if msk[r, c] > 0.5 and np.isfinite(val[r, c]) else None
            out.append({
                "t0": cur.isoformat().replace("+00:00", "Z"),
                "t1": end.isoformat().replace("+00:00", "Z"),
                "value": v
            })
        except Exception as e:
            out.append({
                "t0": cur.isoformat().replace("+00:00", "Z"),
                "t1": end.isoformat().replace("+00:00", "Z"),
                "value": None,
                "error": str(e)[:200]
            })
        cur = end + dt.timedelta(seconds=1)

    return {
        "index": index,
        "lat": lat, "lon": lon,
        "from": _ensure_iso8601(from_iso, end=False),
        "to":   _ensure_iso8601(to_iso, end=True),
        "step_days": int(step_days),
        "series": out
    }


# ----------------------------- Time series (area) -----------------------------

def area_series_index(
    *,
    index: str,
    from_iso: str,
    to_iso: str,
    step_days: int = 10,
    width: int = 256,
    height: int = 256,
    bbox: Optional[List[float]] = None,
    geometry: Optional[Dict] = None,
    cloud_mask: bool = True,
    composite: bool = False,
) -> Dict:
    """
    For each time bin, fetch index+mask over AOI and compute mean/median/p25/p75.
    If composite=True, each bin uses a temporal 'first valid' across scenes.
    """
    import datetime as dt
    t0 = dt.datetime.fromisoformat(_ensure_iso8601(from_iso, end=False).replace("Z", "+00:00"))
    t1 = dt.datetime.fromisoformat(_ensure_iso8601(to_iso, end=True).replace("Z", "+00:00"))
    step = dt.timedelta(days=max(1, int(step_days)))

    out: List[Dict] = []
    cur = t0
    while cur <= t1:
        end = min(cur + step, t1)
        try:
            val, msk = _fetch_index_and_mask(
                index=index,
                from_iso=cur.isoformat().replace("+00:00", "Z"),
                to_iso=end.isoformat().replace("+00:00", "Z"),
                width=width, height=height,
                bbox=bbox, geometry=geometry,
                cloud_mask=cloud_mask,
                composite=composite,
            )
            stats = compute_stats(val, msk)
            out.append({
                "t0": cur.isoformat().replace("+00:00", "Z"),
                "t1": end.isoformat().replace("+00:00", "Z"),
                "stats": stats
            })
        except Exception as e:
            out.append({
                "t0": cur.isoformat().replace("+00:00", "Z"),
                "t1": end.isoformat().replace("+00:00", "Z"),
                "error": str(e)[:200]
            })
        cur = end + dt.timedelta(seconds=1)

    return {
        "index": index,
        "from": _ensure_iso8601(from_iso, end=False),
        "to":   _ensure_iso8601(to_iso, end=True),
        "step_days": int(step_days),
        "width": int(width), "height": int(height),
        "series": out
    }

# ---------------------------------------------------------------------------
# Backward-compat names required by older MCP/tests
# ---------------------------------------------------------------------------

def zonal_timeseries_index(
    index: str,
    geometry: dict,
    from_iso: str,
    to_iso: str,
    step_days: int = 5,
    cloud_mask: bool = False,
    composite: bool = False,
    **kwargs, # absorb bbox etc
):
    """
    Older MCP layer calls this name.
    Delegate to the new, canonical function.
    """
    return area_series_index(
        index=index,
        geometry=geometry,
        from_iso=from_iso,
        to_iso=to_iso,
        step_days=step_days,
        cloud_mask=cloud_mask,
        composite=composite,
        **kwargs,
    )


def zonal_timeseries(
    *,
    index: str,
    geometry: dict,
    from_iso: str,
    to_iso: str,
    step_days: int = 5,
    cloud_mask: bool = False,
    composite: bool = False,
    **kwargs,
):
    """
    Even older name — keep for safety.
    """
    return zonal_timeseries_index(
        index=index,
        geometry=geometry,
        from_iso=from_iso,
        to_iso=to_iso,
        step_days=step_days,
        cloud_mask=cloud_mask,
        composite=composite,
        **kwargs,
    )