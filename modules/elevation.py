# modules/elevation.py
import io
import math
from collections import deque

import numpy as np
import requests
import tifffile
from PIL import Image

# Headless plotting backend (for vector-field visualization)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: robust single-band reads
import rasterio

from modules.sentinel_hub import _get_token, PROCESS_URL


# ───────────────────────────────────────────────────────────────────────────────
# DEM FETCHERS (Sentinel Hub Process API, using same auth flow as you have)
# ───────────────────────────────────────────────────────────────────────────────

def get_elevation(bbox, width=512, height=512):
    """
    Return DEM as viewable grayscale PNG (0–4000m mapped to 0–1).
    Backward-compatible with your existing /elevation.png endpoint.
    """
    headers = {"Authorization": f"Bearer {_get_token()}"}

    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["DEM"],
        output: { bands: 3 }
      };
    }
    function evaluatePixel(sample) {
      let v = sample.DEM;
      if (v < 0) v = 0;
      if (v > 4000) v = 4000;
      let scaled = v / 4000.0;
      return [scaled, scaled, scaled];
    }
    """

    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{"type": "DEM"}]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": evalscript
    }

    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"DEM API error {r.status_code}: {r.text[:300]}")
    return r.content


def get_elevation_raw(bbox, width=512, height=512):
    """
    Return raw float32 DEM GeoTIFF (scientific).
    """
    headers = {"Authorization": f"Bearer {_get_token()}"}

    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["DEM"],
        output: { bands: 1, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(sample) {
      return [sample.DEM];
    }
    """

    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{"type": "DEM"}]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": evalscript
    }

    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"DEM API error {r.status_code}: {r.text[:300]}")
    return r.content


def get_elevation_matrix(bbox, width=256, height=256):
    """
    Return DEM as JSON float matrix (meters).
    """
    tiff_bytes = get_elevation_raw(bbox, width, height)
    with rasterio.open(io.BytesIO(tiff_bytes)) as src:
        arr = src.read(1).astype(float)
    arr = np.nan_to_num(arr).tolist()
    return {
        "bbox": bbox,
        "width": width,
        "height": height,
        "units": "meters",
        "elevation": arr
    }


# ───────────────────────────────────────────────────────────────────────────────
# GEODESY HELPERS & CORE DERIVATIVES (slope, aspect, hillshade, flow)
# ───────────────────────────────────────────────────────────────────────────────

def _meters_per_pixel_from_bbox(bbox, width, height):
    """
    bbox: [minLon, minLat, maxLon, maxLat] (EPSG:4326)
    Returns dx_m, dy_m (meters per pixel in x (lon), y (lat)).
    """
    minLon, minLat, maxLon, maxLat = bbox
    mid_lat_rad = math.radians((minLat + maxLat) / 2.0)

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(mid_lat_rad)

    deg_per_px_lon = (maxLon - minLon) / float(width)
    deg_per_px_lat = (maxLat - minLat) / float(height)

    dx_m = m_per_deg_lon * deg_per_px_lon
    dy_m = m_per_deg_lat * deg_per_px_lat
    return dx_m, dy_m


def _read_dem_array(tiff_bytes):
    """
    Load float32 DEM from bytes → np.ndarray (H,W).
    """
    arr = tifffile.imread(io.BytesIO(tiff_bytes)).astype(np.float32)
    return np.nan_to_num(arr)


def _slope_aspect_from_tiff(tiff_bytes, bbox, width, height):
    """
    Compute slope (deg) and aspect (deg, clockwise from North) from DEM GeoTIFF.
    """
    dem = _read_dem_array(tiff_bytes)
    dx_m, dy_m = _meters_per_pixel_from_bbox(bbox, width, height)

    # numpy.gradient → (rows, cols) ≡ (y, x)
    dz_dy, dz_dx = np.gradient(dem)

    dzdx = dz_dx / max(dx_m, 1e-9)
    dzdy = dz_dy / max(dy_m, 1e-9)

    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # downhill direction = -grad; aspect CW from North
    aspect_rad = np.arctan2(-dzdx, -dzdy)           # x then y
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    aspect_deg = aspect_deg.astype(np.float32)

    return slope_deg, aspect_deg


def slope_tiff_from_dem_tiff(tiff_bytes, bbox, width, height):
    slope_deg, _ = _slope_aspect_from_tiff(tiff_bytes, bbox, width, height)
    buf = io.BytesIO()
    tifffile.imwrite(buf, slope_deg.astype(np.float32))
    buf.seek(0)
    return buf.getvalue()


def aspect_tiff_from_dem_tiff(tiff_bytes, bbox, width, height):
    _, aspect_deg = _slope_aspect_from_tiff(tiff_bytes, bbox, width, height)
    buf = io.BytesIO()
    tifffile.imwrite(buf, aspect_deg.astype(np.float32))
    buf.seek(0)
    return buf.getvalue()


def slope_png_from_dem_tiff(tiff_bytes, bbox, width, height, vmax=60.0):
    """
    Visualize slope degrees to 0..255 with optional vmax clamp.
    """
    slope_deg, _ = _slope_aspect_from_tiff(tiff_bytes, bbox, width, height)
    s = np.clip(slope_deg, 0.0, float(vmax)) / float(vmax)
    s = (s * 255.0).astype(np.uint8)
    im = Image.fromarray(s, mode="L")
    out = io.BytesIO()
    im.save(out, format="PNG")
    out.seek(0)
    return out.getvalue()


def slope_matrix_from_dem_tiff(tiff_bytes, bbox, width, height):
    slope_deg, _ = _slope_aspect_from_tiff(tiff_bytes, bbox, width, height)
    return {
        "bbox": bbox,
        "width": width,
        "height": height,
        "units": "degrees",
        "slope": np.nan_to_num(slope_deg).astype(float).tolist()
    }


def aspect_matrix_from_dem_tiff(tiff_bytes, bbox, width, height):
    _, aspect_deg = _slope_aspect_from_tiff(tiff_bytes, bbox, width, height)
    return {
        "bbox": bbox,
        "width": width,
        "height": height,
        "units": "degrees_clockwise_from_north",
        "aspect": np.nan_to_num(aspect_deg).astype(float).tolist()
    }


def slope_aspect_at_point(tiff_bytes, bbox, width, height, lat, lon):
    """
    Return slope/aspect at the nearest pixel to (lat, lon).
    """
    slope_deg, aspect_deg = _slope_aspect_from_tiff(tiff_bytes, bbox, width, height)
    minLon, minLat, maxLon, maxLat = bbox
    col = int(round((lon - minLon) / (maxLon - minLon) * (width - 1)))
    row = int(round((lat - minLat) / (maxLat - minLat) * (height - 1)))
    row = int(np.clip(row, 0, slope_deg.shape[0] - 1))
    col = int(np.clip(col, 0, slope_deg.shape[1] - 1))
    return float(slope_deg[row, col]), float(aspect_deg[row, col])


# ── Hillshade (PNG, TIFF, MATRIX) ──────────────────────────────────────────────

def _hillshade_array_from_dem(dem, azimuth_deg=315.0, altitude_deg=45.0):
    """
    Compute hillshade array in 0..1 from DEM.
    """
    x, y = np.gradient(dem)
    slope = np.pi/2.0 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)

    azimuth = np.radians(azimuth_deg)
    altitude = np.radians(altitude_deg)
    shaded = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    return np.clip(shaded, 0.0, 1.0).astype(np.float32)


def hillshade_png_from_dem_tiff(tiff_bytes, azimuth_deg=315.0, altitude_deg=45.0):
    dem = _read_dem_array(tiff_bytes)
    shaded = _hillshade_array_from_dem(dem, azimuth_deg, altitude_deg)
    img = Image.fromarray((shaded * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# Backward-compat alias (your server calls this name today):
compute_hillshade_from_tiff = hillshade_png_from_dem_tiff


def hillshade_tiff_from_dem_tiff(tiff_bytes, azimuth_deg=315.0, altitude_deg=45.0):
    dem = _read_dem_array(tiff_bytes)
    shaded = _hillshade_array_from_dem(dem, azimuth_deg, altitude_deg)
    buf = io.BytesIO()
    tifffile.imwrite(buf, shaded.astype(np.float32))
    buf.seek(0)
    return buf.getvalue()


def hillshade_matrix_from_dem_tiff(tiff_bytes, azimuth_deg=315.0, altitude_deg=45.0):
    dem = _read_dem_array(tiff_bytes)
    shaded = _hillshade_array_from_dem(dem, azimuth_deg, altitude_deg)
    return {
        "range": [0.0, 1.0],
        "hillshade": shaded.astype(float).tolist()
    }


# ── Flow Direction & Accumulation (PNG, TIFF, MATRIX) ─────────────────────────

def compute_flow_direction_and_accumulation(tiff_bytes):
    """
    Compute D8 flow direction (0..7) and log-scaled flow accumulation.
    Accumulation is log1p of contributing cells.
    """
    dem = _read_dem_array(tiff_bytes)

    nrows, ncols = dem.shape
    d_row = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    d_col = np.array([1, 1, 0, -1, -1, -1, 0, 1])

    direction = np.full_like(dem, -1, dtype=np.int8)
    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            z = dem[r, c]
            dz = dem[r + d_row, c + d_col] - z
            if np.all(dz >= 0):
                continue  # pit/flat
            direction[r, c] = np.argmin(dz)

    accumulation = np.ones_like(dem, dtype=np.float32)

    inflow_count = np.zeros_like(dem, dtype=np.int16)
    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            d = direction[r, c]
            if d == -1:
                continue
            rr, cc = r + d_row[d], c + d_col[d]
            inflow_count[rr, cc] += 1

    q = deque(zip(*np.where(inflow_count == 0)))
    while q:
        r, c = q.popleft()
        d = direction[r, c]
        if d == -1:
            continue
        rr, cc = r + d_row[d], c + d_col[d]
        accumulation[rr, cc] += accumulation[r, c]
        inflow_count[rr, cc] -= 1
        if inflow_count[rr, cc] == 0:
            q.append((rr, cc))

    acc_log = np.log1p(accumulation)
    return direction, acc_log


def flow_accum_png_from_dem_tiff(tiff_bytes):
    _, acc = compute_flow_direction_and_accumulation(tiff_bytes)
    acc_norm = acc / max(np.max(acc), 1e-9)
    acc_img = (acc_norm * 255).astype(np.uint8)
    im = Image.fromarray(acc_img, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def flow_accum_tiff_from_dem_tiff(tiff_bytes):
    _, acc = compute_flow_direction_and_accumulation(tiff_bytes)
    buf = io.BytesIO()
    tifffile.imwrite(buf, acc.astype(np.float32))
    buf.seek(0)
    return buf.getvalue()


def flow_accum_matrix_from_dem_tiff(tiff_bytes):
    _, acc = compute_flow_direction_and_accumulation(tiff_bytes)
    return {
        "accumulation_log": acc.astype(float).tolist()
    }


# ── Slope Vector Field (PNG, TIFF(2 bands), MATRIX(U,V)) ──────────────────────

def compute_slope_vector_field(tiff_bytes, step=20):
    """
    Visualization only (PNG): arrows show downhill direction & relative steepness.
    """
    arr = _read_dem_array(tiff_bytes)
    dzdx, dzdy = np.gradient(arr)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    direction = np.arctan2(-dzdy, -dzdx)  # downhill

    X, Y = np.meshgrid(np.arange(0, arr.shape[1], step), np.arange(0, arr.shape[0], step))
    U = np.cos(direction[::step, ::step]) * slope[::step, ::step]
    V = np.sin(direction[::step, ::step]) * slope[::step, ::step]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="terrain")
    ax.quiver(X, Y, U, -V, slope[::step, ::step], cmap="inferno", scale=20, pivot="mid")
    ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def slope_vector_field_tiff(tiff_bytes, step=20):
    """
    Return 2-band GeoTIFF (float32) with downsampled U,V components.
    """
    arr = _read_dem_array(tiff_bytes)
    dzdx, dzdy = np.gradient(arr)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    direction = np.arctan2(-dzdy, -dzdx)

    U = (np.cos(direction) * slope)[::step, ::step]
    V = (np.sin(direction) * slope)[::step, ::step]

    stacked = np.stack([U, V]).astype(np.float32)  # (bands, H, W)
    buf = io.BytesIO()
    tifffile.imwrite(buf, stacked)
    buf.seek(0)
    return buf.getvalue()


def slope_vector_field_matrix(tiff_bytes, step=20):
    """
    Return JSON with downsampled U,V arrays and 'step'.
    """
    arr = _read_dem_array(tiff_bytes)
    dzdx, dzdy = np.gradient(arr)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    direction = np.arctan2(-dzdy, -dzdx)

    U = (np.cos(direction) * slope)[::step, ::step]
    V = (np.sin(direction) * slope)[::step, ::step]

    return {
        "step": step,
        "U": U.astype(float).tolist(),
        "V": V.astype(float).tolist()
    }