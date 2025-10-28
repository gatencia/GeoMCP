import io
import requests
import numpy as np
import tifffile
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Headless backend (no GUI)
import matplotlib.pyplot as plt
from modules.sentinel_hub import _get_token, PROCESS_URL
import math
import numpy as np
import tifffile
import io
import numpy as np
import io
import tifffile
from collections import deque


def get_elevation(bbox, width=512, height=512):
    """Fetch elevation (DEM) as viewable grayscale PNG."""
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
      // Normalize DEM roughly 0–4000m → 0–1 range
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


# modules/elevation.py  (add below existing get_elevation)
import io
from fastapi.responses import StreamingResponse

def get_elevation_raw(bbox, width=512, height=512):
    """Fetch raw float32 DEM (GeoTIFF) for scientific use."""
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
        raise RuntimeError(f"DEM API error {r.status_code}: {r.text[:200]}")
    return r.content

def compute_hillshade_from_tiff(tiff_bytes):
    """Compute a simple hillshade from a float32 DEM GeoTIFF."""
    arr = tifffile.imread(io.BytesIO(tiff_bytes)).astype(np.float32)
    arr = np.nan_to_num(arr)

    # Calculate gradients
    x, y = np.gradient(arr)
    slope = np.pi/2.0 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)

    # Light source: 315° azimuth (NW), 45° elevation
    azimuth, altitude = np.radians(315), np.radians(45)
    shaded = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    shaded = np.clip(shaded, 0, 1)

    # Convert to grayscale PNG
    img = Image.fromarray((shaded * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def compute_slope_vector_field(tiff_bytes, step=20):
    """
    Compute and render a slope vector field from DEM GeoTIFF.
    step: spacing between arrows (pixels)
    """
    arr = tifffile.imread(io.BytesIO(tiff_bytes)).astype(np.float32)
    arr = np.nan_to_num(arr)

    # Compute gradients
    dzdx, dzdy = np.gradient(arr)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    direction = np.arctan2(-dzdy, -dzdx)  # downhill direction

    # Downsample for clarity
    X, Y = np.meshgrid(np.arange(0, arr.shape[1], step), np.arange(0, arr.shape[0], step))
    U = np.cos(direction[::step, ::step]) * slope[::step, ::step]
    V = np.sin(direction[::step, ::step]) * slope[::step, ::step]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="terrain")
    ax.quiver(X, Y, U, -V, slope[::step, ::step], cmap="inferno", scale=20, pivot="mid")
    ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _meters_per_pixel_from_bbox(bbox, width, height):
    """
    bbox: [minLon, minLat, maxLon, maxLat] (EPSG:4326)
    Returns dx_m, dy_m (meters per pixel in x (lon), y (lat)).
    """
    minLon, minLat, maxLon, maxLat = bbox
    mid_lat_rad = math.radians((minLat + maxLat) / 2.0)

    # meters per degree
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(mid_lat_rad)

    deg_per_px_lon = (maxLon - minLon) / float(width)
    deg_per_px_lat = (maxLat - minLat) / float(height)

    dx_m = m_per_deg_lon * deg_per_px_lon
    dy_m = m_per_deg_lat * deg_per_px_lat
    return dx_m, dy_m

def _slope_aspect_from_tiff(tiff_bytes, bbox, width, height):
    """
    Compute slope (deg) and aspect (deg 0..360) from float32 DEM GeoTIFF.
    Aspect is degrees clockwise from North (0 = North, 90 = East).
    """
    dem = tifffile.imread(io.BytesIO(tiff_bytes)).astype(np.float32)
    dem = np.nan_to_num(dem)

    dx_m, dy_m = _meters_per_pixel_from_bbox(bbox, width, height)

    # gradients: note numpy.gradient order is (rows(y), cols(x))
    dz_dy, dz_dx = np.gradient(dem)

    # convert to per-meter gradients
    dzdx = dz_dx / max(dx_m, 1e-9)
    dzdy = dz_dy / max(dy_m, 1e-9)

    # slope in radians -> degrees
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # aspect: 0..360, clockwise from north
    # downhill direction = -grad; arctan2(x,y) careful with axes
    aspect_rad = np.arctan2(-dzdx, -dzdy)  # x then y
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
    Quick visualization: map 0..vmax degrees to 0..255.
    """
    slope_deg, _ = _slope_aspect_from_tiff(tiff_bytes, bbox, width, height)
    s = np.clip(slope_deg, 0.0, float(vmax)) / float(vmax)
    s = (s * 255.0).astype(np.uint8)
    from PIL import Image
    im = Image.fromarray(s, mode="L")
    out = io.BytesIO()
    im.save(out, format="PNG")
    out.seek(0)
    return out.getvalue()

def slope_aspect_at_point(tiff_bytes, bbox, width, height, lat, lon):
    """
    Return slope/ aspect at the nearest pixel to (lat, lon).
    """
    slope_deg, aspect_deg = _slope_aspect_from_tiff(tiff_bytes, bbox, width, height)
    minLon, minLat, maxLon, maxLat = bbox
    # compute pixel indices
    col = int(round((lon - minLon) / (maxLon - minLon) * (width - 1)))
    row = int(round((lat - minLat) / (maxLat - minLat) * (height - 1)))
    row = int(np.clip(row, 0, slope_deg.shape[0]-1))
    col = int(np.clip(col, 0, slope_deg.shape[1]-1))
    return float(slope_deg[row, col]), float(aspect_deg[row, col])

def compute_flow_direction_and_accumulation(tiff_bytes):
    """
    Compute D8 flow direction (0–7 encoding) and flow accumulation using a
    single-pass topological approach (no overflow risk).
    """
    dem = tifffile.imread(io.BytesIO(tiff_bytes)).astype(np.float32)
    dem = np.nan_to_num(dem)

    nrows, ncols = dem.shape
    d_row = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    d_col = np.array([1, 1, 0, -1, -1, -1, 0, 1])

    # 1️⃣ Flow direction (index 0–7 of steepest descent)
    direction = np.full_like(dem, -1, dtype=np.int8)
    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            z = dem[r, c]
            dz = dem[r + d_row, c + d_col] - z
            if np.all(dz >= 0):
                continue  # local pit or flat
            direction[r, c] = np.argmin(dz)

    # 2️⃣ Initialize accumulation = 1 for each pixel
    accumulation = np.ones_like(dem, dtype=np.float32)

    # 3️⃣ Count inflows for each cell (number of neighbors draining into it)
    inflow_count = np.zeros_like(dem, dtype=np.int16)
    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            d = direction[r, c]
            if d == -1:
                continue
            rr, cc = r + d_row[d], c + d_col[d]
            inflow_count[rr, cc] += 1

    # 4️⃣ Process cells topologically: start with sources (no inflows)
    from collections import deque
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

    # 5️⃣ Log-scale for visualization
    accumulation = np.log1p(accumulation)
    return direction, accumulation



def flow_accum_png_from_dem_tiff(tiff_bytes):
    _, acc = compute_flow_direction_and_accumulation(tiff_bytes)
    acc_norm = acc / np.max(acc)
    acc_img = (acc_norm * 255).astype(np.uint8)
    from PIL import Image
    im = Image.fromarray(acc_img, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def flow_accum_tiff_from_dem_tiff(tiff_bytes):
    _, acc = compute_flow_direction_and_accumulation(tiff_bytes)
    buf = io.BytesIO()
    tifffile.imwrite(buf, acc)
    buf.seek(0)
    return buf.getvalue()