import io
import requests
import numpy as np
import tifffile
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Headless backend (no GUI)
import matplotlib.pyplot as plt
from modules.sentinel_hub import _get_token, PROCESS_URL


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