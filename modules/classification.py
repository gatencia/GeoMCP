# modules/classification.py
# -----------------------------------------------------------------------------
# Land Cover Classification utilities (Supervised & Unsupervised) for Sentinel-2
# - Uses same auth system as other modules (Sentinel Hub PROCESS endpoint).
# - Fetches B02,B03,B04,B08,B11,B12 + dataMask as FLOAT32 GeoTIFF.
# - Provides PNG / GeoTIFF / JSON matrix outputs for both pipelines.
#
# Supervised expects "training_points": List[{"lat": float, "lon": float, "label": int}]
# Unsupervised uses KMeans with K clusters.
#
# Notes:
#   - dataMask==0 will be set to class -1 (nodata).
#   - PNG is colored with a stable palette for up to ~20 classes.
#   - GeoTIFF is int16 label raster (nodata=-1).
# -----------------------------------------------------------------------------

from __future__ import annotations
import io
import math
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import requests
import tifffile
from PIL import Image

from modules.sentinel_hub import _get_token, PROCESS_URL

from typing import List, Tuple, Dict
import io
import numpy as np
import tifffile
from PIL import Image
import requests
from modules.sentinel_hub import _get_token, PROCESS_URL

# ----------------------------- Config ----------------------------------------

# Default Sentinel-2 bands used for classification (10–20 m)
S2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

# Stable color palette for class visualization (up to 20 distinct colors)
_PALETTE = [
    (230, 25, 75),   # red
    (60, 180, 75),   # green
    (0, 130, 200),   # blue
    (245, 130, 48),  # orange
    (145, 30, 180),  # purple
    (70, 240, 240),  # cyan
    (240, 50, 230),  # magenta
    (210, 245, 60),  # lime
    (250, 190, 190), # pink
    (0, 128, 128),   # teal
    (230, 190, 255), # lavender
    (170, 110, 40),  # brown
    (255, 250, 200), # beige
    (128, 0, 0),     # maroon
    (170, 255, 195), # mint
    (128, 128, 0),   # olive
    (255, 215, 180), # apricot
    (0, 0, 128),     # navy
    (128, 128, 128), # grey
    (255, 255, 255), # white
]

NODATA_LABEL = -1

# ----------------------------- Helpers ---------------------------------------

def _ensure_iso8601(s: str, end: bool = False) -> str:
    """Normalize a date string to full ISO-8601 if needed."""
    if "T" in s:
        return s
    return f"{s}T23:59:59Z" if end else f"{s}T00:00:00Z"

def _fetch_s2_stack_float32(
    bbox: List[float],
    from_iso: str,
    to_iso: str,
    width: int,
    height: int,
    bands: Tuple[str, ...] = S2_BANDS,
) -> np.ndarray:
    """
    Fetch Sentinel-2 bands + dataMask as FLOAT32 GeoTIFF, shape (H, W, B+1).
    Last channel is dataMask in {0,1}. Invalid pixels → mask=0.
    """
    headers = {"Authorization": f"Bearer {_get_token()}"}

    from_iso = _ensure_iso8601(from_iso, end=False)
    to_iso   = _ensure_iso8601(to_iso, end=True)

    band_list = ", ".join(bands)
    nb = len(bands) + 1  # + dataMask

    evalscript = f"""
    //VERSION=3
    function setup() {{
      return {{
        input: [{band_list}, "dataMask"],
        output: {{ bands: {nb}, sampleType: "FLOAT32" }},
        mosaicking: "SIMPLE"
      }};
    }}
    function evaluatePixel(s) {{
      return [{', '.join([f's.{b}' for b in bands])}, s.dataMask];
    }}
    """

    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "S2L2A",
                "dataFilter": {"timeRange": {"from": from_iso, "to": to_iso}},
            }],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }

    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"S2 PROCESS error {r.status_code}: {r.text[:400]}")
    arr = tifffile.imread(io.BytesIO(r.content)).astype(np.float32)

    # Ensure (H, W, C)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr

def _lonlat_to_rc(
    bbox: List[float], width: int, height: int, lat: float, lon: float
) -> Tuple[int, int]:
    """Convert lon/lat to row/col indices (nearest pixel)."""
    minLon, minLat, maxLon, maxLat = bbox
    col = int(round((lon - minLon) / (maxLon - minLon) * (width - 1)))
    row = int(round((lat - minLat) / (maxLat - minLat) * (height - 1)))
    row = int(np.clip(row, 0, height - 1))
    col = int(np.clip(col, 0, width - 1))
    return row, col

def _mask_invalid(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split features (H,W,B) from mask channel, returning (features, valid_mask).
    data shape is (H, W, B+1); last channel is dataMask.
    """
    feats = data[..., :-1]
    mask = data[..., -1] > 0.5
    return feats, mask

def _labels_to_png(labels: np.ndarray, palette: List[Tuple[int, int, int]] = _PALETTE) -> bytes:
    """
    Convert int label raster to RGB PNG using a fixed palette.
    NODATA_LABEL (-1) is rendered as black.
    """
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    unique = np.unique(labels)

    for cls in unique:
        if cls == NODATA_LABEL:
            continue
        color = palette[int(cls) % len(palette)]
        rgb[labels == cls] = color

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def _labels_to_tiff(labels: np.ndarray) -> bytes:
    """Write labels as int16 GeoTIFF bytes (nodata = -1)."""
    buf = io.BytesIO()
    tifffile.imwrite(buf, labels.astype(np.int16))
    buf.seek(0)
    return buf.getvalue()

def _labels_to_matrix_json(labels: np.ndarray) -> Dict:
    """Serialize labels to JSON matrix with summary counts."""
    h, w = labels.shape
    flat = labels.flatten()
    classes, counts = np.unique(flat, return_counts=True)
    stats = {int(k): int(v) for k, v in zip(classes, counts)}
    # Convert to nested Python lists for JSON
    return {
        "width": int(w),
        "height": int(h),
        "nodata": NODATA_LABEL,
        "classes": sorted([int(c) for c in classes if c != NODATA_LABEL]),
        "counts": stats,
        "matrix": labels.tolist(),
    }

# ----------------------- Unsupervised (KMeans) -------------------------------

def _kmeans_numpy(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight KMeans fallback (no sklearn). Returns (centers, labels).
    X: (N, D) float32
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # Random init
    idx = rng.choice(n, size=k, replace=False)
    centers = X[idx].copy()

    for _ in range(max_iter):
        # Assign
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (N, K)
        labels = np.argmin(dists, axis=1)

        # Recompute
        new_centers = np.zeros_like(centers)
        for c in range(k):
            m = labels == c
            if np.any(m):
                new_centers[c] = X[m].mean(axis=0)
            else:
                # re-init empty cluster
                new_centers[c] = X[rng.integers(0, n)]
        if np.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    # Final labels
    dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    return centers, labels

def unsupervised_clusters(
    bbox: List[float],
    from_iso: str,
    to_iso: str,
    width: int = 256,
    height: int = 256,
    k: int = 6,
    bands: Tuple[str, ...] = S2_BANDS,
) -> np.ndarray:
    """
    Run KMeans on valid pixels. Returns label raster (H, W) with NODATA_LABEL for invalid.
    """
    stack = _fetch_s2_stack_float32(bbox, from_iso, to_iso, width, height, bands)
    feats, valid = _mask_invalid(stack)  # (H,W,B), (H,W)
    H, W, B = feats.shape

    X = feats[valid].reshape(-1, B)
    if X.size == 0:
        return np.full((H, W), NODATA_LABEL, dtype=np.int16)

    # Try sklearn KMeans; fallback to numpy
    labels_valid: np.ndarray
    try:
        from sklearn.cluster import KMeans  # type: ignore
        km = KMeans(n_clusters=k, n_init="auto", random_state=0)
        labels_valid = km.fit_predict(X).astype(np.int16)
    except Exception:
        _, labels_valid = _kmeans_numpy(X.astype(np.float32), k=k, max_iter=50, seed=0)
        labels_valid = labels_valid.astype(np.int16)

    labels = np.full((H, W), NODATA_LABEL, dtype=np.int16)
    labels[valid] = labels_valid
    return labels

def get_unsupervised_png(**kwargs) -> bytes:
    labels = unsupervised_clusters(**kwargs)
    return _labels_to_png(labels)

def get_unsupervised_tiff(**kwargs) -> bytes:
    labels = unsupervised_clusters(**kwargs)
    return _labels_to_tiff(labels)

def get_unsupervised_matrix(**kwargs) -> Dict:
    labels = unsupervised_clusters(**kwargs)
    return _labels_to_matrix_json(labels)

# ----------------------- Supervised (RF / KNN) -------------------------------

def _collect_training_samples(
    feats: np.ndarray, valid: np.ndarray, bbox: List[float], width: int, height: int,
    training_points: List[Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract training (X,y) from labeled points.
    training_points: [{"lat":..., "lon":..., "label": int}, ...]
    """
    X_list = []
    y_list = []
    for p in training_points:
        lat = float(p["lat"]); lon = float(p["lon"]); lbl = int(p["label"])
        r, c = _lonlat_to_rc(bbox, width, height, lat, lon)
        if not valid[r, c]:
            # Skip masked pixel
            continue
        X_list.append(feats[r, c])
        y_list.append(lbl)
    if not X_list:
        raise ValueError("No valid training samples intersect the image. Check points/bbox/time window.")
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int16)
    return X, y

def _predict_knn(X_train: np.ndarray, y_train: np.ndarray, X: np.ndarray, k: int = 3) -> np.ndarray:
    """Very small KNN fallback (L2)."""
    # (N, D) vs (M, D) → distances (M, N)
    d2 = np.sum((X[:, None, :] - X_train[None, :, :]) ** 2, axis=2)
    nn_idx = np.argpartition(d2, kth=min(k-1, X_train.shape[0]-1), axis=1)[:, :k]
    votes = y_train[nn_idx]  # (M, k)
    # majority vote (tie → min label)
    out = []
    for row in votes:
        vals, cnts = np.unique(row, return_counts=True)
        out.append(vals[np.argmax(cnts)])
    return np.array(out, dtype=np.int16)

def supervised_lulc_labels(
    bbox: List[float],
    from_iso: str,
    to_iso: str,
    training_points: List[Dict[str, float]],
    width: int = 256,
    height: int = 256,
    bands: Tuple[str, ...] = S2_BANDS,
) -> np.ndarray:
    """
    Train a classifier on labeled points and predict full raster labels.
    Prefers RandomForest (sklearn); falls back to a tiny KNN if sklearn unavailable.
    Returns label raster (H,W) with NODATA_LABEL for invalid pixels.
    """
    stack = _fetch_s2_stack_float32(bbox, from_iso, to_iso, width, height, bands)
    feats, valid = _mask_invalid(stack)  # (H,W,B), (H,W)
    H, W, B = feats.shape

    X_train, y_train = _collect_training_samples(feats, valid, bbox, W, H, training_points)

    # Normalize features (min-max per band) for stability (esp. KNN fallback)
    f = feats.reshape(-1, B)
    fmin = np.nanmin(f, axis=0)
    fmax = np.nanmax(f, axis=0)
    rng = np.maximum(fmax - fmin, 1e-6)
    feats_n = (feats - fmin) / rng
    X_train_n = (X_train - fmin) / rng

    X_valid = feats_n[valid].reshape(-1, B)

    # Try RandomForest
    labels_valid: np.ndarray
    try:
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        rf.fit(X_train_n, y_train)
        labels_valid = rf.predict(X_valid).astype(np.int16)
    except Exception:
        labels_valid = _predict_knn(X_train_n.astype(np.float32), y_train, X_valid.astype(np.float32), k=3)

    labels = np.full((H, W), NODATA_LABEL, dtype=np.int16)
    labels[valid] = labels_valid
    return labels

def get_supervised_png(**kwargs) -> bytes:
    labels = supervised_lulc_labels(**kwargs)
    return _labels_to_png(labels)

def get_supervised_tiff(**kwargs) -> bytes:
    labels = supervised_lulc_labels(**kwargs)
    return _labels_to_tiff(labels)

def get_supervised_matrix(**kwargs) -> Dict:
    labels = supervised_lulc_labels(**kwargs)
    return _labels_to_matrix_json(labels)


RB_CLASSES = {
    0: "water",
    1: "vegetation",
    2: "built_up",
    3: "bare_soil",
    4: "snow_ice",
}
RB_NODATA = -1

def _fetch_s2_stack_rule(bbox: List[float], from_iso: str, to_iso: str, width: int, height: int):
    """
    Fetch B02,B03,B04,B08,B11,B12,SCL,dataMask as FLOAT32 GeoTIFF; returns array (H,W,8)
    Order: [B02,B03,B04,B08,B11,B12,SCL,dataMask]
    """
    if "T" not in from_iso: from_iso = f"{from_iso}T00:00:00Z"
    if "T" not in to_iso:   to_iso   = f"{to_iso}T23:59:59Z"

    headers = {"Authorization": f"Bearer {_get_token()}"}
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B02","B03","B04","B08","B11","B12","SCL","dataMask"],
        output: { bands: 8, sampleType: "FLOAT32" },
        mosaicking: "SIMPLE"
      };
    }
    function evaluatePixel(s) {
      return [s.B02, s.B03, s.B04, s.B08, s.B11, s.B12, s.SCL, s.dataMask];
    }
    """
    body = {
      "input": {
        "bounds": {"bbox": bbox, "properties": {"crs":"http://www.opengis.net/def/crs/EPSG/0/4326"}},
        "data": [{
          "type": "S2L2A",
          "dataFilter": {"timeRange": {"from": from_iso, "to": to_iso}}
        }]
      },
      "output": {
        "width": width,
        "height": height,
        "responses": [{"identifier": "default", "format": {"type":"image/tiff"}}]
      },
      "evalscript": evalscript
    }
    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"S2 rule fetch error {r.status_code}: {r.text[:300]}")
    arr = tifffile.imread(io.BytesIO(r.content)).astype(np.float32)
    if arr.ndim == 2: arr = arr[..., None]
    return arr

def _labels_to_png(labels: np.ndarray) -> bytes:
    palette = [
        (0, 105, 148),   # water
        (60, 180, 75),   # vegetation
        (180, 180, 180), # built-up
        (210, 160, 90),  # bare soil
        (200, 220, 255), # snow/ice
    ]
    h,w = labels.shape
    rgb = np.zeros((h,w,3), dtype=np.uint8)
    for cls, color in enumerate(palette):
        rgb[labels == cls] = color
    # nodata stays black
    img = Image.fromarray(rgb, "RGB")
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return buf.getvalue()

def _labels_to_tiff(labels: np.ndarray) -> bytes:
    buf = io.BytesIO(); tifffile.imwrite(buf, labels.astype(np.int16)); buf.seek(0); return buf.getvalue()

def _labels_to_matrix(labels: np.ndarray) -> Dict:
    h,w = labels.shape
    classes, counts = np.unique(labels, return_counts=True)
    return {
        "width": int(w),
        "height": int(h),
        "classes": {int(c): int(n) for c,n in zip(classes, counts)},
        "labels_map": RB_CLASSES,
        "nodata": RB_NODATA,
        "matrix": labels.tolist()
    }

def rule_based_lulc_labels(
    bbox: List[float],
    from_iso: str,
    to_iso: str,
    width: int = 256,
    height: int = 256
) -> np.ndarray:
    """
    No-training land-cover classifier using NDVI/NDWI/NDBI/NDSI + SCL mask.
    Returns (H,W) int labels: {0..4} or -1 for nodata/cloud/shadow.
    """
    data = _fetch_s2_stack_rule(bbox, from_iso, to_iso, width, height)
    B02, B03, B04, B08, B11, B12, SCL, DM = [data[...,i] for i in range(8)]
    valid = (DM > 0.5)

    # Mask clouds (SCL ~ 8,9,10) and shadows (3), nodata (0/1 vary by product)
    cloud = np.isin(SCL.round().astype(np.int32), [3,8,9,10])  # add 3=shadows
    good = valid & (~cloud)

    # Indices
    eps = 1e-6
    NDVI = (B08 - B04) / (B08 + B04 + eps)
    NDWI = (B03 - B08) / (B03 + B08 + eps)
    NDBI = (B11 - B08) / (B11 + B08 + eps)
    NDSI = (B03 - B11) / (B03 + B11 + eps)

    labels = np.full(B02.shape, RB_NODATA, dtype=np.int16)

    # 0) water
    water = (NDWI > 0.2) & (NDVI < 0.2) & good
    labels[water] = 0

    # 4) snow/ice
    snow = (NDSI > 0.4) & good
    labels[snow] = 4

    # 1) vegetation
    veg = (NDVI > 0.45) & good & (labels == RB_NODATA)
    labels[veg] = 1

    # 2) built-up (strong NDBI, low greenness)
    built = (NDBI > 0.2) & (NDVI < 0.3) & good & (labels == RB_NODATA)
    labels[built] = 2

    # 3) bare soil (low greenness, not water/built/snow)
    bare = (NDVI > 0.05) & (NDVI < 0.3) & (NDBI > -0.1) & (NDBI < 0.25) & good & (labels == RB_NODATA)
    labels[bare] = 3

    return labels

def get_rule_based_png(**kwargs) -> bytes:
    return _labels_to_png(rule_based_lulc_labels(**kwargs))

def get_rule_based_tiff(**kwargs) -> bytes:
    return _labels_to_tiff(rule_based_lulc_labels(**kwargs))

def get_rule_based_matrix(**kwargs) -> Dict:
    return _labels_to_matrix(rule_based_lulc_labels(**kwargs))