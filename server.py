from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List
import io
import numpy as np
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Dict, Any

from modules.zonal import (
    zonal_stats_index,
    point_series_index,
    area_series_index,
)

# Core NDVI util (unchanged)
from modules.sentinel_hub import process_png, NDVI_PNG_EVALSCRIPT
from typing import List, Dict

from pydantic import BaseModel, Field

from modules.cloudfree import (
    get_cloudfree_truecolor_png,
    get_cloudfree_ndvi_png,
    get_cloudfree_ndwi_png,
)

from modules.classification import (
    get_rule_based_png,
    get_rule_based_tiff,
    get_rule_based_matrix,
    get_unsupervised_png,
    get_unsupervised_tiff,
    get_unsupervised_matrix,
    get_supervised_png,
    get_supervised_tiff,
    get_supervised_matrix,
)

# Urban areas and exploration
from modules.ndbi import (
    get_ndbi_png,
    get_ndbi_tiff,
    get_ndbi_matrix,
)

# Elevation + derivatives (now with PNG / TIFF / MATRIX helpers)
from modules.elevation import (
    get_elevation,
    get_elevation_raw,
    # Hillshade
    hillshade_png_from_dem_tiff,
    hillshade_tiff_from_dem_tiff,
    hillshade_matrix_from_dem_tiff,
    # Slope
    slope_png_from_dem_tiff,
    slope_tiff_from_dem_tiff,
    slope_matrix_from_dem_tiff,
    # Aspect
    aspect_tiff_from_dem_tiff,
    aspect_matrix_from_dem_tiff,
    # Flow accumulation
    flow_accum_png_from_dem_tiff,
    flow_accum_tiff_from_dem_tiff,
    flow_accum_matrix_from_dem_tiff,
    # Vector field
    compute_slope_vector_field,
    slope_vector_field_tiff,
    slope_vector_field_matrix,
)

# NDWI (PNG / TIFF / MATRIX)
from modules.ndwi import get_ndwi, get_ndwi_raw, get_ndwi_matrix

from modules.status import get_status

app = FastAPI(title="GeoMCP - Satellite MCP Server")

# --- Add to server.py (helpers for zonal parsers) ----------------------------
from fastapi import Request
from typing import Dict, Any, Optional, List
import json

_ALLOWED_INDICES = {"NDVI", "NDWI", "NDBI"}

def _coerce_bbox(x: Any) -> Optional[List[float]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 4:
        return [float(v) for v in x]
    if isinstance(x, str):
        # accept "minLon,minLat,maxLon,maxLat"
        parts = [p.strip() for p in x.split(",")]
        if len(parts) == 4:
            return [float(v) for v in parts]
    raise ValueError("bbox must be [minLon,minLat,maxLon,maxLat] or 'minLon,minLat,maxLon,maxLat'")

def _ensure_iso8601_day(s: str, end=False) -> str:
    # accept YYYY-MM-DD and expand to full ISO-8601Z
    if "T" in s:
        return s
    return f"{s}T23:59:59Z" if end else f"{s}T00:00:00Z"

def _parse_zonal_body(body: Dict[str, Any]) -> Dict[str, Any]:
    # tolerate from/to aliases
    if "from_date" not in body and "from" in body:
        body["from_date"] = body.pop("from")
    if "to_date" not in body and "to" in body:
        body["to_date"] = body.pop("to")

    # required
    missing = [k for k in ("index", "from_date", "to_date") if k not in body]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required field(s): {', '.join(missing)}")

    # normalize index
    idx = str(body["index"]).upper()
    if idx not in _ALLOWED_INDICES:
        raise HTTPException(status_code=400, detail=f"index must be one of {sorted(_ALLOWED_INDICES)}")
    body["index"] = idx

    # exactly one of geometry or bbox
    geom = body.get("geometry")
    bbox  = body.get("bbox")
    if (geom is None and bbox is None) or (geom is not None and bbox is not None):
        raise HTTPException(status_code=400, detail="Provide exactly one of 'geometry' OR 'bbox'")

    # coerce bbox if present
    if bbox is not None:
        try:
            body["bbox"] = _coerce_bbox(bbox)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # dims
    body["width"]  = int(body.get("width", 256))
    body["height"] = int(body.get("height", 256))
    if body["width"] <= 0 or body["height"] <= 0:
        raise HTTPException(status_code=400, detail="width/height must be positive integers")

    # cloud mask optional
    body["cloud_mask"] = bool(body.get("cloud_mask", True))

    # dates → full ISO
    body["from_date"] = _ensure_iso8601_day(str(body["from_date"]), end=False)
    body["to_date"]   = _ensure_iso8601_day(str(body["to_date"]),   end=True)

    return body

    
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    """
    Returns SentinelHub token status (for debugging auth & uptime).
    """
    return get_status()

@app.get("/elevation.png")
def elevation_png(bbox: str, width: int = 512, height: int = 512):
    """
    Returns an elevation map (SRTM DEM) as grayscale PNG.
    """
    try:
        bb = [float(x.strip()) for x in bbox.split(",")]
        img_bytes = get_elevation(bb, width=width, height=height)
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _parse_bbox(bbox: str) -> List[float]:
    try:
        vals = [float(x.strip()) for x in bbox.split(",")]
        if len(vals) != 4:
            raise ValueError
        return vals
    except Exception:
        raise HTTPException(status_code=400, detail="bbox must be 'minLon,minLat,maxLon,maxLat'")

@app.get("/ndvi.png")
def ndvi_png(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
    width: int = 512,
    height: int = 512,
    collection: str = "S2L2A",
):
    """
    Returns a grayscale NDVI PNG for the requested bbox/time window.
    """
    try:
        bb = _parse_bbox(bbox)
        img_bytes = process_png(
            bbox=bb,
            from_iso=from_date,
            to_iso=to_date,
            width=width,
            height=height,
            evalscript=NDVI_PNG_EVALSCRIPT,
            collection=collection,
        )
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze")
def analyze_ndvi(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
    width: int = 512,
    height: int = 512,
    collection: str = "S2L2A",
):
    """
    Simple NDVI stats over the returned PNG (mean/median/quantiles mapped back to [-1,1]).
    """
    bb = _parse_bbox(bbox)
    img_bytes = process_png(
        bbox=bb,
        from_iso=from_date,
        to_iso=to_date,
        width=width,
        height=height,
        evalscript=NDVI_PNG_EVALSCRIPT,
        collection=collection,
    )
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    arr = np.asarray(img, dtype=np.float32)  # 0..255
    ndvi = (arr / 255.0) * 2.0 - 1.0         # back to [-1,1]

    stats = {
        "mean": float(np.nanmean(ndvi)),
        "median": float(np.nanmedian(ndvi)),
        "p25": float(np.nanpercentile(ndvi, 25)),
        "p75": float(np.nanpercentile(ndvi, 75)),
        "min": float(np.nanmin(ndvi)),
        "max": float(np.nanmax(ndvi)),
    }
    return JSONResponse({"bbox": bb, "from": from_date, "to": to_date, "stats": stats})

@app.get("/elevation/raw")
def elevation_raw(bbox: str, width: int = 512, height: int = 512):
    """
    Returns float32 GeoTIFF (raw DEM values).
    """
    bb = [float(x.strip()) for x in bbox.split(",")]
    data = get_elevation_raw(bb, width, height)
    return StreamingResponse(io.BytesIO(data), media_type="image/tiff")



@app.get("/elevation/gradient")
def elevation_gradient(bbox: str, width: int = 512, height: int = 512):
    """
    Returns hillshade (illumination map) computed from DEM.
    """
    try:
        bb = [float(x.strip()) for x in bbox.split(",")]
        raw_tiff = get_elevation_raw(bb, width, height)
        hillshade_png = compute_hillshade_from_tiff(raw_tiff)
        return StreamingResponse(io.BytesIO(hillshade_png), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/elevation/vectors")
def elevation_vectors(bbox: str, width: int = 512, height: int = 512, step: int = 20):
    """
    Returns slope vector field overlay (arrows show direction & steepness).
    """
    try:
        bb = [float(x.strip()) for x in bbox.split(",")]
        raw_tiff = get_elevation_raw(bb, width, height)
        vector_png = compute_slope_vector_field(raw_tiff, step=step)
        return StreamingResponse(io.BytesIO(vector_png), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flow/accumulation.png")
def flow_accumulation_png(bbox: str, width: int = 512, height: int = 512):
    """
    Returns log-scaled flow accumulation map as PNG (brighter = higher flow).
    """
    bb = [float(x) for x in bbox.split(",")]
    dem = get_elevation_raw(bb, width, height)
    img_bytes = flow_accum_png_from_dem_tiff(dem)
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

@app.get("/flow/accumulation.tif")
def flow_accumulation_tif(bbox: str, width: int = 512, height: int = 512):
    """
    Returns raw flow accumulation (float32) as GeoTIFF.
    """
    bb = [float(x) for x in bbox.split(",")]
    dem = get_elevation_raw(bb, width, height)
    data = flow_accum_tiff_from_dem_tiff(dem)
    return StreamingResponse(io.BytesIO(data), media_type="image/tiff")

@app.get("/ndwi.png")
def ndwi_png(bbox: str, width: int = 512, height: int = 512):
    bb = [float(x) for x in bbox.split(",")]
    img = get_ndwi(bb, width, height)
    return StreamingResponse(io.BytesIO(img), media_type="image/png")


# Elevation JSON
@app.get("/elevation.matrix")
def elevation_matrix(bbox: str, width: int = 256, height: int = 256):
    bb = [float(x) for x in bbox.split(",")]
    return JSONResponse(get_elevation_matrix(bb, width, height))

# NDWI TIFF & JSON
@app.get("/ndwi.tif")
def ndwi_tif(bbox: str, width: int = 512, height: int = 512):
    bb = [float(x) for x in bbox.split(",")]
    data = get_ndwi_raw(bb, width, height)
    return StreamingResponse(io.BytesIO(data), media_type="image/tiff")

@app.get("/ndwi.matrix")
def ndwi_matrix(bbox: str, width: int = 256, height: int = 256):
    bb = [float(x) for x in bbox.split(",")]
    return JSONResponse(get_ndwi_matrix(bb, width, height))


@app.get("/hillshade.png")
def hillshade_png(
    bbox: str,
    width: int = 512,
    height: int = 512,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        img = hillshade_png_from_dem_tiff(dem, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hillshade.tif")
def hillshade_tif(
    bbox: str,
    width: int = 512,
    height: int = 512,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        data = hillshade_tiff_from_dem_tiff(dem, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg)
        return StreamingResponse(io.BytesIO(data), media_type="image/tiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hillshade.matrix")
def hillshade_matrix(
    bbox: str,
    width: int = 256,
    height: int = 256,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        payload = hillshade_matrix_from_dem_tiff(dem, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/aspect.tif")
def aspect_tif(bbox: str, width: int = 512, height: int = 512):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        data = aspect_tiff_from_dem_tiff(dem, bbox=bb, width=width, height=height)
        return StreamingResponse(io.BytesIO(data), media_type="image/tiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/aspect.matrix")
def aspect_matrix(bbox: str, width: int = 256, height: int = 256):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        payload = aspect_matrix_from_dem_tiff(dem, bbox=bb, width=width, height=height)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/flow/accumulation.png")
def flow_accumulation_png(bbox: str, width: int = 512, height: int = 512):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        img = flow_accum_png_from_dem_tiff(dem)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flow/accumulation.tif")
def flow_accumulation_tif(bbox: str, width: int = 512, height: int = 512):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        data = flow_accum_tiff_from_dem_tiff(dem)
        return StreamingResponse(io.BytesIO(data), media_type="image/tiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flow/accumulation.matrix")
def flow_accumulation_matrix(bbox: str, width: int = 256, height: int = 256):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        payload = flow_accum_matrix_from_dem_tiff(dem)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Keep your existing endpoint (PNG) for backward compatibility
@app.get("/elevation/vectors")
def elevation_vectors_legacy(bbox: str, width: int = 512, height: int = 512, step: int = 20):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        img = compute_slope_vector_field(dem, step=step)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New explicit PNG route
@app.get("/elevation/vectors.png")
def elevation_vectors_png(bbox: str, width: int = 512, height: int = 512, step: int = 20):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        img = compute_slope_vector_field(dem, step=step)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/elevation/vectors.tif")
def elevation_vectors_tif(bbox: str, width: int = 512, height: int = 512, step: int = 20):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        data = slope_vector_field_tiff(dem, step=step)  # 2 bands: U,V
        return StreamingResponse(io.BytesIO(data), media_type="image/tiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/elevation/vectors.matrix")
def elevation_vectors_matrix(bbox: str, width: int = 512, height: int = 512, step: int = 20):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        payload = slope_vector_field_matrix(dem, step=step)  # {"U":[[...]], "V":[[...]]}
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ndwi.png")
def ndwi_png(bbox: str, width: int = 512, height: int = 512):
    try:
        bb = _parse_bbox(bbox)
        img = get_ndwi(bb, width, height)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ndwi.tif")
def ndwi_tif(bbox: str, width: int = 512, height: int = 512):
    try:
        bb = _parse_bbox(bbox)
        data = get_ndwi_raw(bb, width, height)
        return StreamingResponse(io.BytesIO(data), media_type="image/tiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ndwi.matrix")
def ndwi_matrix(bbox: str, width: int = 256, height: int = 256):
    try:
        bb = _parse_bbox(bbox)
        payload = get_ndwi_matrix(bb, width, height)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/slope.png")
def slope_png(
    bbox: str,
    width: int = 512,
    height: int = 512,
    vmax: float = 60.0,
):
    """
    PNG visualization of slope (degrees). 0..vmax mapped to 0..255.
    """
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        img = slope_png_from_dem_tiff(dem, bbox=bb, width=width, height=height, vmax=vmax)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/slope.tif")
def slope_tif(bbox: str, width: int = 512, height: int = 512):
    """
    Float32 GeoTIFF of slope (degrees).
    """
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        data = slope_tiff_from_dem_tiff(dem, bbox=bb, width=width, height=height)
        return StreamingResponse(io.BytesIO(data), media_type="image/tiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/slope.matrix")
def slope_matrix(bbox: str, width: int = 256, height: int = 256):
    """
    JSON float matrix of slope (degrees).
    """
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        payload = slope_matrix_from_dem_tiff(dem, bbox=bb, width=width, height=height)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/ndbi.png")
def ndbi_png(
    bbox: str,
    width: int = 512,
    height: int = 512,
    from_date: str | None = Query(None, alias="from"),
    to_date: str | None = Query(None, alias="to"),
):
    """
    PNG visualization of NDBI. If `from`/`to` are provided (ISO 8601), uses that time window.
    Otherwise uses mostRecent mosaic.
    """
    try:
        bb = _parse_bbox(bbox)
        img = get_ndbi_png(bb, width=width, height=height, from_iso=from_date, to_iso=to_date)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ndbi.tif")
def ndbi_tif(
    bbox: str,
    width: int = 512,
    height: int = 512,
    from_date: str | None = Query(None, alias="from"),
    to_date: str | None = Query(None, alias="to"),
):
    """Float32 GeoTIFF of NDBI."""
    try:
        bb = _parse_bbox(bbox)
        data = get_ndbi_tiff(bb, width=width, height=height, from_iso=from_date, to_iso=to_date)
        return StreamingResponse(io.BytesIO(data), media_type="image/tiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ndbi.matrix")
def ndbi_matrix(
    bbox: str,
    width: int = 256,
    height: int = 256,
    from_date: str | None = Query(None, alias="from"),
    to_date: str | None = Query(None, alias="to"),
):
    """JSON float matrix of NDBI values."""
    try:
        bb = _parse_bbox(bbox)
        payload = get_ndbi_matrix(bb, width=width, height=height, from_iso=from_date, to_iso=to_date)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SupervisedBody(BaseModel):
    bbox: List[float]
    from_: str = Field(..., alias="from")
    to: str = Field(..., alias="to")
    width: int = 256
    height: int = 256
    training_points: List[Dict]  # [{ "lat": float, "lon": float, "label": int }, ...]


@app.get("/classify/rule_based.png")
def classify_rule_based_png(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
    width: int = 256,
    height: int = 256,
):
    try:
        bb = _parse_bbox(bbox)
        img = get_rule_based_png(bbox=bb, from_iso=from_date, to_iso=to_date, width=width, height=height)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classify/rule_based.tif")
def classify_rule_based_tif(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
    width: int = 256,
    height: int = 256,
):
    try:
        bb = _parse_bbox(bbox)
        tif = get_rule_based_tiff(bbox=bb, from_iso=from_date, to_iso=to_date, width=width, height=height)
        return StreamingResponse(io.BytesIO(tif), media_type="image/tiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classify/rule_based.matrix")
def classify_rule_based_matrix(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
    width: int = 256,
    height: int = 256,
):
    try:
        bb = _parse_bbox(bbox)
        mat = get_rule_based_matrix(bbox=bb, from_iso=from_date, to_iso=to_date, width=width, height=height)
        return JSONResponse(mat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classify/unsupervised.png")
def classify_unsupervised_png(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
    k: int = 6,
    width: int = 256,
    height: int = 256,
):
    try:
        bb = _parse_bbox(bbox)
        img = get_unsupervised_png(bbox=bb, from_iso=from_date, to_iso=to_date, width=width, height=height, k=k)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classify/unsupervised.tif")
def classify_unsupervised_tif(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
    k: int = 6,
    width: int = 256,
    height: int = 256,
):
    try:
        bb = _parse_bbox(bbox)
        tif = get_unsupervised_tiff(bbox=bb, from_iso=from_date, to_iso=to_date, width=width, height=height, k=k)
        return StreamingResponse(io.BytesIO(tif), media_type="image/tiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classify/unsupervised.matrix")
def classify_unsupervised_matrix(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
    k: int = 6,
    width: int = 256,
    height: int = 256,
):
    try:
        bb = _parse_bbox(bbox)
        mat = get_unsupervised_matrix(bbox=bb, from_iso=from_date, to_iso=to_date, width=width, height=height, k=k)
        return JSONResponse(mat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/supervised.png")
def classify_supervised_png(body: SupervisedBody):
    try:
        img = get_supervised_png(
            bbox=body.bbox,
            from_iso=body.from_,
            to_iso=body.to,
            width=body.width,
            height=body.height,
            training_points=body.training_points,
        )
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/supervised.tif")
def classify_supervised_tif(body: SupervisedBody):
    try:
        tif = get_supervised_tiff(
            bbox=body.bbox,
            from_iso=body.from_,
            to_iso=body.to,
            width=body.width,
            height=body.height,
            training_points=body.training_points,
        )
        return StreamingResponse(io.BytesIO(tif), media_type="image/tiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/supervised.matrix")
def classify_supervised_matrix(body: SupervisedBody):
    try:
        mat = get_supervised_matrix(
            bbox=body.bbox,
            from_iso=body.from_,
            to_iso=body.to,
            width=body.width,
            height=body.height,
            training_points=body.training_points,
        )
        return JSONResponse(mat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import io
from PIL import Image

def _strip_alpha(png_bytes: bytes) -> bytes:
    """Flatten any RGBA/LA image to opaque RGB/8-bit grayscale."""
    bio = io.BytesIO(png_bytes)
    im = Image.open(bio)
    if im.mode in ("RGBA", "LA"):
        # For color images, composite on black (or change to white if you prefer)
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (0, 0, 0))
            bg.paste(im, mask=im.split()[-1])
            out = io.BytesIO()
            bg.save(out, format="PNG")
            return out.getvalue()
        # For grayscale+alpha
        if im.mode == "LA":
            g, a = im.split()
            # Make it opaque grayscale by ignoring alpha
            out = io.BytesIO()
            g.save(out, format="PNG")
            return out.getvalue()
    return png_bytes

    # ------------------------------------------------------------------
# Cloud-free TRUE COLOR (opaque toggle)
# ------------------------------------------------------------------
@app.get("/cloudfree/truecolor.png")
def cloudfree_truecolor_png(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str   = Query(..., alias="to"),
    maxcc: int = 30,
    width: int = 512,
    height: int = 512,
    opaque: bool = False,   # <— NEW
):
    try:
        bb = _parse_bbox(bbox)
        png_bytes = get_cloudfree_truecolor_png(
            bb, from_date, to_date, maxcc=maxcc, width=width, height=height
        )
        if opaque:
            png_bytes = _strip_alpha(png_bytes)
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Cloud-free NDVI (opaque toggle)
# ------------------------------------------------------------------
@app.get("/cloudfree/ndvi.png")
def cloudfree_ndvi_png(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str   = Query(..., alias="to"),
    maxcc: int = 30,
    width: int = 512,
    height: int = 512,
    opaque: bool = False,   # <— NEW
):
    try:
        bb = _parse_bbox(bbox)
        png_bytes = get_cloudfree_ndvi_png(
            bb, from_date, to_date, maxcc=maxcc, width=width, height=height
        )
        if opaque:
            png_bytes = _strip_alpha(png_bytes)
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Cloud-free NDWI (opaque toggle)
# ------------------------------------------------------------------
@app.get("/cloudfree/ndwi.png")
def cloudfree_ndwi_png(
    bbox: str,
    from_date: str = Query(..., alias="from"),
    to_date: str   = Query(..., alias="to"),
    maxcc: int = 30,
    width: int = 512,
    height: int = 512,
    opaque: bool = False,   # <— NEW
):
    try:
        bb = _parse_bbox(bbox)
        png_bytes = get_cloudfree_ndwi_png(
            bb, from_date, to_date, maxcc=maxcc, width=width, height=height
        )
        if opaque:
            png_bytes = _strip_alpha(png_bytes)
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ZonalStatsBody(BaseModel):
    index: str                          # "NDVI" | "NDWI" | "NDBI" | "NDMI" | "EVI" | "RAW:B08" ...
    from_date: str
    to_date: str
    width: int = 256
    height: int = 256
    cloud_mask: bool = True
    bbox: Optional[List[float]] = None  # [minLon,minLat,maxLon,maxLat]
    geometry: Optional[Dict[str, Any]] = None  # GeoJSON geometry

class PointSeriesQuery(BaseModel):
    index: str
    lat: float
    lon: float
    from_date: str
    to_date: str
    step_days: int = 10
    buffer_m: float = 20.0
    cloud_mask: bool = True

class AreaSeriesBody(BaseModel):
    index: str
    from_date: str
    to_date: str
    step_days: int = 10
    width: int = 256
    height: int = 256
    cloud_mask: bool = True
    bbox: Optional[List[float]] = None
    geometry: Optional[Dict[str, Any]] = None



@app.post("/zonal_stats.json")
def post_zonal_stats(body: ZonalStatsBody):
    """
    Zonal statistics for an index over a bbox or GeoJSON geometry.
    Returns { index, from, to, width, height, stats{...} }.
    """
    if (body.bbox is None) and (body.geometry is None):
        raise HTTPException(status_code=400, detail="Provide either 'bbox' or 'geometry'.")

    try:
        out = zonal_stats_index(
            index=body.index,
            from_iso=body.from_date,
            to_iso=body.to_date,
            width=body.width,
            height=body.height,
            bbox=body.bbox,
            geometry=body.geometry,
            cloud_mask=body.cloud_mask,
        )
        return JSONResponse(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/series/point.json")
def post_series_point(body: PointSeriesQuery):
    """
    Point time-series for an index (lat/lon). Splits [from,to] into step_days bins.
    """
    try:
        out = point_series_index(
            index=body.index,
            lat=body.lat, lon=body.lon,
            from_iso=body.from_date, to_iso=body.to_date,
            step_days=body.step_days, buffer_m=body.buffer_m,
            cloud_mask=body.cloud_mask,
        )
        return JSONResponse(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/series/area.json")
def post_series_area(body: AreaSeriesBody):
    """
    Area (bbox/geometry) time-series for an index. Each bin returns zonal stats.
    """
    if (body.bbox is None) and (body.geometry is None):
        raise HTTPException(status_code=400, detail="Provide either 'bbox' or 'geometry'.")

    try:
        out = area_series_index(
            index=body.index,
            from_iso=body.from_date, to_iso=body.to_date,
            step_days=body.step_days,
            width=body.width, height=body.height,
            bbox=body.bbox, geometry=body.geometry,
            cloud_mask=body.cloud_mask,
        )
        return JSONResponse(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))