from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List
import io
import numpy as np
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Dict, Any
import math

from modules.zonal import (
    zonal_stats_index,
    point_series_index,
    area_series_index,
    _fetch_index_and_mask,
)
#Aliases for Zonal analyis
compute_zonal_stats_bbox = lambda **kwargs: zonal_stats_index(**kwargs)
compute_zonal_stats_geometry = lambda **kwargs: zonal_stats_index(**kwargs)
compute_zonal_timeseries_bbox = lambda **kwargs: area_series_index(**kwargs)
compute_zonal_timeseries_geometry = lambda **kwargs: area_series_index(**kwargs)

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
    get_elevation_matrix,  # Added import
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
        hillshade_png = hillshade_png_from_dem_tiff(raw_tiff)
        return StreamingResponse(io.BytesIO(hillshade_png), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/flow/accumulation.tif")
def flow_accumulation_tif(bbox: str, width: int = 512, height: int = 512):
    """
    Returns raw flow accumulation (float32) as GeoTIFF.
    """
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        data = flow_accum_tiff_from_dem_tiff(dem)
        return StreamingResponse(io.BytesIO(data), media_type="image/tiff")
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

@app.get("/flow/accumulation.matrix")
def flow_accumulation_matrix(bbox: str, width: int = 256, height: int = 256):
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        payload = flow_accum_matrix_from_dem_tiff(dem)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/elevation/vectors")
def elevation_vectors_legacy(bbox: str, width: int = 512, height: int = 512, step: int = 20):
    """
    Returns slope vector field overlay (arrows show direction & steepness).
    """
    try:
        bb = _parse_bbox(bbox)
        dem = get_elevation_raw(bb, width, height)
        img = compute_slope_vector_field(dem, step=step)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    composite: bool = False

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
    composite: bool = False


# --- ZONAL STATS (drop-in replacement) --------------------------------------
from fastapi import Request
@app.post("/zonal_stats.json")
async def zonal_stats_json(request: Request):
    """
    Body accepts either:
      - { index, from_date|from, to_date|to, width, height, cloud_mask, bbox:[minLon,minLat,maxLon,maxLat] }
      - { index, from_date|from, to_date|to, width, height, cloud_mask, geometry:<GeoJSON Polygon/MultiPolygon> }
    Exactly one of 'bbox' or 'geometry' must be provided.
    index ∈ {NDVI, NDWI, NDBI} (case-insensitive).
    """
    def _ensure_iso(s: str, end: bool = False) -> str:
        if "T" in s:
            return s
        return f"{s}T23:59:59Z" if end else f"{s}T00:00:00Z"

    def _coerce_bbox(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)) and len(x) == 4:
            return [float(v) for v in x]
        if isinstance(x, str):
            parts = [p.strip() for p in x.split(",")]
            if len(parts) == 4:
                return [float(v) for v in parts]
        raise HTTPException(status_code=400, detail="bbox must be [minLon,minLat,maxLon,maxLat] or 'minLon,minLat,maxLon,maxLat'")

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    # aliases
    if "from_date" not in body and "from" in body:
        body["from_date"] = body.pop("from")
    if "to_date" not in body and "to" in body:
        body["to_date"] = body.pop("to")

    # required
    for k in ("index", "from_date", "to_date"):
        if k not in body:
            raise HTTPException(status_code=400, detail=f"Missing required field: {k}")

    idx = str(body["index"]).upper()
    if idx not in {"NDVI", "NDWI", "NDBI"}:
        raise HTTPException(status_code=400, detail="index must be one of ['NDVI','NDWI','NDBI']")
    from_iso = _ensure_iso(str(body["from_date"]), end=False)
    to_iso   = _ensure_iso(str(body["to_date"]),   end=True)

    width  = int(body.get("width", 256))
    height = int(body.get("height", 256))
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="width/height must be positive integers")
    cloud_mask = bool(body.get("cloud_mask", True))

    geom = body.get("geometry")
    bbox = body.get("bbox")
    if (geom is None and bbox is None) or (geom is not None and bbox is not None):
        raise HTTPException(status_code=400, detail="Provide exactly one of 'geometry' OR 'bbox'")

    try:
        if geom is not None:
            stats = compute_zonal_stats_geometry(
                index=idx,
                geometry=geom,
                from_iso=from_iso,
                to_iso=to_iso,
                width=width,
                height=height,
                cloud_mask=cloud_mask,
            )
        else:
            bb = _coerce_bbox(bbox)
            stats = compute_zonal_stats_bbox(
                index=idx,
                bbox=bb,
                from_iso=from_iso,
                to_iso=to_iso,
                width=width,
                height=height,
                cloud_mask=cloud_mask,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # --- sanitize all floats for JSON compliance ---
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj

    safe_stats = _sanitize(stats)

    return JSONResponse({
        "index": idx,
        "from": from_iso,
        "to": to_iso,
        "width": width,
        "height": height,
        "stats": safe_stats,
    })




from datetime import datetime
import math


# --- ZONAL TIME-SERIES (flattened JSON, NaN-safe) ----------------------------
@app.post("/zonal_timeseries.json")
async def zonal_timeseries_json(request: Request):
    """
    Body accepts either:
      - { index, from_date|from, to_date|to, width, height, cloud_mask, bbox:[minLon,minLat,maxLon,maxLat], step_days?:int }
      - { index, from_date|from, to_date|to, width, height, cloud_mask, geometry:<GeoJSON>,                    step_days?:int }
    Returns: { index, from, to, width, height, step_days, series:[ {t0,t1,stats} ... ] }
    """
    def nan_to_none_deep(x):
        import math
        if isinstance(x, float):
            return None if (math.isnan(x) or math.isinf(x)) else x
        if isinstance(x, dict):
            return {k: nan_to_none_deep(v) for k, v in x.items()}
        if isinstance(x, list):
            return [nan_to_none_deep(v) for v in x]
        return x

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    # aliases
    if "from_date" not in body and "from" in body:
        body["from_date"] = body.pop("from")
    if "to_date" not in body and "to" in body:
        body["to_date"] = body.pop("to")

    # required
    for k in ("index", "from_date", "to_date"):
        if k not in body:
            raise HTTPException(status_code=400, detail=f"Missing required field: {k}")

    idx = str(body["index"]).upper()
    if idx not in {"NDVI", "NDWI", "NDBI"}:
        raise HTTPException(status_code=400, detail="index must be one of ['NDVI','NDWI','NDBI']")

    from_iso = _ensure_iso8601_day(str(body["from_date"]), end=False)
    to_iso   = _ensure_iso8601_day(str(body["to_date"]),   end=True)

    width      = int(body.get("width", 256))
    height     = int(body.get("height", 256))
    cloud_mask = bool(body.get("cloud_mask", True))
    step_days  = int(body.get("step_days", 10))
    if width <= 0 or height <= 0 or step_days <= 0:
        raise HTTPException(status_code=400, detail="width/height/step_days must be positive integers")

    geom = body.get("geometry")
    bbox = body.get("bbox")
    if (geom is None and bbox is None) or (geom is not None and bbox is not None):
        raise HTTPException(status_code=400, detail="Provide exactly one of 'geometry' OR 'bbox'")

    # compute via existing helper
    composite = bool(body.get("composite", False))

    try:
        area = area_series_index(
            index=idx,
            from_iso=from_iso, to_iso=to_iso,
            step_days=step_days,
            width=width, height=height,
            bbox=_coerce_bbox(bbox) if bbox is not None else None,
            geometry=geom,
            cloud_mask=cloud_mask,
            composite=composite,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Flatten: keep metadata, but expose the array directly at top-level 'series'
    payload = {
        "index": idx,
        "from": from_iso,
        "to": to_iso,
        "width": width,
        "height": height,
        "step_days": step_days,
        "series": area.get("series", []),
    }
    return JSONResponse(nan_to_none_deep(payload))

import zipfile
import tempfile
import tifffile
@app.post("/zonal_stats.tif")
async def zonal_stats_tif(request: Request):
    """
    Returns raw GeoTIFF (FLOAT32) of the requested index and AOI.
    """
    try:
        body = await request.json()
        body = _parse_zonal_body(body)

        val, mask = _fetch_index_and_mask(
            index=body["index"],
            from_iso=body["from_date"],
            to_iso=body["to_date"],
            width=body["width"],
            height=body["height"],
            bbox=body.get("bbox"),
            geometry=body.get("geometry"),
            cloud_mask=body["cloud_mask"],
        )

        buf = io.BytesIO()
        tifffile.imwrite(buf, np.dstack([val, mask]).astype(np.float32))
        buf.seek(0)

        # Do not close buf — StreamingResponse reads from it asynchronously
        return StreamingResponse(
            buf,
            media_type="image/tiff",
            headers={"Content-Disposition": "attachment; filename=zonal_stats.tif"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/zonal_stats.png")
async def zonal_stats_png(request: Request):
    """
    Returns a color-mapped PNG visualization of NDVI/NDWI/NDBI.
    Applies appropriate color ramp and transparent clouds.
    """
    import numpy as np
    import io
    from matplotlib import cm
    from PIL import Image

    try:
        body = await request.json()
        body = _parse_zonal_body(body)

        result = _fetch_index_and_mask(
            index=body["index"],
            from_iso=body["from_date"],
            to_iso=body["to_date"],
            width=body["width"],
            height=body["height"],
            bbox=body.get("bbox"),
            geometry=body.get("geometry"),
            cloud_mask=body["cloud_mask"],
        )

        if result is None or len(result) != 2:
            raise HTTPException(status_code=500, detail="_fetch_index_and_mask() returned invalid result")

        val, mask = result

        if val is None or not isinstance(val, np.ndarray):
            raise HTTPException(status_code=500, detail="Invalid data returned from _fetch_index_and_mask")

        # Clean and clip index values
        arr = np.copy(val).astype(np.float32)
        arr[np.isnan(arr)] = np.nan
        arr = np.clip(arr, -1, 1)
        # Auto contrast stretch to emphasize differences
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size > 0:
            lo, hi = np.nanpercentile(finite_vals, [5, 95])
            if hi > lo:
                arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 2 - 1

        # Convert mask to bool safely
        if mask is None or mask.shape != arr.shape:
            mask = np.isfinite(arr)
        else:
            mask = np.asarray(mask).astype(bool)

        # Choose a color ramp depending on index
        idx = body["index"].upper()
        cmap = {
            "NDVI": cm.get_cmap("RdYlGn"),
            "NDWI": cm.get_cmap("BrBG"),
            "NDBI": cm.get_cmap("PuOr"),
        }.get(idx, cm.get_cmap("gray"))

        print("NDVI stats:", np.nanmin(arr), np.nanmax(arr), np.nanmean(arr))

        # Normalize and colorize
        rgba = cmap((arr + 1) / 2.0)
        rgba = (rgba * 255).astype(np.uint8)

        # Apply transparency (clouds or NaNs)
        alpha = np.where(mask & np.isfinite(arr), 255, 0).astype(np.uint8)
        rgba[..., 3] = alpha

        img = Image.fromarray(rgba, mode="RGBA")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=zonal_stats.png"},
        )

    except Exception as e:
        import traceback
        print("❌ zonal_stats.png error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"zonal_stats.png failed: {e}")



# --- ZONAL TIME-SERIES ARCHIVE (MULTI-TIFF) ----------------------------------
@app.post("/zonal_timeseries.zip")
async def zonal_timeseries_zip(request: Request):
    """
    Returns a ZIP containing GeoTIFFs for each time bin.
    Each file = index_YYYYMMDD_YYYYMMDD.tif
    """
    import datetime as dt
    try:
        body = await request.json()
        body = _parse_zonal_body(body)
        step_days = int(body.get("step_days", 10))
        t0 = dt.datetime.fromisoformat(body["from_date"].replace("Z","+00:00"))
        t1 = dt.datetime.fromisoformat(body["to_date"].replace("Z","+00:00"))
        step = dt.timedelta(days=max(1, step_days))

        tmpbuf = io.BytesIO()
        with zipfile.ZipFile(tmpbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            cur = t0
            while cur <= t1:
                end = min(cur + step, t1)
                try:
                    val, mask = _fetch_index_and_mask(
                        index=body["index"],
                        from_iso=cur.isoformat().replace("+00:00","Z"),
                        to_iso=end.isoformat().replace("+00:00","Z"),
                        width=body["width"],
                        height=body["height"],
                        bbox=body.get("bbox"),
                        geometry=body.get("geometry"),
                        cloud_mask=body["cloud_mask"],
                    )
                    tifbuf = io.BytesIO()
                    tifffile.imwrite(tifbuf, np.dstack([val, mask]).astype(np.float32))
                    zf.writestr(
                        f"{body['index']}_{cur.date()}_{end.date()}.tif",
                        tifbuf.getvalue()
                    )
                except Exception as e:
                    # skip problematic time bins
                    zf.writestr(f"error_{cur.date()}_{end.date()}.txt", str(e))
                cur = end + dt.timedelta(seconds=1)
        tmpbuf.seek(0)
        return StreamingResponse(
            tmpbuf, media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=zonal_timeseries.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Helpers to render frames -------------------------------------------------
from matplotlib import cm
from modules.sentinel_hub import _get_token, PROCESS_URL

def _colorize_index_rgba(val: np.ndarray, mask: np.ndarray, index_name: str) -> bytes:
    """
    Colorize an index (e.g., NDVI/NDWI/NDBI) into an RGBA PNG.
    - Clips to [-1, 1], then percentile-stretches (5–95%) to enhance contrast.
    - Alpha = cloud-free & finite pixels; everything else transparent.
    """
    import numpy as _np
    from PIL import Image as _Image
    import io as _io

    arr = _np.asarray(val, dtype=_np.float32)
    arr = _np.clip(arr, -1.0, 1.0)

    finite = _np.isfinite(arr)
    if finite.any():
        lo, hi = _np.nanpercentile(arr[finite], [5, 95])
        if hi > lo:
            arr = _np.clip((arr - lo) / (hi - lo), 0, 1) * 2 - 1

    idx = str(index_name).upper()
    cmap = {
        "NDVI": cm.get_cmap("RdYlGn"),
        "NDWI": cm.get_cmap("BrBG"),
        "NDBI": cm.get_cmap("PuOr"),
    }.get(idx, cm.get_cmap("gray"))

    rgba = (cmap((arr + 1) / 2.0) * 255).astype(_np.uint8)
    alpha = _np.where((_np.asarray(mask) > 0.5) & _np.isfinite(arr), 255, 0).astype(_np.uint8)
    rgba[..., 3] = alpha

    im = _Image.fromarray(rgba, mode="RGBA")
    out = _io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


# --- True-color RGBA fetch (PNG from PROCESS) --------------------------------
def _fetch_truecolor_rgba(
    *,
    from_iso: str,
    to_iso: str,
    width: int,
    height: int,
    bbox: Optional[List[float]] = None,
    geometry: Optional[Dict] = None,
    cloud_mask: bool = True,
) -> bytes:
    """
    Returns a PNG (RGBA). Alpha=0 for clouds/shadows if cloud_mask=True.
    """
    import io
    import requests
    from modules.sentinel_hub import _get_token, PROCESS_URL

    from_iso = from_iso if "T" in from_iso else f"{from_iso}T00:00:00Z"
    to_iso   = to_iso   if "T" in to_iso   else f"{to_iso}T23:59:59Z"

    if geometry is not None:
        bounds_obj = {"geometry": geometry,
                      "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}
    elif bbox is not None:
        bounds_obj = {"bbox": bbox,
                      "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}
    else:
        raise ValueError("Provide bbox or geometry")

    # Linear stretch constants for L2A reflectance
    evalscript = f"""//VERSION=3
function setup() {{
  return {{
    input: ["B02","B03","B04","SCL","dataMask"],
    output: {{ bands: 4, sampleType: "UINT8" }},
    mosaicking: "SIMPLE"
  }};
}}

const L = 0.02;     // low cut (2%)
const H = 0.30;     // high cut (30%)
function clamp01(x) {{ return Math.max(0, Math.min(1, x)); }}

function evaluatePixel(s) {{
  // stretch each band to 0..1 then to 0..255
  let r = clamp01((s.B04 - L) / (H - L));
  let g = clamp01((s.B03 - L) / (H - L));
  let b = clamp01((s.B02 - L) / (H - L));

  let ok = s.dataMask > 0{(' && !(s.SCL==3 || s.SCL==8 || s.SCL==9 || s.SCL==10)' if cloud_mask else '')};
  return [Math.round(r*255), Math.round(g*255), Math.round(b*255), ok ? 255 : 0];
}}
"""

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
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": evalscript
    }

    headers = {"Authorization": f"Bearer {_get_token()}"}
    r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"PROCESS error {r.status_code}: {r.text[:400]}")
    return r.content




# --- Minimal time-series renderer (no matplotlib) ----------------------------
def _render_timeseries_png(
    bins: List[Dict[str, Any]],
    *,
    index_name: str,
    y_key: str = "median",         # one of: mean, median, p25, p75, min, max
    size: tuple[int, int] = (900, 420),
    bg_rgba=(250, 250, 245, 255),
) -> bytes:
    """
    Render a time-series PNG for a list of bins like:
    [{"t0": "...Z", "t1": "...Z", "stats": {"mean":..., "median":..., ...}}, ...]
    - Skips bins with no valid stats (count==0 or metric is NaN/None)
    - Draws axes, a polyline, and point markers
    """
    from PIL import Image, ImageDraw
    import datetime as dt
    import numpy as np
    import io as _io

    W, H = size
    padL, padR, padT, padB = 70, 20, 40, 60
    plotW = max(10, W - padL - padR)
    plotH = max(10, H - padT - padB)

    def _is_num(x):
        return isinstance(x, (int, float)) and np.isfinite(x)

    def _mid_time(t0: str | None, t1: str | None):
        if not t0 or not t1:
            return None
        d0 = dt.datetime.fromisoformat(t0.replace("Z", "+00:00"))
        d1 = dt.datetime.fromisoformat(t1.replace("Z", "+00:00"))
        return d0 + (d1 - d0) / 2

    # Collect mid-times & metric values
    times: List[dt.datetime | None] = []
    vals: List[float | None] = []
    for b in bins:
        stats = (b or {}).get("stats") or {}
        v = stats.get(y_key, None)
        v = float(v) if _is_num(v) else None
        vals.append(v)
        times.append(_mid_time(b.get("t0"), b.get("t1")))

    valid_vals = [v for v in vals if v is not None]
    # If absolutely no valid values, draw a friendly message
    if not valid_vals:
        img = Image.new("RGBA", (W, H), bg_rgba)
        draw = ImageDraw.Draw(img, "RGBA")
        msg = "No valid (cloud-free) pixels for this time window."
        draw.text((20, H // 2 - 10), f"{index_name} time series: {msg}", fill=(30, 30, 30, 255))
        out = _io.BytesIO()
        img.save(out, "PNG")
        return out.getvalue()

    # y-range (pad a touch, clamp to [-1,1] for indices)
    y_min = min(valid_vals)
    y_max = max(valid_vals)
    y_min = max(-1.0, y_min - 0.05)
    y_max = min( 1.0, y_max + 0.05)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or np.isclose(y_min, y_max):
        y_min, y_max = -1.0, 1.0

    # x positions are uniform across bins (we don’t need absolute dates for scaling)
    N = max(1, len(bins))

    def _xy(i: int, y: float):
        x = padL + (i / (N - 1 if N > 1 else 1)) * plotW
        yp = padT + (1.0 - ( (y - y_min) / max(1e-9, (y_max - y_min)) )) * plotH
        return (x, yp)

    # Canvas
    img = Image.new("RGBA", (W, H), bg_rgba)
    draw = ImageDraw.Draw(img, "RGBA")

    # Axes & grid
    # y grid at -1, -0.5, 0, 0.5, 1 if inside range
    y_ticks = []
    for t in (-1.0, -0.5, 0.0, 0.5, 1.0):
        if y_min <= t <= y_max:
            y_ticks.append(t)

    # Axes frame
    draw.rectangle([padL, padT, padL + plotW, padT + plotH], outline=(0, 0, 0, 180), width=1)
    # y grid + labels
    for t in y_ticks:
        x0, y0 = padL, _xy(0, t)[1]
        draw.line([(x0, y0), (padL + plotW, y0)], fill=(0, 0, 0, 40), width=1)
        draw.text((8, y0 - 7), f"{t: .2f}", fill=(60, 60, 60, 255))

    # x tick labels: pick ~6 labels max
    label_cnt = min(6, N)
    step = max(1, N // label_cnt)
    for i in range(0, N, step):
        x, _ = _xy(i, y_min)
        draw.line([(x, padT + plotH), (x, padT + plotH + 5)], fill=(0, 0, 0, 120), width=1)
        tt = times[i]
        if tt:
            draw.text((x - 38, padT + plotH + 10), tt.strftime("%Y-%m-%d"), fill=(60, 60, 60, 255))

    # Polyline + markers (break on gaps)
    prev = None
    for i, v in enumerate(vals):
        if v is None:
            prev = None
            continue
        p = _xy(i, v)
        r = 3
        draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=(24, 120, 24, 255))
        if prev is not None:
            draw.line([prev, p], fill=(24, 120, 24, 255), width=2)
        prev = p

    # Title
    draw.text((padL, 12), f"{index_name} time series ({y_key})", fill=(10, 10, 10, 255))

    out = _io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()

# --- ZONAL TIME-SERIES PNG ---------------------------------------------------
@app.post("/zonal_timeseries.png")
async def zonal_timeseries_png(request: Request):
    """
    PNG chart of the zonal (area) time-series for an index.
    Body accepts either:
      { index, from_date|from, to_date|to, step_days?, width, height, cloud_mask?, bbox:[..] }
      or
      { index, from_date|from, to_date|to, step_days?, width, height, cloud_mask?, geometry:{..} }

    Optional rendering controls:
      - y : which statistic to plot ("median" default; one of mean, median, p25, p75, min, max)
      - chart_width  : PNG width  (default 900)
      - chart_height : PNG height (default 420)
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    # Accept from/to aliases
    if "from_date" not in body and "from" in body:
        body["from_date"] = body.pop("from")
    if "to_date" not in body and "to" in body:
        body["to_date"] = body.pop("to")

    # Required
    for k in ("index", "from_date", "to_date"):
        if k not in body:
            raise HTTPException(status_code=400, detail=f"Missing required field: {k}")

    # Time bin + fetch resolution (used by PROCESS request inside area_series_index)
    step_days   = int(body.get("step_days", 10))
    fetch_w     = int(body.get("width", 256))
    fetch_h     = int(body.get("height", 256))
    cloud_mask  = bool(body.get("cloud_mask", True))

    # Chart rendering controls (separate from fetch_w/fetch_h)
    y_key       = str(body.get("y", "median")).lower()
    if y_key not in ("mean", "median", "p25", "p75", "min", "max"):
        y_key = "median"
    chart_w     = int(body.get("chart_width", 900))
    chart_h     = int(body.get("chart_height", 420))

    geom = body.get("geometry")
    bbox = body.get("bbox")
    if (geom is None and bbox is None) or (geom is not None and bbox is not None):
        raise HTTPException(status_code=400, detail="Provide exactly one of 'geometry' OR 'bbox'")

    composite = bool(body.get("composite", True))
    # Compute JSON series (already NaN-safe)
    try:
        out = area_series_index(
            index=body["index"],
            from_iso=body["from_date"],
            to_iso=body["to_date"],
            step_days=step_days,
            width=fetch_w,
            height=fetch_h,
            bbox=bbox,
            geometry=geom,
            cloud_mask=cloud_mask,
            composite=composite,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Render chart
    try:
        png_bytes = _render_timeseries_png(
            out.get("series", []),
            index_name=str(body["index"]).upper(),
            y_key=y_key,
            size=(chart_w, chart_h),
        )
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render error: {e}")

# --- POINT TIME-SERIES PNG (optional) ----------------------------------------
@app.post("/series/point.png")
def post_series_point_png(body: PointSeriesQuery):
    """
    PNG chart for point time-series (same body as /series/point.json).
    Uses the 'value' field from each bin.
    Optional extras in body:
      - chart_width, chart_height
    """
    try:
        out = point_series_index(
            index=body.index,
            lat=body.lat, lon=body.lon,
            from_iso=body.from_date, to_iso=body.to_date,
            step_days=body.step_days, buffer_m=body.buffer_m,
            cloud_mask=body.cloud_mask,
            composite=body.composite,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Adapt bins to the same shape used by the area renderer
    bins = []
    for b in out.get("series", []):
        v = b.get("value", None)
        stats = None
        if v is None:
            stats = {"median": None}
        else:
            # Fake a "stats" object with a single metric so we can reuse the same renderer
            stats = {"median": float(v)}
        bins.append({"t0": b.get("t0"), "t1": b.get("t1"), "stats": stats})

    chart_w = getattr(body, "chart_width", 900)
    chart_h = getattr(body, "chart_height", 420)

    try:
        png_bytes = _render_timeseries_png(
            bins,
            index_name=str(body.index).upper(),
            y_key="median",
            size=(int(chart_w), int(chart_h)),
        )
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render error: {e}")
    


# --- ZONAL TIME-SERIES FRAMES (PNG ZIP) --------------------------------------
@app.post("/zonal_timeseries.frames.zip")
async def zonal_timeseries_frames_zip(request: Request):
    import datetime as dt, zipfile, io as _io, traceback
    from PIL import Image, ImageDraw

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    body = _parse_zonal_body(body)
    step_days = int(body.get("step_days", 10))
    style = str(body.get("style", "index")).lower()
    annotate = bool(body.get("annotate", True))

    t0 = dt.datetime.fromisoformat(body["from_date"].replace("Z","+00:00"))
    t1 = dt.datetime.fromisoformat(body["to_date"].replace("Z","+00:00"))
    step = dt.timedelta(days=max(1, step_days))

    tmp = _io.BytesIO()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        cur = t0
        while cur <= t1:
            end = min(cur + step, t1)
            fname = f"{body['index']}_{cur.date()}_{end.date()}.png"
            try:
                if style == "truecolor":
                    png_bytes = _fetch_truecolor_rgba(
                        from_iso=cur.isoformat().replace("+00:00","Z"),
                        to_iso=end.isoformat().replace("+00:00","Z"),
                        width=body["width"], height=body["height"],
                        bbox=body.get("bbox"), geometry=body.get("geometry"),
                        cloud_mask=body["cloud_mask"],
                    )
                else:
                    val, msk = _fetch_index_and_mask(
                        index=body["index"],
                        from_iso=cur.isoformat().replace("+00:00","Z"),
                        to_iso=end.isoformat().replace("+00:00","Z"),
                        width=body["width"], height=body["height"],
                        bbox=body.get("bbox"), geometry=body.get("geometry"),
                        cloud_mask=body["cloud_mask"],
                    )
                    png_bytes = _colorize_index_rgba(val, msk, body["index"])

                if annotate:
                    im = Image.open(_io.BytesIO(png_bytes)).convert("RGBA")   # <-- FIXED
                    draw = ImageDraw.Draw(im, "RGBA")
                    label = f"{cur.date()} → {end.date()} ({body['index'] if style!='truecolor' else 'TRUECOLOR'})"
                    draw.rectangle([(0, 0), (im.width, 24)], fill=(0, 0, 0, 90))
                    draw.text((6, 4), label, fill=(255, 255, 255, 230))
                    out = _io.BytesIO()                                       # <-- FIXED
                    im.save(out, "PNG")
                    png_bytes = out.getvalue()

                zf.writestr(fname, png_bytes)

            except Exception as e:
                err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(f"[frames.zip] {fname} ERROR:\n{err}")
                zf.writestr(f"errors/{fname.replace('.png','.txt')}", err)

            cur = end + dt.timedelta(seconds=1)

    tmp.seek(0)
    return StreamingResponse(
        tmp, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=zonal_timeseries_frames.zip"}
    )

# --- CONTACT SHEET (grid) ----------------------------------------------------
@app.post("/zonal_timeseries.contact_sheet.png")
async def zonal_timeseries_contact_sheet_png(request: Request):
    import datetime as dt, io as _io, math as _math, traceback
    from PIL import Image, ImageDraw

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    body = _parse_zonal_body(body)
    step_days = int(body.get("step_days", 10))
    cols = int(body.get("columns", 6))
    style = str(body.get("style", "index")).lower()

    t0 = dt.datetime.fromisoformat(body["from_date"].replace("Z","+00:00"))
    t1 = dt.datetime.fromisoformat(body["to_date"].replace("Z","+00:00"))
    step = dt.timedelta(days=max(1, step_days))

    frames, labels = [], []

    cur = t0
    while cur <= t1:
        end = min(cur + step, t1)
        try:
            if style == "truecolor":
                png_bytes = _fetch_truecolor_rgba(
                    from_iso=cur.isoformat().replace("+00:00","Z"),
                    to_iso=end.isoformat().replace("+00:00","Z"),
                    width=body["width"], height=body["height"],
                    bbox=body.get("bbox"), geometry=body.get("geometry"),
                    cloud_mask=body["cloud_mask"],
                )
            else:
                val, msk = _fetch_index_and_mask(
                    index=body["index"],
                    from_iso=cur.isoformat().replace("+00:00","Z"),
                    to_iso=end.isoformat().replace("+00:00","Z"),
                    width=body["width"], height=body["height"],
                    bbox=body.get("bbox"), geometry=body.get("geometry"),
                    cloud_mask=body["cloud_mask"],
                )
                png_bytes = _colorize_index_rgba(val, msk, body["index"])

            frames.append(Image.open(_io.BytesIO(png_bytes)).convert("RGBA"))  # <-- FIXED
            labels.append(f"{cur.date()}→{end.date()}")

        except Exception as e:
            print(f"[contact_sheet] {cur.date()}→{end.date()} ERROR:\n{traceback.format_exc()}")
            ph = Image.new("RGBA", (body["width"], body["height"]), (200,200,200,255))
            # paint “error” band
            d = ImageDraw.Draw(ph, "RGBA")
            d.rectangle([(0, ph.height-24), (ph.width, ph.height)], fill=(80,80,80,255))
            d.text((6, ph.height-20), "error", fill=(240,240,240,255))
            frames.append(ph)
            labels.append("error")

        cur = end + dt.timedelta(seconds=1)

    if not frames:
        raise HTTPException(status_code=400, detail="No frames to render")

    W, H = frames[0].width, frames[0].height
    cols = max(1, cols)
    rows = int(_math.ceil(len(frames) / cols))
    pad, label_h = 6, 18

    sheet = Image.new("RGBA",
                      (cols*W + (cols+1)*pad, rows*(H+label_h) + (rows+1)*pad),
                      (245,245,240,255))
    draw = ImageDraw.Draw(sheet, "RGBA")

    for i, (im, lab) in enumerate(zip(frames, labels)):
        r, c = divmod(i, cols)
        x = pad + c*(W + pad)
        y = pad + r*(H + label_h + pad)
        sheet.paste(im, (x, y), im)
        draw.rectangle([(x, y+H), (x+W, y+H+label_h)], fill=(0,0,0,80))
        draw.text((x+6, y+H+3), lab, fill=(255,255,255,230))

    out = _io.BytesIO()
    sheet.save(out, "PNG")
    out.seek(0)
    return StreamingResponse(out, media_type="image/png")

