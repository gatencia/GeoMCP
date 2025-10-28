# server.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List
import io
import numpy as np
from PIL import Image
from modules.elevation import (
    get_elevation, 
    get_elevation_raw, 
    compute_hillshade_from_tiff,
    compute_slope_vector_field
)

from modules.status import get_status
from modules.sentinel_hub import process_png, NDVI_PNG_EVALSCRIPT


app = FastAPI(title="GeoMCP - Satellite MCP Server")

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