from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List
import io
import numpy as np
from PIL import Image

# Core NDVI util (unchanged)
from modules.sentinel_hub import process_png, NDVI_PNG_EVALSCRIPT

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