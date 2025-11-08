# modules/ndvi.py
from __future__ import annotations

import io
from typing import Dict, List, Optional

import numpy as np
import requests
import tifffile
from fastapi import HTTPException

from modules.sentinel_hub import _get_token, PROCESS_URL


NDVI_TIFF_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
	input: ["B04", "B08"],
	output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  let red = s.B04;
  let nir = s.B08;
  let denom = red + nir;
  if (Math.abs(denom) < 1e-6) {
	return [0.0];
  }
  return [(nir - red) / denom];
}
"""


def _normalize_iso(date_str: Optional[str], *, is_end: bool = False) -> Optional[str]:
	if not date_str:
		return None
	if "T" in date_str:
		return date_str
	return f"{date_str}T23:59:59Z" if is_end else f"{date_str}T00:00:00Z"


def _build_process_body(
	*,
	bbox: List[float],
	width: int,
	height: int,
	from_iso: Optional[str],
	to_iso: Optional[str],
	maxcc: int,
) -> Dict[str, object]:
	data_filter: Dict[str, object] = {
		"mosaickingOrder": "leastCC",
		"maxCloudCoverage": int(maxcc),
	}

	start = _normalize_iso(from_iso, is_end=False)
	end = _normalize_iso(to_iso, is_end=True)
	if start and end:
		data_filter["timeRange"] = {"from": start, "to": end}

	return {
		"input": {
			"bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
			"data": [{"type": "S2L2A", "dataFilter": data_filter}],
		},
		"output": {
			"width": int(width),
			"height": int(height),
			"responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
		},
		"evalscript": NDVI_TIFF_EVALSCRIPT,
	}


def _post_process(body: Dict[str, object]) -> bytes:
	headers = {"Authorization": f"Bearer {_get_token()}"}
	r = requests.post(PROCESS_URL, headers=headers, json=body, timeout=180)
	if r.status_code != 200:
		raise HTTPException(status_code=r.status_code, detail=r.text[:300])
	return r.content


def get_ndvi_tiff(
	bbox: List[float],
	*,
	width: int = 512,
	height: int = 512,
	from_iso: Optional[str] = None,
	to_iso: Optional[str] = None,
	maxcc: int = 20,
) -> bytes:
	body = _build_process_body(
		bbox=bbox,
		width=width,
		height=height,
		from_iso=from_iso,
		to_iso=to_iso,
		maxcc=maxcc,
	)
	return _post_process(body)


def get_ndvi_matrix(
	bbox: List[float],
	*,
	width: int = 256,
	height: int = 256,
	from_iso: Optional[str] = None,
	to_iso: Optional[str] = None,
	maxcc: int = 20,
) -> Dict[str, object]:
	tiff_bytes = get_ndvi_tiff(
		bbox,
		width=width,
		height=height,
		from_iso=from_iso,
		to_iso=to_iso,
		maxcc=maxcc,
	)

	arr = tifffile.imread(io.BytesIO(tiff_bytes)).astype(np.float32)
	arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

	return {
		"bbox": bbox,
		"width": int(arr.shape[1]),
		"height": int(arr.shape[0]),
		"matrix": arr.tolist(),
		"index": "NDVI",
		"range": [-1.0, 1.0],
		"max_cloud_coverage": int(maxcc),
		"time_range": {"from": from_iso, "to": to_iso},
	}
