# tests/test_01_sentinel_ndvi.py

import importlib
import os
import pytest


@pytest.mark.skipif(
    os.environ.get("SENTINELHUB_CLIENT_ID") is None,
    reason="Sentinel credentials not set"
)
def test_ndvi_png_small_bbox(tmp_path):
    sh = importlib.import_module("modules.sentinel_hub")

    bbox = [15.2, -0.2, 15.25, -0.15]  # your Congo-ish test box
    png_bytes = sh.process_png(
        bbox=bbox,
        from_iso="2025-01-01",
        to_iso="2025-01-05",
        width=128,
        height=128,
        evalscript=sh.NDVI_PNG_EVALSCRIPT,
        collection="S2L2A",
    )

    out = tmp_path / "ndvi.png"
    out.write_bytes(png_bytes)

    assert len(png_bytes) > 100, "NDVI PNG should not be empty"