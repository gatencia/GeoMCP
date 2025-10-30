# tests/test_02_cloudfree.py

import importlib
import os
import pytest


@pytest.mark.skipif(
    os.environ.get("SENTINELHUB_CLIENT_ID") is None,
    reason="Sentinel credentials not set"
)
def test_cloudfree_local_or_skip(tmp_path):
    cloudfree = importlib.import_module("modules.cloudfree")

    bbox = [15.2, -0.2, 15.25, -0.15]

    if hasattr(cloudfree, "truecolor_png"):
        png_bytes = cloudfree.truecolor_png(
            bbox=bbox,
            from_iso="2025-01-01",
            to_iso="2025-01-31",
            width=256,
            height=256,
            maxcc=20,
        )
        (tmp_path / "cloudfree.png").write_bytes(png_bytes)
        assert len(png_bytes) > 100
    else:
        pytest.skip("modules.cloudfree.truecolor_png missing → need HTTP fallback in MCP")