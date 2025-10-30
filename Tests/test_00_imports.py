# tests/test_00_imports.py

import importlib
import pytest


def test_import_server():
    mod = importlib.import_module("server")
    assert hasattr(mod, "app"), "server.py should expose FastAPI app=..."


@pytest.mark.parametrize("name", ["modules.sentinel_hub", "modules.cloudfree", "modules.zonal"])
def test_import_modules(name):
    mod = importlib.import_module(name)
    assert mod is not None


def test_sentinel_has_process_png():
    sh = importlib.import_module("modules.sentinel_hub")
    assert hasattr(sh, "process_png"), "modules/sentinel_hub.py must define process_png(...)"
    assert hasattr(sh, "NDVI_PNG_EVALSCRIPT"), "need NDVI_PNG_EVALSCRIPT for MCP color/preview"


def test_cloudfree_has_something():
    cloudfree = importlib.import_module("modules.cloudfree")
    # we accept either this:
    ok = hasattr(cloudfree, "truecolor_png") or hasattr(cloudfree, "truecolor") or hasattr(cloudfree, "build_cloudfree")
    assert ok, "modules/cloudfree.py must expose one of: truecolor_png, truecolor, build_cloudfree"


def test_zonal_has_timeseries():
    zonal = importlib.import_module("modules.zonal")
    assert hasattr(zonal, "zonal_timeseries_index") or hasattr(zonal, "zonal_timeseries"), \
        "modules/zonal.py must expose zonal_timeseries_index(...) or zonal_timeseries(...)"