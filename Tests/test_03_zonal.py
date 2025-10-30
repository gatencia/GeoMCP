# tests/test_03_zonal.py

import importlib
import pytest


def _test_polygon():
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [15.2, -0.2],
                [15.25, -0.2],
                [15.25, -0.15],
                [15.2, -0.15],
                [15.2, -0.2],
            ]
        ],
    }


def test_zonal_timeseries_exists_and_runs():
    zonal = importlib.import_module("modules.zonal")

    geom = _test_polygon()

    if hasattr(zonal, "zonal_timeseries_index"):
        out = zonal.zonal_timeseries_index(
            index="ndvi",
            geometry=geom,
            from_iso="2025-01-01",
            to_iso="2025-01-15",
            step_days=5,
            cloud_mask=False,
            composite=False,
        )
    elif hasattr(zonal, "zonal_timeseries"):
        out = zonal.zonal_timeseries(
            index="ndvi",
            geometry=geom,
            from_iso="2025-01-01",
            to_iso="2025-01-15",
            step_days=5,
            cloud_mask=False,
            composite=False,
        )
    else:
        pytest.fail("modules.zonal has neither zonal_timeseries_index nor zonal_timeseries")

    assert isinstance(out, dict), "zonal should return a dict/json-like"
    assert "series" in out or "data" in out, "zonal output should contain 'series' or 'data'"