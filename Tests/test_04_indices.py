import importlib
import os

import numpy as np
import pytest

IOWA_CROP_BBOX = [-93.776, 41.939, -93.722, 41.985]
FROM_ISO = "2024-07-01"
TO_ISO = "2024-07-10"
IMAGE_SIZE = 64
MAX_CC = 50


@pytest.mark.skipif(
    os.environ.get("SENTINELHUB_CLIENT_ID") is None,
    reason="Sentinel credentials not set",
)
@pytest.mark.parametrize(
    "func_name, expected_index",
    [
        ("get_ndre_matrix", "NDRE"),
        ("get_evi_matrix", "EVI"),
        ("get_msavi_matrix", "MSAVI"),
        ("get_nbr_matrix", "NBR"),
    ],
)
def test_indices_matrix_shapes(func_name: str, expected_index: str):
    indices_mod = importlib.import_module("modules.indices")

    func = getattr(indices_mod, func_name, None)
    assert callable(func), f"modules.indices must expose {func_name}"

    result = func(
        bbox=IOWA_CROP_BBOX,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        from_iso=FROM_ISO,
        to_iso=TO_ISO,
        maxcc=MAX_CC,
    )

    assert isinstance(result, dict), "tool output should be a dict"
    assert result.get("index") == expected_index

    matrix = result.get("matrix")
    assert isinstance(matrix, list) and matrix, "matrix payload must be a non-empty list"
    assert len(matrix) == IMAGE_SIZE, "matrix rows must match requested height"
    assert all(isinstance(row, list) for row in matrix), "matrix rows must be lists"
    assert all(len(row) == IMAGE_SIZE for row in matrix), "matrix cols must match requested width"

    np_matrix = np.asarray(matrix, dtype=np.float32)
    assert np_matrix.shape == (IMAGE_SIZE, IMAGE_SIZE), "matrix must convert to IMAGE_SIZE x IMAGE_SIZE"
    assert np.issubdtype(np_matrix.dtype, np.floating), "matrix entries must be floats"
    assert np.all(np.isfinite(np_matrix)), "matrix must not contain NaN or Inf"
    assert np.count_nonzero(np_matrix) > 0, "crop tile should contain non-zero signal"

    time_range = result.get("time_range", {})
    assert time_range.get("from") == FROM_ISO
    assert time_range.get("to") == TO_ISO
    assert result.get("max_cloud_coverage") == MAX_CC
