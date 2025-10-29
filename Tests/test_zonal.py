# Tests/test_zonal.py
"""
Tests for zonal statistics and time-series endpoints.
"""
import pytest


class TestZonalStats:
    """Tests for /zonal_stats.* endpoints."""

    def test_zonal_stats_json_bbox(self, client, sample_bbox_list, sample_date_range):
        """Test zonal stats JSON with bbox."""
        payload = {
            "index": "NDVI",
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "bbox": sample_bbox_list,
            "width": 64,
            "height": 64,
            "cloud_mask": True
        }

        response = client.post("/zonal_stats.json", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "index" in data
        assert "from" in data
        assert "to" in data
        assert "stats" in data

        stats = data["stats"]
        assert "mean" in stats or "median" in stats
        # Other expected fields depend on implementation

    def test_zonal_stats_json_geometry(self, client, sample_geometry, sample_date_range):
        """Test zonal stats JSON with GeoJSON geometry."""
        payload = {
            "index": "NDWI",
            "from_date": sample_date_range["from"],
            "to_date": sample_date_range["to"],
            "geometry": sample_geometry,
            "width": 64,
            "height": 64,
            "cloud_mask": True
        }

        response = client.post("/zonal_stats.json", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["index"] == "NDWI"

    def test_zonal_stats_json_invalid_index(self, client, sample_bbox_list, sample_date_range):
        """Test zonal stats with invalid index (should error)."""
        payload = {
            "index": "INVALID_INDEX",
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "bbox": sample_bbox_list
        }

        response = client.post("/zonal_stats.json", json=payload)
        assert response.status_code == 400

    def test_zonal_stats_json_missing_fields(self, client, sample_bbox_list):
        """Test zonal stats with missing required fields."""
        payload = {
            "bbox": sample_bbox_list
            # Missing index, from, to
        }

        response = client.post("/zonal_stats.json", json=payload)
        assert response.status_code == 400

    def test_zonal_stats_json_both_bbox_and_geometry(self, client, sample_bbox_list, sample_geometry, sample_date_range):
        """Test zonal stats with both bbox and geometry (should error)."""
        payload = {
            "index": "NDVI",
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "bbox": sample_bbox_list,
            "geometry": sample_geometry  # Can't have both
        }

        response = client.post("/zonal_stats.json", json=payload)
        assert response.status_code == 400

    def test_zonal_stats_tif(self, client, sample_bbox_list, sample_date_range):
        """Test zonal stats TIFF export."""
        payload = {
            "index": "NDVI",
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "bbox": sample_bbox_list,
            "width": 64,
            "height": 64
        }

        response = client.post("/zonal_stats.tif", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_zonal_stats_png(self, client, sample_bbox_list, sample_date_range):
        """Test zonal stats PNG visualization."""
        payload = {
            "index": "NDVI",
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "bbox": sample_bbox_list,
            "width": 64,
            "height": 64
        }

        response = client.post("/zonal_stats.png", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


class TestZonalTimeSeries:
    """Tests for /zonal_timeseries.* endpoints."""

    def test_zonal_timeseries_json(self, client, sample_bbox_list, sample_date_range):
        """Test zonal time-series JSON."""
        payload = {
            "index": "NDVI",
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "bbox": sample_bbox_list,
            "width": 64,
            "height": 64,
            "step_days": 7
        }

        response = client.post("/zonal_timeseries.json", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "index" in data
        assert "from" in data
        assert "to" in data
        assert "step_days" in data
        assert "series" in data

        assert isinstance(data["series"], list)
        if len(data["series"]) > 0:
            # Check first entry structure
            entry = data["series"][0]
            assert "t0" in entry or "stats" in entry

    def test_zonal_timeseries_png(self, client, sample_bbox_list, sample_date_range):
        """Test zonal time-series PNG chart."""
        payload = {
            "index": "NDWI",
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "bbox": sample_bbox_list,
            "width": 64,
            "height": 64,
            "step_days": 10,
            "chart_width": 800,
            "chart_height": 400
        }

        response = client.post("/zonal_timeseries.png", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_zonal_timeseries_zip(self, client, sample_bbox_list, sample_date_range):
        """Test zonal time-series ZIP of TIFFs."""
        payload = {
            "index": "NDVI",
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "bbox": sample_bbox_list,
            "width": 64,
            "height": 64,
            "step_days": 15
        }

        response = client.post("/zonal_timeseries.zip", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        # Verify we got some content
        assert len(response.content) > 0


class TestPointSeries:
    """Tests for /series/point.png endpoint."""

    def test_point_series_png(self, client, sample_date_range):
        """Test point time-series PNG chart."""
        payload = {
            "index": "NDVI",
            "lat": 37.55,
            "lon": -122.45,
            "from_date": sample_date_range["from"],
            "to_date": sample_date_range["to"],
            "step_days": 10,
            "buffer_m": 30.0,
            "cloud_mask": True
        }

        response = client.post("/series/point.png", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_point_series_invalid_index(self, client, sample_date_range):
        """Test point series with invalid index."""
        payload = {
            "index": "INVALID",
            "lat": 37.55,
            "lon": -122.45,
            "from_date": sample_date_range["from"],
            "to_date": sample_date_range["to"]
        }

        response = client.post("/series/point.png", json=payload)
        # Depends on validation - might be 400 or 422
        assert response.status_code in [400, 422, 500]
