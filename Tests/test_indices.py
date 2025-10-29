# Tests/test_indices.py
"""
Tests for vegetation and spectral indices (NDVI, NDWI, NDBI).
"""
import pytest
import io
from PIL import Image


class TestNDVI:
    """Tests for NDVI endpoints."""

    def test_ndvi_png(self, client, sample_bbox, sample_date_range, small_dimensions):
        """Test NDVI PNG endpoint."""
        response = client.get(
            f"/ndvi.png?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

        # Verify it's a valid image
        img = Image.open(io.BytesIO(response.content))
        assert img.format == "PNG"

    def test_ndvi_analyze(self, client, sample_bbox, sample_date_range, small_dimensions):
        """Test NDVI analyze endpoint."""
        response = client.get(
            f"/analyze?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200

        data = response.json()
        assert "bbox" in data
        assert "from" in data
        assert "to" in data
        assert "stats" in data

        stats = data["stats"]
        assert "mean" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p25" in stats
        assert "p75" in stats

        # Verify NDVI values are in valid range [-1, 1]
        assert -1 <= stats["min"] <= 1
        assert -1 <= stats["max"] <= 1

    def test_ndvi_invalid_date_range(self, client, sample_bbox):
        """Test NDVI with missing dates."""
        response = client.get(f"/ndvi.png?bbox={sample_bbox}")
        assert response.status_code == 422  # FastAPI validation error


class TestNDWI:
    """Tests for NDWI endpoints."""

    def test_ndwi_png(self, client, sample_bbox, small_dimensions):
        """Test NDWI PNG endpoint."""
        response = client.get(
            f"/ndwi.png?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_ndwi_tif(self, client, sample_bbox, small_dimensions):
        """Test NDWI TIFF endpoint."""
        response = client.get(
            f"/ndwi.tif?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_ndwi_matrix(self, client, sample_bbox):
        """Test NDWI matrix endpoint."""
        response = client.get(f"/ndwi.matrix?bbox={sample_bbox}&width=32&height=32")
        assert response.status_code == 200

        data = response.json()
        assert "bbox" in data
        assert "width" in data
        assert "height" in data
        assert "range" in data
        assert "index" in data
        assert "values" in data

        assert data["index"] == "NDWI"
        assert data["range"] == [-1.0, 1.0]
        assert isinstance(data["values"], list)


class TestNDBI:
    """Tests for NDBI (built-up index) endpoints."""

    def test_ndbi_png(self, client, sample_bbox, small_dimensions):
        """Test NDBI PNG endpoint."""
        response = client.get(
            f"/ndbi.png?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_ndbi_png_with_dates(self, client, sample_bbox, sample_date_range, small_dimensions):
        """Test NDBI PNG with date range."""
        response = client.get(
            f"/ndbi.png?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_ndbi_tif(self, client, sample_bbox, small_dimensions):
        """Test NDBI TIFF endpoint."""
        response = client.get(
            f"/ndbi.tif?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_ndbi_matrix(self, client, sample_bbox):
        """Test NDBI matrix endpoint."""
        response = client.get(f"/ndbi.matrix?bbox={sample_bbox}&width=32&height=32")
        assert response.status_code == 200

        data = response.json()
        assert "bbox" in data
        assert "width" in data
        assert "height" in data
        assert "index" in data
        assert "values" in data

        assert data["index"] == "NDBI"
