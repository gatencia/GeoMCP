# Tests/test_elevation.py
"""
Tests for elevation-related endpoints.
"""
import pytest
import io
from PIL import Image


class TestElevationPNG:
    """Tests for /elevation.png endpoint."""

    def test_elevation_png_basic(self, client, sample_bbox, small_dimensions):
        """Test basic elevation PNG request."""
        response = client.get(
            f"/elevation.png?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

        # Verify it's a valid PNG
        img = Image.open(io.BytesIO(response.content))
        assert img.format == "PNG"
        assert img.size == (small_dimensions['width'], small_dimensions['height'])

    def test_elevation_png_invalid_bbox(self, client):
        """Test elevation PNG with invalid bbox."""
        response = client.get("/elevation.png?bbox=invalid")
        assert response.status_code == 400


class TestElevationRaw:
    """Tests for /elevation/raw endpoint."""

    def test_elevation_raw_tiff(self, client, sample_bbox, small_dimensions):
        """Test raw elevation TIFF request."""
        response = client.get(
            f"/elevation/raw?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"


class TestElevationMatrix:
    """Tests for /elevation.matrix endpoint."""

    def test_elevation_matrix(self, client, sample_bbox):
        """Test elevation matrix JSON response."""
        response = client.get(f"/elevation.matrix?bbox={sample_bbox}&width=32&height=32")
        assert response.status_code == 200

        data = response.json()
        assert "bbox" in data
        assert "width" in data
        assert "height" in data
        assert "units" in data
        assert "elevation" in data

        assert data["width"] == 32
        assert data["height"] == 32
        assert data["units"] == "meters"
        assert isinstance(data["elevation"], list)
        assert len(data["elevation"]) == 32  # 32 rows


class TestElevationGradient:
    """Tests for /elevation/gradient endpoint."""

    def test_elevation_gradient(self, client, sample_bbox, small_dimensions):
        """Test elevation gradient (hillshade) PNG."""
        response = client.get(
            f"/elevation/gradient?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


class TestElevationVectors:
    """Tests for /elevation/vectors endpoints."""

    def test_elevation_vectors_png(self, client, sample_bbox, small_dimensions):
        """Test elevation vectors PNG."""
        response = client.get(
            f"/elevation/vectors.png?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}&step=10"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_elevation_vectors_tif(self, client, sample_bbox, small_dimensions):
        """Test elevation vectors TIFF."""
        response = client.get(
            f"/elevation/vectors.tif?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_elevation_vectors_matrix(self, client, sample_bbox):
        """Test elevation vectors matrix."""
        response = client.get(f"/elevation/vectors.matrix?bbox={sample_bbox}&width=32&height=32&step=5")
        assert response.status_code == 200

        data = response.json()
        assert "step" in data
        assert "U" in data
        assert "V" in data
        assert isinstance(data["U"], list)
        assert isinstance(data["V"], list)
