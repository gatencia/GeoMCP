# Tests/test_terrain.py
"""
Tests for terrain analysis endpoints (slope, aspect, hillshade, flow).
"""
import pytest


class TestSlope:
    """Tests for slope endpoints."""

    def test_slope_png(self, client, sample_bbox, small_dimensions):
        """Test slope PNG endpoint."""
        response = client.get(
            f"/slope.png?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_slope_png_custom_vmax(self, client, sample_bbox, small_dimensions):
        """Test slope PNG with custom vmax parameter."""
        response = client.get(
            f"/slope.png?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}&vmax=45.0"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_slope_tif(self, client, sample_bbox, small_dimensions):
        """Test slope TIFF endpoint."""
        response = client.get(
            f"/slope.tif?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_slope_matrix(self, client, sample_bbox):
        """Test slope matrix endpoint."""
        response = client.get(f"/slope.matrix?bbox={sample_bbox}&width=32&height=32")
        assert response.status_code == 200

        data = response.json()
        assert "bbox" in data
        assert "width" in data
        assert "height" in data
        assert "units" in data
        assert "slope" in data

        assert data["units"] == "degrees"
        assert isinstance(data["slope"], list)


class TestAspect:
    """Tests for aspect endpoints."""

    def test_aspect_tif(self, client, sample_bbox, small_dimensions):
        """Test aspect TIFF endpoint."""
        response = client.get(
            f"/aspect.tif?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_aspect_matrix(self, client, sample_bbox):
        """Test aspect matrix endpoint."""
        response = client.get(f"/aspect.matrix?bbox={sample_bbox}&width=32&height=32")
        assert response.status_code == 200

        data = response.json()
        assert "bbox" in data
        assert "width" in data
        assert "height" in data
        assert "units" in data
        assert "aspect" in data

        assert data["units"] == "degrees_clockwise_from_north"
        assert isinstance(data["aspect"], list)


class TestHillshade:
    """Tests for hillshade endpoints."""

    def test_hillshade_png(self, client, sample_bbox, small_dimensions):
        """Test hillshade PNG endpoint."""
        response = client.get(
            f"/hillshade.png?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_hillshade_png_custom_angles(self, client, sample_bbox, small_dimensions):
        """Test hillshade PNG with custom azimuth and altitude."""
        response = client.get(
            f"/hillshade.png?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}&azimuth_deg=270&altitude_deg=30"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_hillshade_tif(self, client, sample_bbox, small_dimensions):
        """Test hillshade TIFF endpoint."""
        response = client.get(
            f"/hillshade.tif?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_hillshade_matrix(self, client, sample_bbox):
        """Test hillshade matrix endpoint."""
        response = client.get(f"/hillshade.matrix?bbox={sample_bbox}&width=32&height=32")
        assert response.status_code == 200

        data = response.json()
        assert "range" in data
        assert "hillshade" in data

        assert data["range"] == [0.0, 1.0]
        assert isinstance(data["hillshade"], list)


class TestFlowAccumulation:
    """Tests for flow accumulation endpoints."""

    def test_flow_accumulation_png(self, client, sample_bbox, small_dimensions):
        """Test flow accumulation PNG endpoint."""
        response = client.get(
            f"/flow/accumulation.png?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_flow_accumulation_tif(self, client, sample_bbox, small_dimensions):
        """Test flow accumulation TIFF endpoint."""
        response = client.get(
            f"/flow/accumulation.tif?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_flow_accumulation_matrix(self, client, sample_bbox):
        """Test flow accumulation matrix endpoint."""
        response = client.get(f"/flow/accumulation.matrix?bbox={sample_bbox}&width=32&height=32")
        assert response.status_code == 200

        data = response.json()
        assert "accumulation_log" in data
        assert isinstance(data["accumulation_log"], list)
