# Tests/test_cloudfree.py
"""
Tests for cloud-free composite endpoints.
"""
import pytest


class TestCloudfreeTrue color:
    """Tests for /cloudfree/truecolor.png endpoint."""

    def test_cloudfree_truecolor_basic(self, client, sample_bbox, sample_date_range, small_dimensions):
        """Test basic cloud-free truecolor request."""
        response = client.get(
            f"/cloudfree/truecolor.png?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_cloudfree_truecolor_custom_maxcc(self, client, sample_bbox, sample_date_range, small_dimensions):
        """Test cloud-free truecolor with custom max cloud coverage."""
        response = client.get(
            f"/cloudfree/truecolor.png?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width={small_dimensions['width']}&height={small_dimensions['height']}&maxcc=20"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_cloudfree_truecolor_opaque(self, client, sample_bbox, sample_date_range, small_dimensions):
        """Test cloud-free truecolor with opaque flag."""
        response = client.get(
            f"/cloudfree/truecolor.png?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width={small_dimensions['width']}&height={small_dimensions['height']}&opaque=true"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


class TestCloudfreeNDVI:
    """Tests for /cloudfree/ndvi.png endpoint."""

    def test_cloudfree_ndvi_basic(self, client, sample_bbox, sample_date_range, small_dimensions):
        """Test basic cloud-free NDVI request."""
        response = client.get(
            f"/cloudfree/ndvi.png?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_cloudfree_ndvi_custom_params(self, client, sample_bbox, sample_date_range, small_dimensions):
        """Test cloud-free NDVI with custom parameters."""
        response = client.get(
            f"/cloudfree/ndvi.png?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width={small_dimensions['width']}&height={small_dimensions['height']}&maxcc=15&opaque=true"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


class TestCloudfreeNDWI:
    """Tests for /cloudfree/ndwi.png endpoint."""

    def test_cloudfree_ndwi_basic(self, client, sample_bbox, sample_date_range, small_dimensions):
        """Test basic cloud-free NDWI request."""
        response = client.get(
            f"/cloudfree/ndwi.png?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width={small_dimensions['width']}&height={small_dimensions['height']}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_cloudfree_ndwi_missing_dates(self, client, sample_bbox):
        """Test cloud-free NDWI with missing dates (should error)."""
        response = client.get(f"/cloudfree/ndwi.png?bbox={sample_bbox}")
        assert response.status_code == 422  # FastAPI validation error
