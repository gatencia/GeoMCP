# Tests/test_classification.py
"""
Tests for land cover classification endpoints.
"""
import pytest


class TestRuleBasedClassification:
    """Tests for rule-based classification endpoints."""

    def test_rule_based_png(self, client, sample_bbox, sample_date_range):
        """Test rule-based classification PNG."""
        response = client.get(
            f"/classify/rule_based.png?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width=64&height=64"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_rule_based_tif(self, client, sample_bbox, sample_date_range):
        """Test rule-based classification TIFF."""
        response = client.get(
            f"/classify/rule_based.tif?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width=64&height=64"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_rule_based_matrix(self, client, sample_bbox, sample_date_range):
        """Test rule-based classification matrix."""
        response = client.get(
            f"/classify/rule_based.matrix?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&width=32&height=32"
        )
        assert response.status_code == 200

        data = response.json()
        assert "labels" in data or "classes" in data or "classification" in data
        # Exact structure depends on implementation


class TestUnsupervisedClassification:
    """Tests for unsupervised (K-means) classification endpoints."""

    def test_unsupervised_png(self, client, sample_bbox, sample_date_range):
        """Test unsupervised classification PNG."""
        response = client.get(
            f"/classify/unsupervised.png?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&k=4&width=64&height=64"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_unsupervised_tif(self, client, sample_bbox, sample_date_range):
        """Test unsupervised classification TIFF."""
        response = client.get(
            f"/classify/unsupervised.tif?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&k=6&width=64&height=64"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_unsupervised_matrix(self, client, sample_bbox, sample_date_range):
        """Test unsupervised classification matrix."""
        response = client.get(
            f"/classify/unsupervised.matrix?bbox={sample_bbox}&from={sample_date_range['from']}&to={sample_date_range['to']}&k=5&width=32&height=32"
        )
        assert response.status_code == 200

        data = response.json()
        # Exact structure depends on implementation
        assert isinstance(data, dict)


class TestSupervisedClassification:
    """Tests for supervised classification endpoints."""

    def test_supervised_png(self, client, sample_bbox_list, sample_date_range, sample_training_points):
        """Test supervised classification PNG."""
        payload = {
            "bbox": sample_bbox_list,
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "width": 64,
            "height": 64,
            "training_points": sample_training_points
        }

        response = client.post("/classify/supervised.png", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_supervised_tif(self, client, sample_bbox_list, sample_date_range, sample_training_points):
        """Test supervised classification TIFF."""
        payload = {
            "bbox": sample_bbox_list,
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "width": 64,
            "height": 64,
            "training_points": sample_training_points
        }

        response = client.post("/classify/supervised.tif", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    def test_supervised_matrix(self, client, sample_bbox_list, sample_date_range, sample_training_points):
        """Test supervised classification matrix."""
        payload = {
            "bbox": sample_bbox_list,
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "width": 32,
            "height": 32,
            "training_points": sample_training_points
        }

        response = client.post("/classify/supervised.matrix", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)

    def test_supervised_missing_training_points(self, client, sample_bbox_list, sample_date_range):
        """Test supervised classification without training points (should error)."""
        payload = {
            "bbox": sample_bbox_list,
            "from": sample_date_range["from"],
            "to": sample_date_range["to"],
            "width": 64,
            "height": 64
        }

        response = client.post("/classify/supervised.png", json=payload)
        assert response.status_code == 422  # Validation error
