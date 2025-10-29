# Tests/conftest.py
"""
Pytest configuration and shared fixtures for GeoMCP tests.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path so we can import server
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import app


@pytest.fixture(scope="session")
def client():
    """Create a FastAPI test client for the entire test session."""
    return TestClient(app)


@pytest.fixture
def sample_bbox():
    """Sample bounding box (San Francisco area)."""
    return "-122.5,37.5,-122.4,37.6"


@pytest.fixture
def sample_bbox_list():
    """Sample bounding box as list."""
    return [-122.5, 37.5, -122.4, 37.6]


@pytest.fixture
def sample_date_range():
    """Sample date range for testing."""
    return {
        "from": "2024-01-01",
        "to": "2024-01-31"
    }


@pytest.fixture
def sample_geometry():
    """Sample GeoJSON polygon for testing."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [-122.5, 37.5],
            [-122.4, 37.5],
            [-122.4, 37.6],
            [-122.5, 37.6],
            [-122.5, 37.5]
        ]]
    }


@pytest.fixture
def small_dimensions():
    """Small dimensions for faster tests."""
    return {"width": 64, "height": 64}


@pytest.fixture
def sample_training_points():
    """Sample training points for supervised classification."""
    return [
        {"lat": 37.55, "lon": -122.45, "label": 0},  # water
        {"lat": 37.56, "lon": -122.46, "label": 1},  # vegetation
        {"lat": 37.54, "lon": -122.44, "label": 2},  # urban
    ]
