import requests
import pytest

BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Tests if the server is running and healthy."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

def test_status_check():
    """Tests the /status endpoint."""
    response = requests.get(f"{BASE_URL}/status")
    assert response.status_code == 200
    # Check for a key that exists in the response
    assert "active" in response.json()

# A sample bounding box for testing image/data endpoints
# Using a small area in San Francisco
SF_BBOX = "-122.4905,37.7522,-122.4505,37.7722"
NY_BBOX = "-74.01,40.70,-73.99,40.72"
DATE_FROM = "2023-01-01"
DATE_TO = "2023-02-01"

@pytest.mark.parametrize("bbox", [SF_BBOX, NY_BBOX])
def test_elevation_png(bbox):
    """Tests the /elevation.png endpoint."""
    params = {"bbox": bbox, "width": 128, "height": 128}
    response = requests.get(f"{BASE_URL}/elevation.png", params=params)
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'

@pytest.mark.parametrize("bbox", [SF_BBOX, NY_BBOX])
def test_ndvi_png(bbox):
    """Tests the /ndvi.png endpoint."""
    params = {
        "bbox": bbox,
        "from": DATE_FROM,
        "to": DATE_TO,
        "width": 128,
        "height": 128,
    }
    response = requests.get(f"{BASE_URL}/ndvi.png", params=params)
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'

@pytest.mark.parametrize("bbox", [SF_BBOX, NY_BBOX])
def test_analyze_ndvi(bbox):
    """Tests the /analyze endpoint."""
    params = {
        "bbox": bbox,
        "from": DATE_FROM,
        "to": DATE_TO,
        "width": 128,
        "height": 128,
    }
    response = requests.get(f"{BASE_URL}/analyze", params=params)
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "mean" in data["stats"]

def test_zonal_stats_json_bbox():
    """Tests the /zonal_stats.json endpoint with a bbox."""
    payload = {
        "index": "NDVI",
        "from_date": DATE_FROM,
        "to_date": DATE_TO,
        "bbox": [float(x) for x in SF_BBOX.split(",")],
    }
    response = requests.post(f"{BASE_URL}/zonal_stats.json", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    # Correct the assertion to check the nested stats object
    assert "mean" in data["stats"]["stats"]

def test_zonal_timeseries_json_bbox():
    """Tests the /zonal_timeseries.json endpoint with a bbox."""
    payload = {
        "index": "NDVI",
        "from_date": "2023-01-01",
        "to_date": "2023-03-01",
        "step_days": 30,
        "bbox": [float(x) for x in SF_BBOX.split(",")],
    }
    response = requests.post(f"{BASE_URL}/zonal_timeseries.json", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "series" in data
    assert len(data["series"]) > 0
