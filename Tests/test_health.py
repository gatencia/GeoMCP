# Tests/test_health.py
"""
Tests for health and status endpoints.
"""
import pytest


def test_health_endpoint(client):
    """Test /health endpoint returns ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "ok" in data
    assert data["ok"] is True


def test_status_endpoint(client):
    """Test /status endpoint returns token status."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()

    # Check expected fields exist
    assert "token_cached" in data
    assert "expires_in_sec" in data
    assert "minutes_remaining" in data
    assert "active" in data

    # Type checks
    assert isinstance(data["token_cached"], bool)
    assert isinstance(data["expires_in_sec"], (int, float))
    assert isinstance(data["minutes_remaining"], (int, float))
    assert isinstance(data["active"], bool)
