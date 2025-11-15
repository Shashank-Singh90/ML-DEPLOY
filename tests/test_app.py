import pytest
from fastapi.testclient import TestClient
from typing import Iterator

from app.main import app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Provide a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""

    response = client.get("/health")
    assert response.status_code in {200, 503}
    data = response.json()
    assert data["status"] in {"healthy", "degraded"}


def test_home_endpoint(client: TestClient) -> None:
    """Test home endpoint."""

    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "IoT Threat Detection API"
    assert data["status"] == "running"


def test_predict_valid_input(client: TestClient) -> None:
    """Test simple prediction with valid input."""

    test_data = {
        "packet_count": 100,
        "byte_count": 50000,
        "duration": 5.0,
        "syn_flags": 2,
        "fin_flags": 1,
        "ack_flags": 10,
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "threat_score" in data


def test_predict_invalid_input(client: TestClient) -> None:
    """Test prediction with invalid input."""

    test_data = {
        "packet_count": -100,  # Invalid negative value
        "byte_count": 50000,
        "duration": 5.0,
        "syn_flags": 2,
        "fin_flags": 1,
        "ack_flags": 10,
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "validation_failed"
    assert "error" in data


def test_predict_missing_features(client: TestClient) -> None:
    """Test prediction with missing features."""

    test_data = {
        "packet_count": 100
        # Missing required fields
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "validation_failed"
    assert "missing_fields" in data