"""Comprehensive endpoint tests for IoT Threat Detection API"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Provide a FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


class TestAllEndpoints:
    """Test all API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "IoT Threat Detection API"
        assert "endpoints" in data
        assert "/predict" in data["endpoints"]

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code in {200, 503}
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data

    def test_predict_endpoint_valid(self, client):
        """Test predict endpoint with valid data."""
        data = {
            "packet_count": 150,
            "byte_count": 75000,
            "duration": 5.0,
            "syn_flags": 3,
            "fin_flags": 2,
            "ack_flags": 12,
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result
        assert "confidence" in result
        assert "threat_score" in result
        assert "risk_level" in result
        assert result["prediction"] in ["threat", "normal"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["threat_score"] <= 1.0

    def test_predict_missing_field(self, client):
        """Test predict with missing required field."""
        data = {
            "packet_count": 100,
            # Missing other required fields
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 400
        assert "missing_fields" in response.json()

    def test_explain_endpoint(self, client):
        """Test explain endpoint."""
        data = {
            "packet_count": 150,
            "byte_count": 75000,
            "duration": 5.0,
            "syn_flags": 3,
            "fin_flags": 2,
            "ack_flags": 12,
        }
        response = client.post("/explain", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "explanation" in result
        assert "status" in result

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code in {200, 503}
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "feature_count" in data
            assert "feature_names" in data

    def test_stats_endpoint(self, client):
        """Test statistics endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "total_predictions" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        # Check for expected metrics
        metrics_text = response.text
        assert "iot_predictions_total" in metrics_text
        assert "iot_prediction_duration_seconds" in metrics_text

    def test_mlflow_info_endpoint(self, client):
        """Test MLflow info endpoint."""
        response = client.get("/mlflow/info")
        assert response.status_code == 200
        data = response.json()
        assert "mlflow_tracking_uri" in data
        assert "status" in data


class TestContentTypes:
    """Test content type handling."""

    def test_predict_wrong_content_type(self, client):
        """Test predict with wrong content type."""
        response = client.post(
            "/predict",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 400

    def test_predict_malformed_json(self, client):
        """Test predict with malformed JSON."""
        response = client.post(
            "/predict",
            data="{invalid json}",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400


class TestConcurrency:
    """Test concurrent requests."""

    def test_multiple_predictions(self, client):
        """Test multiple prediction requests."""
        data = {
            "packet_count": 100,
            "byte_count": 50000,
            "duration": 5.0,
            "syn_flags": 2,
            "fin_flags": 1,
            "ack_flags": 10,
        }

        # Make multiple requests
        responses = [client.post("/predict", json=data) for _ in range(10)]

        # All should succeed
        for response in responses:
            assert response.status_code == 200

        # Check stats updated
        stats_response = client.get("/stats")
        stats = stats_response.json()
        assert stats["total_predictions"] >= 10
