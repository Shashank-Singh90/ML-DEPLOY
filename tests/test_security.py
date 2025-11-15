"""Security tests for IoT Threat Detection API"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.security import sanitize_error_message, validate_numeric_range


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_numeric_range_validation_max(self):
        """Test that values exceeding maximum are rejected."""
        with pytest.raises(Exception):
            validate_numeric_range(1e13, "test_field")

    def test_numeric_range_validation_min(self):
        """Test that values below minimum are rejected."""
        with pytest.raises(Exception):
            validate_numeric_range(-1e13, "test_field")

    def test_numeric_range_validation_valid(self):
        """Test that valid values are accepted."""
        validate_numeric_range(1000.0, "test_field")  # Should not raise

    def test_error_message_sanitization(self):
        """Test that error messages are sanitized."""
        error = ValueError("Sensitive internal error message")
        sanitized = sanitize_error_message(error, show_details=False)
        assert "internal" not in sanitized.lower()
        assert sanitized == "Invalid input data"

    def test_error_message_details_in_dev(self):
        """Test that detailed errors shown in dev mode."""
        error = ValueError("Detailed error")
        detailed = sanitize_error_message(error, show_details=True)
        assert "Detailed error" in detailed


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(app) as test_client:
            yield test_client

    def test_predict_zero_duration(self, client):
        """Test prediction with zero duration."""
        data = {
            "packet_count": 100,
            "byte_count": 5000,
            "duration": 0.0,  # Invalid: must be > 0
            "syn_flags": 2,
            "fin_flags": 1,
            "ack_flags": 10,
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 400

    def test_predict_negative_values(self, client):
        """Test prediction with negative packet count."""
        data = {
            "packet_count": -100,
            "byte_count": 5000,
            "duration": 5.0,
            "syn_flags": 2,
            "fin_flags": 1,
            "ack_flags": 10,
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 400

    def test_predict_very_large_values(self, client):
        """Test prediction with very large values."""
        data = {
            "packet_count": 1e10,
            "byte_count": 1e10,
            "duration": 1e6,
            "syn_flags": 1000,
            "fin_flags": 1000,
            "ack_flags": 1000,
        }
        response = client.post("/predict", json=data)
        # Should either accept or reject gracefully
        assert response.status_code in [200, 400]

    def test_predict_infinity(self, client):
        """Test that infinity values are rejected."""
        data = {
            "packet_count": float('inf'),
            "byte_count": 5000,
            "duration": 5.0,
            "syn_flags": 2,
            "fin_flags": 1,
            "ack_flags": 10,
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 400

    def test_predict_nan(self, client):
        """Test that NaN values are rejected."""
        data = {
            "packet_count": 100,
            "byte_count": float('nan'),
            "duration": 5.0,
            "syn_flags": 2,
            "fin_flags": 1,
            "ack_flags": 10,
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 400

    def test_predict_minimal_values(self, client):
        """Test prediction with minimal valid values."""
        data = {
            "packet_count": 1,
            "byte_count": 1,
            "duration": 0.001,
            "syn_flags": 0,
            "fin_flags": 0,
            "ack_flags": 0,
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 200
