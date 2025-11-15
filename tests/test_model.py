"""Model quality and performance tests"""

import pytest
import numpy as np
import pandas as pd

from app.runtime import ModelService, EXPECTED_FEATURES


class TestModelQuality:
    """Test ML model quality and behavior."""

    @pytest.fixture(scope="class")
    def model_service(self):
        """Provide a loaded model service."""
        return ModelService()

    def test_model_loads_successfully(self, model_service):
        """Test that model loads without errors."""
        assert model_service.model_loaded
        assert model_service.model is not None
        assert model_service.scaler is not None

    def test_model_features_match_expected(self, model_service):
        """Test that model uses expected features."""
        assert len(model_service.feature_names) > 0
        # Check feature names are from expected set
        for feature in model_service.feature_names:
            assert feature in EXPECTED_FEATURES

    def test_model_prediction_format(self, model_service):
        """Test that predictions return expected format."""
        test_input = {feature: 1.0 for feature in EXPECTED_FEATURES}
        result = model_service.predict(test_input)

        assert hasattr(result, 'prediction')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'threat_score')
        assert result.prediction in [0, 1]
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.threat_score <= 1.0

    def test_model_consistency(self, model_service):
        """Test that model gives consistent predictions for same input."""
        test_input = {feature: 1.0 for feature in EXPECTED_FEATURES}

        result1 = model_service.predict(test_input)
        result2 = model_service.predict(test_input)

        assert result1.prediction == result2.prediction
        assert result1.confidence == result2.confidence
        assert result1.threat_score == result2.threat_score

    def test_model_feature_importance(self, model_service):
        """Test that model returns feature importance."""
        top_features = model_service.top_features(limit=5)

        assert len(top_features) <= 5
        assert all('feature' in f for f in top_features)
        assert all('importance' in f for f in top_features)
        assert all(0.0 <= f['importance'] <= 1.0 for f in top_features)

    def test_model_handles_edge_cases(self, model_service):
        """Test model behavior with edge case inputs."""
        # All zeros
        zero_input = {feature: 0.0 for feature in EXPECTED_FEATURES}
        result_zero = model_service.predict(zero_input)
        assert result_zero.prediction in [0, 1]

        # All ones
        ones_input = {feature: 1.0 for feature in EXPECTED_FEATURES}
        result_ones = model_service.predict(ones_input)
        assert result_ones.prediction in [0, 1]

        # Mixed values
        mixed_input = {feature: np.random.random() for feature in EXPECTED_FEATURES}
        result_mixed = model_service.predict(mixed_input)
        assert result_mixed.prediction in [0, 1]

    def test_model_integrity_verification(self, model_service):
        """Test that model integrity checks are in place."""
        # Model should have been verified during loading
        assert model_service.model_loaded
        # If checksums exist, they should have been verified
        checksum_file = model_service.model_dir / "model_checksums.json"
        if checksum_file.exists():
            import json
            with open(checksum_file) as f:
                checksums = json.load(f)
                assert 'model' in checksums
                assert 'scaler' in checksums


class TestModelPerformance:
    """Test model performance metrics."""

    @pytest.fixture(scope="class")
    def model_service(self):
        return ModelService()

    def test_prediction_speed(self, model_service):
        """Test that predictions are fast enough for production."""
        import time

        test_input = {feature: 1.0 for feature in EXPECTED_FEATURES}

        start = time.time()
        for _ in range(100):
            model_service.predict(test_input)
        elapsed = time.time() - start

        avg_time = elapsed / 100
        assert avg_time < 0.1  # Should be < 100ms per prediction

    def test_batch_prediction_capability(self, model_service):
        """Test that model can handle multiple predictions."""
        test_inputs = [
            {feature: np.random.random() for feature in EXPECTED_FEATURES}
            for _ in range(10)
        ]

        results = [model_service.predict(inp) for inp in test_inputs]

        assert len(results) == 10
        assert all(r.prediction in [0, 1] for r in results)
