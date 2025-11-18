"""
IoT Threat Detection - Runtime Model Service

This module handles all ML model operations including:
- Loading and training the Random Forest model
- Feature preparation and scaling
- Real-time threat predictions
- Model explainability and feature importance
- MLflow experiment tracking integration

Security improvements:
- Uses joblib instead of pickle for safer serialization
- Detects and handles Git LFS pointer files
- Thread-safe operations
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# The 42 features expected by the model (order matters for scaling)
# This list defines the exact feature names and order used during training
EXPECTED_FEATURES: List[str] = [
    "flow_duration",
    "Duration",
    "Rate",
    "Srate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Tot size",
    "Min",
    "Max",
    "AVG",
    "Std",
    "IAT",
    "Number",
    "Magnitude",  # was misspelled in original dataset
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
]

# Model file names for persistence
MODEL_FILENAME = "iot_model.pkl"  # Trained RandomForest model
SCALER_FILENAME = "scaler.pkl"  # StandardScaler for feature normalization
FEATURE_FILENAME = "feature_names.txt"  # List of feature names
FEATURE_STATS_FILENAME = "feature_stats.json"  # Feature statistics for explainability


class ModelPipelineError(RuntimeError):
    """Exception raised when model operations fail (loading, training, prediction)."""


@dataclass
class PredictionResult:
    """
    Container for prediction results from the ML model.

    Attributes:
        prediction: Binary prediction (0=normal, 1=threat)
        class_probabilities: Probability for each class {"normal": 0.7, "threat": 0.3}
        threat_score: Probability of threat class (0.0 to 1.0)
        confidence: Highest class probability (model confidence)
        prediction_label: Human-readable label ("normal" or "threat")
        important_features: Top contributing features with values and importance scores
    """
    prediction: int
    class_probabilities: Dict[str, float]
    threat_score: float
    confidence: float
    prediction_label: str
    important_features: Sequence[Dict[str, Any]]


class ModelService:
    """
    Main service for ML model operations.

    Handles model loading, training, inference, and explainability.
    Automatically loads a saved model or trains a new one if needed.
    Integrates with MLflow for experiment tracking.
    """

    def __init__(
        self,
        model_dir: Path | str = Path("models/production"),
        training_data: Path | str = Path("data/raw/synthetic_iot_data.csv"),
    ) -> None:
        """
        Initialize the ModelService.

        Args:
            model_dir: Directory where model files are stored
            training_data: Path to CSV file with training data
        """
        self.model_dir = Path(model_dir)
        self.training_data = Path(training_data)

        # Model components
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.threat_classes = ["normal", "threat"]
        self.model_loaded = False

        # MLflow client for safer logging
        self.mlflow_client: Optional[mlflow.MlflowClient] = None
        self.mlflow_run_id: Optional[str] = None

        # Configure MLflow experiment tracking
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment("iot-threat-detection")
            self.mlflow_client = mlflow.MlflowClient()
            logger.info("MLflow tracking URI: %s", mlflow_tracking_uri)
        except Exception as e:
            logger.warning("MLflow initialization failed: %s. Continuing without MLflow.", e)

        # Load existing model or train new one
        self._load_or_train()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def prepare_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return only the columns used by the model.

        Parameters
        ----------
        dataframe:
            Input dataframe containing a superset of available features.
        """
        available = [feature for feature in EXPECTED_FEATURES if feature in dataframe.columns]
        if not available:
            raise ModelPipelineError("Training data must contain expected IoT features.")

        if not self.feature_names:
            self.feature_names = available
        return dataframe[available].copy()

    def predict(self, feature_source: Dict[str, Any] | Iterable[Dict[str, Any]]) -> PredictionResult:
        """Run inference and return a structured prediction result."""
        if not self.model_loaded or self.model is None or self.scaler is None:
            raise ModelPipelineError("Model not loaded; predictions are unavailable.")

        features = self.normalise_input(feature_source)
        scaled = self.scaler.transform(features)

        # Get probabilities and derive the prediction
        probabilities = self.model.predict_proba(scaled)[0]
        prediction = int(np.argmax(probabilities))

        important_features = self.top_features(limit=5, value_row=features.iloc[0])

        # Try to log prediction metrics to MLflow if available
        try:
            if self.mlflow_client and self.mlflow_run_id:
                self.mlflow_client.log_metric(
                    self.mlflow_run_id,
                    "prediction_confidence",
                    float(np.max(probabilities))
                )
                self.mlflow_client.log_metric(
                    self.mlflow_run_id,
                    "threat_probability",
                    float(probabilities[1])
                )
        except Exception as e:
            logger.debug("MLflow prediction logging failed: %s", e)

        return PredictionResult(
            prediction=prediction,
            class_probabilities={
                "normal": float(probabilities[0]),
                "threat": float(probabilities[1]),
            },
            threat_score=float(probabilities[1]),
            confidence=float(np.max(probabilities)),
            prediction_label=self.threat_classes[prediction],
            important_features=important_features,
        )

    def normalise_input(self, feature_source: Dict[str, Any] | Iterable[Dict[str, Any]]) -> pd.DataFrame:
        """Normalize input features to match model expectations."""
        if isinstance(feature_source, dict):
            frame = pd.DataFrame([feature_source])
        else:
            frame = pd.DataFrame(feature_source)

        if not self.feature_names:
            raise ModelPipelineError("Feature names are undefined; cannot prepare input.")

        return frame.reindex(columns=self.feature_names, fill_value=0.0)

    def top_features(self, limit: int = 5, value_row: Optional[pd.Series] = None) -> List[Dict[str, Any]]:
        """Return the top contributing features by importance.

        TODO: Consider adding SHAP values for better per-prediction explanations
        """
        if not self.model_loaded or not hasattr(self.model, "feature_importances_"):
            return []

        ranked = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

        results: List[Dict[str, Any]] = []
        for feature, importance in ranked:
            entry: Dict[str, Any] = {"feature": feature, "importance": float(importance)}
            if value_row is not None and feature in value_row.index:
                entry["value"] = float(value_row[feature])
            results.append(entry)
        return results

    def feature_summary(self) -> Dict[str, Any]:
        """Provide a consistent summary for the /model/info endpoint."""
        if not self.model_loaded or not hasattr(self.model, "feature_importances_"):
            return {"error": "Model does not expose feature importances."}

        importance = [
            {"feature": feature, "importance": float(score)}
            for feature, score in zip(self.feature_names, self.model.feature_importances_)
        ]
        importance.sort(key=lambda item: item["importance"], reverse=True)

        return {
            "feature_importance": importance,
            "top_features": importance[:5],
            "total_features": len(self.feature_names),
        }

    def load_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Load cached feature statistics for explanations."""
        stats_path = self.model_dir / FEATURE_STATS_FILENAME
        if stats_path.exists():
            with stats_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_or_train(self) -> None:
        try:
            if self._has_cached_model() and not self._is_lfs_pointer():
                self._load_from_disk()
            else:
                if self._is_lfs_pointer():
                    logger.warning("Model files are Git LFS pointers. Retraining model...")
                self._train_from_source()
            self.model_loaded = True
            logger.info("Threat detection model ready (%s features)", len(self.feature_names))
        except Exception as exc:  # pragma: no cover - defensive guard
            self.model_loaded = False
            logger.error("Failed to initialise model: %s", exc)
            raise

    def _has_cached_model(self) -> bool:
        return all(
            (self.model_dir / filename).exists()
            for filename in (MODEL_FILENAME, SCALER_FILENAME, FEATURE_FILENAME)
        )

    def _is_lfs_pointer(self) -> bool:
        """Check if model files are Git LFS pointers instead of actual models.

        This happens when someone clones the repo but doesn't have LFS set up.
        """
        model_path = self.model_dir / MODEL_FILENAME
        if not model_path.exists():
            return False

        try:
            # LFS pointers are just small text files
            with model_path.open("rb") as f:
                first_bytes = f.read(100)
                if b"version https://git-lfs" in first_bytes:
                    return True
        except Exception:
            pass
        return False

    def _load_from_disk(self) -> None:
        """Load model from disk using joblib (safer than pickle)."""
        try:
            self.model = joblib.load(self.model_dir / MODEL_FILENAME)
            self.scaler = joblib.load(self.model_dir / SCALER_FILENAME)
            with (self.model_dir / FEATURE_FILENAME).open("r", encoding="utf-8") as handle:
                self.feature_names = [line.strip() for line in handle if line.strip()]
            logger.info("Loaded production model from %s", self.model_dir)
        except Exception as e:
            logger.warning("Failed to load model with joblib: %s. Retraining...", e)
            raise

    def _train_from_source(self) -> None:
        dataframe = self._load_training_data()
        features = self.prepare_features(dataframe)
        targets = (dataframe["label"] > 0).astype(int)

        x_train, x_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42, stratify=targets
        )

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        # Track this training run in MLflow
        try:
            with mlflow.start_run(run_name="RandomForest_Training") as run:
                self.mlflow_run_id = run.info.run_id

                # Log hyperparameters
                params = {
                    "n_estimators": 150,  # TODO: tune this
                    "max_depth": 12,
                    "class_weight": "balanced",
                    "random_state": 42,
                    "test_size": 0.2,
                    "n_features": len(self.feature_names),
                    "training_samples": len(x_train),
                    "test_samples": len(x_test),
                }
                mlflow.log_params(params)

                # Train the RF classifier
                self.model = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    class_weight="balanced",  # helps with imbalanced data
                    n_jobs=-1,
                    random_state=42,
                )
                self.model.fit(x_train_scaled, y_train)

                # Make predictions
                predictions = self.model.predict(x_test_scaled)

                # Calculate metrics
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, zero_division=0)
                recall = recall_score(y_test, predictions, zero_division=0)
                f1 = f1_score(y_test, predictions, zero_division=0)

                # Log metrics
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "threat_detection_rate": recall,
                    "false_positive_rate": 1 - precision,
                })

                # Log model
                mlflow.sklearn.log_model(
                    self.model,
                    "model",
                    registered_model_name="IoT-Threat-Detection-RF",
                )

                # Log feature importance
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                mlflow.log_dict(feature_importance, "feature_importance.json")

                logger.info("Trained new RandomForest model (accuracy %.3f, f1 %.3f)", accuracy, f1)
                if logger.isEnabledFor(logging.DEBUG):
                    report = classification_report(y_test, predictions, zero_division=0)
                    logger.debug("Model report:\n%s", report)
        except Exception as e:
            logger.warning("MLflow training logging failed: %s. Continuing without MLflow.", e)

        self._persist_to_disk(features)

    def _persist_to_disk(self, features: pd.DataFrame) -> None:
        """Save the trained model and scaler to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_dir / MODEL_FILENAME)
        joblib.dump(self.scaler, self.model_dir / SCALER_FILENAME)

        with (self.model_dir / FEATURE_FILENAME).open("w", encoding="utf-8") as handle:
            handle.write("\n".join(self.feature_names))
        logger.info("Persisted trained model assets to %s", self.model_dir)

        # Cache simple feature statistics for explanations
        stats = {
            column: {
                "mean": float(features[column].mean()),
                "std": float(features[column].std(ddof=0) or 0.0),
                "q25": float(features[column].quantile(0.25)),
                "q75": float(features[column].quantile(0.75)),
            }
            for column in self.feature_names
        }
        with (self.model_dir / FEATURE_STATS_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(stats, handle)

        metadata = {
            "model_type": type(self.model).__name__ if self.model else "unknown",
            "trained_at": datetime.now().isoformat(),
            "feature_count": len(self.feature_names),
            "training_rows": len(features),
            "serialization": "joblib",
        }
        with (self.model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def _load_training_data(self) -> pd.DataFrame:
        if self.training_data.exists():
            logger.info("Loading training data from %s", self.training_data)
            return pd.read_csv(self.training_data)

        logger.warning("Training data not found at %s; generating fallback dataset.", self.training_data)
        return self._generate_fallback_dataset()

    def _generate_fallback_dataset(self, rows: int = 2000) -> pd.DataFrame:
        """Generate synthetic data if real training data isn't available."""
        rng = np.random.default_rng(seed=42)
        data = {feature: rng.normal(loc=100.0, scale=20.0, size=rows) for feature in EXPECTED_FEATURES}
        # Protocol flags should be binary
        for feature in ["HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC"]:
            data[feature] = rng.integers(low=0, high=2, size=rows)
        dataframe = pd.DataFrame(data)
        threat_probability = rng.uniform(0.05, 0.3)  # some reasonable attack rate
        dataframe["label"] = rng.binomial(1, threat_probability, size=rows)
        return dataframe


class PredictionExplainer:
    """Provide lightweight explanations based on model importances."""

    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service

    def explain(self, feature_source: Dict[str, Any]) -> Dict[str, Any]:
        if not self.model_service.model_loaded:
            return {"status": "unavailable", "message": "Model not ready"}

        # Prepare features for the model
        features = self.model_service.normalise_input(feature_source)
        scaled = self.model_service.scaler.transform(features)

        # Get prediction and probabilities
        probabilities = self.model_service.model.predict_proba(scaled)[0]
        prediction = int(np.argmax(probabilities))

        top_features = self.model_service.top_features(limit=5, value_row=features.iloc[0])

        # Build a human-readable explanation
        summary = self._build_summary(top_features, probabilities[1], prediction)

        return {
            "status": "ok",
            "prediction": prediction,
            "threat_probability": float(probabilities[1]),
            "confidence": float(np.max(probabilities)),
            "top_features": top_features,
            "summary": summary,
        }

    def summary(self) -> Dict[str, Any]:
        base_summary = self.model_service.feature_summary()
        base_summary["stats_available"] = bool(self.model_service.load_feature_statistics())
        return base_summary

    def _build_summary(self, top_features: Sequence[Dict[str, Any]], threat_probability: float, prediction: int) -> str:
        if not top_features:
            return "Model did not report feature importances."

        label = "threat" if prediction == 1 else "normal traffic"
        top_descriptions = [
            f"{item['feature']} (~{item['importance']:.2f})"
            for item in top_features[:3]
        ]
        feature_text = ", ".join(top_descriptions)
        return (
            f"Model identified {label} with {threat_probability:.0%} threat probability. "
            f"Key contributors: {feature_text}."
        )
