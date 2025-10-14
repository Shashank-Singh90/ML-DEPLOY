"""REST API entry point for the IoT threat detection service."""

from __future__ import annotations

import math
import time
import uuid
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Dict

from flask import Flask, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from prometheus_flask_exporter import PrometheusMetrics

from app.runtime import ModelService, PredictionExplainer

# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
metrics = PrometheusMetrics(app)

prediction_counter = Counter(
    "iot_predictions_total",
    "Total number of threat predictions",
    ["prediction_class", "risk_level"],
)
prediction_duration = Histogram(
    "iot_prediction_duration_seconds",
    "Prediction processing time",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
threat_score_gauge = Gauge("iot_current_threat_score", "Average threat score for recent predictions")
error_counter = Counter("iot_prediction_errors_total", "Total prediction errors", ["error_type"])

recent_predictions: list[Dict[str, Any]] = []
app_start_time = datetime.now()

logger.info("Loading model pipeline ...")
model_service = ModelService()
explainer = PredictionExplainer(model_service)


# ---------------------------------------------------------------------------
# Validation and feature handling
# ---------------------------------------------------------------------------
BASIC_REQUEST_FIELDS = (
    "packet_count",
    "byte_count",
    "duration",
    "syn_flags",
    "fin_flags",
    "ack_flags",
)


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def validate_prediction_input(endpoint_function):
    """Validate the six-field prediction payload and pass a clean dict downstream."""

    @wraps(endpoint_function)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return (
                jsonify({
                    "error": "Invalid content type",
                    "expected": "application/json",
                    "status": "validation_failed",
                }),
                400,
            )

        payload = request.get_json(silent=True) or {}
        missing = [field for field in BASIC_REQUEST_FIELDS if field not in payload]
        if missing:
            return (
                jsonify({
                    "error": "Missing required fields",
                    "missing_fields": missing,
                    "status": "validation_failed",
                }),
                400,
            )

        cleaned: Dict[str, float] = {}
        for field in BASIC_REQUEST_FIELDS:
            raw_value = payload[field]
            if not _is_finite_number(raw_value):
                return (
                    jsonify({
                        "error": f"{field} must be a finite number",
                        "status": "validation_failed",
                    }),
                    400,
                )

            value = float(raw_value)
            if field == "duration" and value <= 0:
                return (
                    jsonify({
                        "error": "duration must be greater than zero",
                        "status": "validation_failed",
                    }),
                    400,
                )
            if field != "duration" and value < 0:
                return (
                    jsonify({
                        "error": f"{field} must be non-negative",
                        "status": "validation_failed",
                    }),
                    400,
                )
            cleaned[field] = value

        kwargs["validated_payload"] = cleaned
        return endpoint_function(*args, **kwargs)

    return wrapper


def convert_simple_to_advanced_features(simple_input: Dict[str, float]) -> Dict[str, float]:
    """Expand the short-form payload to the 42-feature model format."""

    duration = simple_input["duration"]
    packet_count = simple_input["packet_count"]
    byte_count = simple_input["byte_count"]
    syn_flags = simple_input["syn_flags"]
    fin_flags = simple_input["fin_flags"]
    ack_flags = simple_input["ack_flags"]

    packet_rate = packet_count / duration if duration > 0 else 0.0
    average_packet_size = byte_count / packet_count if packet_count > 0 else 0.0

    return {
        "flow_duration": duration,
        "Duration": duration,
        "Rate": packet_rate,
        "Srate": packet_rate * 0.8,
        "IAT": max(duration / max(packet_count, 1), 0.001),
        "fin_flag_number": fin_flags,
        "syn_flag_number": syn_flags,
        "rst_flag_number": 0.0,
        "psh_flag_number": 2.0,
        "ack_flag_number": ack_flags,
        "ece_flag_number": 0.0,
        "cwr_flag_number": 0.0,
        "ack_count": ack_flags,
        "syn_count": syn_flags,
        "fin_count": fin_flags,
        "rst_count": 0.0,
        "HTTP": 1.0,
        "HTTPS": 0.0,
        "DNS": 0.0,
        "Telnet": 0.0,
        "SMTP": 0.0,
        "SSH": 0.0,
        "IRC": 0.0,
        "TCP": 1.0,
        "UDP": 0.0,
        "DHCP": 0.0,
        "ARP": 0.0,
        "ICMP": 0.0,
        "IPv": 1.0,
        "LLC": 0.0,
        "Tot sum": byte_count,
        "Tot size": byte_count,
        "Min": average_packet_size * 0.5,
        "Max": average_packet_size * 1.5,
        "AVG": average_packet_size,
        "Std": average_packet_size * 0.3,
        "Number": packet_count,
        "Magnitue": packet_rate,
        "Radius": 25.0,
        "Covariance": 0.1,
        "Variance": 0.2,
        "Weight": 1.0,
    }


def calculate_risk_level(confidence_score: float, prediction: int) -> str:
    if prediction == 0:
        return "low"
    if confidence_score < 0.7:
        return "medium"
    if confidence_score < 0.9:
        return "high"
    return "critical"


def track_metrics(endpoint_function):
    """Record latency and error counters for decorated endpoints."""

    @wraps(endpoint_function)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            response = endpoint_function(*args, **kwargs)
            prediction_duration.observe(time.time() - start_time)
            return response
        except Exception as exc:  # pragma: no cover - defensive guard
            error_counter.labels(error_type=type(exc).__name__).inc()
            raise

    return wrapper


@app.route("/")
def home():
    """Expose service metadata."""

    return jsonify(
        {
            "service": "IoT Threat Detection API",
            "version": "1.0.0",
            "status": "running",
            "model_status": "loaded" if model_service.model_loaded else "not_loaded",
            "endpoints": [
                "/health",
                "/predict",
                "/explain",
                "/model/info",
                "/stats",
                "/metrics",
            ],
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    """Report the current health of the API and model pipeline."""

    try:
        uptime_seconds = (datetime.now() - app_start_time).total_seconds()
        status = "healthy" if model_service.model_loaded else "degraded"
        return (
            jsonify(
                {
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                    "service": "IoT Threat Detection API",
                    "uptime_seconds": uptime_seconds,
                    "model_status": "loaded" if model_service.model_loaded else "not_loaded",
                    "total_predictions": len(recent_predictions),
                }
            ),
            200 if model_service.model_loaded else 503,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Health check failed: %s", exc)
        return (
            jsonify({"status": "unhealthy", "message": "Internal error"}),
            503,
        )


@app.route("/predict", methods=["POST"])
@validate_prediction_input
@track_metrics
def predict(validated_payload: Dict[str, float]):
    """Run a threat prediction using the simplified six-field payload."""

    try:
        feature_payload = convert_simple_to_advanced_features(validated_payload)
        prediction = model_service.predict(feature_payload)
        risk_level = calculate_risk_level(prediction.confidence, prediction.prediction)

        prediction_counter.labels(
            prediction_class=prediction.prediction_label,
            risk_level=risk_level,
        ).inc()

        recent_predictions.append(
            {
                "timestamp": datetime.now(),
                "prediction": prediction.prediction,
                "threat_score": prediction.threat_score,
                "confidence": prediction.confidence,
            }
        )
        if len(recent_predictions) > 100:
            recent_predictions.pop(0)

        if recent_predictions:
            average_threat = sum(item["threat_score"] for item in recent_predictions) / len(recent_predictions)
            threat_score_gauge.set(average_threat)

        response = {
            "status": "success",
            "prediction": "threat" if prediction.prediction == 1 else "normal",
            "confidence": prediction.confidence,
            "risk_level": risk_level,
            "threat_score": prediction.threat_score,
            "important_features": prediction.important_features,
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
        return jsonify(response), 200
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        return (
            jsonify({"status": "error", "message": "Internal server error", "request_id": str(uuid.uuid4())}),
            500,
        )


@app.route("/explain", methods=["POST"])
@validate_prediction_input
def explain_prediction(validated_payload: Dict[str, float]):
    """Return a concise explanation for the current prediction."""

    try:
        feature_payload = convert_simple_to_advanced_features(validated_payload)
        explanation = explainer.explain(feature_payload)
        return (
            jsonify({"status": "success", "explanation": explanation, "timestamp": datetime.now().isoformat()}),
            200,
        )
    except Exception as exc:
        logger.error("Explanation error: %s", exc)
        return (
            jsonify({"status": "error", "message": "Internal server error"}),
            500,
        )


@app.route("/model/info", methods=["GET"])
def model_info():
    """Expose basic runtime and feature metadata for the model."""

    if not model_service.model_loaded:
        return jsonify({"status": "unavailable", "error": "Model not loaded"}), 503

    summary = explainer.summary()
    return (
        jsonify(
            {
                "model_type": type(model_service.model).__name__ if model_service.model else "unknown",
                "feature_count": len(model_service.feature_names),
                "feature_names": model_service.feature_names,
                "feature_importance": summary.get("feature_importance", []),
                "threat_classes": model_service.threat_classes,
                "stats_available": summary.get("stats_available", False),
            }
        ),
        200,
    )


@app.route("/stats", methods=["GET"])
def statistics():
    """Return lightweight operational statistics."""

    uptime_seconds = (datetime.now() - app_start_time).total_seconds()
    attack_count = sum(1 for record in recent_predictions if record["prediction"] == 1)
    prediction_total = len(recent_predictions)
    average_confidence = (
        sum(record["confidence"] for record in recent_predictions) / prediction_total if prediction_total else 0.0
    )
    average_threat = (
        sum(record["threat_score"] for record in recent_predictions) / prediction_total if prediction_total else 0.0
    )

    stats_payload = {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime_seconds,
        "total_predictions": prediction_total,
        "attacks_detected": attack_count,
        "attack_rate": attack_count / prediction_total if prediction_total else 0.0,
        "average_confidence": average_confidence,
        "average_threat_score": average_threat,
    }
    return jsonify(stats_payload), 200


@app.route("/metrics")
def metrics_endpoint():
    """Expose Prometheus metrics."""

    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
