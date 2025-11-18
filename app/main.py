"""
IoT Threat Detection API - Main Application Entry Point

This module provides a FastAPI-based REST API for real-time cybersecurity
threat detection in IoT network traffic using machine learning.

Security improvements:
- API key authentication
- Rate limiting
- Thread-safe operations
- Request size limits
- Information disclosure prevention
- CORS configuration
"""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional, cast

import mlflow
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.runtime import ModelService, PredictionExplainer

# Configure logging with timestamps and log levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Security configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEYS = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else set()

# Enable API key authentication if API_KEYS environment variable is set
ENABLE_AUTH = len(API_KEYS) > 0 and "" not in API_KEYS

if not ENABLE_AUTH:
    logger.warning(
        "⚠️  API running WITHOUT authentication! "
        "Set API_KEYS environment variable for production use."
    )

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address)

# Thread-safe prediction storage
predictions_lock = threading.Lock()
recent_predictions: deque = deque(maxlen=100)


# Security: API key validation
async def get_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    """Validate API key if authentication is enabled."""
    if not ENABLE_AUTH:
        return "no-auth-required"

    if api_key is None or api_key not in API_KEYS:
        logger.warning("Unauthorized access attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return api_key


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown.

    Initializes the ML model, explainer, and application state on startup.
    """
    logger.info("Starting IoT Threat Detection API...")
    logger.info("Loading ML model pipeline...")

    # Initialize model service and explainer
    model_service = ModelService()
    explainer = PredictionExplainer(model_service)

    # Store services in application state for dependency injection
    application.state.model_service = model_service
    application.state.explainer = explainer
    application.state.app_start_time = datetime.now()

    logger.info("API ready to serve requests")

    try:
        yield
    finally:
        # Cleanup on shutdown
        logger.info("API shutting down...")
        pass


app = FastAPI(
    title="IoT Threat Detection API",
    version="1.0.0",
    description="Cybersecurity threat detection service for IoT telemetry",
    lifespan=lifespan,
)

# Configure CORS - TODO: restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup rate limiting to prevent abuse
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Note: FastAPI has built-in request size limits, so we're good here
app.add_middleware(
    lambda app: app,  # placeholder middleware
)

# Prometheus metrics for monitoring
prediction_counter = Counter(
    "iot_predictions_total",
    "Total number of threat predictions made",
    ["prediction_class", "risk_level"],
)

prediction_duration = Histogram(
    "iot_prediction_duration_seconds",
    "Time taken to process prediction requests",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

threat_score_gauge = Gauge(
    "iot_current_threat_score",
    "Rolling average threat score from recent predictions"
)

error_counter = Counter(
    "iot_prediction_errors_total",
    "Total number of prediction errors encountered",
    ["error_type"]
)


# Dependency injection helpers
def get_model_service(request: Request) -> ModelService:
    """Retrieve the ML model service from application state."""
    model_service = getattr(request.app.state, "model_service", None)
    if model_service is None:
        logger.error("Model service not initialized")
        raise HTTPException(status_code=503, detail="Model service not available")
    return cast(ModelService, model_service)


def get_explainer(request: Request) -> PredictionExplainer:
    """Retrieve the prediction explainer from application state."""
    explainer = getattr(request.app.state, "explainer", None)
    if explainer is None:
        logger.error("Prediction explainer not initialized")
        raise HTTPException(status_code=503, detail="Explainer not available")
    return cast(PredictionExplainer, explainer)


def get_app_start_time(request: Request) -> datetime:
    """Retrieve application start time for uptime calculation."""
    return getattr(request.app.state, "app_start_time", datetime.now())


class ValidationErrorResponse(Exception):
    """Custom exception for consistent validation error payloads."""

    def __init__(self, content: Dict[str, Any], status_code: int = 400) -> None:
        self.content = content
        self.status_code = status_code


@app.exception_handler(ValidationErrorResponse)
async def handle_validation_error(_: Request, exc: ValidationErrorResponse) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.content)


# Input validation constants and helpers
# The 6 required fields for threat prediction requests
BASIC_REQUEST_FIELDS = (
    "packet_count",  # Number of packets in the network flow
    "byte_count",    # Total bytes transferred
    "duration",      # Duration of the flow in seconds
    "syn_flags",     # Number of SYN TCP flags
    "fin_flags",     # Number of FIN TCP flags
    "ack_flags",     # Number of ACK TCP flags
)

# Optional protocol indicators that can be provided in requests
# These default to 0.0 if not specified
OPTIONAL_PROTOCOL_FIELDS = {
    "HTTP": 0.0, "HTTPS": 0.0, "DNS": 0.0, "Telnet": 0.0,
    "SMTP": 0.0, "SSH": 0.0, "IRC": 0.0, "TCP": 0.0,
    "UDP": 0.0, "DHCP": 0.0, "ARP": 0.0, "ICMP": 0.0,
    "IPv": 0.0, "LLC": 0.0,
}

# Statistical defaults for the feature expansion
# These values were tuned based on the training dataset characteristics
STATISTICAL_DEFAULTS = {
    "Radius": 25.0,       # spatial analysis radius
    "Covariance": 0.1,    # packet feature covariance
    "Variance": 0.2,      # distribution variance
    "Weight": 1.0,        # feature weight
}


def _is_finite_number(value: Any) -> bool:
    """
    Check if a value is a finite numeric value.

    Returns False for NaN, infinity, or non-numeric values.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


async def validate_prediction_input(request: Request) -> Dict[str, float]:
    """
    Validate and clean incoming prediction request data.

    Ensures the request contains all 6 required numeric fields with valid values.
    Returns a dictionary of validated float values.

    Raises:
        ValidationErrorResponse: If validation fails for any reason.
    """
    # Check content type
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        raise ValidationErrorResponse(
            {
                "error": "Invalid content type",
                "expected": "application/json",
                "status": "validation_failed",
            }
        )

    # Check request size to prevent memory issues
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
        raise ValidationErrorResponse(
            {
                "error": "Request too large",
                "max_size": "10MB",
                "status": "validation_failed",
            },
            status_code=413
        )

    # Parse JSON payload
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise ValidationErrorResponse(
            {
                "error": "Malformed JSON payload",
                "status": "validation_failed",
            }
        ) from None
    except Exception:
        # Don't expose internal errors to clients
        raise ValidationErrorResponse(
            {
                "error": "Invalid request format",
                "status": "validation_failed",
            }
        ) from None

    # Ensure payload is a dictionary
    if not isinstance(payload, dict):
        raise ValidationErrorResponse(
            {
                "error": "Payload must be a JSON object",
                "status": "validation_failed",
            }
        )

    # Check for missing fields
    missing_fields = [field for field in BASIC_REQUEST_FIELDS if field not in payload]
    if missing_fields:
        raise ValidationErrorResponse(
            {
                "error": "Missing required fields",
                "missing_fields": missing_fields,
                "status": "validation_failed",
            }
        )

    # Validate and clean each field
    validated_data: Dict[str, float] = {}
    for field in BASIC_REQUEST_FIELDS:
        raw_value = payload[field]

        # Check if value is a finite number
        if not _is_finite_number(raw_value):
            raise ValidationErrorResponse(
                {
                    "error": f"{field} must be a finite number (not NaN or infinity)",
                    "status": "validation_failed",
                }
            )

        value = float(raw_value)

        # Duration must be positive
        if field == "duration" and value <= 0:
            raise ValidationErrorResponse(
                {
                    "error": "duration must be greater than zero",
                    "status": "validation_failed",
                }
            )

        # Other fields must be non-negative
        if field != "duration" and value < 0:
            raise ValidationErrorResponse(
                {
                    "error": f"{field} must be non-negative",
                    "status": "validation_failed",
                }
            )

        validated_data[field] = value

    # Allow optional protocol indicators to be passed through
    for protocol_field, default_value in OPTIONAL_PROTOCOL_FIELDS.items():
        if protocol_field in payload:
            if _is_finite_number(payload[protocol_field]):
                validated_data[protocol_field] = float(payload[protocol_field])
            else:
                validated_data[protocol_field] = default_value
        # Defaults are handled in feature conversion

    return validated_data


def convert_simple_to_advanced_features(simple_input: Dict[str, float]) -> Dict[str, float]:
    """
    Convert simplified 6-field input to the 42 features the model expects.

    This is basically a bridge between the simple API and the actual model,
    which was trained on a much richer feature set.

    Args:
        simple_input: Dictionary with 6+ basic network metrics

    Returns:
        Dictionary with 42 expanded features for ML model
    """
    # Extract input values
    duration = simple_input["duration"]
    packet_count = simple_input["packet_count"]
    byte_count = simple_input["byte_count"]
    syn_flags = simple_input["syn_flags"]
    fin_flags = simple_input["fin_flags"]
    ack_flags = simple_input["ack_flags"]

    # Calculate some basic derived features
    packet_rate = packet_count / duration if duration > 0 else 0.0
    average_packet_size = byte_count / packet_count if packet_count > 0 else 0.0
    # TODO: might want to add more sophisticated feature engineering here

    # Build the 42-feature dictionary for the ML model
    features = {
        # Time-based features
        "flow_duration": duration,
        "Duration": duration,
        "IAT": max(duration / max(packet_count, 1), 0.001),  # Inter-arrival time

        # Rate features
        "Rate": packet_rate,
        "Srate": packet_rate * 0.8,  # Scaled rate

        # TCP flags (from input)
        "fin_flag_number": fin_flags,
        "syn_flag_number": syn_flags,
        "ack_flag_number": ack_flags,

        # TCP flags (defaults for missing data)
        "rst_flag_number": 0.0,
        "psh_flag_number": 2.0,
        "ece_flag_number": 0.0,
        "cwr_flag_number": 0.0,

        # Packet counts
        "ack_count": ack_flags,
        "syn_count": syn_flags,
        "fin_count": fin_flags,
        "rst_count": 0.0,
        "Number": packet_count,

        # Packet size statistics
        "Tot sum": byte_count,
        "Tot size": byte_count,
        "Min": average_packet_size * 0.5,
        "Max": average_packet_size * 1.5,
        "AVG": average_packet_size,
        "Std": average_packet_size * 0.3,

        # Statistical features
        "Magnitude": packet_rate,  # Note: model might have been trained with typo, verify this
        **STATISTICAL_DEFAULTS,
    }

    # Handle protocol indicators - use provided values or make educated guesses
    for protocol_field, default_value in OPTIONAL_PROTOCOL_FIELDS.items():
        if protocol_field in simple_input:
            features[protocol_field] = simple_input[protocol_field]
        else:
            # Make reasonable guesses based on what we know about the traffic
            if protocol_field == "TCP":
                features[protocol_field] = 1.0 if syn_flags > 0 or ack_flags > 0 else 0.0
            elif protocol_field == "IPv":
                features[protocol_field] = 1.0  # pretty much everything is IP these days
            elif protocol_field == "HTTP":
                features[protocol_field] = 0.3  # moderate chance it's HTTP
            else:
                features[protocol_field] = 0.0

    return features


def calculate_risk_level(confidence_score: float, prediction: int) -> str:
    """
    Determine risk level based on prediction and confidence score.

    TODO: These thresholds might need tuning based on production data

    Args:
        confidence_score: Model confidence (0.0 to 1.0)
        prediction: Binary prediction (0=normal, 1=threat)

    Returns:
        Risk level: "low", "medium", "high", or "critical"
    """
    if prediction == 0:
        return "low"
    if confidence_score < 0.7:
        return "medium"  # borderline threats
    if confidence_score < 0.9:
        return "high"
    return "critical"


# API Endpoints
@app.get("/")
async def home(
    model_service: ModelService = Depends(get_model_service),
    api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Root endpoint - provides API information and available endpoints.

    Returns service metadata, model status, and list of available endpoints.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    return {
        "service": "IoT Threat Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_status": "loaded" if model_service.model_loaded else "not_loaded",
        "mlflow_tracking_uri": mlflow_uri,
        "authentication": "enabled" if ENABLE_AUTH else "disabled",
        "endpoints": [
            "/health",
            "/predict",
            "/explain",
            "/model/info",
            "/stats",
            "/metrics",
            "/mlflow/info",
        ],
    }


@app.get("/health")
@limiter.limit("100/minute")
async def health_check(
    request: Request,
    model_service: ModelService = Depends(get_model_service),
) -> JSONResponse:
    """Report the current health of the API and model pipeline."""

    try:
        uptime_seconds = (datetime.now() - get_app_start_time(request)).total_seconds()
        status_value = "healthy" if model_service.model_loaded else "degraded"

        # Need thread safety here since multiple requests could hit this
        with predictions_lock:
            total_predictions = len(recent_predictions)

        payload = {
            "status": status_value,
            "timestamp": datetime.now().isoformat(),
            "service": "IoT Threat Detection API",
            "uptime_seconds": uptime_seconds,
            "model_status": "loaded" if model_service.model_loaded else "not_loaded",
            "total_predictions": total_predictions,
        }
        status_code = 200 if model_service.model_loaded else 503
        return JSONResponse(status_code=status_code, content=payload)
    except Exception:
        # Something went wrong with health check itself
        logger.exception("Health check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Service unavailable"}
        )


@app.post("/predict")
@limiter.limit("60/minute")
async def predict(
    request: Request,
    model_service: ModelService = Depends(get_model_service),
    api_key: str = Depends(get_api_key)
) -> JSONResponse:
    """Run a threat prediction using the simplified six-field payload."""

    start_time = time.perf_counter()
    try:
        validated_payload = await validate_prediction_input(request)
        feature_payload = convert_simple_to_advanced_features(validated_payload)
        prediction = model_service.predict(feature_payload)
        risk_level = calculate_risk_level(prediction.confidence, prediction.prediction)

        prediction_counter.labels(
            prediction_class=prediction.prediction_label,
            risk_level=risk_level,
        ).inc()

        # Store prediction for stats (with thread safety)
        with predictions_lock:
            recent_predictions.append(
                {
                    "timestamp": datetime.now(),
                    "prediction": prediction.prediction,
                    "threat_score": prediction.threat_score,
                    "confidence": prediction.confidence,
                }
            )

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
        return JSONResponse(status_code=200, content=response)
    except ValidationErrorResponse:
        raise
    except Exception as exc:
        error_counter.labels(error_type=type(exc).__name__).inc()
        # Log internally but don't expose details to client
        logger.exception("Prediction processing failed")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error",
                "request_id": str(uuid.uuid4()),
            },
        )
    finally:
        prediction_duration.observe(time.perf_counter() - start_time)


@app.post("/explain")
@limiter.limit("30/minute")
async def explain_prediction(
    request: Request,
    explainer: PredictionExplainer = Depends(get_explainer),
    api_key: str = Depends(get_api_key)
) -> JSONResponse:
    """Return a concise explanation for the current prediction."""

    try:
        validated_payload = await validate_prediction_input(request)
        feature_payload = convert_simple_to_advanced_features(validated_payload)
        explanation = explainer.explain(feature_payload)
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "explanation": explanation,
                "timestamp": datetime.now().isoformat(),
            },
        )
    except ValidationErrorResponse:
        raise
    except Exception:
        error_counter.labels(error_type="ExplanationError").inc()
        logger.exception("Explanation generation failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error"},
        )


@app.get("/model/info")
@limiter.limit("30/minute")
async def model_info(
    request: Request,
    explainer: PredictionExplainer = Depends(get_explainer),
    model_service: ModelService = Depends(get_model_service),
    api_key: str = Depends(get_api_key)
) -> JSONResponse:
    """Expose basic runtime and feature metadata for the model."""

    if not model_service.model_loaded:
        return JSONResponse(status_code=503, content={"status": "unavailable", "error": "Model not loaded"})

    summary = explainer.summary()
    return JSONResponse(
        status_code=200,
        content={
            "model_type": type(model_service.model).__name__ if model_service.model else "unknown",
            "feature_count": len(model_service.feature_names),
            "feature_names": model_service.feature_names,
            "feature_importance": summary.get("feature_importance", []),
            "threat_classes": model_service.threat_classes,
            "stats_available": summary.get("stats_available", False),
        },
    )


@app.get("/stats")
@limiter.limit("30/minute")
async def statistics(
    request: Request,
    api_key: str = Depends(get_api_key)
) -> JSONResponse:
    """Return lightweight operational statistics."""

    # Grab a snapshot of recent predictions
    with predictions_lock:
        recent_preds = list(recent_predictions)

    uptime_seconds = (datetime.now() - get_app_start_time(request)).total_seconds()
    attack_count = sum(1 for record in recent_preds if record["prediction"] == 1)
    prediction_total = len(recent_preds)
    average_confidence = (
        sum(record["confidence"] for record in recent_preds) / prediction_total if prediction_total else 0.0
    )
    average_threat = (
        sum(record["threat_score"] for record in recent_preds) / prediction_total if prediction_total else 0.0
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
    return JSONResponse(status_code=200, content=stats_payload)


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Expose Prometheus metrics (public - no auth required for monitoring)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/mlflow/info")
@limiter.limit("30/minute")
async def mlflow_info(request: Request, api_key: str = Depends(get_api_key)) -> JSONResponse:
    """Expose MLflow tracking information."""

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    try:
        # Get current experiment info
        experiment = mlflow.get_experiment_by_name("iot-threat-detection")
        experiment_info = {
            "experiment_id": experiment.experiment_id if experiment else None,
            "experiment_name": "iot-threat-detection",
            "artifact_location": experiment.artifact_location if experiment else None,
        } if experiment else {
            "experiment_id": None,
            "experiment_name": "iot-threat-detection",
            "artifact_location": None,
        }

        return JSONResponse(
            status_code=200,
            content={
                "mlflow_tracking_uri": mlflow_uri,
                "mlflow_ui_url": mlflow_uri,
                "experiment": experiment_info,
                "status": "connected" if experiment else "disconnected",
            },
        )
    except Exception:
        logger.exception("MLflow info retrieval failed")
        return JSONResponse(
            status_code=200,
            content={
                "mlflow_tracking_uri": mlflow_uri,
                "mlflow_ui_url": mlflow_uri,
                "status": "unavailable",
                "error": "Unable to retrieve MLflow information",
            },
        )


def run() -> None:  # pragma: no cover - console entry point helper
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, log_level="info")


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    run()
