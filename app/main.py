"""
IoT Threat Detection API

A machine learning-powered REST API for detecting cybersecurity threats
in IoT network traffic using Random Forest classification.
"""

from flask import Flask, request, jsonify
import logging
import uuid
import time
from datetime import datetime
from functools import wraps

from app.validators import validate_prediction_input
from app.models.model_service import ModelService
from app.models.explainer import ModelExplainer
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Initialize Prometheus metrics for monitoring
metrics = PrometheusMetrics(app)

# Custom metrics for IoT threat monitoring
prediction_counter = Counter(
    'iot_predictions_total',
    'Total number of threat predictions made',
    ['prediction_class', 'risk_level']
)

prediction_duration = Histogram(
    'iot_prediction_duration_seconds',
    'Time spent processing predictions',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

threat_score_gauge = Gauge(
    'iot_current_threat_score',
    'Current average threat score from recent predictions'
)

error_counter = Counter(
    'iot_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# Application state tracking
recent_predictions = []
app_start_time = datetime.now()

# Initialize ML model services
logger.info("Initializing IoT Threat Detection API...")
model_service = ModelService()
explainer = ModelExplainer(model_service)
logger.info(f"Model Status: {'Loaded' if model_service.model_loaded else 'Not Loaded'}")


def calculate_risk_level(confidence_score, prediction):
    """
    Determine risk level based on prediction confidence.

    Args:
        confidence_score: Model confidence (0-1)
        prediction: Binary prediction (0=Normal, 1=Threat)

    Returns:
        Risk level string: 'low', 'medium', 'high', or 'critical'
    """
    if prediction == 0:  # Normal traffic
        return 'low'
    elif confidence_score < 0.7:
        return 'medium'
    elif confidence_score < 0.9:
        return 'high'
    else:
        return 'critical'


def convert_simple_to_advanced_features(simple_input):
    """
    Convert simplified 6-field input to complete 42-field feature set.

    This allows the API to accept simplified inputs while maintaining
    compatibility with the full ML model feature requirements.

    Args:
        simple_input: Dictionary with basic network metrics

    Returns:
        Dictionary with complete 42 IoT network features
    """
    # Extract basic network metrics
    packet_count = simple_input.get('packet_count', 100)
    byte_count = simple_input.get('byte_count', 50000)
    duration = simple_input.get('duration', 5.0)
    syn_flags = simple_input.get('syn_flags', 2)
    fin_flags = simple_input.get('fin_flags', 1)
    ack_flags = simple_input.get('ack_flags', 10)

    # Calculate derived metrics
    packet_rate = packet_count / duration if duration > 0 else 0
    avg_packet_size = byte_count / packet_count if packet_count > 0 else 0

    # Build complete feature set with reasonable defaults
    return {
        # Timing features
        'flow_duration': duration,
        'Duration': duration,
        'Rate': packet_rate,
        'Srate': packet_rate * 0.8,  # Estimated response rate
        'IAT': 0.01,  # Inter-arrival time

        # TCP flag counters
        'fin_flag_number': fin_flags,
        'syn_flag_number': syn_flags,
        'rst_flag_number': 0,
        'psh_flag_number': 5,
        'ack_flag_number': ack_flags,
        'ece_flag_number': 0,
        'cwr_flag_number': 0,
        'ack_count': ack_flags,
        'syn_count': syn_flags,
        'fin_count': fin_flags,
        'rst_count': 0,

        # Protocol indicators (binary flags)
        'HTTP': 1, 'HTTPS': 0, 'DNS': 0, 'Telnet': 0,
        'SMTP': 0, 'SSH': 0, 'IRC': 0, 'TCP': 1,
        'UDP': 0, 'DHCP': 0, 'ARP': 0, 'ICMP': 0,
        'IPv': 1, 'LLC': 0,

        # Size and volume metrics
        'Tot sum': byte_count,
        'Tot size': byte_count,
        'Min': avg_packet_size * 0.5,
        'Max': avg_packet_size * 1.5,
        'AVG': avg_packet_size,
        'Std': avg_packet_size * 0.3,

        # Statistical features
        'Number': packet_count,
        'Magnitue': packet_rate,
        'Radius': 50.0,
        'Covariance': 0.5,
        'Variance': 0.25,
        'Weight': 1.0
    }


def track_metrics(func):
    """Decorator to track API call metrics and errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            prediction_duration.observe(duration)
            return result
        except Exception as e:
            error_counter.labels(error_type=type(e).__name__).inc()
            raise
    return wrapper


@app.route('/')
def home():
    """API information and available endpoints."""
    return jsonify({
        "service": "IoT Threat Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_status": "loaded" if model_service.model_loaded else "not_loaded",
        "endpoints": {
            "/health": "Health check with system status",
            "/predict": "Threat prediction (6 basic fields)",
            "/explain": "Get prediction explanation",
            "/model/info": "Model information and features",
            "/metrics": "Prometheus metrics",
            "/stats": "API usage statistics"
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint."""
    try:
        uptime_seconds = (datetime.now() - app_start_time).total_seconds()

        health_data = {
            "status": "healthy" if model_service.model_loaded else "degraded",
            "timestamp": datetime.now().isoformat(),
            "service": "IoT Threat Detection API",
            "uptime_seconds": uptime_seconds,
            "model_status": "loaded" if model_service.model_loaded else "not_loaded",
            "total_predictions": len(recent_predictions),
            "explainer_available": explainer.explainer is not None
        }

        status_code = 200 if model_service.model_loaded else 503
        return jsonify(health_data), status_code

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503


@app.route('/predict', methods=['POST'])
@validate_prediction_input
@track_metrics
def predict():
    """
    Threat prediction endpoint with simple 6-field input.

    Expected input:
    {
        "packet_count": 100,
        "byte_count": 50000,
        "duration": 5.0,
        "syn_flags": 2,
        "fin_flags": 1,
        "ack_flags": 10
    }
    """
    try:
        input_data = request.get_json()
        logger.info(f"Prediction request received")

        # Convert simple input to complete feature set
        complete_features = convert_simple_to_advanced_features(input_data)

        # Get model prediction
        prediction_result = model_service.predict(complete_features)
        prediction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        threat_score = prediction_result['threat_score']

        # Calculate risk assessment
        prediction_label = prediction_result['prediction_label']
        risk_level = calculate_risk_level(confidence, prediction)

        # Update metrics
        prediction_counter.labels(
            prediction_class=prediction_label,
            risk_level=risk_level
        ).inc()

        # Store for monitoring (keep last 100)
        prediction_record = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'threat_score': threat_score,
            'confidence': confidence
        }
        recent_predictions.append(prediction_record)
        if len(recent_predictions) > 100:
            recent_predictions.pop(0)

        # Update threat gauge with current average
        if recent_predictions:
            avg_threat = sum(p['threat_score'] for p in recent_predictions) / len(recent_predictions)
            threat_score_gauge.set(avg_threat)

        # Build response
        response = {
            'status': 'success',
            'prediction': 'threat' if prediction == 1 else 'normal',
            'confidence': float(confidence),
            'risk_level': risk_level,
            'threat_score': float(threat_score),
            'timestamp': datetime.now().isoformat(),
            'request_id': str(uuid.uuid4())
        }

        logger.info(f"Prediction: {response['prediction']} (confidence: {confidence:.3f})")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error_id': str(uuid.uuid4())
        }), 500


@app.route('/explain', methods=['POST'])
@validate_prediction_input
def explain_prediction():
    """Get detailed explanation for a prediction."""
    try:
        input_data = request.get_json()
        complete_features = convert_simple_to_advanced_features(input_data)

        # Get explanation from explainer service
        explanation = explainer.explain_prediction(complete_features)

        return jsonify({
            'status': 'success',
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information and feature list."""
    try:
        if not model_service.model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'unavailable'
            }), 503

        # Get feature summary from explainer
        feature_summary = explainer.get_feature_summary()

        return jsonify({
            'model_type': type(model_service.model).__name__,
            'feature_count': len(model_service.feature_names),
            'feature_names': model_service.feature_names,
            'feature_importance': feature_summary.get('global_feature_importance', []),
            'threat_classes': model_service.threat_classes,
            'model_status': 'loaded'
        }), 200

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def statistics():
    """Get API usage statistics."""
    try:
        uptime_seconds = (datetime.now() - app_start_time).total_seconds()

        if recent_predictions:
            attack_count = sum(1 for p in recent_predictions if p['prediction'] == 1)
            avg_confidence = sum(p['confidence'] for p in recent_predictions) / len(recent_predictions)
            avg_threat = sum(p['threat_score'] for p in recent_predictions) / len(recent_predictions)
        else:
            attack_count = 0
            avg_confidence = 0
            avg_threat = 0

        stats = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime_seconds,
            'total_predictions': len(recent_predictions),
            'attacks_detected': attack_count,
            'attack_rate': attack_count / max(len(recent_predictions), 1),
            'average_confidence': avg_confidence,
            'average_threat_score': avg_threat
        }

        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("IoT Threat Detection API Starting")
    logger.info("=" * 60)
    logger.info(f"Model Status: {'Loaded' if model_service.model_loaded else 'Not Loaded'}")
    logger.info(f"Explainer Status: {'Available' if explainer.explainer else 'Unavailable'}")
    logger.info("Available Endpoints:")
    logger.info("  GET  /           - API information")
    logger.info("  GET  /health     - Health check")
    logger.info("  POST /predict    - Threat prediction")
    logger.info("  POST /explain    - Prediction explanation")
    logger.info("  GET  /model/info - Model details")
    logger.info("  GET  /stats      - Usage statistics")
    logger.info("  GET  /metrics    - Prometheus metrics")
    logger.info("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False)
