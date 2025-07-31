from flask import Flask, request, jsonify
import logging
from datetime import datetime, timedelta
import traceback
import time
from functools import wraps
from models.model_service import ModelService
from models.explainer import ModelExplainer
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Custom Prometheus metrics for IoT security
iot_predictions_total = Counter(
    'iot_predictions_total',
    'Total number of IoT threat predictions',
    ['prediction_class', 'risk_level']
)

iot_prediction_duration = Histogram(
    'iot_prediction_duration_seconds',
    'Time spent on IoT threat prediction',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

iot_model_confidence = Summary(
    'iot_model_confidence',
    'IoT model prediction confidence scores'
)

iot_threat_score_gauge = Gauge(
    'iot_current_threat_score',
    'Current average threat score over last 100 predictions'
)

iot_attack_rate_gauge = Gauge(
    'iot_attack_rate_last_hour',
    'Attack detection rate in the last hour'
)

iot_feature_importance_gauge = Gauge(
    'iot_feature_importance',
    'Feature importance scores',
    ['feature_name']
)

iot_errors_total = Counter(
    'iot_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

iot_explainer_requests = Counter(
    'iot_explainer_requests_total',
    'Total number of explanation requests'
)

# Monitoring data storage
recent_predictions = []
recent_threat_scores = []
start_time = datetime.now()

# Initialize services
logger.info("🚀 Initializing IoT Threat Detection System with Monitoring...")
model_service = ModelService()
explainer = ModelExplainer(model_service)

# Set up feature importance gauges
if hasattr(model_service.model, 'feature_importances_') and model_service.feature_names:
    for feature, importance in zip(model_service.feature_names, model_service.model.feature_importances_):
        iot_feature_importance_gauge.labels(feature_name=feature).set(importance)

def track_prediction_metrics(func):
    """Decorator to track prediction metrics"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Record duration
            duration = time.time() - start_time
            iot_prediction_duration.observe(duration)
            
            return result
            
        except Exception as e:
            # Record error
            iot_errors_total.labels(error_type=type(e).__name__).inc()
            raise
    
    return wrapper

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with monitoring status"""
    try:
        # Check if model is loaded
        if model_service.model is None:
            return jsonify({
                "status": "unhealthy",
                "error": "Model not loaded"
            }), 503
        
        uptime = (datetime.now() - start_time).total_seconds()
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_type": "IoT Threat Detection",
            "features_count": len(model_service.feature_names) if model_service.feature_names else 0,
            "service": "IoT Cybersecurity API",
            "explainability": "enabled" if explainer.explainer is not None else "disabled",
            "monitoring": {
                "enabled": True,
                "uptime_seconds": uptime,
                "total_predictions": len(recent_predictions),
                "metrics_endpoint": "/metrics"
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

@app.route('/predict', methods=['POST'])
@track_prediction_metrics
def predict():
    """IoT threat detection prediction endpoint with monitoring"""
    try:
        # Get features from request
        data = request.json
        
        if not data:
            iot_errors_total.labels(error_type="MissingData").inc()
            return jsonify({
                "error": "No data provided. Please send IoT network traffic features."
            }), 400
        
        # Log prediction request
        logger.info(f"IoT threat prediction requested at {datetime.now().isoformat()}")
        
        # Make prediction
        result = model_service.predict(data)
        
        # Track metrics
        prediction_label = result['prediction_label']
        risk_level = result['risk_level']
        confidence = result['confidence']
        threat_score = result['threat_score']
        
        # Update Prometheus metrics
        iot_predictions_total.labels(
            prediction_class=prediction_label,
            risk_level=risk_level
        ).inc()
        
        iot_model_confidence.observe(confidence)
        
        # Store recent data for gauges
        recent_predictions.append({
            'timestamp': datetime.now(),
            'prediction': result['prediction'],
            'threat_score': threat_score,
            'confidence': confidence
        })
        
        recent_threat_scores.append(threat_score)
        
        # Keep only last 100 predictions
        if len(recent_predictions) > 100:
            recent_predictions.pop(0)
        if len(recent_threat_scores) > 100:
            recent_threat_scores.pop(0)
        
        # Update gauges
        if recent_threat_scores:
            iot_threat_score_gauge.set(sum(recent_threat_scores) / len(recent_threat_scores))
        
        # Calculate attack rate in last hour
        one_hour_ago = datetime.now() - datetime.timedelta(hours=1)
        recent_hour_predictions = [
            p for p in recent_predictions 
            if p['timestamp'] > one_hour_ago
        ]
        
        if recent_hour_predictions:
            attack_count = sum(1 for p in recent_hour_predictions if p['prediction'] == 1)
            attack_rate = attack_count / len(recent_hour_predictions)
            iot_attack_rate_gauge.set(attack_rate)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['model'] = 'IoT Threat Detection'
        
        # Log result for monitoring
        logger.info(f"Prediction: {result['prediction_label']} (confidence: {result['confidence']:.3f})")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        iot_errors_total.labels(error_type="PredictionError").inc()
        
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/explain', methods=['POST'])
def explain_prediction():
    """Get explanation for IoT traffic prediction with monitoring"""
    try:
        # Track explanation requests
        iot_explainer_requests.inc()
        
        # Get features from request
        data = request.json
        
        if not data:
            return jsonify({
                "error": "No data provided. Please send IoT network traffic features."
            }), 400
        
        # Get number of top features to explain (default 10)
        top_k = request.args.get('top_k', 10, type=int)
        
        logger.info(f"Explanation requested for {top_k} top features")
        
        # First get the prediction
        prediction_result = model_service.predict(data)
        
        # Then get the explanation
        explanation_result = explainer.explain_prediction(data, top_k=top_k)
        
        # Combine results
        result = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction_result,
            "explanation": explanation_result,
            "model": "IoT Threat Detection with Explanation"
        }
        
        logger.info(f"Explanation generated for {prediction_result['prediction_label']} prediction")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        logger.error(traceback.format_exc())
        iot_errors_total.labels(error_type="ExplanationError").inc()
        
        return jsonify({
            "error": "Explanation failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict/batch', methods=['POST'])
@track_prediction_metrics
def predict_batch():
    """Batch prediction endpoint for multiple IoT samples with monitoring"""
    try:
        data = request.json
        
        if not data or 'samples' not in data:
            return jsonify({
                "error": "No samples provided. Send data in format: {'samples': [sample1, sample2, ...]}"
            }), 400
        
        samples = data['samples']
        if not isinstance(samples, list):
            return jsonify({
                "error": "Samples must be a list"
            }), 400
        
        # Process each sample
        results = []
        for i, sample in enumerate(samples):
            try:
                prediction = model_service.predict(sample)
                prediction['sample_id'] = i
                results.append(prediction)
                
                # Track batch metrics
                iot_predictions_total.labels(
                    prediction_class=prediction['prediction_label'],
                    risk_level=prediction['risk_level']
                ).inc()
                
            except Exception as e:
                results.append({
                    'sample_id': i,
                    'error': str(e)
                })
                iot_errors_total.labels(error_type="BatchPredictionError").inc()
        
        # Calculate batch statistics
        successful_predictions = [r for r in results if 'error' not in r]
        attack_count = sum(1 for r in successful_predictions if r['prediction'] == 1)
        
        batch_result = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(samples),
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(samples) - len(successful_predictions),
            'attack_count': attack_count,
            'attack_rate': attack_count / max(len(successful_predictions), 1),
            'predictions': results
        }
        
        logger.info(f"Batch prediction: {len(samples)} samples, {attack_count} attacks detected")
        
        return jsonify(batch_result), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        iot_errors_total.labels(error_type="BatchError").inc()
        return jsonify({
            "error": "Batch prediction failed",
            "message": str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the current model"""
    try:
        info = {
            'model_type': 'Random Forest Classifier',
            'task': 'IoT Network Threat Detection',
            'classes': model_service.class_names,
            'feature_count': len(model_service.feature_names) if model_service.feature_names else 0,
            'features': model_service.feature_names if model_service.feature_names else [],
            'description': 'Detects cyber threats in IoT network traffic using machine learning',
            'explainability': {
                'available': explainer.explainer is not None,
                'method': 'Feature Importance + Local Analysis',
                'explanation_types': ['feature_importance', 'local_explanations', 'global_feature_ranking']
            },
            'monitoring': {
                'metrics_enabled': True,
                'tracked_metrics': [
                    'predictions_total', 'prediction_duration', 'model_confidence',
                    'threat_score', 'attack_rate', 'feature_importance', 'errors_total'
                ]
            }
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve model information",
            "message": str(e)
        }), 500

@app.route('/model/features', methods=['GET'])
def feature_importance():
    """Get global feature importance from the model"""
    try:
        feature_summary = explainer.get_feature_summary()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'model': 'IoT Threat Detection',
            'feature_analysis': feature_summary
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve feature importance",
            "message": str(e)
        }), 500

@app.route('/monitoring/stats', methods=['GET'])
def monitoring_stats():
    """Get current monitoring statistics"""
    try:
        uptime = (datetime.now() - start_time).total_seconds()
        
        # Calculate recent statistics
        one_hour_ago = datetime.now() - datetime.timedelta(hours=1)
        recent_hour_predictions = [
            p for p in recent_predictions 
            if p['timestamp'] > one_hour_ago
        ]
        
        attack_count_hour = sum(1 for p in recent_hour_predictions if p['prediction'] == 1)
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'total_predictions': len(recent_predictions),
            'recent_hour': {
                'total_predictions': len(recent_hour_predictions),
                'attack_predictions': attack_count_hour,
                'attack_rate': attack_count_hour / max(len(recent_hour_predictions), 1)
            },
            'current_metrics': {
                'avg_threat_score': sum(recent_threat_scores) / len(recent_threat_scores) if recent_threat_scores else 0,
                'avg_confidence': sum(p['confidence'] for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0
            },
            'system_health': 'healthy' if len(recent_predictions) > 0 else 'no_traffic'
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Monitoring stats error: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve monitoring stats",
            "message": str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test_prediction():
    """Test endpoint with sample IoT data"""
    try:
        # Sample IoT network traffic data
        sample_data = {
            "flow_duration": 1.5,
            "Duration": 2.3,
            "Rate": 150.5,
            "Srate": 100.2,
            "fin_flag_number": 1,
            "syn_flag_number": 2,
            "rst_flag_number": 0,
            "psh_flag_number": 5,
            "ack_flag_number": 10,
            "ece_flag_number": 0,
            "cwr_flag_number": 0,
            "ack_count": 15,
            "syn_count": 3,
            "fin_count": 1,
            "rst_count": 0,
            "HTTP": 1,
            "HTTPS": 0,
            "DNS": 0,
            "Telnet": 0,
            "SMTP": 0,
            "SSH": 0,
            "IRC": 0,
            "TCP": 1,
            "UDP": 0,
            "DHCP": 0,
            "ARP": 0,
            "ICMP": 0,
            "IPv": 1,
            "LLC": 0,
            "Tot sum": 1500.5,
            "Min": 40.0,
            "Max": 1460.0,
            "AVG": 750.25,
            "Std": 450.3,
            "Tot size": 15000,
            "IAT": 0.01,
            "Number": 20,
            "Magnitue": 100.5,
            "Radius": 50.2,
            "Covariance": 0.8,
            "Variance": 0.64,
            "Weight": 1.0
        }
        
        result = model_service.predict(sample_data)
        result['note'] = 'This is a test prediction with sample IoT network data'
        result['sample_data'] = sample_data
        result['available_endpoints'] = {
            'explain': '/explain - Get explanation for this prediction',
            'features': '/model/features - Get global feature importance',
            'monitoring': '/monitoring/stats - Get monitoring statistics',
            'metrics': '/metrics - Prometheus metrics endpoint'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Test prediction error: {str(e)}")
        return jsonify({
            "error": "Test prediction failed",
            "message": str(e)
        }), 500

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    logger.info("🚀 Starting Advanced IoT Cybersecurity API with Full Monitoring")
    logger.info("🔍 SHAP Explainability: ENABLED")
    logger.info("📊 Prometheus Metrics: ENABLED")
    logger.info("📈 Real-time Monitoring: ENABLED")
    logger.info("📡 Available Endpoints:")
    logger.info("   - GET  /health - Health check with monitoring status")
    logger.info("   - POST /predict - Single threat prediction")
    logger.info("   - POST /explain - Feature explanation")
    logger.info("   - POST /predict/batch - Batch predictions")
    logger.info("   - GET  /model/info - Model information")
    logger.info("   - GET  /model/features - Global feature importance")
    logger.info("   - GET  /monitoring/stats - Monitoring statistics")
    logger.info("   - GET  /metrics - Prometheus metrics")
    logger.info("   - GET  /test - Test with sample data")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
    