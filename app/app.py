from flask import Flask, request, jsonify
import logging
from app.model import ModelService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize model service
model_service = ModelService()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return jsonify({"status": "ok"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get features from request
        data = request.json
        features = data.get('features', [])

        # Log the prediction request
        logger.info(f"Prediction requested with features: {features}")

        # Validate input
        if not features or len(features) != 4:
            return jsonify({
                "error": "Invalid input. Please provide 4 features."
            }), 400

        # Make prediction
        result = model_service.predict(features)

        logger.info(f"Prediction result: {result}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
