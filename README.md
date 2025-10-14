# IoT Threat Detection API

A production-ready machine learning API for detecting cybersecurity threats in IoT network traffic using Random Forest classification.

## Features

- **Real-time Threat Detection**: ML-based classification of IoT network traffic
- **Simple API**: Easy-to-use REST endpoints with 6-field input
- **Model Explainability**: Feature importance analysis for predictions
- **Production Monitoring**: Prometheus metrics and Grafana dashboards
- **Docker Support**: Containerized deployment with health checks

## Quick Start

### Using Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# API available at http://localhost:5000
# Prometheus at http://localhost:9090
# Grafana at http://localhost:3000
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app/main.py
```

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and status |
| GET | `/health` | Health check with system metrics |
| POST | `/predict` | Threat prediction (6 network fields) |
| POST | `/explain` | Get prediction explanation |
| GET | `/model/info` | Model information and features |
| GET | `/stats` | Usage statistics |
| GET | `/metrics` | Prometheus metrics |

### Example: Threat Prediction

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "packet_count": 150,
    "byte_count": 75000,
    "duration": 5.0,
    "syn_flags": 15,
    "fin_flags": 2,
    "ack_flags": 10
  }'
```

**Response:**
```json
{
  "status": "success",
  "prediction": "threat",
  "confidence": 0.87,
  "risk_level": "high",
  "threat_score": 0.87,
  "timestamp": "2024-10-04T12:00:00Z",
  "request_id": "uuid"
}
```

## Input Format

### Simple Prediction (6 fields)

```json
{
  "packet_count": 100,
  "byte_count": 50000,
  "duration": 5.0,
  "syn_flags": 2,
  "fin_flags": 1,
  "ack_flags": 10
}
```

These basic metrics are automatically converted to the complete 42-feature set required by the ML model.

## Project Structure

```
.
├── app/                      # Application code
│   ├── main.py              # Flask API server
│   ├── validators.py        # Input validation
│   └── models/              # ML model components
│       ├── model_service.py # Model loading and prediction
│       └── explainer.py     # Prediction explanations
├── models/                   # Trained model storage
│   └── production/          # Production models
├── scripts/                  # Utility scripts
│   ├── generate_sample_data.py  # Generate synthetic data
│   └── train_model.py           # Train new model
├── tests/                    # Test suite
├── prometheus/              # Monitoring configuration
├── grafana/                 # Dashboard configurations
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service orchestration
└── requirements.txt         # Python dependencies
```

## Model Training

To train a new model with your own data:

```bash
# Generate synthetic training data
python scripts/generate_sample_data.py

# Train the model
python scripts/train_model.py
```

The model is automatically saved to `models/production/` and loaded on startup.

## Monitoring

The API includes comprehensive monitoring:

- **Prometheus Metrics**: Request counts, response times, error rates
- **Grafana Dashboards**: Visual monitoring and alerting
- **Health Checks**: Automated container health monitoring
- **Structured Logging**: Detailed application logs

Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

## Development

### Running Tests

```bash
# Run test suite
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

### Code Quality

```bash
# Format code
black app/

# Lint code
flake8 app/
```

## Security Features

- Non-root container user
- Input validation on all endpoints
- No sensitive data logging
- Secure model loading
- Health check monitoring

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONPATH` | Python path | `/app` |
| `PYTHONDONTWRITEBYTECODE` | Disable .pyc files | `1` |
| `PYTHONUNBUFFERED` | Unbuffered output | `1` |

## Production Deployment

The application is production-ready with:

- Automatic container restarts
- Health check endpoints
- Prometheus metrics export
- Resource limits and optimization
- Security best practices

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review the test suite for examples
