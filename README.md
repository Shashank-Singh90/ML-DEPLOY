# IoT Threat Detection API

A production-ready machine learning API for detecting cybersecurity threats in IoT network traffic using Random Forest classification.

## Features

- **Real-time Threat Detection**: ML-based classification of IoT network traffic
- **Simple API**: Easy-to-use REST endpoints with 6-field input
- **Model Explainability**: Feature importance analysis for predictions
- **MLflow Integration**: Experiment tracking, model versioning, and ML lifecycle management
- **Production Monitoring**: Prometheus metrics and Grafana dashboards
- **Docker Support**: Containerized deployment with health checks

## Quick Start

### Using Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Services available at:
# - API: http://localhost:5000
# - MLflow UI: http://localhost:5001
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data
python scripts/generate_sample_data.py

# Train model (optional - model included)
python scripts/train_model.py

# Run the application
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and status |
| GET | `/health` | Health check with system metrics |
| POST | `/predict` | Threat prediction (6 network fields) |
| POST | `/explain` | Get prediction explanation |
| GET | `/model/info` | Model information and features |
| GET | `/stats` | Usage statistics |
| GET | `/metrics` | Prometheus metrics |
| GET | `/mlflow/info` | MLflow tracking information |

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

The API accepts 6 simple network metrics that are automatically expanded to 42 model features:

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

## Project Structure

```
.
├── app/
│   ├── main.py              # FastAPI server and routing
│   └── runtime.py           # Model service, inference, explanations
├── models/production/       # Trained model assets
├── scripts/
│   ├── generate_sample_data.py  # Synthetic data generator
│   └── train_model.py           # Model training with MLflow
├── tests/
│   └── test_app.py         # API tests
├── prometheus/             # Prometheus configuration
├── grafana/                # Grafana dashboards
├── Dockerfile              # API container definition
├── Dockerfile.mlflow       # MLflow container definition
├── docker-compose.yml      # Multi-service orchestration
└── requirements.txt        # Python dependencies
```

## MLflow Integration

MLflow provides experiment tracking and model versioning:

### Accessing MLflow UI
Visit http://localhost:5001 to view:
- Training runs and metrics
- Model registry
- Feature importance
- Experiment comparisons

### What's Tracked

**Training Metrics:**
- accuracy, precision, recall, f1_score
- Model parameters (n_estimators, max_depth, etc.)
- Feature importance

**Prediction Metrics:**
- prediction_confidence
- threat_probability

### Check MLflow Status
```bash
curl http://localhost:5000/mlflow/info
```

## Docker Deployment

### Service Architecture

```
┌─────────────────────────────────────┐
│       Docker Network: monitoring     │
├─────────────────────────────────────┤
│  ┌──────────┐  ┌─────────────┐     │
│  │ IoT API  │  │   MLflow    │     │
│  │  :5000   │  │   :5001     │     │
│  └────┬─────┘  └──────┬──────┘     │
│       │               │             │
│  ┌────▼──────┐  ┌────▼──────┐     │
│  │Prometheus │  │  Grafana  │     │
│  │  :9090    │  │   :3000   │     │
│  └───────────┘  └───────────┘     │
└─────────────────────────────────────┘
```

### Common Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Rebuild containers
docker-compose up -d --build

# Check health
docker-compose ps
curl http://localhost:5000/health
```

### Volume Management

```bash
# List volumes
docker volume ls | grep mldeployiotcybersecurity

# Backup MLflow data
docker run --rm \
  -v mldeployiotcybersecurity_mlflow-artifacts:/source \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/mlflow-artifacts.tar.gz -C /source .

# Remove all volumes (WARNING: Data loss!)
docker-compose down -v
```

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

## Monitoring

### Prometheus Metrics
- Request counts by prediction class and risk level
- Response time histograms
- Error counters
- Average threat scores

Access at: http://localhost:9090

### Grafana Dashboards
Pre-configured dashboards for:
- API performance metrics
- Threat detection rates
- System health monitoring

Access at: http://localhost:3000 (admin/admin)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URL | `http://mlflow:5001` |
| `PYTHONPATH` | Python path | `/app` |
| `PYTHONDONTWRITEBYTECODE` | Disable .pyc files | `1` |
| `PYTHONUNBUFFERED` | Unbuffered output | `1` |

See `.env.example` for complete configuration options.

## Security Features

- Non-root container user
- Input validation on all endpoints
- No sensitive data logging
- Secure model loading
- Health check monitoring
- Network isolation via Docker

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs api

# Restart service
docker-compose restart api
```

### MLflow Connection Issues
```bash
# Verify MLflow is running
docker-compose ps mlflow

# Test connection from API
docker-compose exec api curl http://mlflow:5001/health
```

### Port Conflicts
```bash
# Check port usage (Windows)
netstat -ano | findstr :5000

# Change ports in docker-compose.yml if needed
```

## Production Deployment

The application is production-ready with:
- Automatic container restarts
- Health check endpoints
- Prometheus metrics export
- Resource limits and optimization
- Security best practices

## License

MIT License

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review the test suite for examples
