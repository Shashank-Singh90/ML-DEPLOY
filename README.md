# IoT Threat Detection API

ML-based API for detecting cybersecurity threats in IoT network traffic.

## Features

- Real-time threat detection with Random Forest classifier
- REST API with simple 6-field input
- MLflow experiment tracking and model versioning
- Prometheus metrics and Grafana dashboards
- Docker deployment with health checks

## Quick Start

### Docker

```bash
docker-compose up --build
```

Services:
- API: http://localhost:5000
- MLflow: http://localhost:5001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Local

```bash
pip install -r requirements.txt
python scripts/generate_sample_data.py
python scripts/train_model.py
uvicorn app.main:app --host 0.0.0.0 --port 5000
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

### Example

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

## MLflow

Access MLflow UI at http://localhost:5001

Tracked metrics:
- Training: accuracy, precision, recall, f1_score
- Model parameters and feature importance
- Prediction confidence and threat probability

## Docker

```bash
# Start
docker-compose up -d

# Logs
docker-compose logs -f api

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build
```

## Development

```bash
pytest tests/
pytest --cov=app tests/
black app/
flake8 app/
```

## Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## License

MIT License
