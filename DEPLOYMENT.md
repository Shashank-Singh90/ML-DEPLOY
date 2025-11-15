# Production Deployment Guide

## Overview

This guide covers deploying the IoT Threat Detection API to production environments.

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Minimum 4GB RAM, 2 CPU cores
- 20GB disk space
- Linux server (Ubuntu 20.04+ recommended)

## Quick Start

### 1. Clone and Configure

```bash
git clone <repository-url>
cd ML-DEPLOY
cp .env.example .env
```

### 2. Set Secure Credentials

```bash
# Edit .env and set secure passwords
nano .env

# Generate secure passwords
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32)
```

### 3. Build and Deploy

```bash
docker-compose up -d --build
```

### 4. Verify Deployment

```bash
# Check all services are running
docker-compose ps

# Test API health
curl http://localhost:5000/health

# Access services
# - API: http://localhost:5000
# - Grafana: http://localhost:3000 (admin/your-password)
# - Prometheus: http://localhost:9090
# - MLflow: http://localhost:5001
```

## Production Configuration

### Environment Variables

Required variables in `.env`:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5001
MLFLOW_BACKEND_STORE_URI=file:///mlflow/backend
MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts

# Grafana (CHANGE THESE!)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=<secure-password>

# API
PYTHONPATH=/app
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
```

### Reverse Proxy (nginx)

```nginx
upstream iot_api {
    server localhost:5000;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://iot_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL Certificates

Using Let's Encrypt:

```bash
# Install certbot
sudo apt-get install certbot

# Get certificate
sudo certbot certonly --standalone -d api.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

## Scaling

### Horizontal Scaling

Run multiple API instances behind a load balancer:

```yaml
# docker-compose.override.yml
services:
  api:
    deploy:
      replicas: 3
```

### Vertical Scaling

Adjust resource limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 4G
```

## Monitoring

### Metrics Collection

Prometheus scrapes metrics from:
- API: http://iot-threat-api:5000/metrics
- Prometheus itself

### Dashboards

Access Grafana at http://localhost:3000:

1. Login with admin credentials
2. Navigate to Dashboards
3. Import `grafana/dashboards/Iot-security-dashboard.json`

### Alerts

Configure alerting in Prometheus:

```bash
# Edit prometheus/alerting_rules.yml
# Configure notification channels in Grafana
```

## Backup Strategy

### Database Backups

```bash
# Backup MLflow database
docker-compose exec mlflow tar czf /tmp/mlflow-backup.tar.gz /mlflow/backend
docker cp iot-mlflow:/tmp/mlflow-backup.tar.gz ./backups/

# Backup Prometheus data
docker-compose exec prometheus tar czf /tmp/prometheus-backup.tar.gz /prometheus
docker cp iot-prometheus:/tmp/prometheus-backup.tar.gz ./backups/
```

### Model Backups

```bash
# Backup production models
tar czf models-backup-$(date +%Y%m%d).tar.gz models/production/
```

### Automated Backups

```bash
# Add to crontab
0 2 * * * /path/to/backup-script.sh
```

## Disaster Recovery

### Recovery Plan

1. **Model Corruption**: Restore from latest backup
2. **Data Loss**: Restore from backups
3. **Service Outage**: Restart services, check logs
4. **Security Breach**: Rotate credentials, check integrity

### Restore Procedure

```bash
# Stop services
docker-compose down

# Restore backups
tar xzf backups/models-backup-YYYYMMDD.tar.gz

# Restart
docker-compose up -d

# Verify
curl http://localhost:5000/health
```

## Health Checks

Monitor these endpoints:

- `/health` - API health status
- `/metrics` - Prometheus metrics
- MLflow UI availability
- Prometheus query execution
- Grafana dashboard loading

## Logging

### Centralized Logging

Configure log shipping to ELK/Loki:

```yaml
# docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Log Rotation

```bash
# /etc/logrotate.d/docker-containers
/var/lib/docker/containers/*/*.log {
  rotate 7
  daily
  compress
  missingok
  delaycompress
  copytruncate
}
```

## Performance Tuning

### API Optimization

- Enable Gunicorn workers: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker`
- Use connection pooling
- Enable response caching
- Optimize model loading

### Database Optimization

- Regular MLflow database cleanup
- Prometheus data retention tuning
- Index optimization

## Security Hardening

See [SECURITY.md](SECURITY.md) for detailed security practices:

- Enable HTTPS
- Configure authentication
- Set up rate limiting
- Restrict network access
- Enable audit logging

## Troubleshooting

### Common Issues

**API won't start**:
```bash
docker-compose logs api
# Check model files exist
ls -la models/production/
```

**Prometheus not scraping**:
```bash
# Check network connectivity
docker-compose exec prometheus wget -O- http://iot-threat-api:5000/metrics
```

**High memory usage**:
```bash
# Check resource usage
docker stats

# Adjust limits in docker-compose.yml
```

**Model loading fails**:
```bash
# Check checksums
cat models/production/model_checksums.json

# Regenerate if needed
python scripts/generate_checksums.py
```

## Updates

### Rolling Updates

```bash
# Build new image
docker-compose build api

# Restart with zero downtime (requires load balancer)
docker-compose up -d --no-deps --scale api=2 api
docker-compose up -d --no-deps --scale api=1 api
```

### Database Migrations

```bash
# Backup first!
./backup-script.sh

# Apply migrations
docker-compose exec api python -m alembic upgrade head
```

## Maintenance Windows

Schedule regular maintenance:

- Weekly: Review logs and metrics
- Monthly: Apply security updates
- Quarterly: Full system backup test
- Annually: Disaster recovery drill

## Support & Monitoring

### On-Call Rotation

- Monitor Prometheus alerts
- Check health endpoints
- Review error logs
- Respond to incidents per [SECURITY.md](SECURITY.md)

### Performance Metrics

Track these KPIs:

- API latency (p50, p95, p99)
- Prediction accuracy
- Error rate
- Resource utilization
- Uptime percentage

## Compliance

Document compliance requirements:

- Data retention policies
- Access control logs
- Audit trail
- Change management
- Incident response

## References

- [Docker Compose Production](https://docs.docker.com/compose/production/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
