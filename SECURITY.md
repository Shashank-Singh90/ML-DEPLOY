# Security Best Practices for IoT Threat Detection API

## Overview

This document outlines security considerations and best practices for deploying and operating the IoT Threat Detection API in production environments.

## Authentication & Authorization

### API Key Authentication

The API supports API key authentication via the `X-API-Key` header. To enable:

1. Generate secure API keys:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

2. Add keys to `app/security.py` in the `VALID_API_KEYS` dictionary
3. In production, load keys from environment variables or a secure key management service

### Recommendations

- **Never commit API keys** to version control
- Rotate API keys regularly (every 90 days minimum)
- Use different keys for different clients/environments
- Implement key revocation mechanism
- Monitor API key usage for anomalies

## Model Security

### Pickle File Integrity

**Risk**: Pickle files can contain malicious code (CWE-502)

**Mitigation**: The application uses SHA-256 checksums to verify model integrity:

- `models/production/model_checksums.json` contains hashes of all model files
- Models are verified before loading
- Any tampering triggers a `ModelPipelineError`

### Best Practices

1. **Protect model files**: Restrict file system permissions
   ```bash
   chmod 600 models/production/*.pkl
   ```

2. **Version control**: Track model versions in MLflow
3. **Audit trail**: Log all model loads and verifications
4. **Secure storage**: Consider encrypting model files at rest

## Network Security

### HTTPS/TLS

**Production deployments MUST use HTTPS**. Configure a reverse proxy (nginx, Traefik):

```nginx
server {
    listen 443 ssl http2;
    server_name api.your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Firewall Rules

- Expose only necessary ports (443 for HTTPS)
- Restrict Prometheus (9090), Grafana (3000), MLflow (5001) to internal network
- Use Docker network isolation

## Input Validation

The API validates all inputs:

- **Type checking**: All fields must be numeric
- **Range validation**: Values limited to ±1e12
- **Required fields**: All 6 network metrics required
- **NaN/Infinity rejection**: Non-finite values rejected

**Custom validation**: Add additional checks in `app/security.py`

## Rate Limiting

Configure rate limits in `app/security.py`:

```python
@limiter.limit("100/minute")  # Adjust per your needs
async def predict(request: Request):
    ...
```

Consider:
- Different limits for different API keys
- Burst allowances
- IP-based limiting for unauthenticated endpoints

## Secrets Management

### Environment Variables

All secrets MUST be in environment variables:

```bash
# .env (NEVER commit this file!)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32)
API_SECRET_KEY=$(openssl rand -base64 32)
MLFLOW_TRACKING_URI=https://mlflow.internal
```

### Production Secret Management

Use a dedicated secrets manager:

- **AWS**: AWS Secrets Manager, Parameter Store
- **Azure**: Azure Key Vault
- **GCP**: Secret Manager
- **Kubernetes**: Sealed Secrets, External Secrets Operator
- **HashiCorp Vault**: For any environment

## Monitoring & Alerting

### Security Monitoring

Monitor these metrics for security events:

- Failed authentication attempts
- Unusual prediction patterns
- High error rates
- Model integrity check failures
- Abnormal resource usage

### Alerts

Configure alerts in `prometheus/alerting_rules.yml`:

- HighErrorRate: May indicate attack
- Unusual Traffic: 3x baseline
- Model tampering detection

## Container Security

### Best Practices

1. **Non-root users**: Containers run as `appuser`/`mlflowuser`
2. **Read-only filesystems**: Where possible
3. **Resource limits**: CPU and memory limits set
4. **Image scanning**: Scan for vulnerabilities
   ```bash
   docker scan iot-threat-api
   ```

5. **Minimal base images**: Using slim Python images
6. **Regular updates**: Update base images monthly

## Data Protection

### Sensitive Data

- **Never log sensitive network traffic details**
- **Anonymize IPs** in logs/metrics
- **Encrypt data at rest**: Model files, MLflow artifacts
- **Secure backups**: Encrypt backup data

### Compliance

Consider regulatory requirements:
- GDPR (EU): Data minimization, right to deletion
- CCPA (California): Consumer data rights
- HIPAA: If handling health data
- PCI DSS: If handling payment data

## Incident Response

### Security Incident Plan

1. **Detection**: Monitor alerts, logs
2. **Containment**: Isolate affected systems
3. **Investigation**: Review logs, identify root cause
4. **Remediation**: Patch vulnerabilities, rotate credentials
5. **Recovery**: Restore from clean backups
6. **Lessons Learned**: Update security controls

### Contact Information

Maintain security contact information:
- Security team email
- On-call rotation
- Escalation procedures

## Vulnerability Disclosure

Report security vulnerabilities to: [security@your-org.com]

- Do NOT open public GitHub issues for vulnerabilities
- Allow 90 days for patching before public disclosure
- Provide detailed reproduction steps

## Security Checklist

Pre-deployment security checklist:

- [ ] Change all default passwords
- [ ] Enable HTTPS/TLS
- [ ] Configure API authentication
- [ ] Set up rate limiting
- [ ] Review and restrict firewall rules
- [ ] Enable security monitoring
- [ ] Configure backup encryption
- [ ] Set up log aggregation
- [ ] Test disaster recovery procedures
- [ ] Document incident response plan
- [ ] Scan containers for vulnerabilities
- [ ] Review and restrict IAM permissions
- [ ] Enable audit logging
- [ ] Configure secret rotation

## Updates & Patches

- **Security updates**: Apply within 7 days
- **Critical vulnerabilities**: Apply within 24 hours
- **Dependency updates**: Monthly review
- **Python version**: Keep current with security patches

## References

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [CWE-502: Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
