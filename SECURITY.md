# Security Policy

## Security Improvements (v1.1.0)

This document outlines the security improvements made to the IoT Threat Detection API.

### Critical Fixes

#### 1. Secure Serialization (CWE-502)
**Issue:** Unsafe pickle deserialization could lead to arbitrary code execution.
**Fix:** Replaced `pickle` with `joblib` for model serialization.
- `app/runtime.py`: Uses joblib for loading/saving models
- Detects Git LFS pointer files to prevent loading invalid data

#### 2. API Authentication
**Issue:** No authentication on any endpoints.
**Fix:** Implemented API key authentication.
- Set `API_KEYS` environment variable with comma-separated keys
- All endpoints (except `/health` and `/metrics`) require `X-API-Key` header
- Generate keys: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

Example:
```bash
curl -H "X-API-Key: your-secret-key" http://localhost:5000/predict
```

#### 3. Race Condition Protection
**Issue:** Thread-unsafe shared state in prediction storage.
**Fix:** Implemented thread-safe access with `threading.Lock()`.
- Uses `deque` with maxlen for automatic size management
- All access to `recent_predictions` is protected by lock

#### 4. Rate Limiting
**Issue:** No protection against DoS attacks.
**Fix:** Implemented rate limiting with `slowapi`.
- `/predict`: 60 requests/minute
- `/explain`: 30 requests/minute
- `/health`: 100 requests/minute
- Other endpoints: 30 requests/minute

#### 5. Information Disclosure
**Issue:** Error messages leaked internal details.
**Fix:** Generic error messages to users, detailed logging internally.

#### 6. Request Size Limits
**Issue:** No limits on request payload size.
**Fix:** 10MB maximum request size enforced.

### Additional Security Features

- **CORS Configuration:** Configurable via `CORS_ORIGINS` environment variable
- **Credential Management:** Environment variables for Grafana passwords
- **Input Validation:** Comprehensive validation of all numeric inputs
- **Error Handling:** Safe error handling without information leakage

## Reporting a Vulnerability

To report a security vulnerability, please email [security contact] with:
1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

## Security Best Practices

### For Production Deployment

1. **Set Strong API Keys**
   ```bash
   # Generate secure keys
   python -c "import secrets; print(','.join([secrets.token_urlsafe(32) for _ in range(3)]))"
   export API_KEYS="key1,key2,key3"
   ```

2. **Change Default Credentials**
   ```bash
   export GRAFANA_ADMIN_USER="your_admin_user"
   export GRAFANA_ADMIN_PASSWORD="strong_password_here"
   ```

3. **Configure CORS Properly**
   ```bash
   # Restrict to specific origins
   export CORS_ORIGINS="https://your-frontend.com,https://api.your-domain.com"
   ```

4. **Use HTTPS**
   - Deploy behind a reverse proxy (nginx, Caddy)
   - Enforce TLS 1.2+ only
   - Use valid SSL certificates

5. **Network Security**
   - Use Docker network isolation
   - Expose only necessary ports
   - Use firewall rules

6. **Monitoring**
   - Enable Prometheus alerts
   - Monitor for unusual request patterns
   - Track error rates

### Security Checklist

- [ ] API keys configured and rotated regularly
- [ ] Default credentials changed
- [ ] HTTPS enabled
- [ ] CORS properly configured
- [ ] Rate limits appropriate for your use case
- [ ] Monitoring and alerting configured
- [ ] Regular security updates applied
- [ ] Logs reviewed regularly
- [ ] Backup and disaster recovery plan in place

## Dependencies

Regular updates recommended for:
- FastAPI
- scikit-learn
- pandas
- numpy
- mlflow
- joblib
- slowapi

Run `pip list --outdated` regularly and update dependencies.

## Known Limitations

1. **API Key Storage:** Keys are stored in environment variables. For enterprise use, consider:
   - HashiCorp Vault
   - AWS Secrets Manager
   - Azure Key Vault

2. **Model Integrity:** While joblib is safer than pickle, consider:
   - Digital signatures for model files
   - Model versioning and rollback capabilities
   - Regular model audits

3. **Logging:** Logs may contain sensitive information. Ensure:
   - Log rotation configured
   - Access controls on log files
   - No PII in logs

## Security Contact

For security concerns, please contact the development team.

Last updated: 2025-11-18
