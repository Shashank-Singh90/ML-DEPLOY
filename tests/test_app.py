import pytest
import json
from app.main import app

@pytest.fixture
def client():
    """Create a test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_home_endpoint(client):
    """Test home endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['service'] == 'IoT Threat Detection API'
    assert data['status'] == 'running'

def test_predict_valid_input(client):
    """Test simple prediction with valid input"""
    test_data = {
        "packet_count": 100,
        "byte_count": 50000,
        "duration": 5.0,
        "syn_flags": 2,
        "fin_flags": 1,
        "ack_flags": 10
    }
    response = client.post('/predict',
                          json=test_data,
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'confidence' in data
    assert 'threat_score' in data

def test_predict_invalid_input(client):
    """Test prediction with invalid input"""
    test_data = {
        "packet_count": -100,  # Invalid negative value
        "byte_count": 50000,
        "duration": 5.0,
        "syn_flags": 2,
        "fin_flags": 1,
        "ack_flags": 10
    }
    response = client.post('/predict',
                          json=test_data,
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_missing_features(client):
    """Test prediction with missing features"""
    test_data = {
        "packet_count": 100
        # Missing required fields
    }
    response = client.post('/predict',
                          json=test_data,
                          content_type='application/json')
    assert response.status_code == 400