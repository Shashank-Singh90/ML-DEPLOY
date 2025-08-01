import pytest
import json
from app.app import app

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
    assert data['status'] == 'ok'

def test_predict_valid_input(client):
    """Test prediction with valid input"""
    test_data = {
        'features': [5.1, 3.5, 1.4, 0.2]  # Iris setosa features
    }
    response = client.post('/predict', 
                          json=test_data,
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'probability' in data
    assert 'class_names' in data

def test_predict_invalid_input(client):
    """Test prediction with invalid input"""
    test_data = {
        'features': [5.1, 3.5]  # Only 2 features instead of 4
    }
    response = client.post('/predict',
                          json=test_data,
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_missing_features(client):
    """Test prediction with missing features"""
    test_data = {}
    response = client.post('/predict',
                          json=test_data,
                          content_type='application/json')
    assert response.status_code == 400