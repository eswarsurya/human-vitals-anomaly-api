"""
Test script for Human Vitals Anomaly Detection API
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_single_predict():
    """Test single prediction"""
    data = {
        "heart_rate": 75,
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "temperature": 36.6,
        "spo2": 98
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Single Prediction (Normal Vitals):")
    print(json.dumps(response.json(), indent=2))
    print()

def test_anomaly_predict():
    """Test anomaly prediction"""
    data = {
        "heart_rate": 150,
        "bp_systolic": 180,
        "bp_diastolic": 110,
        "temperature": 39.5,
        "spo2": 85
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Single Prediction (Anomaly Vitals):")
    print(json.dumps(response.json(), indent=2))
    print()

def test_batch_predict():
    """Test batch prediction"""
    data = {
        "data": [
            [75, 120, 80, 36.6, 98],    # Normal
            [150, 180, 110, 39.5, 85],   # Anomaly
            [80, 125, 82, 37.0, 97]      # Normal
        ]
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Batch Prediction (3 samples):")
    print(json.dumps(response.json(), indent=2))
    print()

def test_explain():
    """Test explain endpoint"""
    data = {
        "heart_rate": 150,
        "bp_systolic": 180,
        "bp_diastolic": 110,
        "temperature": 39.5,
        "spo2": 85
    }
    response = requests.post(f"{BASE_URL}/explain", json=data)
    print("Explain Prediction:")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == '__main__':
    print("=" * 60)
    print("Human Vitals Anomaly Detection API - Test Suite")
    print("=" * 60)
    print()

    try:
        test_health()
        test_single_predict()
        test_anomaly_predict()
        test_batch_predict()
        test_explain()

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API. Make sure the server is running:")
        print("  python app.py")
        print("  or")
        print("  docker-compose up")
