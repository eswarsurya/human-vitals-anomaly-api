"""
Flask API for Human Vitals Anomaly Detection
Isolation Forest model serving endpoint
"""
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import os

app = Flask(__name__)

# Load pre-trained model (we'll create a dummy one if not exists)
MODEL_PATH = os.path.join('model', 'isolation_forest_model.pkl')

def load_or_create_model():
    """Load existing model or create a dummy one for demo"""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        # Create a dummy model for demonstration
        # In production, replace with your actual trained model
        np.random.seed(42)
        dummy_data = np.random.randn(1000, 5)  # 5 features: heart_rate, bp_systolic, bp_diastolic, temperature, spo2
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(dummy_data)
        joblib.dump(model, MODEL_PATH)
        return model

model = load_or_create_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'isolation_forest_v1',
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict anomaly for human vitals

    Expected JSON format:
    {
        "heart_rate": 75,
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "temperature": 36.6,
        "spo2": 98
    }

    Or batch:
    {
        "data": [
            [75, 120, 80, 36.6, 98],
            [150, 180, 110, 39.5, 85]
        ]
    }
    """
    try:
        data = request.get_json()

        # Single prediction
        if 'data' not in data:
            features = np.array([[
                data.get('heart_rate', 0),
                data.get('bp_systolic', 0),
                data.get('bp_diastolic', 0),
                data.get('temperature', 0),
                data.get('spo2', 0)
            ]])
        else:
            # Batch prediction
            features = np.array(data['data'])

        # Predict: -1 = anomaly, 1 = normal
        predictions = model.predict(features)
        scores = model.decision_function(features)

        results = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            results.append({
                'sample_id': i,
                'is_anomaly': bool(pred == -1),
                'anomaly_score': float(score),
                'confidence': float(abs(score)),
                'status': 'ANOMALY' if pred == -1 else 'NORMAL'
            })

        return jsonify({
            'success': True,
            'model': 'isolation_forest_v1',
            'predictions': results,
            'total_samples': len(results),
            'anomalies_detected': sum(1 for r in results if r['is_anomaly'])
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/explain', methods=['POST'])
def explain():
    """
    Simple explanation endpoint (SHAP-like feature importance)
    Returns which features contributed most to anomaly score
    """
    try:
        data = request.get_json()
        features = np.array([[
            data.get('heart_rate', 0),
            data.get('bp_systolic', 0),
            data.get('bp_diastolic', 0),
            data.get('temperature', 0),
            data.get('spo2', 0)
        ]])

        score = model.decision_function(features)[0]

        # Simple feature contribution (approximation)
        feature_names = ['heart_rate', 'bp_systolic', 'bp_diastolic', 'temperature', 'spo2']

        return jsonify({
            'success': True,
            'anomaly_score': float(score),
            'is_anomaly': bool(score < 0),
            'feature_names': feature_names,
            'input_values': features[0].tolist(),
            'note': 'For full SHAP explainability, integrate shap library'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
