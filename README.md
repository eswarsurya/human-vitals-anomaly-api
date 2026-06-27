# Human Vitals Anomaly Detection API

A production-ready Flask REST API for real-time anomaly detection in human vital signs using Isolation Forest machine learning.

## Features

- **Real-time anomaly detection** for heart rate, blood pressure, temperature, and SpO2
- **Batch prediction** support for multiple samples
- **Explainability endpoint** for feature contribution analysis
- **Health check** endpoint for monitoring
- **Docker containerization** for cloud deployment
- **Production-ready** with Gunicorn WSGI server

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py

# Test health endpoint
curl http://localhost:5000/health
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t vitals-anomaly-api .
docker run -p 5000:5000 vitals-anomaly-api
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Predict Anomaly (Single)
```bash
POST /predict
Content-Type: application/json

{
    "heart_rate": 75,
    "bp_systolic": 120,
    "bp_diastolic": 80,
    "temperature": 36.6,
    "spo2": 98
}
```

### Predict Anomaly (Batch)
```bash
POST /predict
Content-Type: application/json

{
    "data": [
        [75, 120, 80, 36.6, 98],
        [150, 180, 110, 39.5, 85]
    ]
}
```

### Explain Prediction
```bash
POST /explain
Content-Type: application/json

{
    "heart_rate": 150,
    "bp_systolic": 180,
    "bp_diastolic": 110,
    "temperature": 39.5,
    "spo2": 85
}
```

## Model Details

- **Algorithm**: Isolation Forest
- **Training Data**: 200,020 multivariate records (MSc thesis dataset)
- **Features**: Heart rate, BP systolic, BP diastolic, temperature, SpO2
- **Contamination**: 5% (expected anomaly rate)
- **Explainability**: SHAP-ready architecture for clinical transparency

## Cloud Deployment

### Azure Container Instances
```bash
az container create \
    --resource-group myResourceGroup \
    --name vitals-anomaly-api \
    --image vitals-anomaly-api:latest \
    --ports 5000
```

### AWS ECS/Fargate
```bash
# Push to ECR and deploy via ECS task definition
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag vitals-anomaly-api:latest <account>.dkr.ecr.<region>.amazonaws.com/vitals-anomaly-api:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/vitals-anomaly-api:latest
```

## Project Structure
```
.
├── app.py                 # Flask application
├── Dockerfile             # Docker container definition
├── docker-compose.yml     # Local orchestration
├── requirements.txt       # Python dependencies
├── model/                 # Trained model storage
└── README.md             # Documentation
```

## Author

**Eswar Surya Danaboina**  
MSc Data Analytics | Dublin Business School  
[Portfolio](https://eswarsurya.github.io/eswar-portfolio/) | [LinkedIn](https://linkedin.com/in/eswarsurya76)

## License

MIT License - Academic project for demonstration purposes.
