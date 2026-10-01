# Human Vitals Anomaly Detection API

A Flask REST API that serves **Isolation Forest anomaly detection** for human vital-sign inputs, based on the machine-learning work from my MSc Data Analytics thesis.

## Why this project matters

This repository extends the thesis work from offline analysis into a **service-oriented ML workflow**. It demonstrates how an anomaly-detection approach can be exposed through HTTP endpoints for prediction, batch processing, explainability, and health monitoring.

## What it demonstrates

- REST API design with Flask
- Machine-learning inference with Isolation Forest
- Batch prediction support
- Explainability endpoint architecture
- Health-check endpoint
- Docker containerization
- Gunicorn production serving
- Deployment-oriented project structure

## Example

~~~bash
curl http://localhost:5000/health
~~~

~~~json
{
  "heart_rate": 75,
  "bp_systolic": 120,
  "bp_diastolic": 80,
  "temperature": 36.6,
  "spo2": 98
}
~~~

## Model context

- **Algorithm:** Isolation Forest
- **Thesis dataset scale:** 200,020 multivariate records
- **Features:** heart rate, blood pressure, temperature, SpO2
- **Explainability:** SHAP-ready architecture

## Run locally

~~~bash
pip install -r requirements.txt
python app.py
~~~

Or use Docker:

~~~bash
docker-compose up --build
~~~

## API surface

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service health check |
| `POST /predict` | Single or batch anomaly prediction |
| `POST /explain` | Feature-contribution analysis |

## Project structure

~~~
.
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── model/
└── README.md
~~~

## Related project

**MSc thesis:** [Human Vitals Anomaly Detection Thesis](https://github.com/eswarsurya/human-vitals-anomaly-detection-thesis)

## Author

**Eswar Surya Danaboina** · MSc Data Analytics · Dublin, Ireland

Portfolio: https://eswardanaboina.vercel.app/  
LinkedIn: https://www.linkedin.com/in/eswarsurya76/

## License

MIT
