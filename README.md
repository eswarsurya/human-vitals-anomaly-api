# Human Vitals Anomaly API

![ci](https://github.com/eswarsurya/human-vitals-anomaly-api/actions/workflows/ci.yml/badge.svg)

A small Flask service that scores vital-sign readings with an Isolation Forest, built from my MSc Data Analytics thesis. The thesis did the analysis in notebooks; this repo turns the model into something you can run, test and ship as a container.

## What it does

- `POST /predict` scores one reading or a batch of up to 1,000, and says which ones look anomalous.
- `POST /explain` shows how far each value in a reading sits from the training average, most unusual first.
- `GET /health` reports that the model is loaded, how many rows it was trained on, and when.
- Bad input (missing fields, text instead of numbers, impossible values like a temperature of 60 °C) gets a 400 with a message saying exactly what's wrong.

## Try it

```bash
docker compose up --build
curl localhost:5000/predict -H 'content-type: application/json' \
  -d '{"heart_rate":150,"bp_systolic":180,"bp_diastolic":110,"temperature":39.5,"spo2":85}'
```

```json
{"anomalies_detected":1,"predictions":[{"anomaly_score":-0.0761,"is_anomaly":true,"sample_id":0}],"success":true,"total_samples":1}
```

Batch: `{"data": [{...}, [75, 120, 80, 36.6, 98]]}` (objects, or lists in the order heart_rate, bp_systolic, bp_diastolic, temperature, spo2).

## The model

- **Algorithm:** Isolation Forest, 200 trees, 5% contamination, fixed random seed.
- **Training data:** the thesis dataset, 200,020 rows (`data/`), using heart rate, systolic and diastolic blood pressure, body temperature and SpO2.
- **Reproducible:** `python train.py` rebuilds `model/vitals_iforest.joblib` from the CSV. The file stores the model with its feature order, training means and standard deviations, row count and library version.
- **Explanations:** `/explain` uses standard-deviation distance from the training average, which is fast and easy to read. The thesis notebooks use SHAP for the full analysis.

## How it's checked

GitHub Actions runs on every push:

1. `ruff` lint and format check.
2. `pytest`: 12 tests covering normal and extreme readings, batches, the explanation order, and every kind of bad input.
3. Builds the Docker image, starts the container, sends a real request, and waits for Docker's own health check to report `healthy`.

The image is multi-stage, runs as a non-root user, contains only the app and the model, and checks its own health with Python (no extra tools installed).

## Run without Docker

```bash
pip install -r requirements-dev.txt
pytest -q
python app.py            # http://localhost:5000
```

## Layout

```
app.py              the API
train.py            rebuilds the model from data/
model/              trained model bundle
tests/              pytest suite
data/               thesis dataset (200,020 rows)
notebooks/          thesis notebooks (analysis, SHAP)
docs/thesis/        thesis report and slides
```

## Limits

This is a portfolio service, not a medical device. The dataset is the one used in the thesis, the model flags statistical outliers rather than diagnosing anything, and there is no authentication, so don't expose it to the internet as-is.

Related: [thesis repository](https://github.com/eswarsurya/human-vitals-anomaly-detection-thesis) · Author: Eswar Surya Danaboina, Dublin · [LinkedIn](https://www.linkedin.com/in/eswarsurya76/)
