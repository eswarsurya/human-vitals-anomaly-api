"""Flask API serving an Isolation Forest anomaly model for human vital signs."""

import math
import os

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("model", "vitals_iforest.joblib"))
MAX_BATCH = 1000

# Physically possible ranges. Values outside these are rejected as bad input,
# not scored: the model should only judge readings a sensor could produce.
VALID_RANGE = {
    "heart_rate": (20, 250),
    "bp_systolic": (50, 260),
    "bp_diastolic": (20, 160),
    "temperature": (30.0, 45.0),
    "spo2": (50.0, 100.0),
}


class BadInput(ValueError):
    pass


def load_bundle(path: str) -> dict:
    bundle = joblib.load(path)
    if bundle.get("features") != list(VALID_RANGE):
        raise RuntimeError(f"Model at {path} expects features {bundle.get('features')}")
    return bundle


def parse_row(row, index: int) -> list[float]:
    """Accept {"heart_rate": ..} or a 5-item list in feature order; return floats."""
    names = list(VALID_RANGE)
    if isinstance(row, dict):
        missing = [n for n in names if n not in row]
        if missing:
            raise BadInput(f"row {index}: missing {', '.join(missing)}")
        values = [row[n] for n in names]
    elif isinstance(row, list) and len(row) == len(names):
        values = row
    else:
        raise BadInput(f"row {index}: expected an object with {', '.join(names)} or a list of 5 numbers")
    out = []
    for name, v in zip(names, values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise BadInput(f"row {index}: {name} must be a number")
        lo, hi = VALID_RANGE[name]
        if not lo <= v <= hi:
            raise BadInput(f"row {index}: {name}={v} is outside the possible range {lo}-{hi}")
        out.append(float(v))
    return out


def parse_body(body) -> np.ndarray:
    if not isinstance(body, dict):
        raise BadInput("send a JSON object")
    rows = body.get("data", [body])
    if not isinstance(rows, list) or not rows:
        raise BadInput("'data' must be a non-empty list")
    if len(rows) > MAX_BATCH:
        raise BadInput(f"at most {MAX_BATCH} rows per request")
    return np.array([parse_row(r, i) for i, r in enumerate(rows)])


def create_app(model_path: str = MODEL_PATH) -> Flask:
    app = Flask(__name__)
    bundle = load_bundle(model_path)
    model = bundle["model"]
    mean, std = np.array(bundle["mean"]), np.array(bundle["std"])
    names = bundle["features"]

    @app.errorhandler(BadInput)
    def bad_input(e):
        return jsonify(success=False, error=str(e)), 400

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            model="isolation_forest",
            trained_rows=bundle["rows"],
            trained_at=bundle["trained_at"],
        )

    @app.post("/predict")
    def predict():
        x = parse_body(request.get_json(silent=True))
        labels, scores = model.predict(x), model.decision_function(x)
        results = [
            {"sample_id": i, "is_anomaly": bool(lab == -1), "anomaly_score": round(float(s), 4)}
            for i, (lab, s) in enumerate(zip(labels, scores))
        ]
        return jsonify(
            success=True,
            predictions=results,
            total_samples=len(results),
            anomalies_detected=sum(r["is_anomaly"] for r in results),
        )

    @app.post("/explain")
    def explain():
        x = parse_body(request.get_json(silent=True))
        if len(x) != 1:
            raise BadInput("/explain takes one reading at a time")
        z = (x[0] - mean) / np.where(std == 0, 1, std)
        ranked = sorted(zip(names, x[0], z), key=lambda t: -abs(t[2]))
        score = float(model.decision_function(x)[0])
        return jsonify(
            success=True,
            is_anomaly=bool(model.predict(x)[0] == -1),
            anomaly_score=round(score, 4),
            # How far each reading sits from the training average, in standard deviations.
            # A simple, fast explanation; the thesis notebooks use SHAP for the full picture.
            contributions=[
                {"feature": n, "value": float(v), "std_from_mean": round(float(s), 2)} for n, v, s in ranked
            ],
        )

    return app


app = create_app() if os.path.exists(MODEL_PATH) else None

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
