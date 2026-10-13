"""Train the Isolation Forest on the thesis dataset and save it with its metadata.

Usage:
    python train.py                      # uses data/human_vital_signs_dataset_2024.csv
    python train.py --contamination 0.05 --out model/vitals_iforest.joblib
"""

import argparse
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

# API field name -> dataset column
FEATURES = {
    "heart_rate": "Heart Rate",
    "bp_systolic": "Systolic Blood Pressure",
    "bp_diastolic": "Diastolic Blood Pressure",
    "temperature": "Body Temperature",
    "spo2": "Oxygen Saturation",
}


def train(csv_path: Path, contamination: float, seed: int) -> dict:
    df = pd.read_csv(csv_path, usecols=list(FEATURES.values())).dropna()
    x = df[list(FEATURES.values())].to_numpy(dtype=float)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    model.fit(x)
    return {
        "model": model,
        "features": list(FEATURES),
        "mean": x.mean(axis=0).tolist(),
        "std": x.std(axis=0).tolist(),
        "rows": len(x),
        "contamination": contamination,
        "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/human_vital_signs_dataset_2024.csv"))
    p.add_argument("--out", type=Path, default=Path("model/vitals_iforest.joblib"))
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    bundle = train(a.data, a.contamination, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, a.out, compress=3)
    print(f"Trained on {bundle['rows']} rows -> {a.out}")


if __name__ == "__main__":
    main()
