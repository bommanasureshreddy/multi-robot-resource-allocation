"""
ml_prediction.py
-----------------
Section 19: ML as a PREDICTION COMPONENT feeding the auction, not the
allocator itself. Trains models to predict:

    predicted_energy   (energy_consumption)
    predicted_time      (completion_time)
    predicted_risk       (task_risk)

from data/dataset/ml/prediction_dataset.csv, using the exact feature set
the spec lists (robot_type, SOC, DoD, SOH, distance, path_length, speed,
task_workload, task_priority, zone_type, zone_risk, current_load [+
temperature/environment_factor, present in this dataset]).

Models (deliberately not overcomplicated, per spec):
    - Linear Regression   (baseline)
    - Random Forest        (recommended primary)
    - Gradient Boosting     (recommended alternative to XGBoost, which is
                              not installed in this environment -- using
                              sklearn's GradientBoostingRegressor instead,
                              documented rather than silently substituted)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES_NUMERIC = [
    "soc", "dod", "soh", "distance", "path_length", "speed",
    "task_workload", "task_priority", "zone_risk", "robot_load",
    "temperature_or_environment_factor",
]
FEATURES_CATEGORICAL = ["robot_type", "zone_type"]
FEATURES_ALL = FEATURES_NUMERIC + FEATURES_CATEGORICAL

TARGETS = ["energy_consumption", "completion_time", "task_risk"]
TARGET_TO_PREDICTION_NAME = {
    "energy_consumption": "predicted_energy",
    "completion_time": "predicted_time",
    "task_risk": "predicted_risk",
}

MODEL_BUILDERS = {
    "linear": lambda: LinearRegression(),
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42),
}


def _build_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL),
    ], remainder="passthrough")
    return Pipeline([("pre", pre), ("model", estimator)])


@dataclass
class TrainResult:
    target: str
    model_name: str
    r2: float
    mae: float


def train_all(dataset_root: str = "data/dataset", models_dir: str = "models",
              test_size: float = 0.2, seed: int = 42) -> List[TrainResult]:
    """Trains every (target x model) combination, evaluates on a held-out
    test split, and saves ONLY the best model per target (by R^2) to
    models_dir, plus a metrics.json summarizing everything (so the
    "recommended initial models" comparison is a real, reported result,
    not asserted)."""
    df = pd.read_csv(os.path.join(dataset_root, "ml", "prediction_dataset.csv"))
    os.makedirs(models_dir, exist_ok=True)

    results: List[TrainResult] = []
    best_per_target: Dict[str, Tuple[float, Pipeline, str]] = {}

    for target in TARGETS:
        X = df[FEATURES_ALL]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        for model_name, builder in MODEL_BUILDERS.items():
            pipe = _build_pipeline(builder())
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            results.append(TrainResult(target=target, model_name=model_name, r2=r2, mae=mae))

            current_best = best_per_target.get(target)
            if current_best is None or r2 > current_best[0]:
                best_per_target[target] = (r2, pipe, model_name)

    metrics_summary = []
    for r in results:
        metrics_summary.append({"target": r.target, "model": r.model_name, "r2": r.r2, "mae": r.mae})
    for target, (r2, pipe, model_name) in best_per_target.items():
        path = os.path.join(models_dir, f"{target}_{model_name}.joblib")
        joblib.dump(pipe, path)
        metrics_summary.append({"target": target, "selected_best_model": model_name, "r2": r2, "path": path})

    with open(os.path.join(models_dir, "ml_prediction_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return results


class Predictor:
    """Loads the best-per-target models saved by train_all() and exposes
    predicted_energy / predicted_time / predicted_risk for a single
    robot-task feature row -- this is what auction.py calls."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        with open(os.path.join(models_dir, "ml_prediction_metrics.json")) as f:
            summary = json.load(f)
        # Reconstruct paths from self.models_dir + the saved FILENAME only.
        # The "path" field in metrics.json was written relative to whatever
        # cwd train_all() ran from, so trusting it verbatim breaks the
        # moment this Predictor is instantiated from a different cwd (e.g.
        # a notebook running with cwd=notebooks/ instead of the project
        # root) -- reconstructing from models_dir makes loading robust to
        # where the caller runs from.
        self.best_model_paths = {
            row["target"]: os.path.join(models_dir, os.path.basename(row["path"]))
            for row in summary if "selected_best_model" in row
        }
        self.models = {t: joblib.load(p) for t, p in self.best_model_paths.items()}

    def predict(self, features: Dict) -> Dict[str, float]:
        row = pd.DataFrame([{k: features[k] for k in FEATURES_ALL}])
        out = {}
        for target, model in self.models.items():
            pred = float(model.predict(row)[0])
            out[TARGET_TO_PREDICTION_NAME[target]] = max(0.0, pred)
        return out


if __name__ == "__main__":
    results = train_all()
    print(f"{'target':22s} {'model':18s} {'R^2':>8s} {'MAE':>10s}")
    for r in sorted(results, key=lambda r: (r.target, -r.r2)):
        print(f"{r.target:22s} {r.model_name:18s} {r.r2:8.3f} {r.mae:10.3f}")

    predictor = Predictor()
    sample = {
        "soc": 79.3, "dod": 20.7, "soh": 0.916, "distance": 20.0, "path_length": 24.0,
        "speed": 3.5, "task_workload": 8.0, "task_priority": 4, "zone_risk": 2.0,
        "robot_load": 0.5, "temperature_or_environment_factor": 25.0,
        "robot_type": "UGV", "zone_type": "LAND",
    }
    pred = predictor.predict(sample)
    print("\nSample prediction:", pred)
    assert set(pred.keys()) == {"predicted_energy", "predicted_time", "predicted_risk"}
    assert all(v >= 0 for v in pred.values())
    print("OK: ml_prediction.py trained, saved, and reloaded successfully.")
