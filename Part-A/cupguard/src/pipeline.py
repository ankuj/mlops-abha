"""CupGuard ML lifecycle — ingest, transform, train, evaluate, package."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

FEATURES = [
    "neutral",
    "is_world_cup",
]
TARGET = "outcome"
STATS_AS_OF = "2025-06-01"

MATCHES_PATH = RAW / "matches_enriched.csv"
RESULTS_PATH = RAW / "results.csv"
FEATURES_PATH = PROCESSED / "match_features.csv"
TEAM_STATS_PATH = PROCESSED / "team_stats_2025.csv"
MLFLOW_DIR = ROOT / "mlruns"
EXPERIMENT_NAME = "cupguard-worldcup"
MODEL_URI_PATH = MODELS / "mlflow_model_uri.txt"


def mlflow_tracking_uri() -> str:
    """Tracking URI for both logging runs and loading models."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def ingest() -> dict:
    """Add labels, and save matches_enriched.csv."""

    matches = pd.read_csv(RESULTS_PATH, parse_dates=["date"])

    # Clean flags and add simple derived columns
    matches["neutral"] = matches["neutral"].astype(str).str.lower().eq("true").astype(int)
    matches["is_world_cup"] = matches["tournament"].str.contains("World Cup", case=False, na=False).astype(int)
    matches["total_goals"] = matches["home_score"] + matches["away_score"]
    matches["outcome"] = np.select(
        [matches["home_score"] > matches["away_score"], matches["home_score"] < matches["away_score"]],
        ["home_win", "away_win"],
        default="draw",
    )
    matches = matches.sort_values("date").reset_index(drop=True)
    matches.to_csv(MATCHES_PATH, index=False)

    return {
        "stage": "ingest",
        "frame": matches,
        "artifacts": {"matches": str(MATCHES_PATH)},
        "rows": len(matches),
        "world_cup_matches": int(matches["is_world_cup"].sum()),
    }

def transform(matches: pd.DataFrame | None = None) -> dict:
    """Build features, save processed CSVs, and split train/test."""
    if matches is None:
        if not MATCHES_PATH.exists():
            raise FileNotFoundError("Run ingest() first.")
        matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])

    matches = matches.copy()
    matches = matches.dropna(subset=FEATURES + [TARGET])

    X = matches[FEATURES]
    y = matches[TARGET]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    train_path = PROCESSED / "train.csv"
    test_path = PROCESSED / "test.csv"
    pd.concat([X_train, pd.Series(y_train, name="outcome_enc")], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, pd.Series(y_test, name="outcome_enc")], axis=1).to_csv(test_path, index=False)

    return {
        "stage": "transform",
        "frame": matches,
        "artifacts": {
            "features": str(FEATURES_PATH),
            "team_stats": str(TEAM_STATS_PATH),
            "train": str(train_path),
            "test": str(test_path),
        },
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder
    }


def train(X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    return {"stage": "train", "model": model}


def evaluate(model, X_test: pd.DataFrame, y_test: np.ndarray, label_encoder: LabelEncoder) -> dict:
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
    }
    report = classification_report(y_test, pred, target_names=label_encoder.classes_, digits=3)
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics)
    return {"stage": "evaluate", "metrics": metrics, "classification_report": report}


def package(_model, metrics: dict, label_encoder: LabelEncoder) -> dict:
    """Log label classes to MLflow (sklearn model logged via autolog)."""
    if mlflow.active_run() is not None:
        mlflow.log_param("outcome_classes", json.dumps(list(label_encoder.classes_)))
    return {
        "stage": "package",
        "artifacts": {"model": "logged via MLflow sklearn autolog"},
        "summary": {
            "model": "multinomial_logreg_outcome",
            "objective": "Predict 2026 World Cup match outcome (home_win / draw / away_win)",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "features": FEATURES,
            "classes": list(label_encoder.classes_),
            "stats_as_of": STATS_AS_OF,
        },
    }

def configure_mlflow() -> None:
    """Connect to MLflow tracking server and enable sklearn autologging."""
    tracking_uri = mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()


def run_lifecycle() -> dict:
    """Execute ingest → transform → train → evaluate → package."""
    configure_mlflow()
    with mlflow.start_run(run_name="lifecycle-run") as run:
        ing = ingest()
        tr = transform(ing["frame"])
        trn = train(tr["X_train"], tr["y_train"])
        ev = evaluate(trn["model"], tr["X_test"], tr["y_test"], tr["label_encoder"])

        pkg = package(trn["model"], ev["metrics"], tr["label_encoder"])
        model_uri = f"runs:/{run.info.run_id}/model"
        MODELS.mkdir(parents=True, exist_ok=True)
        MODEL_URI_PATH.write_text(model_uri)

    return {
        "ingest": ing,
        "transform": tr,
        "train": trn,
        "evaluate": ev,
        "package": pkg,
        "mlflow_run_id": run.info.run_id,
        "model_uri": model_uri,
    }


def load_team_stats(path: Path = TEAM_STATS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _run_id_from_model_uri(model_uri: str) -> str:
    return model_uri.removeprefix("runs:/").split("/")[0]


def load_model(model_uri: str | None = None):
    """Load the promoted model and outcome classes from MLflow."""
    if model_uri is None:
        if not MODEL_URI_PATH.exists():
            raise FileNotFoundError("Run run_lifecycle() first to train and log a model.")
        model_uri = MODEL_URI_PATH.read_text().strip()
    tracking_uri = mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(_run_id_from_model_uri(model_uri))
    if "outcome_classes" not in run.data.params:
        raise FileNotFoundError("outcome_classes not found in MLflow run. Re-run run_lifecycle().")
    classes = np.array(json.loads(run.data.params["outcome_classes"]))
    return pyfunc_model, classes


def fixture_features(home_team: str, away_team: str, neutral: int = 0, is_world_cup: int = 1) -> pd.DataFrame:
    stats = load_team_stats().set_index("team")
    row = {
        "home_win_rate": stats.loc[home_team, "win_rate"],
        "home_avg_gf": stats.loc[home_team, "avg_gf"],
        "home_avg_ga": stats.loc[home_team, "avg_ga"],
        "away_win_rate": stats.loc[away_team, "win_rate"],
        "away_avg_gf": stats.loc[away_team, "avg_gf"],
        "away_avg_ga": stats.loc[away_team, "avg_ga"],
        "win_rate_diff": stats.loc[home_team, "win_rate"] - stats.loc[away_team, "win_rate"],
        "neutral": int(neutral),
        "is_world_cup": int(is_world_cup),
    }
    return pd.DataFrame([row])[FEATURES]


def predict_fixture(home_team: str, away_team: str, neutral: int = 0, is_world_cup: int = 1) -> dict[str, float]:
    pyfunc_model, classes = load_model()
    X = fixture_features(home_team, away_team, neutral, is_world_cup)
    sklearn_model = pyfunc_model._model_impl.sklearn_model
    prob = sklearn_model.predict_proba(X)[0]
    return dict(zip(classes, prob))
