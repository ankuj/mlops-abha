"""CupGuard ML lifecycle — ingest, transform, train, evaluate, package."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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
MODEL_PATH = MODELS / "outcome_model.pkl"
CLASSES_PATH = MODELS / "outcome_classes.json"


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
    return {"stage": "evaluate", "metrics": metrics, "classification_report": report}


def package(model, metrics: dict, label_encoder: LabelEncoder) -> dict:
    """Save the trained model and label classes to disk."""
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    CLASSES_PATH.write_text(json.dumps(list(label_encoder.classes_)))
    return {
        "stage": "package",
        "artifacts": {"model": str(MODEL_PATH), "classes": str(CLASSES_PATH)},
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

def run_lifecycle() -> dict:
    """Execute ingest → transform → train → evaluate → package."""
    ing = ingest()
    tr = transform(ing["frame"])
    trn = train(tr["X_train"], tr["y_train"])
    ev = evaluate(trn["model"], tr["X_test"], tr["y_test"], tr["label_encoder"])
    pkg = package(trn["model"], ev["metrics"], tr["label_encoder"])

    return {
        "ingest": ing,
        "transform": tr,
        "train": trn,
        "evaluate": ev,
        "package": pkg,
    }


def load_team_stats(path: Path = TEAM_STATS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_model():
    """Load the saved model and outcome classes from disk."""
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("Run run_lifecycle() first to train and save a model.")
    model = joblib.load(MODEL_PATH)
    classes = np.array(json.loads(CLASSES_PATH.read_text()))
    return model, classes


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
    model, classes = load_model()
    X = fixture_features(home_team, away_team, neutral, is_world_cup)
    prob = model.predict_proba(X)[0]
    return dict(zip(classes, prob))
