"""pytest suite for CupGuard pipeline functions (Part-B)."""

import sys
from pathlib import Path

PART_B_SRC = Path(__file__).resolve().parents[2] / "Part-B" / "cupguard" / "src"
sys.path.insert(0, str(PART_B_SRC))

import pytest

import pipeline
from pipeline import FEATURES, MODEL_PATH, fixture_features, predict_fixture


def test_ingest_adds_outcome_labels(tmp_path, monkeypatch):
    """Unit: synthetic 2-match CSV; ingest() labels outcomes and flags correctly."""
    results = tmp_path / "results.csv"
    results.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2020-01-01,Brazil,France,2,1,FIFA World Cup,Rio,Brazil,True\n"
        "2020-02-01,France,Brazil,0,1,Friendly,Paris,France,False\n"
    )
    matches_path = tmp_path / "matches_enriched.csv"
    monkeypatch.setattr(pipeline, "RESULTS_PATH", results)
    monkeypatch.setattr(pipeline, "MATCHES_PATH", matches_path)

    out = pipeline.ingest()
    matches = out["frame"]

    assert list(matches["outcome"]) == ["home_win", "away_win"]
    assert matches["neutral"].tolist() == [1, 0]
    assert matches["is_world_cup"].tolist() == [1, 0]
    assert matches["total_goals"].tolist() == [3, 1]


def test_fixture_features_returns_expected_columns():
    X = fixture_features("Brazil", "France", neutral=1)
    assert list(X.columns) == FEATURES
    assert len(X) == 1


def test_predict_fixture_output_contract():
    probs = predict_fixture("Brazil", "France", neutral=1)
    assert set(probs) == {"home_win", "draw", "away_win"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6


@pytest.mark.parametrize(
    "home,away",
    [("Brazil", "France"), ("Argentina", "Germany")],
)
def test_predict_fixture_parametrized(home, away):
    probs = predict_fixture(home, away, neutral=1)
    assert set(probs) == {"home_win", "draw", "away_win"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_predict_fixture_raises_without_model(monkeypatch):
    missing = Path("/tmp/cupguard_missing_model.pkl")
    monkeypatch.setattr(pipeline, "MODEL_PATH", missing)
    with pytest.raises(FileNotFoundError, match="Run run_lifecycle"):
        predict_fixture("Brazil", "France", neutral=1)


def test_model_artifact_exists():
    assert MODEL_PATH.exists(), "Train Part-B model first: run_lifecycle()"
