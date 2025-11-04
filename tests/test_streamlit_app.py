from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit_app  # noqa: E402


def test_streamlit_player_profiles_keep_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    features = pd.DataFrame(
        {
            "match_id": [1, 2],
            "player": ["Player X", "Player X"],
            "team": ["Team A", "Team A"],
            "season": [2020, 2021],
            "balls_faced": [10, 20],
            "balls_bowled": [24, 18],
            "batting_index": [1.0, 2.0],
            "bowling_index": [1.5, 1.2],
        }
    )

    batting_metrics = pd.DataFrame(
        {
            "match_id": [1, 2],
            "player": ["Player X", "Player X"],
            "team": ["Team A", "Team A"],
            "season": [2020, 2021],
            "balls_faced": [10, 20],
            "batting_strike_rate": [120.0, 125.0],
            "batting_average": [40.0, 35.0],
            "boundary_percentage": [50.0, 45.0],
            "dot_percentage": [30.0, 33.0],
        }
    )

    bowling_metrics = pd.DataFrame(
        {
            "match_id": [1, 2],
            "player": ["Player X", "Player X"],
            "team": ["Team A", "Team A"],
            "season": [2020, 2021],
            "balls_bowled": [24, 18],
            "bowling_economy": [8.0, 7.5],
            "bowling_strike_rate": [20.0, 18.0],
            "wickets_per_match": [1.0, 2.0],
            "phase_efficacy": [0.5, 0.6],
        }
    )

    model_df = pd.DataFrame(
        {
            "match_id": [1, 2],
            "player": ["Player X", "Player X"],
            "team": ["Team A", "Team A"],
            "season": [2020, 2021],
            "team_won": [1, 0],
            "feature_one": [0.2, 0.4],
            "batting_index": [1.0, 2.0],
            "bowling_index": [1.5, 1.2],
        }
    )

    class DummyArtifacts:
        def __init__(self) -> None:
            self.model = "model"
            self.X_train = model_df[["feature_one"]]
            self.metrics = {"accuracy": 1.0}

    monkeypatch.setattr(streamlit_app, "ensure_dataset_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(streamlit_app, "load_data", lambda *_args, **_kwargs: "raw")
    monkeypatch.setattr(streamlit_app, "prepare_ball_by_ball", lambda *_args, **_kwargs: "processed")
    monkeypatch.setattr(streamlit_app, "aggregate_player_match_stats", lambda *_args, **_kwargs: ("stats", "phases"))
    monkeypatch.setattr(streamlit_app, "compute_batting_metrics", lambda *_args, **_kwargs: batting_metrics)
    monkeypatch.setattr(streamlit_app, "compute_bowling_metrics", lambda *_args, **_kwargs: bowling_metrics)
    monkeypatch.setattr(
        streamlit_app, "compute_composite_indices", lambda *_args, **_kwargs: (features, ["feature_one"])
    )
    monkeypatch.setattr(streamlit_app, "build_model_matrix", lambda *_args, **_kwargs: (model_df, ["feature_one"]))
    monkeypatch.setattr(streamlit_app, "train_impact_model", lambda *_args, **_kwargs: DummyArtifacts())
    monkeypatch.setattr(
        streamlit_app,
        "compute_shap_values",
        lambda *_args, **_kwargs: (pd.DataFrame({"feature_one": [0.3, 0.5]}), model_df[["feature_one"]]),
    )
    monkeypatch.setattr(streamlit_app, "cluster_batters", lambda df: (df.assign(batter_cluster=0), SimpleNamespace()))
    monkeypatch.setattr(streamlit_app, "cluster_bowlers", lambda df: (df.assign(bowler_cluster=0), SimpleNamespace()))
    monkeypatch.setattr(streamlit_app, "label_batter_clusters", lambda *_args, **_kwargs: {0: "Finisher"})
    monkeypatch.setattr(streamlit_app, "label_bowler_clusters", lambda *_args, **_kwargs: {0: "Strike Bowler"})
    monkeypatch.setattr(streamlit_app, "select_best_xi", lambda df: df.head(1))
    monkeypatch.setattr(streamlit_app, "generate_classification_report", lambda *_args, **_kwargs: {})

    outputs = streamlit_app.run_pipeline.__wrapped__("unused.csv")

    profiles = outputs["player_profiles"]
    assert len(profiles) == 2
    assert set(profiles["season"]) == {2020, 2021}
    assert profiles.set_index("season")["balls_faced"].to_dict() == {2020: 10, 2021: 20}
    assert profiles.set_index("season")["balls_bowled"].to_dict() == {2020: 24, 2021: 18}

    season_2020 = profiles[profiles["season"] == 2020]
    season_2021 = profiles[profiles["season"] == 2021]
    assert not season_2020.empty and not season_2021.empty
    assert season_2020.iloc[0]["impact_rating"] != season_2021.iloc[0]["impact_rating"]


def test_find_best_model_files_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that find_best_model_files correctly finds model and metadata files."""
    # Create test files
    model_file = tmp_path / "best_model.json"
    metadata_file = tmp_path / "best_model_metadata.json"

    model_file.write_text("{}")
    metadata_file.write_text("{}")

    # Mock the __file__ location
    monkeypatch.setattr(streamlit_app, "__file__", str(tmp_path / "streamlit_app.py"))

    model_path, metadata_path = streamlit_app.find_best_model_files()

    assert model_path == model_file
    assert metadata_path == metadata_file


def test_find_best_model_files_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that find_best_model_files returns None when no files exist."""
    # Mock the __file__ location to empty directory
    monkeypatch.setattr(streamlit_app, "__file__", str(tmp_path / "streamlit_app.py"))

    model_path, metadata_path = streamlit_app.find_best_model_files()

    assert model_path is None
    assert metadata_path is None


def test_find_best_model_files_with_dash_pattern(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that find_best_model_files finds best-model-*.json pattern."""
    # Create test files with dash pattern
    model_file = tmp_path / "best-model-v1.json"
    model_file.write_text("{}")

    # Mock the __file__ location
    monkeypatch.setattr(streamlit_app, "__file__", str(tmp_path / "streamlit_app.py"))

    model_path, _ = streamlit_app.find_best_model_files()

    assert model_path == model_file


def test_load_pretrained_model_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that load_pretrained_model returns None when no model is found."""
    monkeypatch.setattr(streamlit_app, "find_best_model_files", lambda: (None, None))

    model_df = pd.DataFrame({"feature1": [1, 2], "team_won": [0, 1]})
    result = streamlit_app.load_pretrained_model(model_df, ["feature1"])

    assert result is None
