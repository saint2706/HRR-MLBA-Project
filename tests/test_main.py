from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import main


def test_attempt_kaggle_download_includes_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded_command: list[str] = []

    def fake_run_command(command, cwd=None):  # type: ignore[override]
        nonlocal recorded_command
        recorded_command = list(command)
        output_path = Path(command[command.index("-o") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"zip")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    monkeypatch.setenv("KAGGLE_KEY", "test-key")
    monkeypatch.setattr(main.shutil, "which", lambda _: "curl")
    monkeypatch.setattr(main, "_run_command", fake_run_command)
    monkeypatch.setattr(main, "_extract_csv_from_zip", lambda *args, **kwargs: True)

    result = main._attempt_kaggle_download(  # pylint: disable=protected-access
        Path("dataset.csv"),
        download_url="https://example.com/dataset.zip",
        downloads_dir=tmp_path,
    )

    assert result is True
    assert "-u" in recorded_command
    assert "test-user:test-key" in recorded_command
    assert "X-Kaggle-User: test-user" in recorded_command
    assert "X-Kaggle-Key: test-key" in recorded_command
    assert "--fail" in recorded_command
    assert "--show-error" in recorded_command


def test_main_player_profiles_preserve_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
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
            "balls_faced": [10, 20],
            "batting_strike_rate": [120.0, 125.0],
            "batting_average": [40.0, 35.0],
            "boundary_percentage": [50.0, 45.0],
            "dot_percentage": [30.0, 33.0],
            "batting_index": [1.0, 2.0],
        }
    )

    bowling_metrics = pd.DataFrame(
        {
            "match_id": [1, 2],
            "player": ["Player X", "Player X"],
            "team": ["Team A", "Team A"],
            "balls_bowled": [24, 18],
            "bowling_economy": [8.0, 7.5],
            "bowling_strike_rate": [20.0, 18.0],
            "wickets_per_match": [1.0, 2.0],
            "phase_efficacy": [0.5, 0.6],
            "bowling_index": [1.5, 1.2],
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

    captured: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(main, "ensure_dataset_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "load_data", lambda *_args, **_kwargs: "raw")
    monkeypatch.setattr(main, "prepare_ball_by_ball", lambda *_args, **_kwargs: "processed")
    monkeypatch.setattr(main, "aggregate_player_match_stats", lambda *_args, **_kwargs: ("stats", "phases"))
    monkeypatch.setattr(main, "compute_batting_metrics", lambda *_args, **_kwargs: batting_metrics)
    monkeypatch.setattr(main, "compute_bowling_metrics", lambda *_args, **_kwargs: bowling_metrics)
    monkeypatch.setattr(main, "compute_composite_indices", lambda *_args, **_kwargs: (features, ["feature_one"]))
    monkeypatch.setattr(main, "build_model_matrix", lambda *_args, **_kwargs: (model_df, ["feature_one"]))
    monkeypatch.setattr(main, "train_impact_model", lambda *_args, **_kwargs: DummyArtifacts())
    monkeypatch.setattr(
        main,
        "compute_shap_values",
        lambda *_args, **_kwargs: (pd.DataFrame({"feature_one": [0.3, 0.5]}), model_df[["feature_one"]]),
    )
    monkeypatch.setattr(main, "plot_shap_summary", lambda *_args, **_kwargs: "shap.png")
    monkeypatch.setattr(main, "plot_cluster_scatter", lambda *_args, **_kwargs: "plot.png")
    monkeypatch.setattr(main, "generate_classification_report", lambda *_args, **_kwargs: {})

    monkeypatch.setattr(main, "cluster_batters", lambda df: (df.assign(batter_cluster=0), SimpleNamespace()))
    monkeypatch.setattr(main, "cluster_bowlers", lambda df: (df.assign(bowler_cluster=0), SimpleNamespace()))
    monkeypatch.setattr(main, "label_batter_clusters", lambda *_args, **_kwargs: {0: "Finisher"})
    monkeypatch.setattr(main, "label_bowler_clusters", lambda *_args, **_kwargs: {0: "Strike Bowler"})

    def fake_select_best_xi(df: pd.DataFrame) -> pd.DataFrame:
        captured["profiles"] = df.copy()
        return df.head(1)

    monkeypatch.setattr(main, "select_best_xi", fake_select_best_xi)

    outputs = main.main("unused.csv")

    assert "profiles" in captured
    profiles = captured["profiles"]
    assert len(profiles) == 2
    assert set(profiles["season"]) == {2020, 2021}
    assert profiles.set_index("season")["balls_faced"].to_dict() == {2020: 10, 2021: 20}
    assert profiles.set_index("season")["balls_bowled"].to_dict() == {2020: 24, 2021: 18}

    # The pipeline should still produce the expected outputs dictionary
    assert "player_ratings" in outputs
