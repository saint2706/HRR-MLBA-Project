"""Utility script to retrain the impact model and persist it for inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing import (
    PreprocessingConfig,
    aggregate_player_match_stats,
    load_data,
    prepare_ball_by_ball,
)
from feature_engineering import (
    build_model_matrix,
    compute_batting_metrics,
    compute_bowling_metrics,
    compute_composite_indices,
)
from main import ensure_dataset_available
from model_training import train_impact_model


def retrain_and_save(
    data_path: str,
    model_output: str,
    metadata_output: str | None = None,
) -> None:
    """Retrain the impact model and save it alongside lightweight metadata."""
    ensure_dataset_available(data_path)

    raw_df = load_data(data_path)
    processed_df = prepare_ball_by_ball(raw_df, PreprocessingConfig())

    player_stats, phase_stats = aggregate_player_match_stats(processed_df)
    batting_metrics = compute_batting_metrics(player_stats)
    bowling_metrics = compute_bowling_metrics(player_stats, phase_stats)

    features, metric_columns = compute_composite_indices(batting_metrics, bowling_metrics)
    model_df, feature_cols = build_model_matrix(features, metric_columns)

    artifacts = train_impact_model(model_df, feature_cols)

    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.model.save_model(model_path)

    metadata = {
        "feature_columns": feature_cols,
        "metrics": artifacts.metrics,
    }

    metadata_path = Path(metadata_output) if metadata_output else model_path.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the IPL impact model and overwrite the saved artifact.")
    parser.add_argument(
        "--data",
        dest="data_path",
        default="IPL.csv",
        help="Path to the IPL ball-by-ball dataset (default: IPL.csv)",
    )
    parser.add_argument(
        "--output",
        dest="model_output",
        default="models/best_model.json",
        help="Destination for the trained XGBoost model (default: models/best_model.json)",
    )
    parser.add_argument(
        "--metadata",
        dest="metadata_output",
        default=None,
        help="Optional path to store model metadata such as metrics and feature columns.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retrain_and_save(args.data_path, args.model_output, args.metadata_output)
