"""Main orchestration script for the IPL player impact project."""
from __future__ import annotations

import argparse
import json
import logging
from typing import Dict

import pandas as pd

from clustering_roles import (
    cluster_batters,
    cluster_bowlers,
    label_batter_clusters,
    label_bowler_clusters,
)
from data_preprocessing import (
    PreprocessingConfig,
    aggregate_player_match_stats,
    load_data,
    prepare_ball_by_ball,
)
from feature_engineering import (
    aggregate_player_ratings,
    build_model_matrix,
    compute_batting_metrics,
    compute_bowling_metrics,
    compute_composite_indices,
)
from model_training import generate_classification_report, train_impact_model
from shap_analysis import compute_shap_values, summarise_shap_importance
from team_selection import determine_primary_role, select_best_xi
from visualization import plot_cluster_scatter, plot_shap_summary


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(data_path: str) -> Dict[str, object]:
    logger.info("Starting pipeline with data path: %s", data_path)
    raw_df = load_data(data_path)
    processed_df = prepare_ball_by_ball(raw_df, PreprocessingConfig())

    player_stats, phase_stats = aggregate_player_match_stats(processed_df)

    batting_metrics = compute_batting_metrics(player_stats)
    bowling_metrics = compute_bowling_metrics(player_stats, phase_stats)

    features, metric_columns = compute_composite_indices(batting_metrics, bowling_metrics)
    model_df, feature_cols = build_model_matrix(features, metric_columns)

    logger.info("Training model with %d samples and %d features", len(model_df), len(feature_cols))
    artifacts = train_impact_model(model_df, feature_cols)

    shap_frame, shap_sample = compute_shap_values(artifacts.model, artifacts.X_train[feature_cols])
    shap_summary = summarise_shap_importance(shap_frame)
    shap_weights = dict(zip(shap_summary["feature"], shap_summary["mean_abs_shap"]))

    if not shap_weights:
        shap_weights = {col: 1.0 for col in feature_cols}

    player_ratings = aggregate_player_ratings(model_df, shap_weights)

    batter_base = batting_metrics[[
        "player",
        "team",
        "balls_faced",
        "batting_strike_rate",
        "batting_average",
        "boundary_percentage",
        "dot_percentage",
        "batting_index",
    ]]
    batter_clustered, batter_result = cluster_batters(batter_base)
    batter_labels = label_batter_clusters(batter_result)
    batter_clustered["batter_role"] = batter_clustered["batter_cluster"].map(batter_labels)

    bowler_base = bowling_metrics[[
        "player",
        "team",
        "balls_bowled",
        "bowling_economy",
        "bowling_strike_rate",
        "wickets_per_match",
        "phase_efficacy",
        "bowling_index",
    ]]
    bowler_clustered, bowler_result = cluster_bowlers(bowler_base)
    bowler_labels = label_bowler_clusters(bowler_result)
    bowler_clustered["bowler_role"] = bowler_clustered["bowler_cluster"].map(bowler_labels)

    batter_mode = batter_clustered.groupby("player")["batter_cluster"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    bowler_mode = bowler_clustered.groupby("player")["bowler_cluster"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])

    volume_stats = (
        features[["player", "balls_faced", "balls_bowled"]]
        .fillna(0)
        .groupby("player", as_index=False)
        .sum()
    )

    player_profiles = player_ratings.merge(
        volume_stats,
        on="player",
        how="left",
    ).merge(
        batter_mode.rename("batter_cluster"),
        on="player",
        how="left",
    ).merge(
        bowler_mode.rename("bowler_cluster"),
        on="player",
        how="left",
    )

    player_profiles["balls_faced"] = player_profiles["balls_faced"].fillna(0)
    player_profiles["balls_bowled"] = player_profiles["balls_bowled"].fillna(0)

    player_profiles = (
        player_profiles.sort_values("impact_rating", ascending=False)
        .drop_duplicates("player")
        .reset_index(drop=True)
    )

    player_profiles["batter_role"] = player_profiles["batter_cluster"].map(batter_labels)
    player_profiles["bowler_role"] = player_profiles["bowler_cluster"].map(bowler_labels)

    player_profiles["primary_role"] = player_profiles.apply(
        determine_primary_role,
        axis=1,
        batting_roles=batter_labels,
        bowling_roles=bowler_labels,
    )

    best_xi = select_best_xi(player_profiles)

    shap_plot_path = plot_shap_summary(shap_frame, shap_sample)

    batter_cluster_plot = plot_cluster_scatter(
        batter_clustered,
        x_col="batting_strike_rate",
        y_col="boundary_percentage",
        cluster_col="batter_role",
        title="Batter Archetypes",
        output_filename="batter_clusters.png",
    )

    bowler_cluster_plot = plot_cluster_scatter(
        bowler_clustered,
        x_col="bowling_economy",
        y_col="bowling_strike_rate",
        cluster_col="bowler_role",
        title="Bowler Archetypes",
        output_filename="bowler_clusters.png",
    )

    report = generate_classification_report(artifacts)

    outputs = {
        "model_metrics": artifacts.metrics,
        "classification_report": report,
        "shap_summary": shap_summary,
        "player_ratings": player_ratings.sort_values("impact_rating", ascending=False).head(25),
        "best_xi": best_xi,
        "plots": {
            "shap_summary": shap_plot_path,
            "batter_clusters": batter_cluster_plot,
            "bowler_clusters": bowler_cluster_plot,
        },
        "cluster_labels": {
            "batters": batter_labels,
            "bowlers": bowler_labels,
        },
    }

    logger.info("Pipeline complete")
    return outputs


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run the IPL player impact pipeline.")
    parser.add_argument("data_path", nargs="?", default="IPL.csv", help="Path to the IPL ball-by-ball CSV file")
    parser.add_argument("--export-json", dest="export_json", default=None, help="Optional path to export summary JSON")
    args = parser.parse_args()

    outputs = main(args.data_path)

    if args.export_json:
        serialisable = {
            "model_metrics": outputs["model_metrics"],
            "cluster_labels": outputs["cluster_labels"],
            "plots": outputs["plots"],
        }
        with open(args.export_json, "w", encoding="utf-8") as fp:
            json.dump(serialisable, fp, indent=2)
        logger.info("Exported summary JSON to %s", args.export_json)

    print("Model metrics:\n", outputs["model_metrics"])
    print("Best XI:\n", outputs["best_xi"])  # type: ignore[truthy-bool]


if __name__ == "__main__":
    cli()
