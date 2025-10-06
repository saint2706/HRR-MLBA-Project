"""
Main orchestration script for the IPL player impact project.

This script serves as the entry point for running the entire data analysis pipeline.
It coordinates the execution of several key stages:
1.  **Data Loading and Preprocessing**: Reads the raw IPL dataset, cleans it, and
    derives initial features.
2.  **Feature Engineering**: Computes advanced player metrics for batting and
    bowling, creating composite indices to measure performance.
3.  **Model Training**: Trains a machine learning model to predict match outcomes
    based on player performance.
4.  **SHAP Analysis**: Uses SHAP (SHapley Additive exPlanations) to interpret the
    model's predictions and determine feature importance.
5.  **Player Clustering**: Groups players into distinct archetypes (e.g., "Power
    Hitter," "Death Specialist") based on their performance profiles.
6.  **Team Selection**: Selects a "Best XI" team by balancing player impact
    ratings and role diversity.
7.  **Visualization**: Generates plots to visualize the results, including SHAP
    summaries and cluster distributions.

The script can be executed from the command line and supports exporting the final
results to a JSON file for further analysis.
"""
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


# Configure logging to provide detailed feedback during execution
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(data_path: str) -> Dict[str, object]:
    """
    Executes the end-to-end IPL player analysis pipeline.

    This function orchestrates the entire workflow, from data loading to final
    output generation. It processes the raw data, engineers features, trains a
    model, and produces a dictionary of results, including player ratings, team
    selections, and visualizations.

    Args:
        data_path: The file path to the raw IPL ball-by-ball dataset (CSV format).

    Returns:
        A dictionary containing the key outputs of the pipeline, such as:
        - "model_metrics": Performance metrics of the trained model.
        - "classification_report": A detailed report of the model's predictions.
        - "shap_summary": A summary of feature importance from SHAP analysis.
        - "player_ratings": A DataFrame of players with their calculated impact ratings.
        - "best_xi": A DataFrame representing the selected "Best XI" team.
        - "plots": A dictionary of file paths for the generated visualizations.
        - "cluster_labels": A dictionary mapping cluster IDs to their descriptive names.
    """
    logger.info("Starting pipeline with data path: %s", data_path)

    # Step 1: Load and preprocess the data
    raw_df = load_data(data_path)
    processed_df = prepare_ball_by_ball(raw_df, PreprocessingConfig())

    # Step 2: Aggregate ball-by-ball data to the player-match level
    player_stats, phase_stats = aggregate_player_match_stats(processed_df)

    # Step 3: Engineer features for batting and bowling
    batting_metrics = compute_batting_metrics(player_stats)
    bowling_metrics = compute_bowling_metrics(player_stats, phase_stats)

    # Step 4: Create composite indices and build the model matrix
    features, metric_columns = compute_composite_indices(batting_metrics, bowling_metrics)
    model_df, feature_cols = build_model_matrix(features, metric_columns)

    # Step 5: Train the machine learning model
    logger.info("Training model with %d samples and %d features", len(model_df), len(feature_cols))
    artifacts = train_impact_model(model_df, feature_cols)

    # Step 6: Perform SHAP analysis to get feature importances
    shap_frame, shap_sample = compute_shap_values(artifacts.model, artifacts.X_train[feature_cols])
    shap_summary = summarise_shap_importance(shap_frame)
    shap_weights = dict(zip(shap_summary["feature"], shap_summary["mean_abs_shap"]))

    # If SHAP analysis yields no weights, use a default uniform weighting
    if not shap_weights:
        shap_weights = {col: 1.0 for col in feature_cols}

    # Step 7: Calculate player impact ratings using SHAP weights
    player_ratings = aggregate_player_ratings(model_df, shap_weights)

    # Step 8: Cluster batters into archetypes
    batter_base = batting_metrics[[
        "player", "team", "balls_faced", "batting_strike_rate",
        "batting_average", "boundary_percentage", "dot_percentage", "batting_index",
    ]]
    batter_clustered, batter_result = cluster_batters(batter_base)
    batter_labels = label_batter_clusters(batter_result)
    batter_clustered["batter_role"] = batter_clustered["batter_cluster"].map(batter_labels)

    # Step 9: Cluster bowlers into archetypes
    bowler_base = bowling_metrics[[
        "player", "team", "balls_bowled", "bowling_economy",
        "bowling_strike_rate", "wickets_per_match", "phase_efficacy", "bowling_index",
    ]]
    bowler_clustered, bowler_result = cluster_bowlers(bowler_base)
    bowler_labels = label_bowler_clusters(bowler_result)
    bowler_clustered["bowler_role"] = bowler_clustered["bowler_cluster"].map(bowler_labels)

    # Step 10: Consolidate player profiles with cluster information
    batter_mode = batter_clustered.groupby("player")["batter_cluster"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    bowler_mode = bowler_clustered.groupby("player")["bowler_cluster"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])

    volume_stats = (
        features[["player", "balls_faced", "balls_bowled"]]
        .fillna(0)
        .groupby("player", as_index=False)
        .sum()
    )

    player_profiles = player_ratings.merge(
        volume_stats, on="player", how="left"
    ).merge(
        batter_mode.rename("batter_cluster"), on="player", how="left"
    ).merge(
        bowler_mode.rename("bowler_cluster"), on="player", how="left"
    )

    # Clean up and finalize player profiles
    player_profiles["balls_faced"] = player_profiles["balls_faced"].fillna(0)
    player_profiles["balls_bowled"] = player_profiles["balls_bowled"].fillna(0)
    player_profiles = (
        player_profiles.sort_values("impact_rating", ascending=False)
        .drop_duplicates("player")
        .reset_index(drop=True)
    )

    # Assign roles based on clusters
    player_profiles["batter_role"] = player_profiles["batter_cluster"].map(batter_labels)
    player_profiles["bowler_role"] = player_profiles["bowler_cluster"].map(bowler_labels)
    player_profiles["primary_role"] = player_profiles.apply(
        determine_primary_role, axis=1, batting_roles=batter_labels, bowling_roles=bowler_labels
    )

    # Step 11: Select the "Best XI" team
    best_xi = select_best_xi(player_profiles)

    # Step 12: Generate visualizations
    shap_plot_path = plot_shap_summary(shap_frame, shap_sample)
    batter_cluster_plot = plot_cluster_scatter(
        batter_clustered,
        x_col="batting_strike_rate", y_col="boundary_percentage",
        cluster_col="batter_role", title="Batter Archetypes",
        output_filename="batter_clusters.png",
    )
    bowler_cluster_plot = plot_cluster_scatter(
        bowler_clustered,
        x_col="bowling_economy", y_col="bowling_strike_rate",
        cluster_col="bowler_role", title="Bowler Archetypes",
        output_filename="bowler_clusters.png",
    )

    # Step 13: Generate a classification report for the model
    report = generate_classification_report(artifacts)

    # Step 14: Compile all outputs into a single dictionary
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
    """
    Command-line interface for running the IPL analysis pipeline.

    This function sets up an argument parser to allow users to run the pipeline
    from the command line, specify the data path, and optionally export the
    results to a JSON file.
    """
    parser = argparse.ArgumentParser(description="Run the IPL player impact pipeline.")
    parser.add_argument(
        "data_path",
        nargs="?",
        default="IPL.csv",
        help="Path to the IPL ball-by-ball CSV file (default: IPL.csv)",
    )
    parser.add_argument(
        "--export-json",
        dest="export_json",
        default=None,
        help="Optional path to export summary JSON",
    )
    args = parser.parse_args()

    # Execute the main pipeline
    outputs = main(args.data_path)

    # Export results to JSON if a path is provided
    if args.export_json:
        # Create a serializable version of the outputs
        serialisable = {
            "model_metrics": outputs["model_metrics"],
            "cluster_labels": outputs["cluster_labels"],
            "plots": outputs["plots"],
        }
        with open(args.export_json, "w", encoding="utf-8") as fp:
            json.dump(serialisable, fp, indent=2)
        logger.info("Exported summary JSON to %s", args.export_json)

    # Print key results to the console
    print("Model metrics:\n", outputs["model_metrics"])
    print("Best XI:\n", outputs["best_xi"])  # type: ignore[truthy-bool]


if __name__ == "__main__":
    # This block ensures that the CLI function is called only when the script
    # is executed directly.
    cli()
