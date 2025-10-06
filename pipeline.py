from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np
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
from model_training import train_impact_model
from shap_analysis import compute_shap_values, summarise_shap_importance
from team_selection import determine_primary_role, select_best_xi


DATA_PATH = "IPL.csv"


@lru_cache(maxsize=1)
def load_pipeline_outputs(path: str = DATA_PATH) -> Dict[str, object]:
    """
    Load and process all data, returning a dictionary of outputs.

    This function is cached to ensure the expensive data processing steps are
    run only once.

    Args:
        path: The path to the raw IPL dataset.

    Returns:
        A dictionary containing all the data artifacts from the pipeline.
    """
    raw_df = load_data(path)
    processed_df = prepare_ball_by_ball(raw_df, PreprocessingConfig())

    player_stats, phase_stats = aggregate_player_match_stats(processed_df)

    batting_metrics = compute_batting_metrics(player_stats)
    bowling_metrics = compute_bowling_metrics(player_stats, phase_stats)

    features, metric_columns = compute_composite_indices(batting_metrics, bowling_metrics)
    model_df, feature_cols = build_model_matrix(features, metric_columns)

    artifacts = train_impact_model(model_df, feature_cols)

    shap_frame, _ = compute_shap_values(artifacts.model, artifacts.X_train[feature_cols])
    shap_summary = summarise_shap_importance(shap_frame)

    shap_weights = dict(zip(shap_summary["feature"], shap_summary["mean_abs_shap"]))
    if not shap_weights:
        shap_weights = {column: 1.0 for column in feature_cols}

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

    batter_clustered, batter_result = cluster_batters(batter_base)
    bowler_clustered, bowler_result = cluster_bowlers(bowler_base)

    batter_labels = label_batter_clusters(batter_result)
    bowler_labels = label_bowler_clusters(bowler_result)

    batter_mode = (
        batter_clustered.groupby("player")["batter_cluster"]
        .agg(lambda series: series.mode().iloc[0] if not series.mode().empty else series.iloc[0])
    )
    bowler_mode = (
        bowler_clustered.groupby("player")["bowler_cluster"]
        .agg(lambda series: series.mode().iloc[0] if not series.mode().empty else series.iloc[0])
    )

    volume_stats = (
        features[["player", "balls_faced", "balls_bowled"]]
        .fillna(0)
        .groupby("player", as_index=False)
        .sum()
    )

    player_profiles = (
        player_ratings.merge(volume_stats, on="player", how="left")
        .merge(batter_mode.rename("batter_cluster"), on="player", how="left")
        .merge(bowler_mode.rename("bowler_cluster"), on="player", how="left")
    )

    player_profiles["balls_faced"] = player_profiles["balls_faced"].fillna(0)
    player_profiles["balls_bowled"] = player_profiles["balls_bowled"].fillna(0)

    player_profiles = (
        player_profiles.sort_values("impact_rating", ascending=False)
        .drop_duplicates(["player", "season", "team"], keep="first")
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

    phase_summary = (
        phase_stats.groupby("phase")
        .agg(total_runs=("phase_runs", "sum"), total_balls=("phase_balls", "sum"))
        .reset_index()
    )
    phase_summary["run_rate"] = np.where(
        phase_summary["total_balls"] > 0,
        (phase_summary["total_runs"] * 6) / phase_summary["total_balls"],
        0.0,
    )

    bowler_phase = (
        phase_stats.groupby(["bowler", "phase"])
        .agg(phase_runs=("phase_runs", "sum"), phase_balls=("phase_balls", "sum"))
        .reset_index()
    )
    bowler_phase["economy"] = np.where(
        bowler_phase["phase_balls"] > 0,
        (bowler_phase["phase_runs"] * 6) / bowler_phase["phase_balls"],
        np.nan,
    )

    return {
        "raw_df": raw_df,
        "processed_df": processed_df,
        "player_profiles": player_profiles,
        "best_xi": best_xi,
        "shap_summary": shap_summary,
        "artifacts": artifacts,
        "batter_clusters": batter_clustered,
        "bowler_clusters": bowler_clustered,
        "batter_labels": batter_labels,
        "bowler_labels": bowler_labels,
        "phase_summary": phase_summary,
        "bowler_phase": bowler_phase,
    }