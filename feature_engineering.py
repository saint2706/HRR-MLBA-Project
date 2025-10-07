"""
Feature engineering utilities for player impact modelling.

This module is dedicated to transforming the aggregated player-match statistics
into a rich set of features suitable for machine learning. It is responsible for:
1.  **Calculating Advanced Metrics**: Deriving sophisticated batting and bowling
    metrics that go beyond simple counts, such as strike rates, averages, and
    economy rates.
2.  **Phase-Based Analysis**: Computing a "phase efficacy" score for bowlers,
    which measures their performance in different stages of an innings compared
    to the league average.
3.  **Composite Indices**: Creating weighted indices for batting and bowling by
    combining multiple metrics. The weights can be determined based on their
    correlation with match outcomes.
4.  **Model Matrix Construction**: Assembling the final feature matrix that will be
    used to train the prediction model.
5.  **Player Impact Ratings**: Calculating a final "impact rating" for each player
    by aggregating feature contributions, weighted by their importance as
    determined by SHAP analysis.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Perform element-wise division of two Series, handling division by zero.

    This function replaces any zeros in the denominator with NaN before division
    to prevent errors. The resulting NaNs or infinities are then filled with 0.0.

    Args:
        numerator: The Series containing the numerators.
        denominator: The Series containing the denominators.

    Returns:
        A Series containing the result of the division, with zeros where the
        denominator was zero.
    """
    result = numerator.copy().astype(float)
    # Replace 0 with NaN to avoid ZeroDivisionError
    denom = denominator.replace({0: np.nan})
    result = result.divide(denom)
    # Fill any resulting NaNs (from 0/0) or infs with 0
    return result.fillna(0.0)


def compute_batting_metrics(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Derive advanced batting metrics from aggregated player statistics.

    This function calculates key performance indicators for batters, including:
    - Batting Strike Rate: (Runs Scored / Balls Faced) * 100
    - Batting Average: Runs Scored / Number of Dismissals
    - Boundary Percentage: (Number of Boundaries / Balls Faced) * 100
    - Dot Ball Percentage: (Number of Dot Balls / Balls Faced) * 100

    Args:
        player_stats: DataFrame containing aggregated stats per player per match.

    Returns:
        The input DataFrame with added columns for each batting metric.
    """
    batting = player_stats.copy()
    # Fill NaNs with 0 for key columns to ensure calculations are safe
    batting["balls_faced"] = batting["balls_faced"].fillna(0)
    batting["runs_scored"] = batting["runs_scored"].fillna(0)
    batting["dismissals"] = batting["dismissals"].fillna(0)
    batting["boundaries"] = batting["boundaries"].fillna(0)
    batting["dot_balls"] = batting["dot_balls"].fillna(0)

    # Calculate metrics using the safe_divide helper
    batting["batting_strike_rate"] = _safe_divide(batting["runs_scored"] * 100, batting["balls_faced"])
    # For average, we replace 0 dismissals with NaN to correctly handle not-out innings
    batting["batting_average"] = _safe_divide(batting["runs_scored"], batting["dismissals"].replace({0: np.nan}))
    batting["boundary_percentage"] = _safe_divide(batting["boundaries"] * 100, batting["balls_faced"])
    batting["dot_percentage"] = _safe_divide(batting["dot_balls"] * 100, batting["balls_faced"])
    batting["batting_contribution"] = batting["runs_scored"]

    return batting


def compute_bowling_metrics(player_stats: pd.DataFrame, league_phase_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Derive advanced bowling metrics, including a phase-specific efficacy score.

    This function calculates standard bowling metrics like economy and strike rate,
    as well as a more nuanced "phase efficacy" score. Phase efficacy measures how
    much better or worse a bowler's economy rate is compared to the league
    average in each phase of the game (Powerplay, Middle, Death), weighted by the
    number of balls bowled in that phase.

    Args:
        player_stats: DataFrame with aggregated player-match stats.
        league_phase_summary: DataFrame summarizing league-wide run rates per phase.

    Returns:
        The input DataFrame with added columns for each bowling metric.
    """
    bowling = player_stats.copy()
    # Fill NaNs with 0 for key columns
    bowling["balls_bowled"] = bowling["balls_bowled"].fillna(0)
    bowling["runs_conceded"] = bowling["runs_conceded"].fillna(0)
    bowling["wickets"] = bowling["wickets"].fillna(0)

    # Calculate standard bowling metrics
    bowling["overs_bowled"] = bowling["balls_bowled"] / 6.0
    bowling["bowling_economy"] = _safe_divide(bowling["runs_conceded"], bowling["overs_bowled"].replace({0: np.nan}))
    bowling["bowling_strike_rate"] = _safe_divide(bowling["balls_bowled"], bowling["wickets"].replace({0: np.nan}))
    bowling["wickets_per_match"] = bowling["wickets"]

    # Calculate phase efficacy if league data is available
    if league_phase_summary is not None and not league_phase_summary.empty:
        league_phase_summary = league_phase_summary.copy()
        # Calculate the average run rate for each phase across the league
        league_phase_summary["phase_run_rate"] = _safe_divide(
            league_phase_summary["phase_runs"] * 6,
            league_phase_summary["phase_balls"],
        )
        league_phase = league_phase_summary.groupby("phase")["phase_run_rate"].mean().to_dict()

        def compute_phase_efficacy(row: pd.Series) -> float:
            """Calculate the weighted efficacy for a single player-match record."""
            phase_dict = row.get("phase_runs", {})
            ball_dict = row.get("phase_balls", {})
            if not isinstance(phase_dict, dict) or not phase_dict:
                return 0.0

            weighted_diff = 0.0
            total_balls = 0.0
            for phase, runs in phase_dict.items():
                balls = ball_dict.get(phase, 0.0)
                if balls == 0:
                    continue
                # Compare player's run rate to the league average for the phase
                league_rr = league_phase.get(phase, 0.0)
                player_rr = (runs * 6) / balls
                # The difference is weighted by the number of balls bowled in that phase
                weighted_diff += (league_rr - player_rr) * balls
                total_balls += balls

            if total_balls == 0:
                return 0.0
            return weighted_diff / total_balls

        bowling["phase_efficacy"] = bowling.apply(compute_phase_efficacy, axis=1)
    else:
        # Default to 0 if no phase data is available
        bowling["phase_efficacy"] = 0.0

    return bowling


def compute_composite_indices(
    batting_features: pd.DataFrame,
    bowling_features: pd.DataFrame,
    target_col: str = "team_won",
    weight_method: str = "correlation",
) -> pd.DataFrame:
    """
    Combine batting and bowling metrics into composite indices using weighted sums.

    This function creates `batting_index` and `bowling_index` by combining the
    respective metrics into a single score. The weights for this combination can
    be determined by their correlation with the match outcome (`team_won`),
    providing a data-driven way to value different aspects of performance.

    Args:
        batting_features: DataFrame with batting metrics.
        bowling_features: DataFrame with bowling metrics.
        target_col: The target column used for calculating correlation-based weights.
        weight_method: The method for determining weights ('correlation' or uniform).

    Returns:
        A merged DataFrame containing the composite indices and the list of metric
        columns used.
    """
    # Merge batting and bowling dataframes
    features = batting_features.merge(
        bowling_features,
        on=["match_id", "season", "team", "player", target_col, "winning_team"],
        how="outer",
        suffixes=("_bat", "_bowl"),
    )

    # Define the core metrics to be included in the indices
    metric_columns = [
        "batting_strike_rate", "batting_average", "boundary_percentage", "dot_percentage",
        "bowling_economy", "bowling_strike_rate", "wickets_per_match", "phase_efficacy",
    ]

    # Ensure all metric columns exist, filling with 0 if necessary
    for column in metric_columns:
        if column not in features.columns:
            features[column] = 0.0

    # Only impute numeric columns to avoid upsetting categorical dtypes
    numeric_columns = features.select_dtypes(include=["number"]).columns
    if len(numeric_columns) > 0:
        features[numeric_columns] = features[numeric_columns].fillna(0)

    # Replace missing dictionary-like aggregates with an empty dict so downstream
    # calculations (e.g., phase efficacy) can safely iterate over them.
    for dict_col in ["phase_runs", "phase_balls"]:
        if dict_col in features.columns:
            features[dict_col] = features[dict_col].apply(
                lambda value: value if isinstance(value, dict) else {}
            )

    # Determine weights for each metric
    if weight_method == "correlation" and target_col in features.columns:
        correlations = {}
        target = features[target_col]
        for column in metric_columns:
            # Calculate correlation with the target variable (team_won)
            if features[column].std(ddof=0) == 0 or target.std(ddof=0) == 0:
                correlations[column] = 0.0
            else:
                corr = features[column].corr(target)
                correlations[column] = 0.0 if pd.isna(corr) else corr
        # Use the absolute correlation as the weight (negative correlations are also important)
        weights = np.array([max(correlations[col], 0) for col in metric_columns])
    else:
        # Default to uniform weights if correlation method is not used
        weights = np.ones(len(metric_columns))

    # Normalize weights to sum to 1
    if weights.sum() == 0:
        weights = np.ones(len(weights))
    normalized_weights = weights / weights.sum()

    # Separate metrics for batting and bowling
    batting_metrics = ["batting_strike_rate", "batting_average", "boundary_percentage", "dot_percentage"]
    bowling_metrics = ["bowling_economy", "bowling_strike_rate", "wickets_per_match", "phase_efficacy"]

    # Calculate the weighted composite indices
    features["batting_index"] = features[batting_metrics].mul(normalized_weights[:4], axis=1).sum(axis=1)
    features["bowling_index"] = features[bowling_metrics].mul(normalized_weights[4:], axis=1).sum(axis=1)
    features["overall_index"] = features[["batting_index", "bowling_index"]].sum(axis=1)

    return features, metric_columns


def build_model_matrix(features: pd.DataFrame, metric_columns: Iterable[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct the final modeling matrix with engineered features and the target column.

    This function selects the relevant columns for training the model, including
    the advanced metrics, composite indices, and raw performance stats (like
    runs_scored, wickets). It prepares a clean DataFrame ready for input into
    the machine learning model.

    Args:
        features: The DataFrame containing all engineered features.
        metric_columns: A list of the core metric columns.

    Returns:
        A tuple containing:
        - model_df: The final DataFrame for modeling.
        - feature_cols: A list of the column names used as features.
    """
    # Define all columns to be included in the model matrix
    modelling_columns = list(metric_columns) + [
        "batting_index", "bowling_index", "overall_index",
        "runs_scored", "balls_faced", "wickets", "overs_bowled",
    ]

    # Select only the available columns and prepare the final DataFrame
    available_columns = [col for col in modelling_columns if col in features.columns]
    model_df = features[["match_id", "player", "team", "season", "team_won"] + available_columns].copy()
    model_df = model_df.fillna(0)

    # Define the feature columns (all selected columns except the target)
    feature_cols = [col for col in available_columns if col not in {"team_won"}]
    return model_df, feature_cols


def aggregate_player_ratings(model_df: pd.DataFrame, shap_weights: Dict[str, float]) -> pd.DataFrame:
    """
    Aggregate player impact ratings using SHAP-derived feature importances as weights.

    After the model is trained, SHAP analysis provides the importance of each
    feature. This function uses these importances as weights to calculate a
t    weighted average of a player's feature values, resulting in an "impact score."
    This score is then normalized to a 0-100 scale to produce the final
    "impact rating."

    Args:
        model_df: The DataFrame used for modeling.
        shap_weights: A dictionary mapping feature names to their SHAP importance.

    Returns:
        A DataFrame with aggregated player ratings per season.
    """
    # Normalize SHAP weights to sum to 1
    rating_weights = pd.Series(shap_weights)
    rating_weights = rating_weights / rating_weights.sum()

    # Calculate the weighted impact score for each player-match
    feature_cols = [col for col in rating_weights.index if col in model_df.columns]
    weighted_scores = model_df[feature_cols].mul(rating_weights[feature_cols], axis=1).sum(axis=1)

    # Aggregate scores at the player-team-season level
    player_ratings = (
        model_df.assign(impact_score=weighted_scores)
        .groupby(["player", "team", "season"], as_index=False)
        .agg(
            impact_score=("impact_score", "mean"),
            batting_index=("batting_index", "mean"),
            bowling_index=("bowling_index", "mean")
        )
    )

    # Normalize the impact score to a 0-100 rating
    min_score = player_ratings["impact_score"].min()
    max_score = player_ratings["impact_score"].max()
    if max_score - min_score > 0:
        player_ratings["impact_rating"] = 100 * (player_ratings["impact_score"] - min_score) / (max_score - min_score)
    else:
        # If all scores are the same, assign a default rating of 50
        player_ratings["impact_rating"] = 50

    return player_ratings
