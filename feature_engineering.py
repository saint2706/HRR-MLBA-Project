"""Feature engineering utilities for player impact modelling."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.copy().astype(float)
    denom = denominator.replace({0: np.nan})
    result = result.divide(denom)
    return result.fillna(0.0)


def compute_batting_metrics(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Derive batting metrics and composite batting index."""

    batting = player_stats.copy()
    batting["balls_faced"] = batting["balls_faced"].fillna(0)
    batting["runs_scored"] = batting["runs_scored"].fillna(0)
    batting["dismissals"] = batting["dismissals"].fillna(0)
    batting["boundaries"] = batting["boundaries"].fillna(0)
    batting["dot_balls"] = batting["dot_balls"].fillna(0)

    batting["batting_strike_rate"] = _safe_divide(batting["runs_scored"] * 100, batting["balls_faced"])
    batting["batting_average"] = _safe_divide(batting["runs_scored"], batting["dismissals"].replace({0: np.nan}))
    batting["boundary_percentage"] = _safe_divide(batting["boundaries"] * 100, batting["balls_faced"])
    batting["dot_percentage"] = _safe_divide(batting["dot_balls"] * 100, batting["balls_faced"])
    batting["batting_contribution"] = batting["runs_scored"]

    return batting


def compute_bowling_metrics(player_stats: pd.DataFrame, league_phase_summary: pd.DataFrame) -> pd.DataFrame:
    """Derive bowling metrics including phase efficacy."""

    bowling = player_stats.copy()
    bowling["balls_bowled"] = bowling["balls_bowled"].fillna(0)
    bowling["runs_conceded"] = bowling["runs_conceded"].fillna(0)
    bowling["wickets"] = bowling["wickets"].fillna(0)

    bowling["overs_bowled"] = bowling["balls_bowled"] / 6.0
    bowling["bowling_economy"] = _safe_divide(bowling["runs_conceded"], bowling["overs_bowled"].replace({0: np.nan}))
    bowling["bowling_strike_rate"] = _safe_divide(bowling["balls_bowled"], bowling["wickets"].replace({0: np.nan}))
    bowling["wickets_per_match"] = bowling["wickets"]

    if league_phase_summary is not None and not league_phase_summary.empty:
        league_phase_summary = league_phase_summary.copy()
        league_phase_summary["phase_run_rate"] = _safe_divide(
            league_phase_summary["phase_runs"] * 6,
            league_phase_summary["phase_balls"],
        )
        league_phase = league_phase_summary.groupby("phase")["phase_run_rate"].mean().to_dict()

        def compute_phase_efficacy(row: pd.Series) -> float:
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
                league_rr = league_phase.get(phase, 0.0)
                player_rr = (runs * 6) / balls
                weighted_diff += (league_rr - player_rr) * balls
                total_balls += balls
            if total_balls == 0:
                return 0.0
            return weighted_diff / total_balls

        bowling["phase_efficacy"] = bowling.apply(compute_phase_efficacy, axis=1)
    else:
        bowling["phase_efficacy"] = 0.0

    return bowling


def compute_composite_indices(
    batting_features: pd.DataFrame,
    bowling_features: pd.DataFrame,
    target_col: str = "team_won",
    weight_method: str = "correlation",
) -> pd.DataFrame:
    """Combine batting and bowling metrics with learned or correlation based weights."""

    features = batting_features.merge(
        bowling_features,
        on=["match_id", "season", "team", "player", target_col, "winning_team"],
        how="outer",
        suffixes=("_bat", "_bowl"),
    )

    metric_columns = [
        "batting_strike_rate",
        "batting_average",
        "boundary_percentage",
        "dot_percentage",
        "bowling_economy",
        "bowling_strike_rate",
        "wickets_per_match",
        "phase_efficacy",
    ]

    for column in metric_columns:
        if column not in features.columns:
            features[column] = 0.0
    features = features.fillna(0)

    if weight_method == "correlation" and target_col in features.columns:
        correlations = {}
        target = features[target_col]
        for column in metric_columns:
            if features[column].std(ddof=0) == 0 or target.std(ddof=0) == 0:
                correlations[column] = 0.0
            else:
                corr = features[column].corr(target)
                correlations[column] = 0.0 if pd.isna(corr) else corr
        weights = np.array([max(correlations[col], 0) for col in metric_columns])
    else:
        weights = np.ones(len(metric_columns))

    if weights.sum() == 0:
        weights = np.ones(len(weights))
    normalized_weights = weights / weights.sum()

    batting_metrics = ["batting_strike_rate", "batting_average", "boundary_percentage", "dot_percentage"]
    bowling_metrics = ["bowling_economy", "bowling_strike_rate", "wickets_per_match", "phase_efficacy"]

    features["batting_index"] = features[batting_metrics].mul(normalized_weights[:4], axis=1).sum(axis=1)
    features["bowling_index"] = features[bowling_metrics].mul(normalized_weights[4:], axis=1).sum(axis=1)
    features["overall_index"] = features[["batting_index", "bowling_index"]].sum(axis=1)

    return features, metric_columns


def build_model_matrix(features: pd.DataFrame, metric_columns: Iterable[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Construct modelling matrix with engineered features and target column."""

    modelling_columns = list(metric_columns) + [
        "batting_index",
        "bowling_index",
        "overall_index",
        "runs_scored",
        "balls_faced",
        "wickets",
        "overs_bowled",
    ]

    available_columns = [col for col in modelling_columns if col in features.columns]
    model_df = features[["match_id", "player", "team", "season", "team_won"] + available_columns].copy()
    model_df = model_df.fillna(0)

    feature_cols = [col for col in available_columns if col not in {"team_won"}]
    return model_df, feature_cols


def aggregate_player_ratings(model_df: pd.DataFrame, shap_weights: Dict[str, float]) -> pd.DataFrame:
    """Aggregate player impact ratings using SHAP-derived weights."""

    rating_weights = pd.Series(shap_weights)
    rating_weights = rating_weights / rating_weights.sum()

    feature_cols = [col for col in rating_weights.index if col in model_df.columns]
    weighted_scores = model_df[feature_cols].mul(rating_weights[feature_cols], axis=1).sum(axis=1)

    player_ratings = (
        model_df.assign(impact_score=weighted_scores)
        .groupby(["player", "team", "season"], as_index=False)
        .agg(impact_score=("impact_score", "mean"),
             batting_index=("batting_index", "mean"),
             bowling_index=("bowling_index", "mean"))
    )

    min_score = player_ratings["impact_score"].min()
    max_score = player_ratings["impact_score"].max()
    if max_score - min_score > 0:
        player_ratings["impact_rating"] = 100 * (player_ratings["impact_score"] - min_score) / (max_score - min_score)
    else:
        player_ratings["impact_rating"] = 50

    return player_ratings
