"""Streamlit dashboard for IPL player impact insights."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

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


st.set_page_config(
    page_title="IPL Impact Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "IPL.csv"

@lru_cache(maxsize=1)
def _prepare_pipeline_outputs(path: str) -> Dict[str, object]:
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


def _filter_profiles(
    profiles: pd.DataFrame,
    seasons: Tuple[str, ...],
    teams: Tuple[str, ...],
    min_balls: int,
) -> pd.DataFrame:
    filtered = profiles.copy()
    if seasons:
        filtered = filtered[filtered["season"].astype(str).isin(seasons)]
    if teams:
        filtered = filtered[filtered["team"].isin(teams)]
    if min_balls:
        workload = filtered[["balls_faced", "balls_bowled"]].fillna(0).sum(axis=1)
        filtered = filtered[workload >= min_balls]
    return filtered


def _render_dataset_overview(raw_df: pd.DataFrame) -> None:
    total_matches = raw_df["match_id"].nunique()
    total_seasons = raw_df["season"].nunique()
    total_teams = raw_df["batting_team"].nunique()

    st.subheader("Dataset at a glance")
    cols = st.columns(4)
    cols[0].metric("Deliveries", f"{len(raw_df):,}")
    cols[1].metric("Matches", f"{total_matches:,}")
    cols[2].metric("Seasons", total_seasons)
    cols[3].metric("Teams", total_teams)

    st.dataframe(raw_df.head(100))


def _render_model_performance(outputs: Dict[str, object]) -> None:
    st.subheader("Model diagnostics")
    artifacts = outputs["artifacts"]

    metric_cols = st.columns(len(artifacts.metrics)) if artifacts.metrics else []
    for idx, (metric, value) in enumerate(artifacts.metrics.items()):
        metric_cols[idx].metric(metric.upper(), f"{value:.3f}")

    shap_summary = outputs["shap_summary"]
    top_features = shap_summary.head(10)

    shap_chart = (
        alt.Chart(top_features)
        .mark_bar(color="#F97316")
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Mean |SHAP|"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=["feature", alt.Tooltip("mean_abs_shap", format=".4f")],
        )
        .properties(height=320)
    )
    st.altair_chart(shap_chart, use_container_width=True)


def _render_player_insights(filtered_profiles: pd.DataFrame, best_xi: pd.DataFrame) -> None:
    st.subheader("Player impact leaders")
    top_players = filtered_profiles.sort_values("impact_rating", ascending=False).head(15)

    chart = (
        alt.Chart(top_players)
        .mark_bar()
        .encode(
            x=alt.X("impact_rating:Q", title="Impact rating"),
            y=alt.Y("player:N", sort="-x"),
            color=alt.Color("primary_role:N", legend=alt.Legend(title="Primary role")),
            tooltip=["player", "team", "season", alt.Tooltip("impact_rating", format=".1f"), "primary_role"],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

    role_breakdown = (
        filtered_profiles.groupby("primary_role")["impact_rating"].mean().reset_index().sort_values("impact_rating")
    )
    st.markdown("**Average impact by role**")
    role_chart = (
        alt.Chart(role_breakdown)
        .mark_circle(size=180, color="#10B981")
        .encode(
            x=alt.X("impact_rating:Q", title="Average rating"),
            y=alt.Y("primary_role:N", sort="-x"),
            tooltip=["primary_role", alt.Tooltip("impact_rating", format=".1f")],
        )
        .properties(height=260)
    )
    st.altair_chart(role_chart, use_container_width=True)

    st.markdown("**Suggested Best XI**")
    st.dataframe(best_xi[["player", "team", "impact_rating", "primary_role", "balls_faced", "balls_bowled"]])


def _render_team_and_phase_insights(filtered_profiles: pd.DataFrame, outputs: Dict[str, object]) -> None:
    st.subheader("Team and phase insights")

    team_summary = (
        filtered_profiles.groupby("team")["impact_rating"].mean().reset_index().sort_values("impact_rating", ascending=False)
    )
    team_chart = (
        alt.Chart(team_summary)
        .mark_bar(color="#2563EB")
        .encode(
            x=alt.X("impact_rating:Q", title="Avg impact rating"),
            y=alt.Y("team:N", sort="-x"),
            tooltip=["team", alt.Tooltip("impact_rating", format=".1f")],
        )
        .properties(height=360)
    )

    season_summary = (
        filtered_profiles.groupby("season")["impact_rating"].mean().reset_index().sort_values("season")
    )
    season_chart = (
        alt.Chart(season_summary)
        .mark_line(point=True, color="#7C3AED")
        .encode(
            x=alt.X("season:N", title="Season"),
            y=alt.Y("impact_rating:Q", title="Avg impact rating"),
            tooltip=["season", alt.Tooltip("impact_rating", format=".2f")],
        )
        .properties(height=360)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(team_chart, use_container_width=True)
    with col2:
        st.altair_chart(season_chart, use_container_width=True)

    phase_summary = outputs["phase_summary"]
    phase_chart = (
        alt.Chart(phase_summary)
        .mark_bar(color="#F59E0B")
        .encode(
            x=alt.X("phase:N", title="Phase"),
            y=alt.Y("run_rate:Q", title="League run rate"),
            tooltip=["phase", alt.Tooltip("run_rate", format=".2f")],
        )
        .properties(height=320)
    )

    bowler_phase = outputs["bowler_phase"]
    death_specialists = (
        bowler_phase[bowler_phase["phase"] == "Death"]
        .dropna(subset=["economy"])
        .sort_values("economy")
        .head(10)
    )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**League scoring rate by phase**")
        st.altair_chart(phase_chart, use_container_width=True)
    with col4:
        st.markdown("**Top 10 death-over economies**")
        st.dataframe(
            death_specialists[["bowler", "economy", "phase_runs", "phase_balls"]]
            .rename(columns={"bowler": "player"})
            .assign(economy=lambda df: df["economy"].round(2))
        )


def main() -> None:
    outputs = _prepare_pipeline_outputs(DATA_PATH)
    raw_df = outputs["raw_df"]
    player_profiles = outputs["player_profiles"]

    seasons = tuple(sorted(player_profiles["season"].dropna().astype(str).unique()))
    teams = tuple(sorted(player_profiles["team"].dropna().unique()))

    st.sidebar.header("Filters")
    selected_seasons = st.sidebar.multiselect("Seasons", seasons, default=seasons[-3:] if len(seasons) > 3 else seasons)
    selected_teams = st.sidebar.multiselect("Teams", teams, default=teams)
    min_balls = st.sidebar.slider("Minimum combined balls faced/bowled", 0, int(player_profiles[["balls_faced", "balls_bowled"]].sum(axis=1).max()), 60, step=10)

    filtered_profiles = _filter_profiles(player_profiles, tuple(selected_seasons), tuple(selected_teams), min_balls)

    st.title("IPL Impact Intelligence Dashboard")
    st.caption("Trained on the Kaggle IPL ball-by-ball dataset to surface player roles and match impact.")

    _render_dataset_overview(raw_df)
    _render_model_performance(outputs)
    _render_player_insights(filtered_profiles, outputs["best_xi"])
    _render_team_and_phase_insights(filtered_profiles, outputs)


if __name__ == "__main__":
    main()

