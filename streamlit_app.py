"""Streamlit dashboard that surfaces IPL Impact Intelligence pipeline outputs.

This module orchestrates the full analytics workflow inside Streamlit, caching the
data-intensive computation step and exposing a suite of interactive views that let
analysts explore player impact ratings, archetypes, and model diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

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
from main import ensure_dataset_available
from model_training import ModelArtifacts, generate_classification_report, train_impact_model
from shap_analysis import compute_shap_values, summarise_shap_importance
from team_selection import determine_primary_role, select_best_xi

st.set_page_config(
    page_title="IPL Impact Intelligence Dashboard",
    page_icon="🏏",
    layout="wide",
)


def _format_team_name(team: str | float) -> str:
    """Return a human-friendly representation for team names displayed in tables."""
    if isinstance(team, str) and team:
        return team
    return "Unlisted"


def find_best_model_files() -> Tuple[Optional[Path], Optional[Path]]:
    """Search for best-model*.json and best_model*.json files in the repository.

    Returns:
        A tuple of (model_path, metadata_path) if found, otherwise (None, None).
    """
    repo_root = Path(__file__).parent

    # Search for model files with various patterns
    model_patterns = ["best-model*.json", "best_model*.json"]

    for pattern in model_patterns:
        model_files = list(repo_root.glob(pattern))
        # Filter out metadata files
        model_files = [f for f in model_files if "metadata" not in f.name]

        if model_files:
            # Use the first match
            model_path = model_files[0]

            # Look for corresponding metadata file
            metadata_candidates = [
                model_path.with_name(model_path.stem + "_metadata.json"),
                model_path.with_suffix(".metadata.json"),
            ]

            metadata_path = None
            for candidate in metadata_candidates:
                if candidate.exists():
                    metadata_path = candidate
                    break

            return model_path, metadata_path

    return None, None


def load_pretrained_model(model_df: pd.DataFrame, feature_cols: list[str]) -> Optional[ModelArtifacts]:
    """Load a pre-trained model if available in the repository.

    Args:
        model_df: The DataFrame with features and target for fallback metrics.
        feature_cols: List of feature column names.

    Returns:
        ModelArtifacts if model is found and loaded successfully, None otherwise.
    """
    model_path, metadata_path = find_best_model_files()

    if model_path is None:
        return None

    try:
        # Load the XGBoost model
        model = XGBClassifier()
        model.load_model(model_path)

        # Load metadata if available
        metrics = {}
        if metadata_path and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metrics = metadata.get("metrics", {})
            except json.JSONDecodeError as json_err:
                st.warning(f"⚠️ Metadata file has invalid JSON: {json_err}. Using model without metadata.")

        # Create mock train/test splits for compatibility
        # We just need the model for predictions, not for retraining
        X = model_df[feature_cols]
        y = model_df.get("team_won", pd.Series([0] * len(model_df), index=model_df.index))

        # Create minimal artifacts structure
        artifacts = ModelArtifacts(
            model=model,
            feature_cols=feature_cols,
            X_train=X,
            X_test=X.iloc[:0],  # Empty test set
            y_train=y,
            y_test=y.iloc[:0],  # Empty test set
            metrics=metrics,
            predictions=np.array([], dtype=int),
            probabilities=np.array([], dtype=float),
        )

        st.info(f"✅ Loaded pre-trained model from {model_path.name}")
        return artifacts

    except Exception as e:
        st.warning(f"⚠️ Failed to load pre-trained model: {e}. Training from scratch...")
        return None


@st.cache_data(show_spinner="Crunching numbers...", ttl=60 * 30)
def run_pipeline(data_path: str) -> Dict[str, Any]:
    """Run the full analysis pipeline and prepare artifacts for the dashboard.

    The underlying pipeline is identical to the command-line workflow defined in
    :mod:`main`, but the results are cached within Streamlit so that UI interactions
    remain snappy. When the dataset path changes, Streamlit invalidates the cache and
    recomputes the required artifacts.

    Args:
        data_path: Location of the ball-by-ball dataset file. The default value of
            ``"IPL.csv"`` matches the repository layout when Git LFS assets have been
            pulled.

    Returns:
        A dictionary containing all pre-computed tables needed by downstream view
        functions (e.g., player profiles, SHAP summaries, clustering assignments).
    """
    ensure_dataset_available(data_path)

    raw_df = load_data(data_path)
    processed_df = prepare_ball_by_ball(raw_df, PreprocessingConfig())

    player_stats, phase_stats = aggregate_player_match_stats(processed_df)
    batting_metrics = compute_batting_metrics(player_stats)
    bowling_metrics = compute_bowling_metrics(player_stats, phase_stats)

    features, metric_columns = compute_composite_indices(batting_metrics, bowling_metrics)
    model_df, feature_cols = build_model_matrix(features, metric_columns)

    batting_with_index = batting_metrics.merge(
        features[["match_id", "season", "team", "player", "batting_index"]],
        on=["match_id", "season", "team", "player"],
        how="left",
    )
    bowling_with_index = bowling_metrics.merge(
        features[["match_id", "season", "team", "player", "bowling_index"]],
        on=["match_id", "season", "team", "player"],
        how="left",
    )

    # Try to load pre-trained model first, fall back to training if not available
    artifacts = load_pretrained_model(model_df, feature_cols)
    if artifacts is None:
        st.info("🔧 No pre-trained model found. Training model from scratch...")
        artifacts = train_impact_model(model_df, feature_cols)

    shap_frame, _ = compute_shap_values(artifacts.model, artifacts.X_train[feature_cols])
    shap_summary = summarise_shap_importance(shap_frame)

    shap_weights = dict(zip(shap_summary["feature"], shap_summary["mean_abs_shap"]))
    if not shap_weights:
        shap_weights = {col: 1.0 for col in feature_cols}

    player_ratings = aggregate_player_ratings(model_df, shap_weights)

    batter_base = batting_with_index[
        [
            "player",
            "team",
            "balls_faced",
            "batting_strike_rate",
            "batting_average",
            "boundary_percentage",
            "dot_percentage",
            "batting_index",
        ]
    ]
    batter_clustered, batter_result = cluster_batters(batter_base)
    batter_labels = label_batter_clusters(batter_result)
    if batter_clustered.empty:
        batter_clustered["batter_role"] = pd.Series(dtype="object", index=batter_clustered.index)
    else:
        batter_clustered["batter_role"] = batter_clustered["batter_cluster"].map(batter_labels)

    bowler_base = bowling_with_index[
        [
            "player",
            "team",
            "balls_bowled",
            "bowling_economy",
            "bowling_strike_rate",
            "wickets_per_match",
            "phase_efficacy",
            "bowling_index",
        ]
    ]
    bowler_clustered, bowler_result = cluster_bowlers(bowler_base)
    bowler_labels = label_bowler_clusters(bowler_result)
    if bowler_clustered.empty:
        bowler_clustered["bowler_role"] = pd.Series(dtype="object", index=bowler_clustered.index)
    else:
        bowler_clustered["bowler_role"] = bowler_clustered["bowler_cluster"].map(bowler_labels)

    batter_mode = batter_clustered.groupby("player")["batter_cluster"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
    )
    bowler_mode = bowler_clustered.groupby("player")["bowler_cluster"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
    )

    volume_group_cols = [col for col in ["player", "team", "season"] if col in batting_with_index.columns]
    batting_volume = (
        batting_with_index[volume_group_cols + ["balls_faced"]].groupby(volume_group_cols, as_index=False).sum()
    )
    bowling_volume = (
        bowling_with_index[volume_group_cols + ["balls_bowled"]].groupby(volume_group_cols, as_index=False).sum()
    )
    volume_stats = batting_volume.merge(bowling_volume, on=volume_group_cols, how="outer").fillna(0)

    player_profiles = (
        player_ratings.merge(volume_stats, on=volume_group_cols, how="left")
        .merge(batter_mode.rename("batter_cluster"), on="player", how="left")
        .merge(bowler_mode.rename("bowler_cluster"), on="player", how="left")
    )
    player_profiles["balls_faced"] = player_profiles["balls_faced"].fillna(0)
    player_profiles["balls_bowled"] = player_profiles["balls_bowled"].fillna(0)
    player_profiles = player_profiles.sort_values("impact_rating", ascending=False).reset_index(drop=True)

    player_profiles["batter_role"] = player_profiles["batter_cluster"].map(batter_labels)
    player_profiles["bowler_role"] = player_profiles["bowler_cluster"].map(bowler_labels)
    player_profiles["primary_role"] = player_profiles.apply(
        determine_primary_role,
        axis=1,
        batting_roles=batter_labels,
        bowling_roles=bowler_labels,
    )

    best_xi = select_best_xi(player_profiles)

    classification = generate_classification_report(artifacts)

    outputs: Dict[str, Any] = {
        "model_metrics": artifacts.metrics,
        "classification_report": classification,
        "shap_summary": shap_summary,
        "player_ratings": player_ratings.sort_values("impact_rating", ascending=False),
        "player_profiles": player_profiles,
        "best_xi": best_xi,
        "batter_clustered": batter_clustered,
        "bowler_clustered": bowler_clustered,
        "cluster_labels": {"batters": batter_labels, "bowlers": bowler_labels},
    }
    return outputs


def render_overview(outputs: Dict[str, Any]) -> None:
    """Render the high-level model snapshot, best XI, and SHAP summary visuals."""
    st.subheader("Model Snapshot")
    metrics = outputs["model_metrics"]
    cols = st.columns(len(metrics) or 1)
    for (name, value), col in zip(metrics.items(), cols):
        col.metric(name.replace("_", " ").title(), f"{value:.3f}" if isinstance(value, (int, float)) else value)

    st.write("\n")
    best_xi = outputs["best_xi"].copy()
    best_xi["team"] = best_xi["team"].map(_format_team_name)
    st.markdown("### Data-Driven Best XI")
    st.dataframe(
        best_xi[["player", "team", "impact_rating", "primary_role", "batting_index", "bowling_index"]]
        .rename(
            columns={
                "player": "Player",
                "team": "Team",
                "impact_rating": "Impact Rating",
                "primary_role": "Primary Role",
                "batting_index": "Batting Index",
                "bowling_index": "Bowling Index",
            }
        )
        .style.format({"Impact Rating": "{:.1f}", "Batting Index": "{:.2f}", "Bowling Index": "{:.2f}"})
    )

    shap_summary = outputs["shap_summary"].head(15)
    st.markdown("### Top Feature Drivers")
    chart = (
        alt.Chart(shap_summary)
        .mark_bar()
        .encode(
            x=alt.X("mean_abs_shap", title="Mean |SHAP|"),
            y=alt.Y("feature", sort="-x", title="Feature"),
            tooltip=["feature", alt.Tooltip("mean_abs_shap", format=".3f", title="Mean |SHAP|")],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)


def render_player_explorer(outputs: Dict[str, Any]) -> None:
    """Display an interactive table for filtering and comparing player profiles."""
    profiles = outputs["player_profiles"].copy()
    profiles["team"] = profiles["team"].map(_format_team_name)

    st.subheader("Player Explorer")
    with st.expander("Filters", expanded=True):
        seasons = sorted(profiles["season"].dropna().unique())
        teams = sorted(profiles["team"].dropna().unique())
        roles = sorted(profiles["primary_role"].dropna().unique())

        default_seasons = seasons[-3:] if len(seasons) >= 3 else seasons
        season_filter = st.multiselect("Seasons", seasons, default=default_seasons)
        team_filter = st.multiselect("Teams", teams)
        role_filter = st.multiselect("Primary Roles", roles)

        max_balls_faced = int((profiles["balls_faced"].max(skipna=True) or 0))
        max_balls_bowled = int((profiles["balls_bowled"].max(skipna=True) or 0))

        min_balls_batted = st.slider(
            "Minimum Balls Faced",
            0,
            max_balls_faced,
            min(30, max_balls_faced),
        )
        min_balls_bowled = st.slider(
            "Minimum Balls Bowled",
            0,
            max_balls_bowled,
            min(12, max_balls_bowled),
        )

    filtered = profiles.copy()
    if season_filter:
        filtered = filtered[filtered["season"].isin(season_filter)]
    if team_filter:
        filtered = filtered[filtered["team"].isin(team_filter)]
    if role_filter:
        filtered = filtered[filtered["primary_role"].isin(role_filter)]
    filtered = filtered[(filtered["balls_faced"] >= min_balls_batted) | (filtered["balls_bowled"] >= min_balls_bowled)]

    st.metric("Players Returned", len(filtered))
    st.dataframe(
        filtered[
            [
                "player",
                "team",
                "season",
                "impact_rating",
                "primary_role",
                "batter_role",
                "bowler_role",
                "batting_index",
                "bowling_index",
                "balls_faced",
                "balls_bowled",
            ]
        ]
        .rename(
            columns={
                "player": "Player",
                "team": "Team",
                "season": "Season",
                "impact_rating": "Impact Rating",
                "primary_role": "Primary Role",
                "batter_role": "Batter Archetype",
                "bowler_role": "Bowler Archetype",
                "batting_index": "Batting Index",
                "bowling_index": "Bowling Index",
                "balls_faced": "Balls Faced",
                "balls_bowled": "Balls Bowled",
            }
        )
        .style.format(
            {
                "Impact Rating": "{:.1f}",
                "Batting Index": "{:.2f}",
                "Bowling Index": "{:.2f}",
            }
        )
    )


def render_cluster_section(outputs: Dict[str, Any]) -> None:
    """Visualise batter and bowler archetypes using interactive Altair scatter plots."""
    st.subheader("Role Archetypes")
    batter_clusters = outputs["batter_clustered"].copy()
    bowler_clusters = outputs["bowler_clustered"].copy()

    col1, col2 = st.columns(2)

    with col1:
        if batter_clusters.empty:
            st.info("No batter data available for clustering.")
        else:
            batter_clusters["team"] = batter_clusters["team"].map(_format_team_name)
            batter_chart = (
                alt.Chart(batter_clusters)
                .mark_circle(size=60, opacity=0.7)
                .encode(
                    x=alt.X("batting_strike_rate", title="Strike Rate"),
                    y=alt.Y("boundary_percentage", title="Boundary %"),
                    color=alt.Color("batter_role", title="Archetype"),
                    tooltip=["player", "team", "batting_average", "batting_index"],
                )
                .interactive()
                .properties(title="Batter Archetypes", height=400)
            )
            st.altair_chart(batter_chart, use_container_width=True)

    with col2:
        if bowler_clusters.empty:
            st.info("No bowler data available for clustering.")
        else:
            bowler_clusters["team"] = bowler_clusters["team"].map(_format_team_name)
            bowler_chart = (
                alt.Chart(bowler_clusters)
                .mark_circle(size=60, opacity=0.7)
                .encode(
                    x=alt.X("bowling_economy", title="Economy"),
                    y=alt.Y("bowling_strike_rate", title="Strike Rate"),
                    color=alt.Color("bowler_role", title="Archetype"),
                    tooltip=["player", "team", "wickets_per_match", "phase_efficacy"],
                )
                .interactive()
                .properties(title="Bowler Archetypes", height=400)
            )
            st.altair_chart(bowler_chart, use_container_width=True)


def render_model_diagnostics(outputs: Dict[str, Any]) -> None:
    """Show model quality diagnostics, including the classification report."""
    st.subheader("Model Diagnostics")
    st.markdown("#### Classification Report")
    st.text(outputs["classification_report"])


def _show_error(message: str) -> None:
    """Present errors prominently in the Streamlit UI."""

    st.error(message)


def main() -> None:
    """Entry-point for launching the Streamlit dashboard."""
    st.title("IPL Impact Intelligence Dashboard")
    st.caption("Explore player impact scores, archetypes, and model diagnostics in one place.")

    with st.sidebar:
        st.header("Configuration")
        data_path = st.text_input("Dataset path", value="IPL.csv")
        st.markdown("Use the default path if you have downloaded the Kaggle dataset via Git LFS.")

    try:
        outputs = run_pipeline(data_path)
    except Exception as exc:  # noqa: BLE001
        _show_error(str(exc))
        st.stop()

    overview_tab, players_tab, clusters_tab, shap_tab = st.tabs(
        ["Overview", "Player Explorer", "Role Archetypes", "Diagnostics"]
    )

    with overview_tab:
        render_overview(outputs)
    with players_tab:
        render_player_explorer(outputs)
    with clusters_tab:
        render_cluster_section(outputs)
    with shap_tab:
        render_model_diagnostics(outputs)


if __name__ == "__main__":
    main()
