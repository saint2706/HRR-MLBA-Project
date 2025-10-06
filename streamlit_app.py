"""Streamlit dashboard for IPL player impact insights."""
from __future__ import annotations

from typing import Dict, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from pipeline import load_pipeline_outputs


st.set_page_config(
    page_title="IPL Impact Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _filter_profiles(
    profiles: pd.DataFrame,
    seasons: Tuple[str, ...],
    teams: Tuple[str, ...],
    min_balls: int,
) -> pd.DataFrame:
    """Filter player profiles based on sidebar inputs."""
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
    """Render metrics and a sample of the raw dataset."""
    st.subheader("📊 Dataset Overview", divider="rainbow")
    total_matches = raw_df["match_id"].nunique()
    total_seasons = raw_df["season"].nunique()
    total_teams = raw_df["batting_team"].nunique()

    cols = st.columns(4)
    cols[0].metric("Total Deliveries", f"{len(raw_df):,}")
    cols[1].metric("Total Matches", f"{total_matches:,}")
    cols[2].metric("Seasons Analyzed", total_seasons)
    cols[3].metric("Participating Teams", total_teams)

    with st.expander("Explore the raw dataset"):
        st.dataframe(raw_df.head(100))


def _render_model_performance(outputs: Dict[str, object]) -> None:
    """Render model performance metrics and SHAP summary."""
    st.subheader("🤖 Model Performance & Insights", divider="rainbow")
    artifacts = outputs["artifacts"]

    st.markdown("**Model Accuracy Metrics**")
    metric_cols = st.columns(len(artifacts.metrics)) if artifacts.metrics else []
    for idx, (metric, value) in enumerate(artifacts.metrics.items()):
        metric_cols[idx].metric(metric.replace("_", " ").title(), f"{value:.3f}")

    st.markdown("**Top Predictors of Match-Winning Performances (SHAP Analysis)**")
    shap_summary = outputs["shap_summary"]
    top_features = shap_summary.head(10)

    shap_chart = (
        alt.Chart(top_features)
        .mark_bar(cornerRadius=5, color="#F97316")
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Average Impact on Prediction"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=[
                "feature",
                alt.Tooltip("mean_abs_shap", title="Mean |SHAP|", format=".4f"),
            ],
        )
        .properties(height=320)
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(shap_chart, use_container_width=True)


def _render_cluster_plots(outputs: Dict[str, object]) -> None:
    """Render interactive scatter plots for player clusters."""
    batter_clustered = outputs["batter_clusters"]
    batter_labels = outputs["batter_labels"]
    bowler_clustered = outputs["bowler_clusters"]
    bowler_labels = outputs["bowler_labels"]

    # Map cluster labels to readable names for legend
    batter_clustered["batter_role"] = batter_clustered["batter_cluster"].map(batter_labels)
    bowler_clustered["bowler_role"] = bowler_clustered["bowler_cluster"].map(bowler_labels)

    tab1, tab2 = st.tabs(["Batter Archetypes", "Bowler Archetypes"])

    with tab1:
        batter_chart = (
            alt.Chart(batter_clustered)
            .mark_circle(size=80, opacity=0.8)
            .encode(
                x=alt.X("batting_strike_rate:Q", title="Strike Rate", scale=alt.Scale(zero=False)),
                y=alt.Y("batting_average:Q", title="Average", scale=alt.Scale(zero=False)),
                color=alt.Color("batter_role:N", legend=alt.Legend(title="Archetype")),
                tooltip=["player", "team", "batting_strike_rate", "batting_average", "batter_role"],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(batter_chart, use_container_width=True)

    with tab2:
        bowler_chart = (
            alt.Chart(bowler_clustered)
            .mark_circle(size=80, opacity=0.8)
            .encode(
                x=alt.X("bowling_economy:Q", title="Economy", scale=alt.Scale(zero=False)),
                y=alt.Y("bowling_strike_rate:Q", title="Strike Rate", scale=alt.Scale(zero=False)),
                color=alt.Color("bowler_role:N", legend=alt.Legend(title="Archetype")),
                tooltip=["player", "team", "bowling_economy", "bowling_strike_rate", "bowler_role"],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(bowler_chart, use_container_width=True)


def _render_player_insights(
    filtered_profiles: pd.DataFrame,
    best_xi: pd.DataFrame,
    outputs: Dict[str, object],
) -> None:
    """Render player rankings, Best XI, and cluster plots."""
    st.subheader("🌟 Player Insights & Archetypes", divider="rainbow")

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.markdown("**Impact Player Rankings**")
        top_players = filtered_profiles.sort_values("impact_rating", ascending=False).head(15)
        chart = (
            alt.Chart(top_players)
            .mark_bar(cornerRadius=5)
            .encode(
                x=alt.X("impact_rating:Q", title="Impact Rating"),
                y=alt.Y("player:N", sort="-x", title=None),
                color=alt.Color(
                    "primary_role:N",
                    legend=alt.Legend(title="Primary Role", orient="top"),
                    scale=alt.Scale(scheme="category10"),
                ),
                tooltip=[
                    "player",
                    "team",
                    "season",
                    alt.Tooltip("impact_rating", format=".1f"),
                    "primary_role",
                ],
            )
            .properties(height=420)
            .configure_axis(grid=False)
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown("**Data-Driven Best XI**")
        best_xi_html = "<div>"
        for _, row in best_xi.iterrows():
            best_xi_html += (
                f"<li><b>{row['player']}</b> ({row['team']})<br>"
                f"<small>Role: {row['primary_role']} | Rating: {row['impact_rating']:.1f}</small></li>"
            )
        best_xi_html += "</div>"
        st.markdown(best_xi_html, unsafe_allow_html=True)

        st.markdown("<br>**Player Archetype Explorer**", unsafe_allow_html=True)
        _render_cluster_plots(outputs)


def _render_team_and_phase_insights(filtered_profiles: pd.DataFrame, outputs: Dict[str, object]) -> None:
    """Render team-level analysis and top death bowlers."""
    st.subheader("📈 Team & Phase Analysis", divider="rainbow")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Team Impact Ratings**")
        team_summary = (
            filtered_profiles.groupby("team")["impact_rating"]
            .mean()
            .reset_index()
            .sort_values("impact_rating", ascending=False)
        )
        team_chart = (
            alt.Chart(team_summary)
            .mark_bar(cornerRadius=5, color="#2563EB")
            .encode(
                x=alt.X("impact_rating:Q", title="Average Impact Rating"),
                y=alt.Y("team:N", sort="-x", title=None),
                tooltip=["team", alt.Tooltip("impact_rating", format=".1f")],
            )
            .properties(height=360)
            .configure_axis(grid=False)
        )
        st.altair_chart(team_chart, use_container_width=True)

    with col2:
        st.markdown("**Top Death Over Bowlers (by Economy)**")
        bowler_phase = outputs["bowler_phase"]
        death_specialists = (
            bowler_phase[bowler_phase["phase"] == "Death"]
            .dropna(subset=["economy"])
            .sort_values("economy")
            .head(10)
        )
        death_chart = (
            alt.Chart(death_specialists)
            .mark_bar(cornerRadius=5, color="#D946EF")
            .encode(
                x=alt.X("economy:Q", title="Economy Rate (Death Overs)"),
                y=alt.Y("bowler:N", sort="-x", title=None),
                tooltip=["bowler", alt.Tooltip("economy", format=".2f"), "phase_balls"],
            )
            .properties(height=360)
            .configure_axis(grid=False)
        )
        st.altair_chart(death_chart, use_container_width=True)


def main() -> None:
    """Main function to run the Streamlit dashboard."""
    outputs = load_pipeline_outputs()
    player_profiles = outputs["player_profiles"]

    st.sidebar.title("Filters")
    seasons = tuple(sorted(player_profiles["season"].dropna().astype(str).unique()))
    teams = tuple(sorted(player_profiles["team"].dropna().unique()))

    selected_seasons = st.sidebar.multiselect(
        "Seasons",
        seasons,
        default=seasons[-3:] if len(seasons) > 3 else seasons,
    )
    selected_teams = st.sidebar.multiselect("Teams", teams, default=teams)
    min_balls_played = int(player_profiles[["balls_faced", "balls_bowled"]].sum(axis=1).max())
    min_balls = st.sidebar.slider(
        "Minimum combined balls faced/bowled",
        0,
        min_balls_played,
        60,
        step=10,
    )

    filtered_profiles = _filter_profiles(
        player_profiles,
        tuple(selected_seasons),
        tuple(selected_teams),
        min_balls,
    )

    st.title("🏏 IPL Impact Intelligence")
    st.markdown(
        "Welcome to the IPL Impact Intelligence dashboard. This platform analyzes "
        "player performance to identify archetypes, measure on-field impact, and "
        "select a data-driven 'Best XI' team. Use the filters in the sidebar to "
        "explore the data across different seasons and teams."
    )

    with st.container(border=True):
        _render_dataset_overview(outputs["raw_df"])

    with st.container(border=True):
        _render_model_performance(outputs)

    with st.container(border=True):
        _render_player_insights(filtered_profiles, outputs["best_xi"], outputs)

    with st.container(border=True):
        _render_team_and_phase_insights(filtered_profiles, outputs)


if __name__ == "__main__":
    main()

