"""
Data loading and preprocessing utilities for the IPL player impact project.

This module provides a comprehensive set of functions for handling the initial
stages of the data pipeline. It is responsible for:
1.  **Loading Data**: Reading the raw IPL ball-by-ball dataset from a CSV file.
2.  **Column Canonicalization**: Standardizing column names to ensure consistency,
    as different datasets may use varying naming conventions (e.g., "batter" vs.
    "striker").
3.  **Feature Derivation**: Creating new, insightful features from the raw data,
    such as identifying dot balls, boundaries, and wickets. It also assigns each
    delivery to a specific phase of the game (Powerplay, Middle, Death).
4.  **Data Aggregation**: Transforming the granular ball-by-ball data into a
    more structured format by aggregating statistics at the player-match level.
    This creates a summary of each player's performance in every match they played.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing behaviour, specifically for defining game phases.

    This dataclass allows for easy customization of how an innings is divided into
    different phases (e.g., Powerplay, Middle, Death overs). The default settings
    are based on standard T20 cricket conventions.

    Attributes:
        phase_bins: A dictionary where keys are phase names (str) and values are
                    range objects defining the overs for that phase.
    """

    phase_bins: Dict[str, range] = None

    def __post_init__(self) -> None:
        """Initializes default phase bins if they are not provided."""
        if self.phase_bins is None:
            self.phase_bins = {
                "Powerplay": range(1, 7),  # Overs 1-6
                "Middle": range(7, 16),   # Overs 7-15
                "Death": range(16, 21),  # Overs 16-20
            }


# This dictionary maps canonical column names to a set of possible aliases found
# in different IPL datasets. This allows the pipeline to be robust to variations
# in column naming and ensures a standardized output.
COLUMN_ALIASES = {
    "match_id": {"match_id", "Match_Id", "matchid"},
    "season": {"season", "Season", "match_season"},
    "date": {"date", "match_date", "start_date"},
    "venue": {"venue", "stadium"},
    "city": {"city", "match_city"},
    "innings": {"innings", "inning"},
    "over": {"over", "overs"},
    "ball_in_over": {"ball", "ball_number", "ball_in_over", "ball_no"},
    "batter": {"batter", "striker", "batsman"},
    "non_striker": {"non_striker", "non-striker"},
    "bowler": {"bowler"},
    "batting_team": {"batting_team", "team_batting", "bat_team"},
    "bowling_team": {"bowling_team", "team_bowling", "bowl_team"},
    "runs_batter": {"runs_batter", "batsman_runs", "runs_off_bat", "runs_batsman", "batter_runs"},
    "runs_total": {"runs_total", "total_runs", "total"},
    "runs_extras": {"runs_extras", "extras", "extra_runs"},
    "extra_type": {"extra_type", "extras_type", "extra_kind"},
    "player_out": {"player_out", "dismissed_player", "player_dismissed"},
    "wicket_kind": {"wicket_kind", "dismissal_kind", "kind"},
    "fielder": {"fielder", "fielders_involved", "fielders"},
    "is_wicket": {"is_wicket", "wicket", "striker_out"},
    "win_team": {"match_won_by", "winner", "winning_team", "match_winner"},
}


def _select_first_available(df: pd.DataFrame, candidates: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    """
    Return the first column from a list of candidates that exists in a DataFrame.

    This helper function is used to find the correct column name from a set of
    possible aliases. It iterates through the candidates and returns the first one
    that is present in the DataFrame's columns.

    Args:
        df: The DataFrame to check for columns.
        candidates: An iterable of potential column names.
        default: The value to return if no candidate is found.

    Returns:
        The first matching column name, or the default value if none are found.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return default


def load_data(file_path: str, **read_csv_kwargs) -> pd.DataFrame:
    """
    Load the raw IPL ball-by-ball dataset from a CSV file.

    This function handles the initial data loading. It also cleans up the loaded
    DataFrame by removing any unnamed columns that are often created by spreadsheet
    programs. It raises an error if the loaded data is empty, which can happen
    if the dataset is not correctly downloaded (e.g., with Git LFS).

    Args:
        file_path: The path to the CSV file.
        **read_csv_kwargs: Additional keyword arguments to pass to `pd.read_csv`.

    Returns:
        A DataFrame containing the raw ball-by-ball data.

    Raises:
        ValueError: If the loaded DataFrame is empty.
    """
    defaults = {"low_memory": False}
    defaults.update(read_csv_kwargs)
    logger.info("Loading dataset from %s", file_path)
    df = pd.read_csv(file_path, **defaults)

    # Remove any unnamed columns that may be present
    unnamed_columns = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed_columns:
        logger.debug("Dropping unnamed columns: %s", unnamed_columns)
        df = df.drop(columns=unnamed_columns)

    # Ensure the dataset is not empty, which can be an issue with Git LFS
    if df.empty:
        raise ValueError("Loaded dataframe is empty. Ensure the dataset is available via Git LFS.")
    return df


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to a canonical format using the COLUMN_ALIASES map.

    This function iterates through the `COLUMN_ALIASES` dictionary and renames
    any matching columns in the DataFrame to their canonical form. This ensures
    that the rest of the pipeline can work with a consistent set of column names.

    Args:
        df: The DataFrame with potentially non-standard column names.

    Returns:
        A DataFrame with standardized column names.
    """
    df = df.copy()
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue  # Skip if the canonical name already exists
        selected = _select_first_available(df, aliases)
        if selected:
            rename_map[selected] = canonical
    df = df.rename(columns=rename_map)
    return df


def _add_over_and_phase(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """
    Derive over, ball number, and game phase from the raw data.

    This function is responsible for inferring the over and ball number for each
    delivery, as some datasets might not provide these explicitly. It then uses
    the `PreprocessingConfig` to assign each delivery to a game phase (e.g.,
    "Powerplay").

    Args:
        df: The DataFrame to process.
        config: The configuration object defining the game phases.

    Returns:
        The DataFrame with added 'over', 'ball_in_over', and 'phase' columns.
    """
    df = df.copy()

    # Get existing over and ball information, if available
    existing_over = (
        pd.to_numeric(df["over"], errors="coerce") if "over" in df.columns else pd.Series(np.nan, index=df.index)
    )
    existing_ball_in_over = (
        pd.to_numeric(df["ball_in_over"], errors="coerce")
        if "ball_in_over" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    derived_over = pd.Series(np.nan, index=df.index)
    derived_ball = pd.Series(np.nan, index=df.index)

    # Try to derive over and ball from columns like 'ball_no' or 'ball' (e.g., 6.1)
    if "ball_no" in df.columns:
        ball_no = pd.to_numeric(df["ball_no"], errors="coerce")
        if ball_no.notna().any():
            floor = np.floor(ball_no)
            derived_over = floor + 1
            derived_ball = (ball_no - floor) * 10
    elif "ball" in df.columns:
        ball_str = df["ball"].astype(str)
        if ball_str.str.contains(".").any():
            derived_over = pd.to_numeric(ball_str.str.split(".").str[0], errors="coerce")
            derived_ball = pd.to_numeric(ball_str.str.split(".").str[1], errors="coerce")

    # Combine derived and existing data, prioritizing derived values
    if derived_over.notna().any():
        over_series = pd.to_numeric(derived_over, errors="coerce")
        over_series = over_series.fillna(existing_over)
    else:
        over_series = existing_over

    # Fill any remaining NaNs and ensure the column is of integer type
    if over_series.isna().all() and "over" not in df.columns:
        over_series = pd.Series(0, index=df.index, dtype="int64")
    else:
        over_series = over_series.fillna(0)
    df["over"] = over_series.astype(int).clip(lower=0)

    if derived_ball.notna().any():
        ball_series = pd.to_numeric(derived_ball, errors="coerce").round().astype("Int64")
        if "ball" in df.columns:
            fallback_ball = pd.to_numeric(df["ball"], errors="coerce")
            ball_series = ball_series.where(ball_series.notna() & (ball_series > 0), fallback_ball)
        ball_series = ball_series.fillna(existing_ball_in_over)
    else:
        if "ball" in df.columns:
            ball_series = pd.to_numeric(df["ball"], errors="coerce")
        else:
            ball_series = existing_ball_in_over

    ball_series = ball_series.fillna(0)
    df["ball_in_over"] = ball_series.astype(int).clip(lower=0)

    # Assign game phase based on the over number
    def assign_phase(over: int) -> str:
        for phase, rng in config.phase_bins.items():
            if over in rng:
                return phase
        return "Other"

    df["phase"] = df["over"].apply(assign_phase)
    return df


def _infer_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create boolean flag columns for key events like dot balls and wickets.

    This function adds several boolean columns to the DataFrame to simplify
    downstream calculations. It identifies valid deliveries, dot balls, and
    wickets.

    Args:
        df: The DataFrame to process.

    Returns:
        The DataFrame with added boolean columns ('valid_ball', 'is_dot_ball', 'is_wicket').
    """
    df = df.copy()
    runs_total_col = _select_first_available(df, COLUMN_ALIASES["runs_total"], "runs_total")
    runs_total = pd.to_numeric(df.get(runs_total_col, 0), errors="coerce").fillna(0)

    # A "valid ball" is a delivery that counts towards the over (i.e., not a wide or no-ball)
    valid_col = "valid_ball" if "valid_ball" in df.columns else None
    if valid_col:
        valid_series = pd.to_numeric(df[valid_col], errors="coerce").fillna(0).astype(int)
    else:
        valid_series = pd.Series(1, index=df.index, dtype=int)
    df["valid_ball"] = valid_series

    # A dot ball is a valid delivery where no runs are scored
    df["is_dot_ball"] = (runs_total == 0) & df["valid_ball"].astype(bool)

    # A wicket is recorded if a player is dismissed
    wicket_flag_col = _select_first_available(df, COLUMN_ALIASES["is_wicket"], None)
    if wicket_flag_col:
        df["is_wicket"] = df[wicket_flag_col].fillna(False).astype(bool)
    else:
        # If no explicit wicket flag exists, infer it from the 'player_out' column
        df["is_wicket"] = df[_select_first_available(df, COLUMN_ALIASES["player_out"], "player_out")].notna()

    return df


def prepare_ball_by_ball(df: pd.DataFrame, config: Optional[PreprocessingConfig] = None) -> pd.DataFrame:
    """
    Standardize column names and add derived features to the ball-by-ball data.

    This function serves as a wrapper that orchestrates the main preprocessing
    steps. It ensures that the raw data is cleaned, standardized, and enriched
    with all the necessary features for the aggregation stage.

    Args:
        df: The raw ball-by-ball DataFrame.
        config: The preprocessing configuration for defining game phases.

    Returns:
        A fully preprocessed DataFrame ready for aggregation.

    Raises:
        KeyError: If essential columns are missing after canonicalization.
    """
    if config is None:
        config = PreprocessingConfig()

    df = _canonicalise_columns(df)

    # Convert date column to datetime objects for time-series analysis
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = _add_over_and_phase(df, config)
    df = _infer_boolean_columns(df)

    # Validate that all required columns are present
    for required in ("match_id", "batter", "bowler", "batting_team", "bowling_team"):
        if required not in df.columns:
            raise KeyError(f"Required column '{required}' not found after canonicalisation.")

    return df


def aggregate_player_match_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball-level data to player-match level batting and bowling features.

    This function is a critical step in the pipeline. It transforms the granular
    ball-by-ball data into a summarized view of each player's performance in a
    given match. It calculates key statistics for both batting and bowling.

    Args:
        df: The preprocessed ball-by-ball DataFrame.

    Returns:
        A tuple containing two DataFrames:
        - player_stats: A DataFrame with aggregated stats for each player per match.
        - phase_stats: A DataFrame summarizing bowler performance in each game phase.
    """
    df = df.copy()

    # Find the correct column names using the alias map
    runs_batter_col = _select_first_available(df, COLUMN_ALIASES["runs_batter"], "runs_batter")
    runs_total_col = _select_first_available(df, COLUMN_ALIASES["runs_total"], "runs_total")
    player_out_col = _select_first_available(df, COLUMN_ALIASES["player_out"], "player_out")
    wicket_kind_col = _select_first_available(df, COLUMN_ALIASES["wicket_kind"], "wicket_kind")

    # Ensure numeric columns are in the correct format
    df["runs_batter"] = pd.to_numeric(df.get(runs_batter_col, 0), errors="coerce").fillna(0)
    df["runs_total"] = pd.to_numeric(df.get(runs_total_col, 0), errors="coerce").fillna(0)
    df["valid_ball"] = pd.to_numeric(df.get("valid_ball", 1), errors="coerce").fillna(0).astype(int)

    # Create flags for specific events (boundaries, dismissals, etc.)
    df["is_boundary"] = df["runs_batter"].isin([4, 6])
    df["is_six"] = df["runs_batter"] == 6
    df["is_four"] = df["runs_batter"] == 4
    df["batter_dismissed"] = df[player_out_col].fillna("__none__") == df["batter"]

    # A "bowler wicket" is a dismissal where the bowler gets credit (e.g., not a run-out)
    dismissal_exclusions = {"run out", "retired hurt", "retired out", "obstructing the field"}
    wicket_kind = df.get(wicket_kind_col, "").astype(str).str.lower()
    df["is_bowler_wicket"] = df["batter_dismissed"] & ~wicket_kind.fillna("").isin(dismissal_exclusions)

    df["dot_ball_flag"] = df["is_dot_ball"] & df["valid_ball"].astype(bool)

    # Prepare data for phase-based aggregation
    df["phase_runs_tuple"] = list(zip(df["phase"], df["runs_total"]))
    df["phase_balls_tuple"] = list(zip(df["phase"], df["valid_ball"]))

    # Helper function to sum values within each phase
    def _phase_sum(values):
        store: Dict[str, float] = {}
        for phase, val in values:
            if pd.isna(phase):
                continue
            store[phase] = store.get(phase, 0.0) + float(val)
        return store

    # Aggregate batting statistics for each player in each match
    batter_group_cols = ["match_id", "season", "batting_team", "batter"]
    batting_agg = (
        df.groupby(batter_group_cols)
        .agg(
            runs_scored=("runs_batter", "sum"),
            balls_faced=("valid_ball", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            boundaries=("is_boundary", "sum"),
            dot_balls=("dot_ball_flag", "sum"),
            dismissals=("batter_dismissed", "sum"),
            phases=("phase_runs_tuple", _phase_sum),
        )
        .reset_index()
    )
    batting_agg = batting_agg.rename(columns={"batting_team": "team", "batter": "player"})

    # Aggregate bowling statistics for each player in each match
    bowler_group_cols = ["match_id", "season", "bowling_team", "bowler"]
    bowling_agg = (
        df.groupby(bowler_group_cols)
        .agg(
            runs_conceded=("runs_total", "sum"),
            balls_bowled=("valid_ball", "sum"),
            wickets=("is_bowler_wicket", "sum"),
            dot_balls_bowling=("dot_ball_flag", "sum"),
            phase_runs=("phase_runs_tuple", _phase_sum),
            phase_balls=("phase_balls_tuple", _phase_sum),
        )
        .reset_index()
    )

    # Create a separate summary of bowler performance by phase
    if "phase" in df.columns:
        phase_stats = (
            df.groupby(["bowler", "phase"])
            .agg(phase_runs=("runs_total", "sum"), phase_balls=("valid_ball", "sum"))
            .reset_index()
        )
    else:
        phase_stats = pd.DataFrame(columns=["bowler", "phase", "phase_runs", "phase_balls"])

    bowling_agg = bowling_agg.rename(columns={"bowling_team": "team", "bowler": "player"})

    # Merge the batting and bowling aggregates into a single DataFrame
    player_stats = pd.merge(batting_agg, bowling_agg, on=["match_id", "season", "team", "player"], how="outer", indicator=True)

    # Clean up temporary columns created during aggregation
    for tmp_col in ["phase_runs_tuple_x", "phase_runs_tuple_y", "phase_balls_tuple", "dot_ball_flag_x", "dot_ball_flag_y"]:
        if tmp_col in player_stats.columns:
            player_stats = player_stats.drop(columns=tmp_col)
    if "dot_ball_flag" in player_stats.columns:
        player_stats = player_stats.drop(columns=["dot_ball_flag"])

    # Add information about whether the player's team won the match
    win_col = _select_first_available(df, COLUMN_ALIASES["win_team"], None)
    if win_col:
        match_winners = df[["match_id", win_col]].drop_duplicates().rename(columns={win_col: "winning_team"})
        player_stats = player_stats.merge(match_winners, on="match_id", how="left")
        player_stats["team_won"] = (player_stats["team"] == player_stats["winning_team"]).astype(int)
    else:
        player_stats["winning_team"] = None
        player_stats["team_won"] = 0

    return player_stats, phase_stats
