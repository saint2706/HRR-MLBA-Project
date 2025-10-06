"""Data loading and preprocessing utilities for the IPL player impact project."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing behaviour."""

    phase_bins: Dict[str, range] = None

    def __post_init__(self) -> None:
        if self.phase_bins is None:
            self.phase_bins = {
                "Powerplay": range(1, 7),
                "Middle": range(7, 16),
                "Death": range(16, 21),
            }


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
    """Return the first column from candidates that exists in the frame."""

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return default


def load_data(file_path: str, **read_csv_kwargs) -> pd.DataFrame:
    """Load the raw IPL ball-by-ball dataset."""

    defaults = {"low_memory": False}
    defaults.update(read_csv_kwargs)
    logger.info("Loading dataset from %s", file_path)
    df = pd.read_csv(file_path, **defaults)
    unnamed_columns = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed_columns:
        logger.debug("Dropping unnamed columns: %s", unnamed_columns)
        df = df.drop(columns=unnamed_columns)
    if df.empty:
        raise ValueError("Loaded dataframe is empty. Ensure the dataset is available via Git LFS.")
    return df


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        selected = _select_first_available(df, aliases)
        if selected:
            rename_map[selected] = canonical
    df = df.rename(columns=rename_map)
    return df


def _add_over_and_phase(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    df = df.copy()

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

    if derived_over.notna().any():
        over_series = pd.to_numeric(derived_over, errors="coerce")
        over_series = over_series.fillna(existing_over)
    else:
        over_series = existing_over

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

    def assign_phase(over: int) -> str:
        for phase, rng in config.phase_bins.items():
            if over in rng:
                return phase
        return "Other"

    df["phase"] = df["over"].apply(assign_phase)
    return df


def _infer_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    runs_total_col = _select_first_available(df, COLUMN_ALIASES["runs_total"], "runs_total")
    runs_total = pd.to_numeric(df.get(runs_total_col, 0), errors="coerce").fillna(0)

    valid_col = "valid_ball" if "valid_ball" in df.columns else None
    if valid_col:
        valid_series = pd.to_numeric(df[valid_col], errors="coerce").fillna(0).astype(int)
    else:
        valid_series = pd.Series(1, index=df.index, dtype=int)
    df["valid_ball"] = valid_series

    df["is_dot_ball"] = (runs_total == 0) & df["valid_ball"].astype(bool)

    wicket_flag_col = _select_first_available(df, COLUMN_ALIASES["is_wicket"], None)
    if wicket_flag_col:
        df["is_wicket"] = df[wicket_flag_col].fillna(False).astype(bool)
    else:
        df["is_wicket"] = df[_select_first_available(df, COLUMN_ALIASES["player_out"], "player_out")].notna()

    return df


def prepare_ball_by_ball(df: pd.DataFrame, config: Optional[PreprocessingConfig] = None) -> pd.DataFrame:
    """Standardise column names and add derived features."""

    if config is None:
        config = PreprocessingConfig()

    df = _canonicalise_columns(df)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = _add_over_and_phase(df, config)
    df = _infer_boolean_columns(df)

    for required in ("match_id", "batter", "bowler", "batting_team", "bowling_team"):
        if required not in df.columns:
            raise KeyError(f"Required column '{required}' not found after canonicalisation.")

    return df


def aggregate_player_match_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ball-level data to player-match level batting and bowling features."""

    df = df.copy()

    runs_batter_col = _select_first_available(df, COLUMN_ALIASES["runs_batter"], "runs_batter")
    runs_total_col = _select_first_available(df, COLUMN_ALIASES["runs_total"], "runs_total")
    player_out_col = _select_first_available(df, COLUMN_ALIASES["player_out"], "player_out")
    wicket_kind_col = _select_first_available(df, COLUMN_ALIASES["wicket_kind"], "wicket_kind")

    df["runs_batter"] = pd.to_numeric(df.get(runs_batter_col, 0), errors="coerce").fillna(0)
    df["runs_total"] = pd.to_numeric(df.get(runs_total_col, 0), errors="coerce").fillna(0)
    df["valid_ball"] = pd.to_numeric(df.get("valid_ball", 1), errors="coerce").fillna(0).astype(int)

    df["is_boundary"] = df["runs_batter"].isin([4, 6])
    df["is_six"] = df["runs_batter"] == 6
    df["is_four"] = df["runs_batter"] == 4

    df["batter_dismissed"] = df[player_out_col].fillna("__none__") == df["batter"]

    dismissal_exclusions = {"run out", "retired hurt", "retired out", "obstructing the field"}
    wicket_kind = df.get(wicket_kind_col, "").astype(str).str.lower()
    df["is_bowler_wicket"] = df["batter_dismissed"] & ~wicket_kind.fillna("").isin(dismissal_exclusions)

    df["dot_ball_flag"] = df["is_dot_ball"] & df["valid_ball"].astype(bool)

    df["phase_runs_tuple"] = list(zip(df["phase"], df["runs_total"]))
    df["phase_balls_tuple"] = list(zip(df["phase"], df["valid_ball"]))

    def _phase_sum(values):
        store: Dict[str, float] = {}
        for phase, val in values:
            if pd.isna(phase):
                continue
            store[phase] = store.get(phase, 0.0) + float(val)
        return store

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

    if "phase" in df.columns:
        phase_stats = (
            df.groupby(["bowler", "phase"])
            .agg(phase_runs=("runs_total", "sum"), phase_balls=("valid_ball", "sum"))
            .reset_index()
        )
    else:
        phase_stats = pd.DataFrame(columns=["bowler", "phase", "phase_runs", "phase_balls"])

    bowling_agg = bowling_agg.rename(columns={"bowling_team": "team", "bowler": "player"})

    player_stats = pd.merge(batting_agg, bowling_agg, on=["match_id", "season", "team", "player"], how="outer", indicator=True)

    for tmp_col in ["phase_runs_tuple_x", "phase_runs_tuple_y", "phase_balls_tuple", "dot_ball_flag_x", "dot_ball_flag_y"]:
        if tmp_col in player_stats.columns:
            player_stats = player_stats.drop(columns=tmp_col)

    if "dot_ball_flag" in player_stats.columns:
        player_stats = player_stats.drop(columns=["dot_ball_flag"])

    win_col = _select_first_available(df, COLUMN_ALIASES["win_team"], None)
    if win_col:
        match_winners = df[["match_id", win_col]].drop_duplicates().rename(columns={win_col: "winning_team"})
        player_stats = player_stats.merge(match_winners, on="match_id", how="left")
        player_stats["team_won"] = (player_stats["team"] == player_stats["winning_team"]).astype(int)
    else:
        player_stats["winning_team"] = None
        player_stats["team_won"] = 0

    return player_stats, phase_stats
