"""Team selection and role assignment helpers for the IPL impact project."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd


def determine_primary_role(
    row: pd.Series,
    batting_roles: Dict[int, str],
    bowling_roles: Dict[int, str],
) -> str:
    """Infer a player's headline role by combining batting and bowling signals."""

    has_batting = row.get("balls_faced", 0) > 30
    has_bowling = row.get("balls_bowled", 0) > 18

    batter_cluster = row.get("batter_cluster")
    bowler_cluster = row.get("bowler_cluster")

    batter_label = batting_roles.get(batter_cluster) if batter_cluster is not None else None
    bowler_label = bowling_roles.get(bowler_cluster) if bowler_cluster is not None else None

    if has_batting and has_bowling:
        return "Allrounder"
    if has_batting and batter_label:
        return batter_label
    if has_bowling and bowler_label:
        return bowler_label
    return batter_label or bowler_label or "Specialist"


def select_best_xi(player_profiles: pd.DataFrame) -> pd.DataFrame:
    """Select a balanced Best XI using impact ratings and role coverage."""

    selected_players: List[str] = []

    def _select_players(df: pd.DataFrame, count: int) -> List[str]:
        chosen: List[str] = []
        for _, candidate in df.sort_values("impact_rating", ascending=False).iterrows():
            name = candidate["player"]
            if name in selected_players:
                continue
            selected_players.append(name)
            chosen.append(name)
            if len(chosen) == count:
                break
        return chosen

    batting_roles = {"Power Hitter", "Anchor", "Accumulator", "Finisher"}
    bowling_roles = {"Death Specialist", "Strike Bowler", "Powerplay Controller", "Middle Overs"}

    batters = player_profiles[player_profiles["primary_role"].isin(batting_roles)]
    _select_players(batters, min(5, len(batters)))

    allrounders = player_profiles[player_profiles["primary_role"] == "Allrounder"]
    _select_players(allrounders, min(2, len(allrounders)))

    bowlers = player_profiles[player_profiles["primary_role"].isin(bowling_roles)]
    _select_players(bowlers, 11 - len(selected_players))

    if len(selected_players) < 11:
        remaining = player_profiles[~player_profiles["player"].isin(selected_players)]
        _select_players(remaining, 11 - len(selected_players))

    return (
        player_profiles[player_profiles["player"].isin(selected_players)]
        .sort_values("impact_rating", ascending=False)
        .reset_index(drop=True)
    )

