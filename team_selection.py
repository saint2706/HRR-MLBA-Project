"""
Team selection and role assignment helpers for the IPL impact project.

This module provides the logic for the final, practical application of the analysis:
selecting a "Best XI" team. It uses the player profiles, impact ratings, and
role archetypes to construct a balanced and high-performing team.

The key functions are:
1.  **determine_primary_role**: Assigns a single, primary role to each player
    (e.g., "Power Hitter," "Allrounder," "Strike Bowler") based on their on-field
    workload and clustered archetype.
2.  **select_best_xi**: Implements a selection algorithm that picks a team of 11
    players, prioritizing those with the highest impact ratings while ensuring
    the team has a good mix of batters, bowlers, and all-rounders.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd


def determine_primary_role(
    row: pd.Series,
    batting_roles: Dict[int, str],
    bowling_roles: Dict[int, str],
) -> str:
    """
    Infer a player's primary role based on their workload and cluster assignments.

    This function determines a single, headline role for a player by considering
    whether they have a significant batting and/or bowling workload and what their
    assigned cluster labels are.

    Args:
        row: A Series representing a single player's profile data.
        batting_roles: A dictionary mapping batter cluster IDs to their labels.
        bowling_roles: A dictionary mapping bowler cluster IDs to their labels.

    Returns:
        A string representing the player's primary role (e.g., "Allrounder",
        "Power Hitter", "Specialist").
    """
    # Define workload thresholds to determine if a player is a significant batter or bowler
    has_batting = row.get("balls_faced", 0) > 30
    has_bowling = row.get("balls_bowled", 0) > 18

    # Get the player's assigned cluster labels
    batter_cluster = row.get("batter_cluster")
    bowler_cluster = row.get("bowler_cluster")
    batter_label = batting_roles.get(batter_cluster) if batter_cluster is not None else None
    bowler_label = bowling_roles.get(bowler_cluster) if bowler_cluster is not None else None

    # Apply a set of rules to determine the primary role
    if has_batting and has_bowling:
        return "Allrounder"
    if has_batting and batter_label:
        return batter_label
    if has_bowling and bowler_label:
        return bowler_label

    # As a fallback, use any available label or default to "Specialist"
    return batter_label or bowler_label or "Specialist"


def select_best_xi(player_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Select a balanced "Best XI" team using impact ratings and role coverage.

    This function implements a greedy selection algorithm to construct a team of 11
    players. It follows a structured approach to ensure the team is well-balanced:
    1. Selects a core of specialist batters.
    2. Adds all-rounders to the mix.
    3. Fills the remaining spots with specialist bowlers.
    4. If the team is still not full, it adds the best remaining players based on
       impact rating, regardless of their role.

    Args:
        player_profiles: A DataFrame containing the profiles of all players,
                         including their impact ratings and primary roles.

    Returns:
        A DataFrame representing the selected Best XI, sorted by impact rating.
    """
    selected_players: List[str] = []

    def _select_players(df: pd.DataFrame, count: int) -> List[str]:
        """A helper function to select the top `count` players from a given DataFrame."""
        chosen: List[str] = []
        # Iterate through players sorted by impact rating in descending order
        for _, candidate in df.sort_values("impact_rating", ascending=False).iterrows():
            name = candidate["player"]
            # Skip players who have already been selected
            if name in selected_players:
                continue
            selected_players.append(name)
            chosen.append(name)
            # Stop once the desired number of players has been chosen
            if len(chosen) == count:
                break
        return chosen

    # Define the set of roles for batters and bowlers
    batting_roles = {"Power Hitter", "Anchor", "Accumulator", "Finisher"}
    bowling_roles = {"Death Specialist", "Strike Bowler", "Powerplay Controller", "Middle Overs"}

    # Step 1: Select up to 5 specialist batters
    batters = player_profiles[player_profiles["primary_role"].isin(batting_roles)]
    _select_players(batters, min(5, len(batters)))

    # Step 2: Select up to 2 all-rounders
    allrounders = player_profiles[player_profiles["primary_role"] == "Allrounder"]
    _select_players(allrounders, min(2, len(allrounders)))

    # Step 3: Fill the remaining spots (up to 11) with specialist bowlers
    bowlers = player_profiles[player_profiles["primary_role"].isin(bowling_roles)]
    _select_players(bowlers, 11 - len(selected_players))

    # Step 4: If the team is still not full, top up with the best remaining players
    if len(selected_players) < 11:
        remaining = player_profiles[~player_profiles["player"].isin(selected_players)]
        _select_players(remaining, 11 - len(selected_players))

    # Return the final selected team, sorted by impact rating
    return (
        player_profiles[player_profiles["player"].isin(selected_players)]
        .sort_values("impact_rating", ascending=False)
        .reset_index(drop=True)
    )

