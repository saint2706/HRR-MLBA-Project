import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from feature_engineering import (
    aggregate_player_ratings,
    compute_composite_indices,
)


def test_negative_correlation_metric_retains_weight():
    batting_data = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "season": [2024] * 4,
            "team": ["A"] * 4,
            "player": ["Player"] * 4,
            "team_won": [1, 0, 1, 0],
            "winning_team": ["A", "B", "A", "B"],
            "batting_strike_rate": [0.0] * 4,
            "batting_average": [0.0] * 4,
            "boundary_percentage": [0.0] * 4,
            "dot_percentage": [10.0, 30.0, 15.0, 35.0],
        }
    )

    bowling_data = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "season": [2024] * 4,
            "team": ["A"] * 4,
            "player": ["Player"] * 4,
            "team_won": [1, 0, 1, 0],
            "winning_team": ["A", "B", "A", "B"],
            "bowling_economy": [0.0] * 4,
            "bowling_strike_rate": [0.0] * 4,
            "wickets_per_match": [0.0] * 4,
            "phase_efficacy": [0.0] * 4,
        }
    )

    features, _ = compute_composite_indices(
        batting_data,
        bowling_data,
        target_col="team_won",
        weight_method="correlation",
    )

    # When the dot_percentage metric is negatively correlated with the target,
    # the absolute-value weighting should retain its influence, resulting in a
    # non-zero batting index that mirrors the metric's contribution.
    pd.testing.assert_series_equal(
        features["batting_index"],
        features["dot_percentage"],
        check_names=False,
        check_dtype=False,
    )

    assert (features["batting_index"] != 0).any()


def test_aggregate_player_ratings_handles_zero_weights():
    model_df = pd.DataFrame(
        {
            "match_id": [1, 2],
            "player": ["Player"] * 2,
            "team": ["A"] * 2,
            "season": [2024] * 2,
            "team_won": [1, 0],
            "feature_one": [1.0, 2.0],
            "feature_two": [3.0, 4.0],
            "batting_index": [1.0, 1.0],
            "bowling_index": [0.5, 0.5],
        }
    )

    shap_weights = {"feature_one": 0.0, "feature_two": 0.0, "unused": 0.0}

    player_ratings = aggregate_player_ratings(model_df, shap_weights)

    # Impact scores should be the average of the two features with uniform weights.
    expected_score = model_df[["feature_one", "feature_two"]].mean(axis=1).mean()
    assert np.isclose(player_ratings["impact_score"].iat[0], expected_score)

    # Ratings should remain finite even when supplied with zero weights.
    assert np.isfinite(player_ratings["impact_rating"]).all()
