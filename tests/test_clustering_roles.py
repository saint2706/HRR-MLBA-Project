from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clustering_roles


def test_fit_kmeans_returns_empty_result_without_fitting(monkeypatch):
    empty_features = pd.DataFrame(columns=["feature_a", "feature_b"])

    def _raise_standard_scaler():
        raise AssertionError("StandardScaler should not be instantiated for empty input")

    def _raise_kmeans(*_, **__):
        raise AssertionError("KMeans should not be instantiated for empty input")

    monkeypatch.setattr(clustering_roles, "StandardScaler", _raise_standard_scaler)
    monkeypatch.setattr(clustering_roles, "KMeans", _raise_kmeans)

    result = clustering_roles._fit_kmeans(empty_features, n_clusters=3)  # pylint: disable=protected-access

    assert result.labels.empty
    assert result.labels.dtype == "int64"
    assert result.centroids.empty
    assert result.feature_columns == ["feature_a", "feature_b"]
    assert result.model is None


def test_cluster_batters_preserves_empty_schema():
    empty_batting = pd.DataFrame(
        columns=[
            "player",
            "team",
            "balls_faced",
            "batting_strike_rate",
            "batting_average",
            "boundary_percentage",
            "dot_percentage",
            "batting_index",
        ]
    )

    clustered, result = clustering_roles.cluster_batters(empty_batting, n_clusters=4)

    assert clustered.empty
    assert "batter_cluster" in clustered.columns
    assert clustered["batter_cluster"].dtype == "int64"
    assert result.labels.empty
    assert result.labels.dtype == "int64"
