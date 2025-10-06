"""Role-aware clustering for batters and bowlers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    labels: pd.Series
    centroids: pd.DataFrame
    feature_columns: List[str]
    model: KMeans


def _fit_kmeans(features: pd.DataFrame, n_clusters: int, random_state: int = 42) -> ClusteringResult:
    scaler = StandardScaler()
    clean_features = features.fillna(0)
    effective_clusters = max(1, min(n_clusters, len(clean_features)))
    scaled = scaler.fit_transform(clean_features)
    model = KMeans(n_clusters=effective_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(scaled)
    centroids = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=features.columns,
    )
    return ClusteringResult(labels=pd.Series(labels, index=features.index), centroids=centroids, feature_columns=list(features.columns), model=model)


def cluster_batters(batting_df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, ClusteringResult]:
    """Cluster batters into archetypes."""

    feature_cols = ["batting_strike_rate", "batting_average", "boundary_percentage", "dot_percentage", "batting_index"]
    available = batting_df.reindex(columns=feature_cols).fillna(0)
    result = _fit_kmeans(available, n_clusters=n_clusters)

    clustered = batting_df.copy()
    clustered["batter_cluster"] = result.labels

    return clustered, result


def cluster_bowlers(bowling_df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, ClusteringResult]:
    """Cluster bowlers into archetypes."""

    feature_cols = ["bowling_economy", "bowling_strike_rate", "wickets_per_match", "phase_efficacy", "bowling_index"]
    available = bowling_df.reindex(columns=feature_cols).fillna(0)
    result = _fit_kmeans(available, n_clusters=n_clusters)

    clustered = bowling_df.copy()
    clustered["bowler_cluster"] = result.labels

    return clustered, result


def label_batter_clusters(result: ClusteringResult) -> Dict[int, str]:
    """Assign descriptive names to batter clusters based on centroids."""

    centroids = result.centroids.copy()
    labels = {}
    for cluster_id, centroid in centroids.iterrows():
        sr = centroid["batting_strike_rate"]
        boundary = centroid["boundary_percentage"]
        dot = centroid["dot_percentage"]
        average = centroid["batting_average"]
        if sr >= np.percentile(centroids["batting_strike_rate"], 75):
            labels[cluster_id] = "Power Hitter"
        elif dot >= np.percentile(centroids["dot_percentage"], 75):
            labels[cluster_id] = "Anchor"
        elif average >= np.percentile(centroids["batting_average"], 75):
            labels[cluster_id] = "Accumulator"
        else:
            labels[cluster_id] = "Finisher"
    return labels


def label_bowler_clusters(result: ClusteringResult) -> Dict[int, str]:
    """Assign descriptive names to bowler clusters based on centroids."""

    centroids = result.centroids.copy()
    labels = {}
    for cluster_id, centroid in centroids.iterrows():
        economy = centroid["bowling_economy"]
        strike = centroid["bowling_strike_rate"]
        wickets = centroid["wickets_per_match"]
        phase = centroid["phase_efficacy"]
        if phase >= np.percentile(centroids["phase_efficacy"], 75):
            labels[cluster_id] = "Death Specialist"
        elif wickets >= np.percentile(centroids["wickets_per_match"], 75):
            labels[cluster_id] = "Strike Bowler"
        elif economy <= np.percentile(centroids["bowling_economy"], 25):
            labels[cluster_id] = "Powerplay Controller"
        else:
            labels[cluster_id] = "Middle Overs"
    return labels
