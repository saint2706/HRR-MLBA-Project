"""
Role-aware clustering for batters and bowlers.

This module uses unsupervised machine learning (K-Means clustering) to group
players into distinct archetypes based on their performance metrics. By analyzing
the statistical profiles of players, it identifies natural groupings that correspond
to different roles on the field (e.g., "Power Hitter," "Anchor," "Death Specialist").

The key steps are:
1.  **Standardize Features**: Player metrics are scaled to ensure that no single
    feature dominates the clustering process.
2.  **Fit K-Means**: The K-Means algorithm is applied to the scaled features to
    partition players into a predefined number of clusters.
3.  **Label Clusters**: Each cluster is assigned a human-readable, descriptive
    label by analyzing the characteristics of its centroid. For example, a
    cluster with a high average strike rate and boundary percentage would be
    labeled a "Power Hitter."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    """
    A container for the results of a K-Means clustering operation.

    This dataclass holds all the important outputs from the clustering process,
    making it easy to pass them between functions for further analysis and labeling.

    Attributes:
        labels: A Series containing the cluster label for each data point (player).
        centroids: A DataFrame representing the center of each cluster. The values
                   are in the original, unscaled feature space.
        feature_columns: A list of the feature names used for clustering.
        model: The fitted KMeans model instance from scikit-learn.
    """
    labels: pd.Series
    centroids: pd.DataFrame
    feature_columns: List[str]
    model: Optional[KMeans]


def _fit_kmeans(features: pd.DataFrame, n_clusters: int, random_state: int = 42) -> ClusteringResult:
    """
    Fit a K-Means clustering model to the provided features.

    This private helper function performs the core clustering logic. It first
    standardizes the features to have zero mean and unit variance, then fits the
    K-Means algorithm, and finally returns the results in a structured format.

    Args:
        features: A DataFrame where each row is a player and each column is a metric.
        n_clusters: The number of clusters to form.
        random_state: A seed for the random number generator for reproducibility.

    Returns:
        A `ClusteringResult` object containing the cluster labels, centroids,
        feature columns, and the fitted model.
    """
    if features.empty:
        empty_labels = pd.Series(dtype="int64", index=features.index)
        empty_centroids = pd.DataFrame(columns=features.columns, dtype=float)
        return ClusteringResult(
            labels=empty_labels,
            centroids=empty_centroids,
            feature_columns=list(features.columns),
            model=None,
        )

    # Standardize features to give them equal importance in the clustering algorithm
    scaler = StandardScaler()
    clean_features = features.fillna(0)
    # Ensure n_clusters is not greater than the number of samples
    effective_clusters = max(1, min(n_clusters, len(clean_features)))
    scaled = scaler.fit_transform(clean_features)

    # Fit the K-Means model
    model = KMeans(n_clusters=effective_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(scaled)

    # Inverse transform the centroids to get their values in the original feature space
    centroids = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=features.columns,
    )

    # Package the results into the dataclass
    return ClusteringResult(
        labels=pd.Series(labels, index=features.index),
        centroids=centroids,
        feature_columns=list(features.columns),
        model=model,
    )


def cluster_batters(batting_df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, ClusteringResult]:
    """
    Cluster batters into distinct archetypes based on their performance metrics.

    This function selects a predefined set of batting metrics, applies K-Means
    clustering, and assigns a cluster label to each batter.

    Args:
        batting_df: A DataFrame containing batting statistics for each player.
        n_clusters: The desired number of batter archetypes to identify.

    Returns:
        A tuple containing:
        - clustered: The original DataFrame with an added 'batter_cluster' column.
        - result: A `ClusteringResult` object with the detailed outputs of the
                  clustering process.
    """
    # Define the features that characterize different batting styles
    feature_cols = ["batting_strike_rate", "batting_average", "boundary_percentage", "dot_percentage", "batting_index"]
    available = batting_df.reindex(columns=feature_cols).fillna(0)
    result = _fit_kmeans(available, n_clusters=n_clusters)
    if not result.labels.empty:
        result.labels = result.labels.astype("int64")

    # Add the cluster labels back to the original DataFrame
    clustered = batting_df.copy()
    clustered["batter_cluster"] = result.labels

    return clustered, result


def cluster_bowlers(bowling_df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, ClusteringResult]:
    """
    Cluster bowlers into distinct archetypes based on their performance metrics.

    This function selects a predefined set of bowling metrics, applies K-Means
    clustering, and assigns a cluster label to each bowler.

    Args:
        bowling_df: A DataFrame containing bowling statistics for each player.
        n_clusters: The desired number of bowler archetypes to identify.

    Returns:
        A tuple containing:
        - clustered: The original DataFrame with an added 'bowler_cluster' column.
        - result: A `ClusteringResult` object with the detailed outputs of the
                  clustering process.
    """
    # Define the features that characterize different bowling styles
    feature_cols = ["bowling_economy", "bowling_strike_rate", "wickets_per_match", "phase_efficacy", "bowling_index"]
    available = bowling_df.reindex(columns=feature_cols).fillna(0)
    result = _fit_kmeans(available, n_clusters=n_clusters)
    if not result.labels.empty:
        result.labels = result.labels.astype("int64")

    # Add the cluster labels back to the original DataFrame
    clustered = bowling_df.copy()
    clustered["bowler_cluster"] = result.labels

    return clustered, result


def label_batter_clusters(result: ClusteringResult) -> Dict[int, str]:
    """
    Assign descriptive, human-readable names to batter clusters by analyzing their centroids.

    This function uses a rule-based approach to label each cluster based on the
    characteristics of its center. For example, a cluster with a high strike rate
    is labeled "Power Hitter," while one with a high dot ball percentage is
    labeled "Anchor."

    Args:
        result: The `ClusteringResult` object for batters.

    Returns:
        A dictionary mapping each cluster ID to its descriptive label (e.g., {0: "Power Hitter"}).
    """
    centroids = result.centroids.copy()
    if centroids.empty:
        return {}

    labels = {}
    # Iterate through each cluster's centroid to determine its defining characteristic
    for cluster_id, centroid in centroids.iterrows():
        # Heuristic rules based on percentile rankings of centroid values
        if centroid["batting_strike_rate"] >= np.percentile(centroids["batting_strike_rate"], 75):
            labels[cluster_id] = "Power Hitter"
        elif centroid["dot_percentage"] >= np.percentile(centroids["dot_percentage"], 75):
            labels[cluster_id] = "Anchor"
        elif centroid["batting_average"] >= np.percentile(centroids["batting_average"], 75):
            labels[cluster_id] = "Accumulator"
        else:
            labels[cluster_id] = "Finisher"  # Default category
    return labels


def label_bowler_clusters(result: ClusteringResult) -> Dict[int, str]:
    """
    Assign descriptive, human-readable names to bowler clusters by analyzing their centroids.

    This function uses a rule-based approach to label each cluster based on the
    characteristics of its center. For example, a cluster with high phase efficacy
    is labeled "Death Specialist," while one with a low economy rate is labeled
    "Powerplay Controller."

    Args:
        result: The `ClusteringResult` object for bowlers.

    Returns:
        A dictionary mapping each cluster ID to its descriptive label (e.g., {0: "Strike Bowler"}).
    """
    centroids = result.centroids.copy()
    if centroids.empty:
        return {}

    labels = {}
    # Iterate through each cluster's centroid to determine its defining characteristic
    for cluster_id, centroid in centroids.iterrows():
        # Heuristic rules based on percentile rankings of centroid values
        if centroid["phase_efficacy"] >= np.percentile(centroids["phase_efficacy"], 75):
            labels[cluster_id] = "Death Specialist"
        elif centroid["wickets_per_match"] >= np.percentile(centroids["wickets_per_match"], 75):
            labels[cluster_id] = "Strike Bowler"
        elif centroid["bowling_economy"] <= np.percentile(centroids["bowling_economy"], 25):
            labels[cluster_id] = "Powerplay Controller"
        else:
            labels[cluster_id] = "Middle Overs"  # Default category
    return labels
