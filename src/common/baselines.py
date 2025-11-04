"""Baseline models for comparison and benchmarking.

This module implements simple baseline models to establish performance benchmarks
for the main impact prediction model.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


class VenueHomeBaseline:
    """Baseline that predicts based on home team advantage at venues.

    This baseline learns the historical win rate for home teams at each venue
    and uses that as the predicted probability.
    """

    def __init__(self):
        """Initialize the venue/home baseline."""
        self.venue_priors: Dict[str, float] = {}
        self.global_prior: float = 0.5

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "VenueHomeBaseline":
        """Fit the baseline on training data.

        Args:
            X: DataFrame with at least 'venue' column.
            y: Binary target (1 = team won, 0 = team lost).

        Returns:
            Self for chaining.
        """
        if "venue" not in X.columns:
            # Fallback to global prior if no venue info
            self.global_prior = float(y.mean())
            return self

        # Calculate win rate per venue
        data = X.copy()
        data["target"] = y

        venue_stats = data.groupby("venue")["target"].agg(["mean", "count"])
        self.venue_priors = venue_stats["mean"].to_dict()
        self.global_prior = float(y.mean())

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for test data.

        Args:
            X: DataFrame with venue information.

        Returns:
            Array of predicted probabilities for the positive class.
        """
        if "venue" not in X.columns:
            return np.full(len(X), self.global_prior)

        probs = X["venue"].map(self.venue_priors).fillna(self.global_prior).values
        return probs


class RecentFormBaseline:
    """Baseline that predicts based on recent match performance.

    This baseline uses the team's win rate over their last N matches as the
    predicted probability.
    """

    def __init__(self, window: int = 5):
        """Initialize the recent form baseline.

        Args:
            window: Number of recent matches to consider.
        """
        self.window = window
        self.team_history: Dict[str, list] = {}
        self.global_prior: float = 0.5

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RecentFormBaseline":
        """Fit the baseline on training data.

        Args:
            X: DataFrame with 'team' and optionally 'date' or 'match_id'.
            y: Binary target.

        Returns:
            Self for chaining.
        """
        self.global_prior = float(y.mean())

        if "team" not in X.columns:
            return self

        # Build team history
        data = X.copy()
        data["target"] = y

        # Sort by date or match_id if available
        if "date" in data.columns:
            data = data.sort_values("date")
        elif "match_id" in data.columns:
            data = data.sort_values("match_id")

        for _, row in data.iterrows():
            team = row["team"]
            outcome = row["target"]

            if team not in self.team_history:
                self.team_history[team] = []

            self.team_history[team].append(outcome)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities based on recent form.

        Args:
            X: DataFrame with team information.

        Returns:
            Array of predicted probabilities.
        """
        if "team" not in X.columns:
            return np.full(len(X), self.global_prior)

        probs = []
        for _, row in X.iterrows():
            team = row["team"]
            history = self.team_history.get(team, [])

            if len(history) >= self.window:
                recent = history[-self.window :]
                prob = np.mean(recent)
            elif len(history) > 0:
                prob = np.mean(history)
            else:
                prob = self.global_prior

            probs.append(prob)

        return np.array(probs)


class SimpleLogisticBaseline:
    """Simple logistic regression baseline on a small feature set.

    This baseline uses only the most basic pre-match features for prediction.
    """

    def __init__(self, random_state: int = 42):
        """Initialize the simple logistic baseline.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.feature_cols = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SimpleLogisticBaseline":
        """Fit the baseline on training data.

        Args:
            X: DataFrame with features.
            y: Binary target.

        Returns:
            Self for chaining.
        """
        # Select simple pre-match features
        candidate_features = [
            "batting_strike_rate",
            "batting_average",
            "bowling_economy",
            "bowling_strike_rate",
            "batting_index",
            "bowling_index",
        ]

        self.feature_cols = [col for col in candidate_features if col in X.columns]

        if not self.feature_cols:
            # Fallback: use all numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            excluded = {"match_id", "season", "team_won"}
            self.feature_cols = [col for col in numeric_cols if col not in excluded]

        if not self.feature_cols:
            raise ValueError("No suitable features found for SimpleLogisticBaseline")

        X_subset = X[self.feature_cols].fillna(0)
        self.model.fit(X_subset, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using the logistic model.

        Args:
            X: DataFrame with features.

        Returns:
            Array of predicted probabilities for the positive class.
        """
        X_subset = X[self.feature_cols].fillna(0)
        return self.model.predict_proba(X_subset)[:, 1]


def evaluate_baseline(
    baseline: any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate a baseline model on test data.

    Args:
        baseline: Fitted baseline model with predict_proba method.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred_proba = baseline.predict_proba(X_test)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_binary),
    }

    # Add probabilistic metrics if we have both classes
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        metrics["log_loss"] = log_loss(y_test, y_pred_proba)

    return metrics


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Perform paired bootstrap test to compare model vs baseline.

    Args:
        y_true: True labels.
        y_pred_model: Predictions from the main model.
        y_pred_baseline: Predictions from the baseline.
        metric_fn: Metric function (higher is better).
        n_bootstrap: Number of bootstrap samples.
        random_state: Random seed.

    Returns:
        Tuple of (delta_mean, delta_ci_lower, delta_ci_upper).
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    # Compute actual delta
    model_score = metric_fn(y_true, y_pred_model)
    baseline_score = metric_fn(y_true, y_pred_baseline)
    delta = model_score - baseline_score

    # Bootstrap
    deltas = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n_samples, n_samples)

        y_true_boot = y_true[indices]
        y_pred_model_boot = y_pred_model[indices]
        y_pred_baseline_boot = y_pred_baseline[indices]

        # Skip if only one class
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            model_boot_score = metric_fn(y_true_boot, y_pred_model_boot)
            baseline_boot_score = metric_fn(y_true_boot, y_pred_baseline_boot)
            deltas.append(model_boot_score - baseline_boot_score)
        except (ValueError, ZeroDivisionError):
            continue

    if deltas:
        delta_lower = np.percentile(deltas, 2.5)
        delta_upper = np.percentile(deltas, 97.5)
    else:
        delta_lower = delta
        delta_upper = delta

    return float(delta), float(delta_lower), float(delta_upper)
