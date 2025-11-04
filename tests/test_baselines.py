"""Tests for baseline models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.baselines import (
    RecentFormBaseline,
    SimpleLogisticBaseline,
    VenueHomeBaseline,
    evaluate_baseline,
    paired_bootstrap_test,
)


def test_venue_home_baseline():
    """Test venue/home baseline model."""
    # Create sample data
    X_train = pd.DataFrame({
        "venue": ["Stadium A", "Stadium A", "Stadium B", "Stadium B"] * 10,
    })
    y_train = np.array([1, 1, 0, 1] * 10)

    baseline = VenueHomeBaseline()
    baseline.fit(X_train, y_train)

    # Test prediction
    X_test = pd.DataFrame({"venue": ["Stadium A", "Stadium B"]})
    probs = baseline.predict_proba(X_test)

    assert len(probs) == 2
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_recent_form_baseline():
    """Test recent form baseline model."""
    X_train = pd.DataFrame({
        "team": ["Team A"] * 10 + ["Team B"] * 10,
        "match_id": range(20),
    })
    y_train = np.array([1, 1, 0, 1, 0] * 4)

    baseline = RecentFormBaseline(window=3)
    baseline.fit(X_train, y_train)

    # Test prediction
    X_test = pd.DataFrame({"team": ["Team A", "Team B"]})
    probs = baseline.predict_proba(X_test)

    assert len(probs) == 2
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_simple_logistic_baseline():
    """Test simple logistic baseline."""
    X_train = pd.DataFrame({
        "batting_strike_rate": np.random.uniform(80, 150, 100),
        "batting_average": np.random.uniform(20, 50, 100),
        "bowling_economy": np.random.uniform(6, 10, 100),
    })
    y_train = np.random.randint(0, 2, 100)

    baseline = SimpleLogisticBaseline()
    baseline.fit(X_train, y_train)

    # Test prediction
    X_test = pd.DataFrame({
        "batting_strike_rate": [120, 130],
        "batting_average": [30, 35],
        "bowling_economy": [7, 8],
    })
    probs = baseline.predict_proba(X_test)

    assert len(probs) == 2
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_evaluate_baseline():
    """Test baseline evaluation function."""
    X_train = pd.DataFrame({"venue": ["A", "B"] * 50})
    y_train = np.array([1, 0] * 50)

    baseline = VenueHomeBaseline()
    baseline.fit(X_train, y_train)

    X_test = pd.DataFrame({"venue": ["A", "B"] * 10})
    y_test = np.array([1, 0] * 10)

    metrics = evaluate_baseline(baseline, X_test, y_test)

    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0
    assert metrics["accuracy"] <= 1


def test_paired_bootstrap_test():
    """Test paired bootstrap significance test."""
    np.random.seed(42)

    y_true = np.random.randint(0, 2, 100)
    y_pred_model = np.random.uniform(0, 1, 100)
    y_pred_baseline = np.random.uniform(0, 1, 100)

    def roc_auc(y_t, y_p):
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_t)) < 2:
            return 0.5
        return roc_auc_score(y_t, y_p)

    delta, lower, upper = paired_bootstrap_test(
        y_true, y_pred_model, y_pred_baseline, roc_auc, n_bootstrap=100, random_state=42
    )

    # Check that CIs bracket the delta
    assert lower <= delta <= upper


def test_venue_baseline_no_venue_column():
    """Test venue baseline handles missing venue column."""
    X_train = pd.DataFrame({"team": ["A", "B"] * 10})
    y_train = np.array([1, 0] * 10)

    baseline = VenueHomeBaseline()
    baseline.fit(X_train, y_train)

    X_test = pd.DataFrame({"team": ["A", "B"]})
    probs = baseline.predict_proba(X_test)

    # Should return global prior for all
    assert len(probs) == 2
    assert probs[0] == probs[1]
