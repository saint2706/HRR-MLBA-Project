"""Tests for metrics computation with confidence intervals."""
from __future__ import annotations

import numpy as np
import pytest

from src.common.metrics import bootstrap_metric, compute_metrics_with_ci, format_metric_with_ci


def test_bootstrap_metric_basic():
    """Test bootstrap metric computation."""
    y_true = np.array([0, 1, 0, 1, 0, 1] * 20)
    y_pred = np.array([0, 1, 0, 1, 0, 1] * 20)

    def accuracy(y_t, y_p):
        return (y_t == y_p).mean()

    value, lower, upper = bootstrap_metric(y_true, y_pred, accuracy, n_bootstrap=100, random_state=42)

    # Check that value is correct
    assert value == 1.0

    # Check that CIs are reasonable
    assert lower <= value <= upper
    assert lower >= 0.0
    assert upper <= 1.0


def test_compute_metrics_with_ci():
    """Test comprehensive metrics with CIs."""
    np.random.seed(42)
    n_samples = 200

    y_true = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.random.uniform(0, 1, n_samples)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    metrics = compute_metrics_with_ci(
        y_true,
        y_pred_binary,
        y_pred_proba,
        n_bootstrap=100,
        random_state=42,
    )

    # Check that all expected metrics are present
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "log_loss" in metrics
    assert "brier_score" in metrics

    # Check structure
    for metric_name, metric_dict in metrics.items():
        assert "value" in metric_dict
        assert "ci_lower" in metric_dict
        assert "ci_upper" in metric_dict

        # Check CI ordering
        assert metric_dict["ci_lower"] <= metric_dict["value"] <= metric_dict["ci_upper"]


def test_format_metric_with_ci():
    """Test metric formatting."""
    metric_dict = {"value": 0.8523, "ci_lower": 0.8421, "ci_upper": 0.8625}

    formatted = format_metric_with_ci(metric_dict, precision=4)

    assert "0.8523" in formatted
    assert "0.8421" in formatted
    assert "0.8625" in formatted


def test_format_metric_with_ci_nan():
    """Test metric formatting with NaN values."""
    metric_dict = {"value": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    formatted = format_metric_with_ci(metric_dict)

    assert formatted == "N/A"


def test_bootstrap_metric_deterministic():
    """Test that bootstrap is deterministic with same seed."""
    y_true = np.array([0, 1, 0, 1, 0, 1] * 20)
    y_pred = np.array([0, 1, 1, 1, 0, 0] * 20)

    def accuracy(y_t, y_p):
        return (y_t == y_p).mean()

    result1 = bootstrap_metric(y_true, y_pred, accuracy, n_bootstrap=100, random_state=42)
    result2 = bootstrap_metric(y_true, y_pred, accuracy, n_bootstrap=100, random_state=42)

    assert result1 == result2
