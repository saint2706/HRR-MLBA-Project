"""Advanced metrics computation with confidence intervals.

This module provides functions to compute various classification metrics
with bootstrap confidence intervals.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Compute a metric with bootstrap confidence intervals.

    Args:
        y_true: True labels.
        y_pred: Predicted labels or probabilities.
        metric_fn: Function that computes the metric (takes y_true, y_pred).
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level for the interval (e.g., 0.95 for 95% CI).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (metric_value, lower_ci, upper_ci).
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    # Compute the actual metric
    metric_value = metric_fn(y_true, y_pred)

    # Bootstrap resampling
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except (ValueError, ZeroDivisionError):
            # Skip invalid bootstrap samples
            continue

    # Compute confidence intervals
    if bootstrap_scores:
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_ci = np.percentile(bootstrap_scores, lower_percentile)
        upper_ci = np.percentile(bootstrap_scores, upper_percentile)
    else:
        # Fallback if no valid bootstrap samples
        lower_ci = metric_value
        upper_ci = metric_value

    return float(metric_value), float(lower_ci), float(upper_ci)


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive classification metrics with confidence intervals.

    Args:
        y_true: True binary labels.
        y_pred_binary: Binary predictions (0 or 1).
        y_pred_proba: Predicted probabilities for the positive class.
        n_bootstrap: Number of bootstrap samples for CI.
        confidence_level: Confidence level for intervals (e.g., 0.95).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping metric names to dicts with 'value', 'ci_lower', 'ci_upper'.
    """
    metrics = {}

    # Accuracy
    acc, acc_lower, acc_upper = bootstrap_metric(
        y_true, y_pred_binary, accuracy_score, n_bootstrap, confidence_level, random_state
    )
    metrics["accuracy"] = {"value": acc, "ci_lower": acc_lower, "ci_upper": acc_upper}

    # F1 Score
    f1, f1_lower, f1_upper = bootstrap_metric(
        y_true, y_pred_binary, f1_score, n_bootstrap, confidence_level, random_state
    )
    metrics["f1"] = {"value": f1, "ci_lower": f1_lower, "ci_upper": f1_upper}

    # ROC AUC
    if len(np.unique(y_true)) > 1:
        roc_auc, roc_lower, roc_upper = bootstrap_metric(
            y_true, y_pred_proba, roc_auc_score, n_bootstrap, confidence_level, random_state
        )
        metrics["roc_auc"] = {"value": roc_auc, "ci_lower": roc_lower, "ci_upper": roc_upper}
    else:
        metrics["roc_auc"] = {"value": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    # PR AUC (Average Precision)
    if len(np.unique(y_true)) > 1:
        pr_auc, pr_lower, pr_upper = bootstrap_metric(
            y_true, y_pred_proba, average_precision_score, n_bootstrap, confidence_level, random_state
        )
        metrics["pr_auc"] = {"value": pr_auc, "ci_lower": pr_lower, "ci_upper": pr_upper}
    else:
        metrics["pr_auc"] = {"value": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    # Log Loss
    if len(np.unique(y_true)) > 1:
        ll, ll_lower, ll_upper = bootstrap_metric(
            y_true, y_pred_proba, log_loss, n_bootstrap, confidence_level, random_state
        )
        metrics["log_loss"] = {"value": ll, "ci_lower": ll_lower, "ci_upper": ll_upper}
    else:
        metrics["log_loss"] = {"value": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    # Brier Score
    if len(np.unique(y_true)) > 1:
        brier, brier_lower, brier_upper = bootstrap_metric(
            y_true, y_pred_proba, brier_score_loss, n_bootstrap, confidence_level, random_state
        )
        metrics["brier_score"] = {"value": brier, "ci_lower": brier_lower, "ci_upper": brier_upper}
    else:
        metrics["brier_score"] = {"value": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    return metrics


def format_metric_with_ci(metric_dict: Dict[str, float], precision: int = 4) -> str:
    """Format a metric with confidence interval as a string.

    Args:
        metric_dict: Dictionary with 'value', 'ci_lower', 'ci_upper' keys.
        precision: Number of decimal places.

    Returns:
        Formatted string like "0.8523 [0.8421, 0.8625]".
    """
    value = metric_dict["value"]
    lower = metric_dict["ci_lower"]
    upper = metric_dict["ci_upper"]

    if np.isnan(value):
        return "N/A"

    return f"{value:.{precision}f} [{lower:.{precision}f}, {upper:.{precision}f}]"
