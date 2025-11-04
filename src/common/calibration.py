"""Probability calibration utilities for model predictions.

This module provides functions to calibrate probability predictions from classification
models using methods like Platt scaling (logistic calibration) and isotonic regression.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def calibrate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: Literal["platt", "isotonic"] = "platt",
) -> Tuple[callable, np.ndarray]:
    """Calibrate probability predictions using validation data.

    Args:
        y_true: True binary labels from validation set.
        y_prob: Predicted probabilities from validation set.
        method: Calibration method ('platt' for logistic or 'isotonic').

    Returns:
        Tuple of (calibrator function, calibrated probabilities on validation data).
    """
    y_prob = y_prob.reshape(-1, 1) if y_prob.ndim == 1 else y_prob

    if method == "platt":
        # Platt scaling: fit logistic regression on predictions
        calibrator = LogisticRegression(random_state=42, max_iter=1000)
        calibrator.fit(y_prob, y_true)
        calibrated = calibrator.predict_proba(y_prob)[:, 1]
    elif method == "isotonic":
        # Isotonic regression: monotonic calibration
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrated = calibrator.fit_transform(y_prob.ravel(), y_true)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return calibrator, calibrated


def apply_calibration(
    calibrator: callable,
    y_prob: np.ndarray,
    method: Literal["platt", "isotonic"] = "platt",
) -> np.ndarray:
    """Apply a fitted calibrator to new probability predictions.

    Args:
        calibrator: Fitted calibration model.
        y_prob: Raw probability predictions to calibrate.
        method: Calibration method used ('platt' or 'isotonic').

    Returns:
        Calibrated probability predictions.
    """
    if method == "platt":
        y_prob = y_prob.reshape(-1, 1) if y_prob.ndim == 1 else y_prob
        return calibrator.predict_proba(y_prob)[:, 1]
    elif method == "isotonic":
        return calibrator.transform(y_prob.ravel())
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve (reliability diagram data).

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for the calibration curve.
        strategy: Strategy for binning ('uniform' or 'quantile').

    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value) for each bin.
    """
    return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and actual outcomes,
    weighted by the number of samples in each bin.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration.

    Returns:
        Expected Calibration Error value.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Compute bin counts
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    # Calculate ECE as weighted average of absolute calibration error per bin
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        if bin_counts[i] > 0 and i < len(fraction_of_positives):
            weight = bin_counts[i] / total_samples
            calibration_error = abs(fraction_of_positives[i] - mean_predicted_value[i])
            ece += weight * calibration_error

    return float(ece)
