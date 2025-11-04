"""Tests for probability calibration utilities."""
from __future__ import annotations

import numpy as np
import pytest

from src.common.calibration import (
    apply_calibration,
    calibrate_probabilities,
    compute_calibration_curve,
    compute_expected_calibration_error,
)


def test_calibrate_probabilities_platt():
    """Test Platt scaling calibration."""
    np.random.seed(42)

    y_true = np.array([0, 1, 0, 1, 0, 1] * 20)
    y_prob = np.array([0.3, 0.7, 0.2, 0.8, 0.1, 0.9] * 20)

    calibrator, calibrated = calibrate_probabilities(y_true, y_prob, method="platt")

    # Check that calibrator is fitted
    assert calibrator is not None

    # Check that calibrated probabilities are in [0, 1]
    assert np.all(calibrated >= 0)
    assert np.all(calibrated <= 1)


def test_calibrate_probabilities_isotonic():
    """Test isotonic regression calibration."""
    np.random.seed(42)

    y_true = np.array([0, 1, 0, 1, 0, 1] * 20)
    y_prob = np.array([0.3, 0.7, 0.2, 0.8, 0.1, 0.9] * 20)

    calibrator, calibrated = calibrate_probabilities(y_true, y_prob, method="isotonic")

    # Check that calibrator is fitted
    assert calibrator is not None

    # Check that calibrated probabilities are in [0, 1]
    assert np.all(calibrated >= 0)
    assert np.all(calibrated <= 1)


def test_apply_calibration():
    """Test applying calibration to new data."""
    np.random.seed(42)

    y_true = np.array([0, 1, 0, 1, 0, 1] * 20)
    y_prob = np.array([0.3, 0.7, 0.2, 0.8, 0.1, 0.9] * 20)

    calibrator, _ = calibrate_probabilities(y_true, y_prob, method="platt")

    # Apply to new data
    y_prob_new = np.array([0.4, 0.6, 0.5])
    calibrated_new = apply_calibration(calibrator, y_prob_new, method="platt")

    assert len(calibrated_new) == len(y_prob_new)
    assert np.all(calibrated_new >= 0)
    assert np.all(calibrated_new <= 1)


def test_compute_calibration_curve():
    """Test calibration curve computation."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1] * 10)
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9] * 10)

    frac_pos, mean_pred = compute_calibration_curve(y_true, y_prob, n_bins=5)

    # Check that arrays have expected length
    assert len(frac_pos) == len(mean_pred)

    # Check that values are in valid ranges
    assert np.all(frac_pos >= 0)
    assert np.all(frac_pos <= 1)
    assert np.all(mean_pred >= 0)
    assert np.all(mean_pred <= 1)


def test_compute_expected_calibration_error():
    """Test ECE computation."""
    # Perfectly calibrated
    y_true = np.array([0, 0, 1, 1] * 25)
    y_prob = np.array([0.0, 0.0, 1.0, 1.0] * 25)

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)

    # Should have low ECE for perfectly calibrated predictions
    assert ece >= 0
    assert ece < 0.2  # Allow some tolerance


def test_calibration_invalid_method():
    """Test that invalid calibration method raises error."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.3, 0.7, 0.2, 0.8])

    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrate_probabilities(y_true, y_prob, method="invalid")
