"""Visualization utilities for model evaluation and calibration.

This module provides functions to create various plots for model evaluation,
including ROC curves, precision-recall curves, and calibration plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "ROC Curve",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> str:
    """Plot ROC curve.

    Args:
        y_true: True binary labels.
        y_pred_proba: Predicted probabilities.
        title: Plot title.
        output_path: Path to save figure. If None, saves to 'reports/figures/roc.png'.
        figsize: Figure size as (width, height).

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        output_path = "reports/figures/roc.png"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    # Calculate AUC
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, linewidth=2, label=f"Model (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(output_path)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> str:
    """Plot precision-recall curve.

    Args:
        y_true: True binary labels.
        y_pred_proba: Predicted probabilities.
        title: Plot title.
        output_path: Path to save figure. If None, saves to 'reports/figures/pr.png'.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        output_path = "reports/figures/pr.png"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    # Calculate average precision
    from sklearn.metrics import average_precision_score

    avg_precision = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=figsize)
    plt.plot(recall, precision, linewidth=2, label=f"Model (AP = {avg_precision:.3f})")
    plt.axhline(y=y_true.mean(), color="k", linestyle="--", linewidth=1, label="Baseline")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(output_path)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Plot (Reliability Diagram)",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> str:
    """Plot calibration curve (reliability diagram).

    Args:
        y_true: True binary labels.
        y_pred_proba: Predicted probabilities.
        n_bins: Number of bins for calibration curve.
        title: Plot title.
        output_path: Path to save figure. If None, saves to 'reports/figures/calibration.png'.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        output_path = "reports/figures/calibration.png"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    # Calculate ECE
    from src.common.calibration import compute_expected_calibration_error

    ece = compute_expected_calibration_error(y_true, y_pred_proba, n_bins=n_bins)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2, label=f"Model (ECE={ece:.4f})")
    ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax1.set_ylabel("Fraction of Positives", fontsize=11)
    ax1.set_title("Reliability Diagram", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # Plot 2: Histogram of predictions
    ax2.hist(y_pred_proba, bins=n_bins, alpha=0.7, edgecolor="black", density=True)
    ax2.set_xlabel("Predicted Probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Distribution of Predictions", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(output_path)


def plot_drift_over_time(
    season_metrics: "pd.DataFrame",
    metric_col: str = "log_loss",
    title: str = "Model Performance Drift Over Time",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> str:
    """Plot model performance metrics over seasons to detect drift.

    Args:
        season_metrics: DataFrame with columns ['season', metric_col].
        metric_col: Name of metric column to plot.
        title: Plot title.
        output_path: Path to save figure. If None, saves to 'reports/figures/drift.png'.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        output_path = "reports/figures/drift.png"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Sort by season
    data = season_metrics.sort_values("season").copy()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot metric over time
    ax.plot(data["season"], data[metric_col], "o-", linewidth=2, markersize=8, label=metric_col.replace("_", " ").title())

    # Add horizontal line for mean
    mean_value = data[metric_col].mean()
    ax.axhline(y=mean_value, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Mean: {mean_value:.4f}")

    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    # Add trend annotation if data has multiple points
    if len(data) > 1:
        first_val = data[metric_col].iloc[0]
        last_val = data[metric_col].iloc[-1]
        change = last_val - first_val
        change_pct = (change / first_val) * 100 if first_val != 0 else 0

        trend_text = f"Change: {change:+.4f} ({change_pct:+.1f}%)"
        ax.text(
            0.02,
            0.98,
            trend_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(output_path)


def create_evaluation_plots(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: str = "reports/figures",
) -> dict:
    """Create all standard evaluation plots.

    Args:
        y_true: True binary labels.
        y_pred_proba: Predicted probabilities.
        output_dir: Directory to save plots.

    Returns:
        Dictionary mapping plot names to file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plots = {}

    # ROC curve
    plots["roc"] = plot_roc_curve(y_true, y_pred_proba, output_path=f"{output_dir}/roc.png")

    # PR curve
    plots["pr"] = plot_precision_recall_curve(y_true, y_pred_proba, output_path=f"{output_dir}/pr.png")

    # Calibration curve
    plots["calibration"] = plot_calibration_curve(y_true, y_pred_proba, output_path=f"{output_dir}/calibration.png")

    return plots
