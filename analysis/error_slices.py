"""Error analysis by slicing data along different dimensions.

This module provides functions to analyze model performance across different
data slices (e.g., by season, venue, matchup) to identify where the model
performs well or poorly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def compute_slice_metrics(
    data: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    y_pred_proba_col: str,
    slice_col: str,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Compute metrics for each slice of data.

    Args:
        data: DataFrame containing predictions and slice information.
        y_true_col: Name of column with true labels.
        y_pred_col: Name of column with binary predictions.
        y_pred_proba_col: Name of column with predicted probabilities.
        slice_col: Name of column to slice by.
        min_samples: Minimum samples required per slice.

    Returns:
        DataFrame with metrics per slice.
    """
    results = []

    for slice_value in data[slice_col].unique():
        slice_data = data[data[slice_col] == slice_value]

        if len(slice_data) < min_samples:
            continue

        y_true = slice_data[y_true_col].values
        y_pred = slice_data[y_pred_col].values
        y_pred_proba = slice_data[y_pred_proba_col].values

        metrics = {
            slice_col: slice_value,
            "n_samples": len(slice_data),
            "accuracy": accuracy_score(y_true, y_pred),
        }

        # Add probabilistic metrics if we have both classes
        if len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
                metrics["log_loss"] = log_loss(y_true, y_pred_proba)
            except ValueError:
                metrics["roc_auc"] = float("nan")
                metrics["log_loss"] = float("nan")
        else:
            metrics["roc_auc"] = float("nan")
            metrics["log_loss"] = float("nan")

        # Class balance
        metrics["positive_rate"] = float(y_true.mean())

        results.append(metrics)

    return pd.DataFrame(results)


def analyze_by_season(
    data: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    y_pred_proba_col: str = "y_pred_proba",
    season_col: str = "season",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Analyze model performance by season.

    Args:
        data: DataFrame with predictions and season information.
        y_true_col: Name of true label column.
        y_pred_col: Name of binary prediction column.
        y_pred_proba_col: Name of probability column.
        season_col: Name of season column.
        output_path: Optional path to save results CSV.

    Returns:
        DataFrame with metrics per season.
    """
    results = compute_slice_metrics(data, y_true_col, y_pred_col, y_pred_proba_col, season_col, min_samples=5)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)

    return results


def analyze_by_venue(
    data: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    y_pred_proba_col: str = "y_pred_proba",
    venue_col: str = "venue",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Analyze model performance by venue.

    Args:
        data: DataFrame with predictions and venue information.
        y_true_col: Name of true label column.
        y_pred_col: Name of binary prediction column.
        y_pred_proba_col: Name of probability column.
        venue_col: Name of venue column.
        output_path: Optional path to save results CSV.

    Returns:
        DataFrame with metrics per venue.
    """
    results = compute_slice_metrics(data, y_true_col, y_pred_col, y_pred_proba_col, venue_col, min_samples=10)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)

    return results


def analyze_by_matchup(
    data: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    y_pred_proba_col: str = "y_pred_proba",
    team1_col: str = "team",
    team2_col: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Analyze model performance by team matchup.

    Args:
        data: DataFrame with predictions and team information.
        y_true_col: Name of true label column.
        y_pred_col: Name of binary prediction column.
        y_pred_proba_col: Name of probability column.
        team1_col: Name of first team column.
        team2_col: Optional name of opponent column.
        output_path: Optional path to save results CSV.

    Returns:
        DataFrame with metrics per team/matchup.
    """
    if team2_col and team2_col in data.columns:
        # Create matchup identifier
        data = data.copy()
        data["matchup"] = data.apply(
            lambda row: f"{row[team1_col]} vs {row[team2_col]}"
            if row[team1_col] < row[team2_col]
            else f"{row[team2_col]} vs {row[team1_col]}",
            axis=1,
        )
        results = compute_slice_metrics(data, y_true_col, y_pred_col, y_pred_proba_col, "matchup", min_samples=5)
    else:
        # Just analyze by team
        results = compute_slice_metrics(data, y_true_col, y_pred_col, y_pred_proba_col, team1_col, min_samples=10)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)

    return results


def generate_drift_report(
    season_metrics: pd.DataFrame,
    output_path: str = "reports/DRIFT.md",
) -> None:
    """Generate a narrative report on temporal drift in model performance.

    Args:
        season_metrics: DataFrame with metrics per season.
        output_path: Path to save the markdown report.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Model Performance Drift Analysis\n\n")
        f.write("This report analyzes how model performance changes over time (across seasons).\n\n")

        f.write("## Season-wise Performance Summary\n\n")
        f.write("| Season | Samples | Accuracy | ROC AUC | Log Loss | Positive Rate |\n")
        f.write("|--------|---------|----------|---------|----------|---------------|\n")

        for _, row in season_metrics.iterrows():
            season = row.get("season", "N/A")
            n_samples = row.get("n_samples", 0)
            accuracy = row.get("accuracy", float("nan"))
            roc_auc = row.get("roc_auc", float("nan"))
            log_loss_val = row.get("log_loss", float("nan"))
            pos_rate = row.get("positive_rate", float("nan"))

            f.write(
                f"| {season} | {n_samples} | {accuracy:.4f} | {roc_auc:.4f} | "
                f"{log_loss_val:.4f} | {pos_rate:.4f} |\n"
            )

        f.write("\n## Observations\n\n")

        # Check for drift
        if "log_loss" in season_metrics.columns:
            log_loss_vals = season_metrics["log_loss"].dropna()
            if len(log_loss_vals) > 1:
                trend = "increasing" if log_loss_vals.iloc[-1] > log_loss_vals.iloc[0] else "decreasing"
                f.write(
                    f"- **Calibration Drift**: Log loss shows a {trend} trend from earliest to latest season.\n"
                )

        if "accuracy" in season_metrics.columns:
            acc_vals = season_metrics["accuracy"].dropna()
            if len(acc_vals) > 1:
                acc_change = acc_vals.iloc[-1] - acc_vals.iloc[0]
                f.write(f"- **Accuracy Change**: {acc_change:+.4f} from first to last season.\n")

        f.write("\n## Recommendations\n\n")
        f.write("1. **Retraining Cadence**: Consider retraining the model annually or at the start of each season "
                "to adapt to evolving team strategies and player performance patterns.\n")
        f.write("2. **Feature Refresh**: Monitor which features are most affected by drift and update feature "
                "engineering to capture recent trends.\n")
        f.write("3. **Threshold Adjustment**: If calibration drift is significant, recalibrate probabilities "
                "on recent validation data.\n")
        f.write("\n**Last Updated**: Auto-generated from season-wise analysis\n")
