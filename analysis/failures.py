"""Analysis of model failure cases (false positives and false negatives).

This module identifies and analyzes the most egregious model errors to help
understand where and why the model fails.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def identify_top_failures(
    data: pd.DataFrame,
    y_true_col: str,
    y_pred_proba_col: str,
    n_top: int = 5,
    include_features: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify top false positives and false negatives.

    Args:
        data: DataFrame with predictions and features.
        y_true_col: Name of true label column.
        y_pred_proba_col: Name of predicted probability column.
        n_top: Number of top failures to return for each type.
        include_features: List of feature columns to include in the output.

    Returns:
        Tuple of (false_positives_df, false_negatives_df).
    """
    data = data.copy()

    # Compute prediction errors
    data["error"] = abs(data[y_true_col] - data[y_pred_proba_col])

    # Identify false positives (predicted 1, actual 0)
    fp_mask = (data[y_true_col] == 0) & (data[y_pred_proba_col] > 0.5)
    false_positives = data[fp_mask].nlargest(n_top, y_pred_proba_col)

    # Identify false negatives (predicted 0, actual 1)
    fn_mask = (data[y_true_col] == 1) & (data[y_pred_proba_col] < 0.5)
    false_negatives = data[fn_mask].nsmallest(n_top, y_pred_proba_col)

    # Select columns to include
    base_cols = ["match_id", "player", "team", "season", y_true_col, y_pred_proba_col, "error"]
    if include_features:
        cols_to_include = [col for col in base_cols + include_features if col in data.columns]
    else:
        cols_to_include = [col for col in base_cols if col in data.columns]

    fp_output = false_positives[cols_to_include] if not false_positives.empty else pd.DataFrame()
    fn_output = false_negatives[cols_to_include] if not false_negatives.empty else pd.DataFrame()

    return fp_output, fn_output


def save_failure_cases(
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
    output_csv: str = "reports/failure_cases.csv",
    output_md: str = "reports/FAILURES.md",
) -> None:
    """Save failure cases to CSV and generate a markdown report.

    Args:
        false_positives: DataFrame of false positive cases.
        false_negatives: DataFrame of false negative cases.
        output_csv: Path to save the combined CSV file.
        output_md: Path to save the markdown report.
    """
    # Save combined CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    fp_with_type = false_positives.copy()
    fp_with_type["failure_type"] = "false_positive"

    fn_with_type = false_negatives.copy()
    fn_with_type["failure_type"] = "false_negative"

    combined = pd.concat([fp_with_type, fn_with_type], ignore_index=True)

    if not combined.empty:
        combined.to_csv(output_csv, index=False)

    # Generate markdown report
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Model Failure Analysis\n\n")
        f.write("This report highlights the most significant prediction errors to help identify "
                "patterns in model failures.\n\n")

        f.write("## False Positives (High Confidence, Wrong Prediction)\n\n")
        f.write("These are cases where the model predicted a win with high confidence, but the team lost.\n\n")

        if not false_positives.empty:
            _write_failure_table(f, false_positives)
        else:
            f.write("*No significant false positives found.*\n\n")

        f.write("## False Negatives (Low Confidence, Missed Win)\n\n")
        f.write("These are cases where the model predicted a loss with high confidence, but the team won.\n\n")

        if not false_negatives.empty:
            _write_failure_table(f, false_negatives)
        else:
            f.write("*No significant false negatives found.*\n\n")

        f.write("## Key Insights\n\n")
        f.write(_generate_failure_insights(false_positives, false_negatives))

        f.write("\n## Recommendations\n\n")
        f.write("1. **Feature Engineering**: Investigate if additional features could capture the patterns "
                "in these failure cases.\n")
        f.write("2. **Data Quality**: Check if any of these failures are due to data quality issues or "
                "unusual match circumstances.\n")
        f.write("3. **Model Capacity**: Consider if the model needs more capacity to capture complex "
                "interaction patterns.\n")
        f.write("4. **Outlier Detection**: Develop an uncertainty estimation method to flag low-confidence "
                "predictions.\n")

        f.write("\n**Last Updated**: Auto-generated from model predictions\n")


def _write_failure_table(f, failures: pd.DataFrame) -> None:
    """Write a markdown table of failure cases.

    Args:
        f: File handle to write to.
        failures: DataFrame of failure cases.
    """
    # Select key columns for display
    display_cols = []
    for col in ["player", "team", "season", "y_true", "y_pred_proba", "error"]:
        if col in failures.columns:
            display_cols.append(col)

    # Add a few feature columns if available
    feature_candidates = [
        "batting_strike_rate",
        "batting_average",
        "bowling_economy",
        "batting_index",
        "bowling_index",
    ]
    for col in feature_candidates:
        if col in failures.columns and len(display_cols) < 10:
            display_cols.append(col)

    if not display_cols:
        f.write("*No columns available to display.*\n\n")
        return

    # Write header
    f.write("| " + " | ".join(display_cols) + " |\n")
    f.write("|" + "---|" * len(display_cols) + "\n")

    # Write rows
    for _, row in failures.iterrows():
        values = []
        for col in display_cols:
            val = row[col]
            if isinstance(val, (int, np.integer)):
                values.append(str(val))
            elif isinstance(val, (float, np.floating)):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))

        f.write("| " + " | ".join(values) + " |\n")

    f.write("\n")


def _generate_failure_insights(fp: pd.DataFrame, fn: pd.DataFrame) -> str:
    """Generate insights from failure analysis.

    Args:
        fp: False positives DataFrame.
        fn: False negatives DataFrame.

    Returns:
        String with markdown-formatted insights.
    """
    insights = []

    if not fp.empty:
        insights.append(f"- Found {len(fp)} significant false positive cases.")

        # Check for common patterns in FPs
        if "team" in fp.columns and len(fp) > 1:
            team_counts = fp["team"].value_counts()
            if len(team_counts) > 0:
                top_team = team_counts.index[0]
                count = team_counts.iloc[0]
                if count > 1:
                    insights.append(
                        f"  - Team **{top_team}** appears {count} times in FPs, suggesting potential overestimation."
                    )

    if not fn.empty:
        insights.append(f"- Found {len(fn)} significant false negative cases.")

        # Check for common patterns in FNs
        if "team" in fn.columns and len(fn) > 1:
            team_counts = fn["team"].value_counts()
            if len(team_counts) > 0:
                top_team = team_counts.index[0]
                count = team_counts.iloc[0]
                if count > 1:
                    insights.append(
                        f"  - Team **{top_team}** appears {count} times in FNs, suggesting potential underestimation."
                    )

    if not insights:
        insights.append("- Analysis complete. Review individual cases for patterns.")

    return "\n".join(insights) + "\n"


def analyze_failures(
    data: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_proba_col: str = "y_pred_proba",
    n_top: int = 8,
    feature_cols: Optional[List[str]] = None,
    output_csv: str = "reports/failure_cases.csv",
    output_md: str = "reports/FAILURES.md",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Complete failure analysis pipeline.

    Args:
        data: DataFrame with predictions and features.
        y_true_col: Name of true label column.
        y_pred_proba_col: Name of predicted probability column.
        n_top: Number of top failures per type.
        feature_cols: List of feature columns to include.
        output_csv: Path to save CSV output.
        output_md: Path to save markdown report.

    Returns:
        Tuple of (false_positives, false_negatives) DataFrames.
    """
    fp, fn = identify_top_failures(data, y_true_col, y_pred_proba_col, n_top, feature_cols)

    save_failure_cases(fp, fn, output_csv, output_md)

    return fp, fn
