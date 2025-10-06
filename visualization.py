"""Plotting utilities for player impact project."""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap


OUTPUT_DIR = "outputs"


def _ensure_output_dir(path: str = OUTPUT_DIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_shap_summary(shap_values, features: pd.DataFrame, output_filename: str = "shap_summary.png") -> str:
    """Create and save a SHAP summary plot."""

    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, output_filename)

    plt.figure(figsize=(10, 6))
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    shap.summary_plot(values, features, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def plot_cluster_scatter(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    title: str,
    output_filename: str,
    hue_order: Optional[list] = None,
) -> str:
    """Plot clusters in two-dimensional space."""

    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, output_filename)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=cluster_col, palette="deep", hue_order=hue_order)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path
