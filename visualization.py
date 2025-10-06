"""
Enhanced plotting utilities for the player impact project.

This module provides functions for creating and saving high-quality, insightful
visualizations. It is responsible for generating two main types of plots:
1.  **SHAP Bar Plots**: These plots visualize global feature importance from SHAP
    analysis, making it easy to see which factors most significantly impact match
    outcomes.
2.  **Annotated Cluster Scatter Plots**: These plots visualize player archetypes
    with annotated cluster centers, providing a clear and aesthetically pleasing
    representation of player groupings.

All plots are saved to a designated output directory with a professional style.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap

# Set a professional and aesthetically pleasing style for all plots.
sns.set_style("whitegrid")
sns.set_context("talk")

# Define the default directory where all generated plots will be saved.
OUTPUT_DIR = "outputs"


def _ensure_output_dir(path: str = OUTPUT_DIR) -> str:
    """
    Ensure that the output directory exists, creating it if necessary.

    Args:
        path: The path to the directory to check/create.

    Returns:
        The path to the ensured directory.
    """
    os.makedirs(path, exist_ok=True)
    return path


def plot_shap_summary(
    shap_values,
    features: pd.DataFrame,
    output_filename: str = "shap_feature_importance.png",
    max_display: int = 15,
) -> str:
    """
    Create and save a SHAP bar plot to visualize global feature importance.

    This function generates a bar plot that displays the mean absolute SHAP value
    for each feature, providing a clear and easy-to-interpret overview of the most
    impactful features on the model's predictions.

    Args:
        shap_values: The SHAP values, typically a numpy array or a DataFrame.
        features: The feature matrix (DataFrame) corresponding to the SHAP values.
        output_filename: The name of the file to save the plot as.
        max_display: The maximum number of features to display in the plot.

    Returns:
        The full path to the saved plot image.
    """
    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, output_filename)

    plt.figure(figsize=(12, 8))
    # Generate a bar plot, which is often clearer for global importance
    shap.summary_plot(shap_values, features, plot_type="bar", show=False, max_display=max_display)
    plt.title("Global Feature Importance (SHAP Values)", fontsize=18, fontweight="bold")
    plt.xlabel("Mean Absolute SHAP Value", fontsize=14)
    plt.tight_layout()
    # Save the plot to the specified file
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()  # Close the plot to free up memory
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
    """
    Plot annotated player clusters in a two-dimensional scatter plot.

    This function creates an aesthetically pleasing scatter plot to visualize how
    players are grouped into different archetypes. Each point represents a
    player, colored by their assigned cluster. Cluster centers are annotated to
    provide a clear, insightful view of the archetypes.

    Args:
        data: The DataFrame containing the player data.
        x_col: The name of the column to use for the x-axis.
        y_col: The name of the column to use for the y-axis.
        cluster_col: The column containing the cluster labels for coloring the points.
        title: The title of the plot.
        output_filename: The name of the file to save the plot as.
        hue_order: An optional list to specify the order of cluster labels in the legend.

    Returns:
        The full path to the saved plot image.
    """
    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, output_filename)

    plt.figure(figsize=(12, 8))
    # Use a color-blind friendly and aesthetically pleasing palette
    palette = sns.color_palette("viridis", n_colors=data[cluster_col].nunique())
    # Create the scatter plot
    ax = sns.scatterplot(
        data=data, x=x_col, y=y_col, hue=cluster_col, palette=palette, hue_order=hue_order, s=100, alpha=0.8
    )
    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=14)
    plt.ylabel(y_col.replace("_", " ").title(), fontsize=14)
    plt.legend(title="Archetype", fontsize=12, title_fontsize=14)

    # Annotate cluster centers for better interpretability
    if hue_order is None:
        hue_order = sorted(data[cluster_col].unique())

    for i, cluster_name in enumerate(hue_order):
        cluster_data = data[data[cluster_col] == cluster_name]
        centroid_x = cluster_data[x_col].mean()
        centroid_y = cluster_data[y_col].mean()
        ax.text(
            centroid_x,
            centroid_y,
            cluster_name,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor=palette[i], alpha=0.9, boxstyle="round,pad=0.5"),
        )

    plt.tight_layout()
    # Save the plot with high resolution
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()  # Close the plot to free up memory
    return output_path
