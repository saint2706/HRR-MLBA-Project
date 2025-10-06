"""
Plotting utilities for the player impact project.

This module provides functions for creating and saving visualizations that help to
interpret the results of the analysis. It is responsible for generating two main
types of plots:
1.  **SHAP Summary Plots**: These plots visualize the global feature importance as
    determined by the SHAP analysis, showing which factors have the most
    significant impact on the model's predictions.
2.  **Cluster Scatter Plots**: These plots help to visualize the player archetypes
    identified by the clustering algorithm, showing how different groups of
    players are separated based on key performance metrics.

All plots are saved to a designated output directory.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap


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


def plot_shap_summary(shap_values, features: pd.DataFrame, output_filename: str = "shap_summary.png") -> str:
    """
    Create and save a SHAP summary plot to visualize feature importance.

    This function generates a summary plot that displays the SHAP values for each
    feature, providing a clear overview of which features are most important for
    the model's predictions and how they influence the output.

    Args:
        shap_values: The SHAP values, typically a numpy array or a DataFrame.
        features: The feature matrix (DataFrame) corresponding to the SHAP values.
        output_filename: The name of the file to save the plot as.

    Returns:
        The full path to the saved plot image.
    """
    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, output_filename)

    plt.figure(figsize=(10, 6))
    # Ensure shap_values is a numpy array, as expected by the shap library
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    # Create the summary plot without displaying it directly
    shap.summary_plot(values, features, show=False)
    plt.tight_layout()
    # Save the plot to the specified file
    plt.savefig(output_path, bbox_inches="tight")
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
    Plot player clusters in a two-dimensional scatter plot.

    This function creates a scatter plot to visualize how players are grouped into
    different archetypes. Each point represents a player, colored by their
    assigned cluster, allowing for easy visual inspection of the separation
    between different roles.

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

    plt.figure(figsize=(8, 6))
    # Create the scatter plot using seaborn for better aesthetics
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=cluster_col, palette="deep", hue_order=hue_order)
    plt.title(title)
    plt.tight_layout()
    # Save the plot to the specified file
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()  # Close the plot to free up memory
    return output_path
