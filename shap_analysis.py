"""
SHAP explainability utilities for player impact modelling.

This module provides functions for interpreting the trained machine learning model
using SHAP (SHapley Additive exPlanations). SHAP is a powerful technique from
cooperative game theory that helps to explain the output of any machine learning
model by assigning an importance value to each feature for each prediction.

The key functions in this module are:
1.  **compute_shap_values**: Calculates the SHAP values for a given model and
    dataset. It uses a sample of the data for efficiency.
2.  **summarise_shap_importance**: Aggregates the SHAP values to provide a global
    summary of feature importance, which is crucial for understanding the main
    drivers of the model's predictions. These importance scores are then used as
    weights for calculating player impact ratings.
"""
from __future__ import annotations

import re
from typing import Tuple

import pandas as pd
import shap


def _safe_float_conversion(value):
    """
    Safely convert a string to float, handling XGBoost 3.x bracketed format.
    
    XGBoost 3.x stores some numeric values in brackets, e.g., '[4.7761565E-1]'.
    This function strips the brackets before converting to float.
    
    Args:
        value: The value to convert (string or numeric).
    
    Returns:
        The float value.
    """
    if isinstance(value, str):
        # Remove brackets if present: '[4.7761565E-1]' -> '4.7761565E-1'
        value = value.strip('[]')
    return float(value)


def _patch_xgboost_base_score():
    """
    Monkey-patch SHAP's XGBTreeModelLoader to handle XGBoost 3.x base_score format.
    
    XGBoost 3.x stores base_score as '[value]' (e.g., '[4.7761565E-1]'), but SHAP's
    TreeExplainer tries to convert it directly to float, which fails. This patch
    wraps the built-in float() function to handle bracketed values.
    """
    try:
        from shap.explainers import _tree
        import builtins
        
        # Store the original __init__ method and original float
        original_init = _tree.XGBTreeModelLoader.__init__
        original_float = builtins.float
        
        def patched_init(self, xgb_model):
            """Patched __init__ that handles bracketed base_score values."""
            # Temporarily replace the built-in float function
            def safe_float(value):
                """float() wrapper that handles bracketed XGBoost values."""
                if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    # Remove brackets: '[4.7761565E-1]' -> '4.7761565E-1'
                    value = value[1:-1]
                return original_float(value)
            
            # Temporarily replace float in builtins
            builtins.float = safe_float
            
            try:
                # Call the original __init__
                original_init(self, xgb_model)
            finally:
                # Restore the original float function
                builtins.float = original_float
        
        # Apply the patch
        _tree.XGBTreeModelLoader.__init__ = patched_init
        return True
    except Exception as e:
        # If patching fails, log it but don't crash
        print(f"Warning: Failed to patch SHAP XGBTreeModelLoader: {e}")
        return False


# Apply the patch when the module is imported
_patch_xgboost_base_score()


def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: int = 2000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute SHAP values for a fitted tree-based model to explain its predictions.

    This function uses the `shap.TreeExplainer` to calculate SHAP values, which
    quantify the contribution of each feature to the model's output for each
    data point. To manage performance, it operates on a sample of the data if
    the dataset is large.

    Args:
        model: The trained, tree-based model (e.g., XGBoost) to be explained.
        X: The feature matrix (DataFrame) for which to compute SHAP values.
        sample_size: The number of samples to use for the SHAP calculation. If
                     the input DataFrame is smaller, the full dataset is used.
        random_state: A seed for the random number generator to ensure
                      reproducibility of the sample selection.

    Returns:
        A tuple containing:
        - shap_frame: A DataFrame of SHAP values, with a row for each sample and
                      a column for each feature.
        - X_sample: The subset of the original data that was used for the
                    calculation.

    Raises:
        ValueError: If the input feature matrix `X` is empty.
    """
    if len(X) == 0:
        raise ValueError("Input feature matrix is empty. Cannot compute SHAP values.")

    # Use a sample of the data for efficiency if the dataset is large
    if sample_size and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
    else:
        X_sample = X

    # Initialize the TreeExplainer, which is optimized for tree-based models
    # The monkey-patch applied at module import handles XGBoost 3.x compatibility
    explainer = shap.TreeExplainer(model)
    # Calculate the SHAP values for the sampled data
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers, shap_values returns a list of two arrays (one for each class).
    # We are interested in the SHAP values for the positive class (class 1).
    if isinstance(shap_values, list):
        shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_array = shap_values

    # Convert the SHAP values array into a more usable DataFrame
    shap_frame = pd.DataFrame(shap_array, columns=X_sample.columns, index=X_sample.index)
    return shap_frame, X_sample


def summarise_shap_importance(shap_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise feature importance by calculating the mean absolute SHAP value for each feature.

    This function provides a global measure of feature importance by averaging the
    absolute SHAP values across all samples. A higher mean absolute SHAP value
    indicates a feature that has a larger average impact on the model's predictions.

    Args:
        shap_frame: A DataFrame containing the SHAP values for each feature and sample.

    Returns:
        A DataFrame summarizing the importance of each feature, sorted in
        descending order. It has two columns: 'feature' and 'mean_abs_shap'.
    """
    # To get a global feature importance, we take the mean of the absolute SHAP values
    # for each feature across all the samples.
    summary = (
        shap_frame.abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return summary
