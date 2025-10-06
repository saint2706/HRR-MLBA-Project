"""SHAP explainability utilities for player impact modelling."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
import shap


def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: int = 2000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute SHAP values for a fitted tree-based model."""

    if len(X) == 0:
        raise ValueError("Input feature matrix is empty. Cannot compute SHAP values.")

    if sample_size and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_array = shap_values

    shap_frame = pd.DataFrame(shap_array, columns=X_sample.columns, index=X_sample.index)
    return shap_frame, X_sample


def summarise_shap_importance(shap_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise feature importance using mean absolute SHAP values."""

    summary = (
        shap_frame.abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return summary
