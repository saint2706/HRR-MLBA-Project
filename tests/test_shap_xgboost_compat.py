"""
Test for XGBoost 3.x compatibility with SHAP TreeExplainer.

This test ensures that the monkey-patch in shap_analysis.py correctly handles
the bracketed base_score format introduced in XGBoost 3.x.
"""
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from shap_analysis import compute_shap_values


def test_shap_with_xgboost_base_score_format():
    """Test that SHAP can compute values with XGBoost 3.x models.
    
    XGBoost 3.x stores base_score as '[value]' (e.g., '[4.7761565E-1]'),
    which causes SHAP to fail without the patch. This test verifies the fix works.
    """
    # Create a simple test dataset
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
    y = np.random.randint(0, 2, 100)
    
    # Train an XGBoost model (which will have the bracketed base_score in 3.x)
    model = XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # This should not raise ValueError about string to float conversion
    shap_frame, shap_sample = compute_shap_values(model, X, sample_size=20, random_state=42)
    
    # Verify the output shape and type
    assert isinstance(shap_frame, pd.DataFrame)
    assert isinstance(shap_sample, pd.DataFrame)
    assert shap_frame.shape == (20, 5)
    assert shap_sample.shape == (20, 5)
    assert list(shap_frame.columns) == ['a', 'b', 'c', 'd', 'e']


def test_shap_values_are_numeric():
    """Test that SHAP values are numeric and not NaN."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(50, 3), columns=['x1', 'x2', 'x3'])
    y = np.random.randint(0, 2, 50)
    
    model = XGBClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    shap_frame, _ = compute_shap_values(model, X, sample_size=10, random_state=42)
    
    # All SHAP values should be numeric
    assert shap_frame.notna().all().all()
    assert np.isfinite(shap_frame.values).all()
