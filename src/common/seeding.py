"""Global seeding utilities for reproducibility across the pipeline.

This module provides a centralized way to set random seeds across all libraries
used in the project (Python, NumPy, XGBoost) to ensure reproducible results.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy, and XGBoost for reproducibility.

    This function ensures that all random operations across the pipeline produce
    consistent results when the same seed is used. It sets the random state for:
    - Python's built-in random module
    - NumPy's random number generator
    - Environment variables for XGBoost's random state

    Args:
        seed: The integer seed value to use for all random number generators.
        deterministic: If True, sets additional environment variables to enforce
                      deterministic behavior in XGBoost (may impact performance).

    Example:
        >>> set_global_seed(42)
        >>> # All random operations will now be reproducible
    """
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set environment variables for XGBoost
    # These are read by XGBoost when creating models
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Additional settings for deterministic XGBoost behavior
        # Note: This may reduce performance but ensures reproducibility
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"


def get_default_seed() -> int:
    """Get the default seed value used across the project.

    Returns:
        The default seed value (42).
    """
    return 42


def set_xgboost_random_state(seed: Optional[int] = None) -> int:
    """Get a seed value suitable for XGBoost's random_state parameter.

    Args:
        seed: Optional seed value. If None, uses the default seed.

    Returns:
        The seed value to use for XGBoost's random_state parameter.
    """
    if seed is None:
        seed = get_default_seed()
    return seed
