"""Tests for time-aware data splitting utilities."""
from __future__ import annotations

import pandas as pd
import pytest

from splits.time_splits import SplitConfig, create_time_splits, rolling_origin_cv_splits


def test_create_time_splits_basic():
    """Test basic time split functionality."""
    # Create sample data
    data = pd.DataFrame({
        "match_id": range(100),
        "season": [2017] * 20 + [2018] * 30 + [2019] * 25 + [2020] * 25,
        "team_won": [0, 1] * 50,
    })

    config = SplitConfig(train_seasons_max=2018, val_seasons=[2019], test_seasons=[2020])

    split = create_time_splits(data, config)

    # Check that splits are not empty
    assert len(split.train) > 0
    assert len(split.val) > 0
    assert len(split.test) > 0

    # Check season boundaries
    assert split.train["season"].max() <= 2018
    assert split.val["season"].tolist() == [2019] * len(split.val)
    assert split.test["season"].tolist() == [2020] * len(split.test)


def test_create_time_splits_summary():
    """Test that split summary is computed correctly."""
    data = pd.DataFrame({
        "match_id": range(100),
        "season": [2018] * 50 + [2019] * 25 + [2020] * 25,
        "team_won": [0, 1] * 50,
    })

    split = create_time_splits(data)

    # Check summary structure
    assert "train" in split.summary
    assert "val" in split.summary
    assert "test" in split.summary

    # Check summary contents
    assert split.summary["train"]["n_samples"] == 50
    assert split.summary["val"]["n_samples"] == 25
    assert split.summary["test"]["n_samples"] == 25


def test_create_time_splits_empty_split_raises():
    """Test that empty splits raise ValueError."""
    data = pd.DataFrame({
        "match_id": range(50),
        "season": [2020] * 50,
        "team_won": [0, 1] * 25,
    })

    config = SplitConfig(train_seasons_max=2018, val_seasons=[2019], test_seasons=[2020])

    with pytest.raises(ValueError, match="Training split is empty"):
        create_time_splits(data, config)


def test_rolling_origin_cv_basic():
    """Test rolling origin cross-validation."""
    data = pd.DataFrame({
        "match_id": range(200),
        "season": [2015] * 40 + [2016] * 40 + [2017] * 40 + [2018] * 40 + [2019] * 40,
        "team_won": [0, 1] * 100,
    })

    splits = rolling_origin_cv_splits(data, min_train_seasons=3)

    # Should have at least 2 folds (test on 2018, 2019)
    assert len(splits) >= 2

    # Check that each split has train, val, test
    for split in splits:
        assert len(split.train) > 0
        assert len(split.test) > 0


def test_split_config_default_values():
    """Test that SplitConfig has correct defaults."""
    config = SplitConfig()

    assert config.train_seasons_max == 2018
    assert config.val_seasons == [2019]
    assert config.test_seasons == [2020]
