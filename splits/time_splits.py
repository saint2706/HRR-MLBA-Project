"""Time-aware data splitting for preventing temporal leakage.

This module implements chronological data splits that respect the temporal
nature of cricket matches. It ensures that training data comes from earlier
seasons than validation and test data, preventing information leakage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class SplitConfig:
    """Configuration for time-based data splits.

    Attributes:
        train_seasons_max: Maximum season year for training data (inclusive).
        val_seasons: List of season years for validation data.
        test_seasons: List of season years for test data.
    """

    train_seasons_max: int = 2018
    val_seasons: List[int] = None
    test_seasons: List[int] = None

    def __post_init__(self) -> None:
        """Initialize default validation and test seasons if not provided."""
        if self.val_seasons is None:
            self.val_seasons = [2019]
        if self.test_seasons is None:
            self.test_seasons = [2020]


@dataclass
class DataSplit:
    """Container for train/val/test data splits.

    Attributes:
        train: Training data DataFrame.
        val: Validation data DataFrame.
        test: Test data DataFrame.
        config: The split configuration used.
        summary: Dictionary with split statistics.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    config: SplitConfig
    summary: Dict[str, any]


def create_time_splits(
    data: pd.DataFrame,
    config: Optional[SplitConfig] = None,
    season_col: str = "season",
    target_col: str = "team_won",
) -> DataSplit:
    """Create train/val/test splits based on chronological seasons.

    This function splits data chronologically to prevent temporal leakage.
    The default configuration uses:
    - Train: seasons <= 2018
    - Validation: season 2019
    - Test: season 2020

    Args:
        data: DataFrame containing the full dataset with a season column.
        config: Optional SplitConfig specifying the split boundaries.
        season_col: Name of the column containing season information.
        target_col: Name of the target column for computing win rates.

    Returns:
        DataSplit object containing train, val, test DataFrames and metadata.

    Raises:
        KeyError: If the season column is not found in the DataFrame.
        ValueError: If any split results in an empty DataFrame.
    """
    if config is None:
        config = SplitConfig()

    if season_col not in data.columns:
        raise KeyError(f"Season column '{season_col}' not found in data")

    # Ensure season is numeric
    data = data.copy()
    data[season_col] = pd.to_numeric(data[season_col], errors="coerce")

    # Drop rows with invalid seasons
    data = data.dropna(subset=[season_col])
    data[season_col] = data[season_col].astype(int)

    # Create splits based on season ranges
    train_mask = data[season_col] <= config.train_seasons_max
    val_mask = data[season_col].isin(config.val_seasons)
    test_mask = data[season_col].isin(config.test_seasons)

    train_df = data[train_mask].copy()
    val_df = data[val_mask].copy()
    test_df = data[test_mask].copy()

    # Validate splits
    if train_df.empty:
        raise ValueError(f"Training split is empty (seasons <= {config.train_seasons_max})")
    if val_df.empty:
        raise ValueError(f"Validation split is empty (seasons {config.val_seasons})")
    if test_df.empty:
        raise ValueError(f"Test split is empty (seasons {config.test_seasons})")

    # Compute summary statistics
    summary = _compute_split_summary(train_df, val_df, test_df, target_col, season_col)

    return DataSplit(train=train_df, val=val_df, test=test_df, config=config, summary=summary)


def _compute_split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    season_col: str,
) -> Dict[str, any]:
    """Compute summary statistics for data splits.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.
        target_col: Name of the target column.
        season_col: Name of the season column.

    Returns:
        Dictionary containing split statistics.
    """

    def _get_stats(df: pd.DataFrame, split_name: str) -> Dict[str, any]:
        """Get statistics for a single split."""
        stats = {
            "split": split_name,
            "n_samples": len(df),
            "n_unique_matches": df["match_id"].nunique() if "match_id" in df.columns else None,
            "seasons": sorted(df[season_col].unique().tolist()) if season_col in df.columns else [],
        }

        if target_col in df.columns:
            stats["win_rate"] = float(df[target_col].mean())
            stats["n_wins"] = int(df[target_col].sum())
            stats["n_losses"] = int((1 - df[target_col]).sum())

        return stats

    return {
        "train": _get_stats(train, "train"),
        "val": _get_stats(val, "val"),
        "test": _get_stats(test, "test"),
        "total_samples": len(train) + len(val) + len(test),
    }


def save_split_manifests(
    split: DataSplit,
    output_dir: str = "reports",
    include_ids: bool = True,
) -> Tuple[Path, Path]:
    """Save split manifests to CSV and JSON files.

    Args:
        split: The DataSplit object to save.
        output_dir: Directory to save the manifest files.
        include_ids: If True, saves a CSV with match_id and split assignment.

    Returns:
        Tuple of (csv_path, json_path) for the saved files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summary JSON
    json_path = output_path / "split_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(split.summary, f, indent=2)

    # Save splits CSV if requested
    csv_path = output_path / "splits.csv"
    if include_ids:
        splits_records = []

        for split_name, df in [("train", split.train), ("val", split.val), ("test", split.test)]:
            if "match_id" in df.columns:
                for match_id in df["match_id"].unique():
                    splits_records.append({"match_id": match_id, "split": split_name})

        if splits_records:
            splits_df = pd.DataFrame(splits_records)
            splits_df.to_csv(csv_path, index=False)
        else:
            # Fallback: save summary as CSV
            _save_summary_as_csv(split.summary, csv_path)
    else:
        _save_summary_as_csv(split.summary, csv_path)

    return csv_path, json_path


def _save_summary_as_csv(summary: Dict[str, any], csv_path: Path) -> None:
    """Save split summary as a CSV file.

    Args:
        summary: Summary dictionary from DataSplit.
        csv_path: Path to save the CSV file.
    """
    records = []
    for split_name in ["train", "val", "test"]:
        if split_name in summary:
            records.append(summary[split_name])

    if records:
        summary_df = pd.DataFrame(records)
        summary_df.to_csv(csv_path, index=False)


def rolling_origin_cv_splits(
    data: pd.DataFrame,
    min_train_seasons: int = 3,
    season_col: str = "season",
    target_col: str = "team_won",
) -> List[DataSplit]:
    """Create rolling-origin cross-validation splits.

    This function creates multiple train/test splits where the training window
    grows incrementally and each subsequent season is used as a test set.

    Args:
        data: DataFrame containing the full dataset.
        min_train_seasons: Minimum number of seasons for the initial training set.
        season_col: Name of the column containing season information.
        target_col: Name of the target column.

    Returns:
        List of DataSplit objects, one for each fold.
    """
    data = data.copy()
    data[season_col] = pd.to_numeric(data[season_col], errors="coerce")
    data = data.dropna(subset=[season_col])
    data[season_col] = data[season_col].astype(int)

    seasons = sorted(data[season_col].unique())

    if len(seasons) < min_train_seasons + 1:
        raise ValueError(
            f"Not enough seasons for rolling CV. Need at least {min_train_seasons + 1}, found {len(seasons)}"
        )

    splits = []

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        # Use the season before test as validation if possible
        if i > min_train_seasons:
            val_season = seasons[i - 1]
            train_seasons_actual = seasons[: i - 1]
        else:
            # For the first fold, use the last training season as validation
            val_season = seasons[i - 1]
            train_seasons_actual = seasons[: i - 1]

        config = SplitConfig(
            train_seasons_max=train_seasons_actual[-1] if train_seasons_actual else train_seasons[0],
            val_seasons=[val_season],
            test_seasons=[test_season],
        )

        train_mask = data[season_col].isin(train_seasons_actual)
        val_mask = data[season_col] == val_season
        test_mask = data[season_col] == test_season

        train_df = data[train_mask].copy()
        val_df = data[val_mask].copy()
        test_df = data[test_mask].copy()

        if not train_df.empty and not test_df.empty:
            summary = _compute_split_summary(train_df, val_df, test_df, target_col, season_col)
            splits.append(DataSplit(train=train_df, val=val_df, test=test_df, config=config, summary=summary))

    return splits
