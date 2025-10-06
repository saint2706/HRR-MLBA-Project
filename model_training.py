"""Model training utilities for player impact prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@dataclass
class ModelArtifacts:
    model: XGBClassifier
    feature_cols: List[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metrics: Dict[str, float]
    predictions: np.ndarray
    probabilities: np.ndarray


def train_impact_model(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "team_won",
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelArtifacts:
    """Train an XGBoost model for player impact prediction."""

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=False,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds) if y_test.nunique() > 1 else float("nan"),
        "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan"),
    }

    return ModelArtifacts(
        model=model,
        feature_cols=feature_cols,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metrics=metrics,
        predictions=preds,
        probabilities=proba,
    )


def generate_classification_report(artifacts: ModelArtifacts) -> str:
    """Generate a detailed classification report."""

    if artifacts.y_test.nunique() <= 1:
        return "Insufficient class variance in test labels for classification report."
    return classification_report(artifacts.y_test, artifacts.predictions, digits=3)
