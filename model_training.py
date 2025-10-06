"""
Model training utilities for player impact prediction.

This module contains the functions and classes necessary for training the machine
learning model that predicts player impact. The core of this module is the
`train_impact_model` function, which uses an XGBoost classifier to learn the
relationship between player performance features and match outcomes.

The key components are:
1.  **ModelArtifacts**: A dataclass to store all the outputs of the training
    process, including the trained model, feature sets, and evaluation metrics.
2.  **train_impact_model**: A function that takes the feature matrix, splits it
    into training and testing sets, trains an XGBoost model, and evaluates its
    performance.
3.  **generate_classification_report**: A utility function to create a detailed
    text-based report of the model's classification performance.
"""
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
    """
    A container for all the objects generated during the model training process.

    This dataclass provides a structured way to return and manage the various
    outputs from the `train_impact_model` function, making it easier to pass
    them to other parts of the pipeline, such as SHAP analysis and reporting.

    Attributes:
        model: The trained XGBoost classifier instance.
        feature_cols: A list of the feature names used for training.
        X_train: The training set features.
        X_test: The testing set features.
        y_train: The training set labels (target variable).
        y_test: The testing set labels (target variable).
        metrics: A dictionary of performance metrics (e.g., accuracy, ROC AUC).
        predictions: The binary predictions made on the test set.
        probabilities: The predicted probabilities for the positive class on the test set.
    """
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
    """
    Train an XGBoost model to predict player impact based on performance features.

    This function orchestrates the model training process. It takes the feature
    matrix, splits it into training and testing sets, and then trains an XGBoost
    classifier. It also evaluates the model on the test set and returns all the
    relevant artifacts in a `ModelArtifacts` object.

    Args:
        data: The DataFrame containing the features and the target variable.
        feature_cols: A list of column names to be used as features.
        target_col: The name of the target variable column.
        test_size: The proportion of the dataset to allocate to the test set.
        random_state: A seed for the random number generator to ensure reproducibility.

    Returns:
        A `ModelArtifacts` object containing the trained model and all related outputs.
    """
    # Step 1: Separate features (X) and target (y)
    X = data[feature_cols]
    y = data[target_col]

    # Step 2: Split the data into training and testing sets
    # Stratification is used to maintain the same proportion of classes in train and test sets,
    # which is important for imbalanced datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    # Step 3: Initialize the XGBoost classifier with a set of optimized hyperparameters
    # These parameters are chosen to balance performance and prevent overfitting.
    model = XGBClassifier(
        n_estimators=300,          # Number of boosting rounds
        max_depth=4,               # Maximum tree depth
        learning_rate=0.05,        # Step size shrinkage
        subsample=0.8,             # Fraction of samples to be used for fitting each tree
        colsample_bytree=0.8,      # Fraction of features to be used for fitting each tree
        reg_lambda=1.0,            # L2 regularization term
        random_state=random_state,
        objective="binary:logistic", # Objective function for binary classification
        eval_metric="auc",         # Evaluation metric for validation data
        tree_method="hist",        # Use histogram-based algorithm for faster training
        enable_categorical=False,
    )

    # Step 4: Train the model on the training data
    model.fit(X_train, y_train)

    # Step 5: Make predictions on the test set
    # Predict probabilities for the positive class (team_won = 1)
    proba = model.predict_proba(X_test)[:, 1]
    # Convert probabilities to binary predictions using a 0.5 threshold
    preds = (proba >= 0.5).astype(int)

    # Step 6: Evaluate the model's performance
    # Calculate accuracy and ROC AUC score, handling cases with no variance in the test labels.
    metrics = {
        "accuracy": accuracy_score(y_test, preds) if y_test.nunique() > 1 else float("nan"),
        "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan"),
    }

    # Step 7: Package all outputs into the ModelArtifacts container
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
    """
    Generate a detailed classification report from the model's predictions.

    This function uses scikit-learn's `classification_report` to create a text
    summary of the main classification metrics (precision, recall, F1-score)
    for each class.

    Args:
        artifacts: The `ModelArtifacts` object containing the test labels and predictions.

    Returns:
        A string containing the formatted classification report. Returns an error
        message if the test data has only one class.
    """
    # The classification report can only be generated if there are at least two classes
    # in the test labels.
    if artifacts.y_test.nunique() <= 1:
        return "Insufficient class variance in test labels for classification report."
    return classification_report(artifacts.y_test, artifacts.predictions, digits=3)
