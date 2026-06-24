"""
Evaluation module for the Bank Marketing project.

Provides:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix visualization
- ROC curve plotting
- Feature importance visualization
- Model comparison utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray or None
        Predicted probabilities for the positive class (for ROC-AUC).

    Returns
    -------
    dict
        Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    pass


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print a formatted classification report (precision, recall, f1-score).

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    """
    pass


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> None:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    title : str
        Plot title.
    """
    pass


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, label: str = "Model") -> None:
    """
    Plot the ROC curve with AUC score.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    label : str
        Legend label for the model.
    """
    pass


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list[str], top_n: int = 20) -> None:
    """
    Plot feature importances for tree-based models (Random Forest, XGBoost).

    Parameters
    ----------
    model : estimator
        Fitted model with feature_importances_ or coef_ attribute.
    feature_names : list[str]
        Names of the features.
    top_n : int
        Number of top features to display.
    """
    pass


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def compare_models(results: dict[str, dict], sort_by: str = "ROC-AUC") -> pd.DataFrame:
    """
    Compare multiple models' performance in a DataFrame.

    Parameters
    ----------
    results : dict
        {model_name: {metric: value}}.
    sort_by : str
        Metric to sort by.

    Returns
    -------
    pd.DataFrame
    """
    pass
