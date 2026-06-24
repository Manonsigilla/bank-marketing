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
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print a formatted classification report.
    """
    print(classification_report(y_true, y_pred, target_names=["No (0)", "Yes (1)"]))


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> None:
    """
    Plot a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No", "Yes"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, label: str = "Model") -> None:
    """
    Plot the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", linewidth=2, color="darkorange")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list[str], top_n: int = 20) -> None:
    """
    Plot feature importances for tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        print("Model does not have feature_importances_ or coef_.")
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def compare_models(results: dict[str, dict], sort_by: str = "ROC-AUC") -> pd.DataFrame:
    """
    Compare multiple models' performance.

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
    df = pd.DataFrame(results).T
    df = df.sort_values(sort_by, ascending=False)
    return df
