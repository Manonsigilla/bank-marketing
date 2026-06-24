"""
Modeling module for the Bank Marketing project.

Provides:
- Baseline models (dummy classifier)
- Classic ML models: Logistic Regression, Random Forest, XGBoost
- Model training with optional class balancing
- Hyperparameter tuning (GridSearchCV)
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------

def get_baseline(strategy: str = "stratified") -> DummyClassifier:
    """
    Create a dummy classifier as a baseline.

    Parameters
    ----------
    strategy : str
        One of 'stratified', 'most_frequent', 'uniform'.

    Returns
    -------
    DummyClassifier
    """
    return DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def get_logistic_regression(
    class_weight: str = "balanced",
    max_iter: int = 1000,
) -> LogisticRegression:
    """
    Create a Logistic Regression model.

    Parameters
    ----------
    class_weight : str
        'balanced' for automatic class weighting, or None.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    LogisticRegression
    """
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def get_random_forest(
    n_estimators: int = 200,
    class_weight: str = "balanced",
    max_depth: int | None = None,
) -> RandomForestClassifier:
    """
    Create a Random Forest model.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    class_weight : str
        'balanced', 'balanced_subsample', or None.
    max_depth : int or None
        Maximum depth of trees.

    Returns
    -------
    RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def get_xgboost(
    scale_pos_weight: float | None = None,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 6,
) -> XGBClassifier:
    """
    Create an XGBoost model.

    Parameters
    ----------
    scale_pos_weight : float or None
        Weight for the positive class. Set to (neg/pos) for imbalance.
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Step size shrinkage.
    max_depth : int
        Maximum tree depth.

    Returns
    -------
    XGBClassifier
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
    )


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

def tune_model(
    model,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Perform hyperparameter tuning with GridSearchCV.

    Parameters
    ----------
    model : estimator
        The model to tune.
    param_grid : dict
        Parameter grid to search over.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training target.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    GridSearchCV (fitted)
    """
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV score ({scoring}): {grid.best_score_:.4f}")
    return grid
