"""
Preprocessing module for the Bank Marketing dataset.

Handles:
- Feature/target separation
- Dropping the 'duration' column for realistic modeling
- Encoding categorical variables
- Scaling numerical features
- Train/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DURATION_COL = "duration"
TARGET_COL = "y"


# ---------------------------------------------------------------------------
# Feature/target separation
# ---------------------------------------------------------------------------

def separate_features_target(
    df: pd.DataFrame,
    drop_duration: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    drop_duration : bool
        If True, drops the 'duration' column (recommended for realistic models).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable (encoded as 0/1).
    """
    pass


# ---------------------------------------------------------------------------
# Column type identification
# ---------------------------------------------------------------------------

def get_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical columns in a DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    num_cols : list[str]
        Numerical column names.
    cat_cols : list[str]
        Categorical column names.
    """
    pass


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float
        Proportion of data to use for testing.
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        If True, uses stratified splitting to preserve class proportions.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    pass


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numerical features
    and one-hot encodes categorical features.

    Parameters
    ----------
    num_cols : list[str]
        Numerical column names.
    cat_cols : list[str]
        Categorical column names.

    Returns
    -------
    ColumnTransformer
    """
    pass
