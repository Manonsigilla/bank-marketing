"""
Exploratory Data Analysis (EDA) module.

Provides functions for:
- Summary statistics
- Target distribution analysis (class imbalance)
- Numerical feature distributions
- Categorical feature distributions
- Correlation analysis
- Missing value analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Styling defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# ---------------------------------------------------------------------------
# Summary & missing values
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> None:
    """
    Print descriptive statistics for numerical and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    """
    pass


def analyze_missing(df: pd.DataFrame) -> pd.Series:
    """
    Analyze 'unknown' values in categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.Series
        Count of 'unknown' values per column.
    """
    pass


# ---------------------------------------------------------------------------
# Target analysis
# ---------------------------------------------------------------------------

def plot_target_distribution(y: pd.Series, title: str = "Target Variable Distribution") -> None:
    """
    Plot the target variable distribution (bar chart + percentages).

    Parameters
    ----------
    y : pd.Series
        Target variable (0/1).
    title : str
        Plot title.
    """
    pass


# ---------------------------------------------------------------------------
# Numerical features
# ---------------------------------------------------------------------------

def plot_numerical_distributions(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """
    Plot histograms for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list[str] or None
        Specific columns to plot. If None, all numerical columns are used.
    """
    pass


# ---------------------------------------------------------------------------
# Categorical features
# ---------------------------------------------------------------------------

def plot_categorical_distributions(
    df: pd.DataFrame,
    target: pd.Series | None = None,
    columns: list[str] | None = None,
    max_categories: int = 15,
) -> None:
    """
    Plot count plots for categorical features, optionally colored by target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : pd.Series or None
        If provided, colors bars by target class.
    columns : list[str] or None
        Specific columns to plot. If None, all categorical columns are used.
    max_categories : int
        Maximum number of categories to display per variable.
    """
    pass


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> None:
    """
    Plot a correlation matrix heatmap for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    method : str
        Correlation method ('pearson', 'spearman', or 'kendall').
    """
    pass
