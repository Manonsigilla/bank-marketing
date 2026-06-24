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
    Print descriptive statistics for all columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    """
    print("Numerical features summary:")
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(df[num_cols].describe().round(2).to_string())
    print(f"\nCategorical features summary:")
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        n_unique = df[col].nunique()
        print(f"  {col}: {n_unique} unique values")


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
    unknown_counts = (df == "unknown").sum()
    unknown_counts = unknown_counts[unknown_counts > 0].sort_values(ascending=False)
    if len(unknown_counts) == 0:
        print("No 'unknown' values found.")
    else:
        print("'unknown' values per column:")
        for col, count in unknown_counts.items():
            pct = 100 * count / len(df)
            print(f"  {col}: {count} ({pct:.1f}%)")
    return unknown_counts


# ---------------------------------------------------------------------------
# Target analysis
# ---------------------------------------------------------------------------

def plot_target_distribution(y: pd.Series, title: str = "Target Variable Distribution") -> None:
    """
    Plot the target variable distribution (bar chart + percentages).
    """
    counts = y.value_counts()
    labels = ["No (0)" if k == 0 else "Yes (1)" for k in counts.index]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, counts.values, color=["steelblue", "darkorange"], edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Number of clients")

    for bar, count in zip(bars, counts.values):
        pct = 100 * count / len(y)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{count:,}\n({pct:.1f}%)", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Numerical features
# ---------------------------------------------------------------------------

def plot_numerical_distributions(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """
    Plot histograms for numerical features.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        df[col].hist(bins=40, ax=axes[i], color="steelblue", edgecolor="white", alpha=0.8)
        axes[i].set_title(col)
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


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
    """
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()
        # Exclude target if present in df
        columns = [c for c in columns if c != "y"]

    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    data = df.copy()
    if target is not None:
        data = data.copy()
        data["__target__"] = target.map({1: "Yes", 0: "No"})
        hue = "__target__"
    else:
        hue = None

    for i, col in enumerate(columns):
        n_unique = data[col].nunique()
        if n_unique > max_categories:
            top = data[col].value_counts().head(max_categories).index
            plot_data = data[data[col].isin(top)]
        else:
            plot_data = data
        sns.countplot(data=plot_data, x=col, hue=hue, ax=axes[i],
                      palette="Set2", order=plot_data[col].value_counts().index)
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> None:
    """
    Plot a correlation matrix heatmap for numerical features.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        print("Not enough numerical columns for correlation matrix.")
        return

    corr = num_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"Correlation Matrix ({method.capitalize()})")
    plt.tight_layout()
    plt.show()
