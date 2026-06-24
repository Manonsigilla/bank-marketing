"""
Data loading module for the Bank Marketing dataset.

Provides functions to load and inspect the two dataset versions:
- bank (UCI original, 17 attributes)
- bank_additional (enriched with social/economic indicators, 21 attributes)
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BANK_DIR = DATA_DIR / "bank"
BANK_ADDITIONAL_DIR = DATA_DIR / "bank_additional"


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_bank(full: bool = True) -> pd.DataFrame:
    """
    Load the original Bank Marketing dataset (UCI).

    Parameters
    ----------
    full : bool
        If True, loads bank-full.csv (45,211 rows).
        If False, loads bank.csv (4,521 rows, 10% sample).

    Returns
    -------
    pd.DataFrame
    """
    filename = "bank-full.csv" if full else "bank.csv"
    filepath = BANK_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, sep=";")


def load_bank_additional(full: bool = True) -> pd.DataFrame:
    """
    Load the enriched Bank Marketing dataset (with social/economic features).

    Parameters
    ----------
    full : bool
        If True, loads bank-additional-full.csv (41,188 rows).
        If False, loads bank-additional.csv (4,119 rows, 10% sample).

    Returns
    -------
    pd.DataFrame
    """
    filename = "bank-additional-full.csv" if full else "bank-additional.csv"
    filepath = BANK_ADDITIONAL_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, sep=";")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_dataset_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print basic information about a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.
    name : str
        Label for display.
    """
    print(f"{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col:<20s}  dtype={df[col].dtype}")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  None (note: 'unknown' is used as a category label)")
    else:
        for col, count in missing.items():
            print(f"  {col}: {count}")
    print(f"\nTarget distribution:")
    if "y" in df.columns:
        print(df["y"].value_counts().to_string())
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    print()
