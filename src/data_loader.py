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
    
    BANK_FILE = BANK_DIR / ("bank-full.csv" if full else "bank.csv")
    df = pd.read_csv(BANK_FILE, sep=";")
    return df


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
    BANK_ADDITIONAL_FILE = BANK_ADDITIONAL_DIR / ("bank-additional-full.csv" if full else "bank-additional.csv")
    df = pd.read_csv(BANK_ADDITIONAL_FILE, sep=";")
    return df


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_dataset_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print basic information about a dataset (shape, columns, dtypes,
    missing values, target distribution, first rows).

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.
    name : str
        Label for display.
    """
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Target distribution:\n{df['y'].value_counts()}")
    print(f"First rows:\n{df.head()}")