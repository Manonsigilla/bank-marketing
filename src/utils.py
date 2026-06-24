"""
Utility functions for the Bank Marketing project.

Includes:
- Setting random seeds for reproducibility
- Timer context manager
- Model saving/loading
"""

import os
import time
import random
import joblib
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and
    optionally TensorFlow/PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    pass


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """
    Context manager to time code blocks.

    Usage:
        with Timer("Model training"):
            model.fit(X, y)
    """
    def __init__(self, label: str = "Operation"):
        self.label = label
        self.start: float = 0.0

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, name: str) -> str:
    """
    Save a trained model to the models/ directory using joblib.

    Parameters
    ----------
    model : object
        Trained model object.
    name : str
        File name (without extension).

    Returns
    -------
    str
        Path to the saved model.
    """
    pass


def load_model(name: str) -> object:
    """
    Load a model from the models/ directory.

    Parameters
    ----------
    name : str
        File name (without extension).

    Returns
    -------
    object
        Loaded model.
    """
    pass
