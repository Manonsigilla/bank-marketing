"""
Utility functions for the Bank Marketing project.

Includes:
- Setting random seeds for reproducibility
- Timer context manager
- Model saving/loading
- Formatting helpers
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
    (potentially) other libraries.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Attempt to set seeds for optional dependencies
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
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
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        if elapsed < 1:
            print(f"[{self.label}] {elapsed*1000:.0f} ms")
        elif elapsed < 60:
            print(f"[{self.label}] {elapsed:.2f} s")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"[{self.label}] {minutes}m {seconds:.1f}s")


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
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return str(path)


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
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
