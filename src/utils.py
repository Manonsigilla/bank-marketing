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
    Set random seeds for reproducibility across Python and NumPy.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    # Fixer les seeds garantit que les résultats sont reproductibles :
    # même code + même seed = mêmes résultats à chaque run.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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
        # Démarre le chronomètre au début du bloc with
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        # Arrête le chronomètre à la fin du bloc with et affiche la durée
        elapsed = time.perf_counter() - self.start
        # Format adapté à la durée : ms si < 1s, secondes si < 60s, minutes sinon
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
    # Sauvegarde le modèle entraîné sur le disque.
    # joblib est plus efficace que pickle pour les gros tableaux numpy.
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")
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
    # Recharge un modèle sauvegardé depuis le disque.
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)
