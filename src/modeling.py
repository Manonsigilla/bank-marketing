"""
Module de modélisation avec PyCaret (AutoML).

POURQUOI PyCaret ?
------------------
PyCaret est une librairie d'AutoML (Automatic Machine Learning) qui automatise
tout le workflow de modélisation :
- Prétraitement (encodage, scaling, gestion des valeurs manquantes)
- Train/test split avec stratification
- Entraînement et comparaison de 15+ modèles en UNE commande
- Évaluation (courbes ROC, matrice de confusion, importance des features)

C'est l'outil idéal pour un débutant : pas besoin de coder chaque modèle
à la main, PyCaret s'occupe de tout et présente les résultats de façon
claire et comparable.

CE QUE CE MODULE CONTIENT :
---------------------------
- setup_pycaret() : initialise l'environnement PyCaret
- compare_all_models() : entraîne et compare tous les modèles disponibles
"""

import pandas as pd
from pycaret.classification import setup, compare_models, pull  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Configuration PyCaret
# ---------------------------------------------------------------------------

def setup_pycaret(
    df: pd.DataFrame,
    target: str = "y",
    train_size: float = 0.8,
    session_id: int = 42,
    fix_imbalance: bool = False,
) -> None:
    """
    Initialise l'environnement PyCaret pour une tâche de classification.

    Cette fonction configure automatiquement :
    - La séparation train/test (80/20 par défaut, avec stratification)
    - Le prétraitement (encodage, scaling, imputation)
    - La validation croisée (10 folds par défaut)
    - La détection du type de chaque colonne (numérique/catégorielle)

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame complet, incluant la colonne cible.
    target : str
        Nom de la colonne cible (par défaut "y").
    train_size : float
        Proportion des données pour l'entraînement (0.8 = 80%).
    session_id : int
        Seed aléatoire pour la reproductibilité.
    fix_imbalance : bool
        Si True, applique SMOTE pour rééquilibrer les classes.
        Laisser False pour utiliser les paramètres par défaut de PyCaret.
    """
    # PyCaret setup : une seule fonction qui configure tout l'environnement.
    # - preprocess=True : active le prétraitement automatique
    #   (standardisation des numériques, one-hot encoding des catégorielles)
    # - session_id : seed pour reproductibilité
    # - train_size : 80% des données pour entraîner, 20% pour tester
    # - silent=True : ne pas demander de confirmation interactive
    # - log_experiment=False : ne pas logger sur MLflow (trop complexe pour débuter)
    setup(
        data=df,
        target=target,
        train_size=train_size,
        session_id=session_id,
        preprocess=True,
        fix_imbalance=fix_imbalance,
        verbose=False,
        log_experiment=False,
    )
    print("✅ Environnement PyCaret initialisé.")
    print(f"   Cible : {target}")
    print(f"   Split : {train_size:.0%} train / {1-train_size:.0%} test")
    print(f"   Prétraitement automatique activé (encodage + scaling)")


# ---------------------------------------------------------------------------
# Comparaison de modèles
# ---------------------------------------------------------------------------

def compare_all_models(
    sort: str = "Accuracy",
    n_select: int = 5,
):
    """
    Entraîne et compare TOUS les modèles de classification disponibles.

    PyCaret va automatiquement :
    1. Entraîner 15+ modèles différents (Régression Logistique, Random Forest,
       XGBoost, LightGBM, Arbre de Décision, etc.)
    2. Évaluer chaque modèle avec la métrique choisie
    3. Les classer du meilleur au moins bon

    Parameters
    ----------
    sort : str
        Métrique utilisée pour classer les modèles.
        Par défaut "Accuracy" comme demandé dans les instructions.
        Alternatives : "AUC", "Recall", "Precision", "F1".
    n_select : int
        Nombre de meilleurs modèles à retourner.

    Returns
    -------
    best_model : le meilleur modèle entraîné (utilisable avec plot_model, predict_model...)
    top_models : list
        Liste des n_select meilleurs modèles (ou un seul modèle si n_select=1).
    results : pd.DataFrame
        Tableau comparatif des modèles trié par la métrique choisie.
    """
    # compare_models() : LA commande magique de PyCaret.
    # En une ligne, elle entraîne et compare tous les modèles disponibles.
    # - sort : la métrique de classement
    # - n_select : combien de modèles garder
    # Si n_select=1 : retourne un modèle unique
    # Si n_select>1 : retourne une liste de modèles
    top_models = compare_models(sort=sort, n_select=n_select)

    # Récupérer le tableau de résultats détaillé
    results = pull()

    # Extraire le meilleur modèle (le 1er de la liste si n_select > 1)
    if isinstance(top_models, list):
        best_model = top_models[0]
    else:
        best_model = top_models

    print(f"\n📊 Comparaison des {n_select} meilleurs modèles (triés par {sort}) :")
    print(results.head(n_select).to_string())

    return best_model, results
