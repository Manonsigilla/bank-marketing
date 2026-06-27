"""
Modeling module for the Bank Marketing project.

Provides:
- Baseline models (dummy classifier)
- Classic ML models: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning (GridSearchCV)
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier  # type: ignore[import-untyped]


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
    # Le modèle le plus bête possible. Sert de point de comparaison.
    # - "stratified" : répond au hasard en respectant la proportion des classes
    #   (ex: 88% de "non", 12% de "oui" → répond "non" 88% du temps)
    # - "most_frequent" : répond TOUJOURS la classe majoritaire ("non")
    # - "uniform" : répond 50/50 au hasard (complètement aléatoire)
    return DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)  # type: ignore[arg-type]


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
    # Régression logistique : le modèle le plus simple et interprétable.
    # Il calcule une probabilité à partir d'une somme pondérée des features :
    #   P(oui) = 1 / (1 + exp(-(w1*age + w2*solde + ... + b)))
    #
    # class_weight="balanced" : compense automatiquement le déséquilibre
    #   en donnant plus de poids aux "oui" (la classe rare).
    # max_iter=1000 : on augmente le nombre d'itérations pour être sûr
    #   que l'algorithme converge.
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
    # Random Forest : une "forêt" d'arbres de décision qui votent ensemble.
    # Chaque arbre voit un sous-ensemble aléatoire des données et des features,
    # ce qui évite le sur-apprentissage (overfitting).
    #
    # n_estimators=200 : 200 arbres dans la forêt. Plus y'en a, plus c'est
    #   robuste, mais plus c'est lent.
    # class_weight="balanced" : compense le déséquilibre des classes.
    # n_jobs=-1 : utilise tous les cœurs du CPU pour aller plus vite.
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,  # type: ignore[arg-type]
        max_depth=max_depth,      # None = laisse les arbres grandir librement
        random_state=RANDOM_STATE,
        n_jobs=-1,                # parallélisation sur tous les cœurs
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
    # XGBoost (eXtreme Gradient Boosting) : le modèle star des compétitions ML.
    # Contrairement à Random Forest où les arbres sont indépendants, XGBoost
    # construit les arbres SÉQUENTIELLEMENT : chaque nouvel arbre se concentre
    # sur les erreurs du précédent. C'est comme un élève qui apprend de ses fautes.
    #
    # scale_pos_weight : gère le déséquilibre. À calculer avec (nb_négatifs / nb_positifs).
    #   Ex: 32000 non / 4200 oui ≈ 7.6 → les "oui" pèsent 7.6× plus.
    # learning_rate=0.1 : chaque arbre ne contribue qu'à 10%. Plus c'est petit,
    #   plus l'apprentissage est fin... mais plus il faut d'arbres.
    # max_depth=6 : profondeur max de chaque arbre. Limite la complexité.
    # eval_metric="logloss" : métrique de perte pour l'évaluation interne.
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
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
    # GridSearchCV : teste TOUTES les combinaisons de paramètres possibles
    # et garde la meilleure (selon le scoring choisi).
    #
    # Exemple de param_grid pour Random Forest :
    #   {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
    #   → va tester 3×3 = 9 combinaisons
    #
    # cv=5 : validation croisée en 5 folds.
    #   Divise le train en 5 parts, entraîne sur 4, teste sur la 5ème,
    #   et répète 5 fois en changeant la part test. Plus fiable qu'un
    #   simple split train/test.
    #
    # scoring="roc_auc" : on optimise l'AUC (aire sous la courbe ROC),
    #   une métrique robuste aux classes déséquilibrées.
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,    # parallélisation
        verbose=1,        # affiche la progression
    )
    grid.fit(X_train, y_train)
    # Afficher les résultats
    print(f"Meilleurs paramètres : {grid.best_params_}")
    print(f"Meilleur score {scoring} : {grid.best_score_:.4f}")
    return grid


# =============================================================================
# Guide : quel modèle choisir ?
# =============================================================================
#
# Situation réelle : on veut prédire quels clients vont dire "oui".
# Deux objectifs possibles, deux modèles différents :
#
# SCÉNARIO A — "Je veux comprendre POURQUOI"
#   → Régression Logistique
#   → On peut lire les coefficients : "un client avec un prêt en cours
#     a 2× moins de chances de dire oui".
#   → Utile pour présenter à un manager, un client, une direction.
#   → Inconvénient : moins performant sur les relations complexes.
#
# SCÉNARIO B — "Je veux la MEILLEURE prédiction possible"
#   → Random Forest ou XGBoost
#   → Meilleur score, capture les interactions entre variables.
#   → Inconvénient : boîte noire, difficile à expliquer.
#
# BONNE PRATIQUE : entraîner les 4 modèles et comparer avec compare_models().
# Si la Régression Logistique est proche du Random Forest → prendre la LR
# (plus simple, plus explicable). Si le RF est meilleur → le prendre.
#
# La baseline (DummyClassifier) sert uniquement de garde-fou :
# si mon meilleur modèle fait moins bien que la baseline, il y a un bug.
