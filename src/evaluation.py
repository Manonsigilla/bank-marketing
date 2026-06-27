"""
Module d'évaluation des modèles.

PyCaret fournit déjà des fonctions d'évaluation intégrées (plot_model, evaluate_model)
qui génèrent automatiquement :
- Matrice de confusion
- Courbe ROC
- Importance des features
- Rapport de classification

Ce module contient des fonctions complémentaires pour :
- Calculer les métriques manuellement (pour comprendre leur signification)
- Afficher le rapport de classification avec des labels lisibles
- Comparer plusieurs modèles dans un tableau récapitulatif

CE QUE CE MODULE CONTIENT :
---------------------------
- get_metrics() : calculer les métriques de classification
- print_classification_report() : rapport détaillé par classe
- compare_models() : tableau comparatif des performances
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """
    Calcule les métriques de classification standards.

    Parameters
    ----------
    y_true : np.ndarray
        Vraies étiquettes (0 ou 1).
    y_pred : np.ndarray
        Étiquettes prédites par le modèle (0 ou 1).
    y_proba : np.ndarray ou None
        Probabilités prédites pour la classe positive (nécessaire pour le ROC-AUC).

    Returns
    -------
    dict
        Dictionnaire contenant Accuracy, Precision, Recall, F1-Score, et ROC-AUC.
    """
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        # zero_division=0 : évite une erreur si le modèle ne prédit jamais "oui"
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
    }
    # Le ROC-AUC nécessite les probabilités, pas juste les prédictions 0/1
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Affiche un rapport de classification détaillé :
    précision, rappel, F1-score pour chaque classe (Non et Oui).

    Parameters
    ----------
    y_true : np.ndarray
        Vraies étiquettes.
    y_pred : np.ndarray
        Étiquettes prédites.
    """
    print(classification_report(y_true, y_pred, target_names=["Non (0)", "Oui (1)"]))


# ---------------------------------------------------------------------------
# Comparaison de modèles
# ---------------------------------------------------------------------------

def compare_models(results: dict[str, dict], sort_by: str = "Accuracy") -> pd.DataFrame:
    """
    Transforme un dictionnaire de résultats en DataFrame lisible.
    Chaque ligne = un modèle, chaque colonne = une métrique.

    Parameters
    ----------
    results : dict
        Dictionnaire de la forme {nom_du_modèle: {métrique: valeur}}.
    sort_by : str
        Métrique utilisée pour trier les modèles (par défaut "Accuracy").

    Returns
    -------
    pd.DataFrame
        Tableau des performances trié de la meilleure à la moins bonne.
    """
    # .T (transpose) : les clés deviennent les lignes du DataFrame
    df = pd.DataFrame(results).T
    # Trier du meilleur au moins bon
    df = df.sort_values(sort_by, ascending=False)
    return df


# =============================================================================
# Guide : quelle métrique regarder ?
# =============================================================================
#
# Les instructions demandent d'interpréter la colonne "Accuracy".
#
# L'ACCURACY c'est : (bonnes prédictions) / (total des prédictions)
#   → Facile à comprendre mais PIÈGE dans ce projet !
#
# Pourquoi c'est un piège ?
#   Nos données ont ~88% de "non" et ~12% de "oui".
#   Un modèle qui répond TOUJOURS "non" aura 88% d'accuracy...
#   et pourtant il est complètement INUTILE (il ne trouve aucun client).
#
# ⚠️ NE PAS se fier à l'Accuracy seule pour ce projet.
#
# Les métriques complémentaires à regarder :
#
#   RECALL (Rappel) — "Parmi les vrais 'oui', combien j'en ai trouvé ?"
#     → Priorité si on veut rater le moins d'opportunités possible.
#
#   PRECISION — "Parmi ceux que j'ai dit 'oui', combien le sont vraiment ?"
#     → Priorité si on veut éviter les faux espoirs.
#
#   ROC-AUC — La métrique la plus fiable pour les données déséquilibrées.
#     → Mesure la capacité globale à séparer les "oui" des "non".
#     → AUC = 0.5 = hasard, AUC = 0.8+ = bon, AUC = 0.9+ = excellent.
#
#   F1-SCORE — Le compromis entre Précision et Rappel.
