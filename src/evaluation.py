"""
Evaluation module for the Bank Marketing project.

Provides:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix visualization
- ROC curve plotting
- Feature importance visualization
- Model comparison utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray or None
        Predicted probabilities for the positive class (for ROC-AUC).

    Returns
    -------
    dict
        Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    # Dictionnaire des métriques de base
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        # zero_division=0 : évite une erreur si le modèle ne prédit jamais "oui"
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
    }
    # ROC-AUC nécessite les probabilités, pas juste les prédictions 0/1
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print a formatted classification report (precision, recall, f1-score).

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    """
    # Rapport détaillé : précision, recall, f1 pour chaque classe (Non et Oui)
    # + moyenne macro (moyenne simple des 2 classes)
    # + moyenne pondérée (pondérée par le nombre d'exemples dans chaque classe)
    print(classification_report(y_true, y_pred, target_names=["Non (0)", "Oui (1)"]))


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> None:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    title : str
        Plot title.
    """
    # Matrice de confusion : un tableau qui compte les bonnes et mauvaises
    # prédictions pour chaque classe.
    #
    #                  | Prédit NON | Prédit OUI |
    #   Vrai NON       |     VN     |     FP     |  (FP = fausse alerte)
    #   Vrai OUI       |     FN     |     VP     |  (FN = opportunité ratée)
    #
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Non", "Oui"]

    _, ax = plt.subplots(figsize=(6, 5))
    # Heatmap avec annotations (nombre dans chaque case)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, label: str = "Model") -> None:
    """
    Plot the ROC curve with AUC score.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    label : str
        Legend label for the model.
    """
    # Courbe ROC : montre le compromis entre "trouver les vrais oui" (TPR)
    # et "ne pas alerter pour rien" (FPR), quel que soit le seuil choisi.
    #
    # Elle répond à : si je change mon seuil de décision (ex: je dis "oui"
    # dès que la proba > 30% au lieu de 50%), comment ça impacte mes résultats ?
    #
    # AUC = aire sous la courbe. 1.0 = parfait, 0.5 = aléatoire (la diagonale).
    # C'est LA métrique de référence pour les problèmes déséquilibrés.

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    _, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", linewidth=2, color="darkorange")
    # Diagonale = classificateur aléatoire (50/50)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs (FP / (FP+VN))")
    ax.set_ylabel("Taux de vrais positifs (VP / (VP+FN))")
    ax.set_title("Courbe ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list[str], top_n: int = 20) -> None:
    """
    Plot feature importances for tree-based models (Random Forest, XGBoost).

    Parameters
    ----------
    model : estimator
        Fitted model with feature_importances_ or coef_ attribute.
    feature_names : list[str]
        Names of the features.
    top_n : int
        Number of top features to display.
    """
    # Récupérer les importances selon le type de modèle
    # Random Forest / XGBoost → feature_importances_
    # Régression Logistique → coefficients (valeur absolue)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Valeur absolue : un coef négatif est tout aussi important qu'un positif
        importances = np.abs(model.coef_).flatten()
    else:
        print("Ce modèle n'a ni feature_importances_ ni coef_.")
        return

    # Créer un DataFrame pour trier facilement
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    # Bar plot horizontal (plus lisible pour les noms de features longs)
    _, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} — Importance des variables")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def compare_models(results: dict[str, dict], sort_by: str = "ROC-AUC") -> pd.DataFrame:
    """
    Compare multiple models' performance in a DataFrame.

    Parameters
    ----------
    results : dict
        {model_name: {metric: value}}.
    sort_by : str
        Metric to sort by.

    Returns
    -------
    pd.DataFrame
    """
    # Transforme {modele: {metrique: valeur}} en DataFrame lisible
    # .T (transpose) : les clés du dict deviennent les lignes du DataFrame
    df = pd.DataFrame(results).T
    # Trier du meilleur au moins bon selon la métrique choisie
    df = df.sort_values(sort_by, ascending=False)
    return df


# =============================================================================
# Guide : quelle métrique regarder en priorité ?
# =============================================================================
#
# Notre problème : prédire si un client va souscrire (oui/non).
# Les classes sont TRÈS déséquilibrées (~12% oui, ~88% non).
#
# ⚠️ NE JAMAIS se fier à l'Accuracy seule dans ce cas.
#    Un modèle qui répond TOUJOURS "non" a 88% d'accuracy...
#    et ne trouve AUCUN client intéressé. Il est inutile.
#
# Les métriques qui comptent VRAIMENT :
#
#   RECALL (Rappel) — "Parmi tous les vrais 'oui', combien j'en ai trouvé ?"
#     → Si la banque veut rater le moins d'opportunités possible.
#     → Ex: recall=0.80 → on a trouvé 80% des clients intéressés.
#
#   PRECISION — "Parmi tous ceux que j'ai appelés 'oui', combien le sont vraiment ?"
#     → Si la banque veut éviter d'appeler des gens pour rien.
#     → Ex: precision=0.50 → 1 appel sur 2 est pertinent.
#
#   F1-SCORE — Le compromis entre Précision et Rappel.
#     → Quand on veut un équilibre entre les deux.
#
#   ROC-AUC — La meilleure métrique GLOBALE pour les données déséquilibrées.
#     → Mesure la capacité du modèle à séparer les "oui" des "non".
#     → AUC=0.5 = hasard, AUC=0.8 = bon, AUC=0.9+ = excellent.
#     → C'est notre métrique de référence pour comparer les modèles.
#
# En pratique : on regarde d'abord le ROC-AUC pour classer les modèles,
# puis on examine le Recall et la Précision selon le besoin métier.
