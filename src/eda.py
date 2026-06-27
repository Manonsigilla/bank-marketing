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
    Print descriptive statistics for numerical and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    """
    
    print(df.select_dtypes(include=[np.number]).describe().T)
    print(df.select_dtypes(include=[object]).describe().T)


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
    # Compte le nombre de "unknown" dans chaque colonne
    unknown_counts = (df == "unknown").sum()
    # Garde uniquement les colonnes qui ont au moins 1 "unknown"
    unknown_counts = unknown_counts[unknown_counts > 0]
    # Trie de la plus touchée à la moins touchée
    unknown_counts = unknown_counts.sort_values(ascending=False)

    # Affichage lisible avec le pourcentage en plus du compte brut
    print("Colonnes contenant la valeur 'unknown' :")
    for col, count in unknown_counts.items():
        pct = 100 * count / len(df)  # calcule le pourcentage
        print(f"  {col}: {count} ({pct:.1f}%)")

    return unknown_counts


# ---------------------------------------------------------------------------
# Target analysis
# ---------------------------------------------------------------------------

def plot_target_distribution(y: pd.Series, title: str = "Target Variable Distribution") -> None:
    """
    Plot the target variable distribution (bar chart + percentages).

    Parameters
    ----------
    y : pd.Series
        Target variable (0/1).
    title : str
        Plot title.
    """
    # 1. Compter combien de 0 et combien de 1
    counts = y.value_counts()

    # 2. Créer des labels lisibles pour l'axe X
    labels = ["Non (0)" if k == 0 else "Oui (1)" for k in counts.index]

    # 3. Créer la figure et les barres
    _, ax = plt.subplots()
    # Bleu pour Non, orange pour Oui
    bars = ax.bar(labels, counts.to_list(), color=["steelblue", "darkorange"], edgecolor="white")

    # 4. Ajouter le nombre et le pourcentage au-dessus de chaque barre
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / len(y)  # pourcentage par rapport au total
        # Position X : centre de la barre, Position Y : un peu au-dessus de la barre
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{count:,}\n({pct:.1f}%)", ha="center", fontweight="bold")

    # 5. Titres et légendes
    ax.set_title(title)
    ax.set_ylabel("Nombre de clients")

    plt.tight_layout()  # ajuste les marges
    plt.show()


# ---------------------------------------------------------------------------
# Numerical features
# ---------------------------------------------------------------------------

def plot_numerical_distributions(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """
    Plot histograms for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list[str] or None
        Specific columns to plot. If None, all numerical columns are used.
    """
    # Si aucune colonne n'est spécifiée, prendre toutes les colonnes numériques
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Si aucune colonne numérique, ne rien faire
    if len(columns) == 0:
        print("Aucune colonne numérique à afficher.")
        return

    n_cols = len(columns)
    # Calculer le nombre de lignes nécessaires (3 graphiques par ligne)
    n_rows = (n_cols + 2) // 3

    # Créer une grille de sous-graphiques
    _, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    # Aplatir le tableau 2D d'axes en 1D pour itérer facilement
    axes = axes.flatten()

    i = 0  # initialiser au cas où la boucle ne s'exécute pas
    for i, col in enumerate(columns):
        # Tracer l'histogramme de la colonne
        axes[i].hist(df[col], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        axes[i].set_title(col)      # Nom de la variable en titre
        axes[i].set_ylabel("Fréquence")

    # Masquer les sous-graphiques vides restants
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()  # Ajuster les espacements
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

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : pd.Series or None
        If provided, colors bars by target class.
    columns : list[str] or None
        Specific columns to plot. If None, all categorical columns are used.
    max_categories : int
        Maximum number of categories to display per variable.
    """
    # Si aucune colonne n'est spécifiée, prendre toutes les colonnes texte
    # sauf la cible 'y' (si elle est dans le DataFrame)
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()
        columns = [c for c in columns if c != "y"]

    if len(columns) == 0:
        print("Aucune colonne catégorielle à afficher.")
        return

    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3  # 3 graphiques par ligne

    # Créer une grille de sous-graphiques
    _, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    # Si la cible est fournie, on l'ajoute temporairement pour colorer les barres
    data = df.copy()
    if target is not None:
        # Convertir 0/1 en labels lisibles pour la légende
        data["__target__"] = target.map({1: "Yes", 0: "No"})
        hue = "__target__"
    else:
        hue = None

    i = 0  # initialiser au cas où la boucle ne s'exécute pas
    for i, col in enumerate(columns):
        # Si la colonne a trop de catégories, garder seulement les plus fréquentes
        n_unique = data[col].nunique()
        if n_unique > max_categories:
            top_categories = data[col].value_counts().head(max_categories).index
            plot_data = data[data[col].isin(top_categories)]
        else:
            plot_data = data

        # Compter les occurrences de chaque catégorie
        sns.countplot(
            data=plot_data, x=col, hue=hue, ax=axes[i],
            palette="Set2",                           # palette de couleurs douces
            order=plot_data[col].value_counts().index # trier par fréquence
        )
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)    # pivoter les labels si longs

    # Masquer les sous-graphiques inutilisés
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    # Nettoyer la colonne temporaire si elle existe
    if target is not None:
        data.drop(columns=["__target__"], inplace=True)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> None:
    """
    Plot a correlation matrix heatmap for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    method : str
        Correlation method ('pearson', 'spearman', or 'kendall').
    """
    # Garder uniquement les colonnes numériques
    num_df = df.select_dtypes(include=[np.number])

    # Vérifier qu'il y a assez de colonnes numériques pour une matrice
    if num_df.shape[1] < 2:
        print("Pas assez de colonnes numériques pour une matrice de corrélation.")
        return

    # Calculer la matrice de corrélation (Pearson par défaut)
    # Pearson mesure la relation linéaire entre deux variables (-1 à +1)
    corr = num_df.corr(method=method)  # type: ignore

    # Créer la figure
    _, ax = plt.subplots(figsize=(12, 10))

    # Masquer le triangle supérieur pour éviter les doublons
    # (la matrice est symétrique, le triangle inférieur suffit)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Heatmap : carte de chaleur avec annotations
    sns.heatmap(
        corr,
        mask=mask,            # cache le triangle supérieur
        annot=True,           # affiche la valeur dans chaque case
        fmt=".2f",            # format à 2 décimales
        cmap="coolwarm",      # bleu = corrélation négative, rouge = positive
        center=0,             # centrer la colormap sur 0
        square=True,          # cases carrées
        linewidths=0.5,       # fine bordure entre les cases
        ax=ax,
        cbar_kws={"shrink": 0.8}  # réduire la taille de la barre de couleur
    )
    ax.set_title(f"Matrice de corrélation ({method.capitalize()})")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Résumé d'analyse exploratoire (à compléter après avoir exécuté le notebook)
# =============================================================================
#
# Après avoir exploré les données avec les fonctions ci-dessus, voici
# les conclusions types qu'un data analyst tirerait :
#
# 1. DÉSÉQUILIBRE DES CLASSES
#    - Environ 12% de "oui" contre 88% de "non"
#    - Un modèle qui prédit toujours "non" aurait 88% de bonnes réponses...
#      et serait complètement inutile.
#    - Il FAUT gérer ce déséquilibre (class_weight='balanced', SMOTE...).
#
# 2. DURATION = VARIABLE PIÈGE
#    - 'duration' (durée de l'appel) est TRÈS corrélée à la cible
#      (les appels longs aboutissent plus souvent à un "oui").
#    - Mais dans la vraie vie, on ne connaît pas la durée AVANT l'appel.
#    - Il faut IMPÉRATIVEMENT l'exclure des modèles prédictifs.
#
# 3. VALEURS 'unknown'
#    - Certaines colonnes sont truffées de "unknown" (ex: poutcome à 82%).
#    - poutcome = résultat de la campagne précédente. Si le client n'a jamais
#      été contacté, c'est normal que ce soit "unknown" — c'est une info en soi.
#    - On garde "unknown" comme une catégorie à part entière.
#
# 4. INDICATEURS SOCIO-ÉCONOMIQUES (dataset Additional uniquement)
#    - nr.employed (nombre d'employés) est le plus corrélé à la cible (~-0.35)
#    - euribor3m (taux d'intérêt) et emp.var.rate (variation de l'emploi)
#      sont aussi liés : en période de crise, les gens placent moins.
#    - Ces 5 variables macro-économiques apportent un contexte utile.
#
# 5. PROFIL TYPE DU CLIENT QUI SOUSCRIT
#    - A tendance à être plus âgé
#    - A un solde bancaire plus élevé
#    - A déjà été contacté auparavant (poutcome = "success")
#    - N'a pas de prêt en cours (loan = "no")
#    - Est contacté via "cellular" plutôt que "telephone"
#
# Ces observations guideront nos choix de modélisation.
