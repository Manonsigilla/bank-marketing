"""
Preprocessing module for the Bank Marketing dataset.

POURQUOI CE FICHIER ?
--------------------
Les modèles de Machine Learning sont des robots mathématiques : ils ne
comprennent QUE les chiffres, et ils les préfèrent à la même échelle.

Or nos données brutes contiennent :
- Du texte ("marié", "célibataire", "divorcé")
- Des chiffres à des échelles très différentes (âge: 18-95, solde: -8000 à 100000)
- Une colonne piège (duration) qu'on ne connaîtra pas en conditions réelles
- Une cible en texte ("yes"/"no") qu'il faut convertir en 0/1

CE QUE CE FICHIER FAIT (dans l'ordre) :
---------------------------------------
1. separate_features_target() — Sépare X (ce qu'on sait) et y (ce qu'on prédit)
   → Supprime aussi 'duration' car inutilisable en vrai
   → Convertit "yes"/"no" en 1/0

2. get_column_types() — Trie les colonnes en numériques vs textuelles
   → Car le traitement n'est pas le même pour les deux

3. split_data() — Découpe en train (80%) et test (20%)
   → Avec stratification : même % de "oui" dans les deux paquets

4. build_preprocessing_pipeline() — Construit la machine à transformer
   → StandardScaler : ramène tous les chiffres à la même échelle
   → OneHotEncoder : transforme le texte en colonnes 0/1

Handles:
- Feature/target separation
- Dropping the 'duration' column for realistic modeling
- Encoding categorical variables
- Scaling numerical features
- Train/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DURATION_COL = "duration"
TARGET_COL = "y"


# ---------------------------------------------------------------------------
# Feature/target separation
# ---------------------------------------------------------------------------

def separate_features_target(
    df: pd.DataFrame,
    drop_duration: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    drop_duration : bool
        If True, drops the 'duration' column (recommended for realistic models).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable (encoded as 0/1).
    """
    # Travailler sur une copie pour ne pas modifier l'original
    df = df.copy()

    # Supprimer la colonne 'duration' si demandé ET si elle existe
    # Pourquoi ? Dans la vraie vie, on ne connaît pas la durée de l'appel
    # avant de l'avoir passé. L'inclure = tricher.
    if drop_duration and DURATION_COL in df.columns:
        df = df.drop(columns=[DURATION_COL])

    # Isoler la cible (y) et la convertir en 0/1
    # Le modèle ne comprend que les chiffres, pas "yes"/"no"
    y = df[TARGET_COL].map({"yes": 1, "no": 0})

    # Retirer la colonne cible des features (X)
    X = df.drop(columns=[TARGET_COL])

    return X, y


# ---------------------------------------------------------------------------
# Column type identification
# ---------------------------------------------------------------------------

def get_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical columns in a DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    num_cols : list[str]
        Numerical column names.
    cat_cols : list[str]
        Categorical column names.
    """
    # Colonnes numériques : types int64, float64, etc.
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # Colonnes catégorielles : types object (texte) ou category
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return num_cols, cat_cols


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float
        Proportion of data to use for testing.
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        If True, uses stratified splitting to preserve class proportions.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # Stratification : garantit que le % de "oui" est le même dans le train
    # et le test. Ex: 12% de oui dans les données → 12% dans train ET dans test.
    # Sans ça, on pourrait avoir 5% dans le train et 20% dans le test = biaisé.
    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,         # ex: 0.2 = 20% pour le test
        random_state=random_state,   # seed fixe pour avoir le même split à chaque run
        stratify=stratify_arg,       # préserver les proportions de classes
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numerical features
    and one-hot encodes categorical features.

    Parameters
    ----------
    num_cols : list[str]
        Numerical column names.
    cat_cols : list[str]
        Categorical column names.

    Returns
    -------
    ColumnTransformer
    """
    # ColumnTransformer : applique une transformation différente à chaque
    # groupe de colonnes, puis recolle le tout en une seule matrice.

    preprocessor = ColumnTransformer(
        transformers=[
            # 1. Colonnes numériques → StandardScaler
            #    Soustrait la moyenne et divise par l'écart-type.
            #    Ex: âge moyen=40, écart-type=10 → âge 50 devient (50-40)/10 = 1.0
            #    Pourquoi ? Sinon le solde (0 à 100 000) écrase l'âge (18 à 95)
            ("num", StandardScaler(), num_cols),

            # 2. Colonnes catégorielles → OneHotEncoder
            #    Transforme chaque catégorie en colonne binaire (0/1).
            #    Ex: "marié"/"célibataire"/"divorcé" →
            #         colonne "marié" (0/1), colonne "divorcé" (0/1)
            #    drop="first" : supprime la 1ère catégorie (évite la redondance)
            #    sparse_output=False : renvoie un tableau dense (plus simple à utiliser)
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ],
        remainder="drop",  # supprime toute colonne non listée (sécurité)
    )
    return preprocessor
