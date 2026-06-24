────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┐
  │            Fichier/Dossier             │                               Rôle                                │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ data/                                  │ Données brutes (déjà importées, non versionnées)                  │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ src/data_loader.py                     │ Charger les 2 datasets (Bank + Bank Additional)                   │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ src/preprocessing.py                   │ Séparer X/y, supprimer duration, encoder/scaler, train/test split │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ src/eda.py                             │ Distributions, corrélations, analyse des valeurs unknown          │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ src/modeling.py                        │ Baseline, Logistic Regression, Random Forest, XGBoost, GridSearch │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ src/evaluation.py                      │ Accuracy, Precision, Recall, F1, ROC-AUC, matrice de confusion    │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ src/utils.py                           │ Seeds, timer, sauvegarde/chargement de modèles                    │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ notebooks/01_eda_bank.ipynb            │ Exploration du dataset UCI original                               │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ notebooks/02_eda_bank_additional.ipynb │ Exploration du dataset enrichi                                    │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ notebooks/03_modeling.ipynb            │ Pipeline complet de modélisation                                  │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ models/                                │ Modèles sauvegardés (joblib)                                      │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ outputs/                               │ Résultats et rapports                                             │
  ├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ figures/                               │ Graphiques exportés                                               │
  └────────────────────────────────────────┴───────────────────────────────────────────────────────────────────┘

  ⚙️  Détails importants

  - L'attribut duration est automatiquement exclu (drop_duration=True par défaut) pour que les modèles soient réalistes
  - Les valeurs unknown sont conservées comme catégorie à part entière (pas de suppression automatique)
  - Les modèles sont configurés avec gestion du déséquilibre des classes (class_weight='balanced' ou scale_pos_weight)

  🚀 Pour lancer un notebook

  cd bank-marketing
  source venv/Scripts/activate
  jupyter notebook notebooks/