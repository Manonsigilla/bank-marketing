# 🏦 Bank Marketing — Prédiction de Souscription à un Dépôt à Terme

> Projet de Machine Learning visant à prédire si un client souscrira à un dépôt à terme bancaire, à partir de données de campagnes de télémarketing d'une banque portugaise.

---

## 📋 Contexte

Les campagnes de marketing direct de la banque sont réalisées par appels téléphoniques. Souvent, plusieurs contacts sont nécessaires pour déterminer si le client va souscrire ou non au produit (dépôt à terme).

**Objectif :** Construire un modèle de classification binaire capable de prédire la variable cible `y` (`yes`/`no`) indiquant si le client a souscrit.

---

## 📊 Données

Deux jeux de données sont disponibles :

| Dataset | Fichier principal | Instances | Attributs | Description |
| --- | --- | --- | --- | --- |
| **Bank** | `data/bank/bank-full.csv` | 45 211 | 16 + cible | Dataset UCI original |
| **Bank Additional** | `data/bank_additional/bank-additional-full.csv` | 41 188 | 20 + cible | Enrichi avec 5 indicateurs socio-économiques |

### Variables clés

- **Données client :** âge, emploi, statut marital, éducation, crédit en défaut, solde moyen, prêt immobilier, prêt personnel
- **Dernier contact :** type de contact, jour, mois, durée
- **Campagne :** nombre de contacts, jours depuis dernier contact, contacts précédents, résultat précédent
- **Indicateurs économiques (Additional uniquement) :** taux de variation de l'emploi, indice des prix, indice de confiance des consommateurs, taux Euribor 3 mois, nombre d'employés

> ⚠️ **Important :** L'attribut `duration` (durée de l'appel) n'est pas connu avant la fin de l'appel — il doit être **exclu** des modèles prédictifs réalistes.

---

## 🗂️ Structure du projet

```text
bank-marketing/
├── data/                       # Données brutes (non versionnées)
│   ├── bank/                   # Dataset original UCI
│   └── bank_additional/        # Dataset enrichi
├── notebooks/                  # Notebooks Jupyter
│   ├── 01_eda.ipynb            # Analyse exploratoire (Bank + Bank Additional)
│   └── 02_modeling.ipynb       # Pipeline de modélisation
├── src/                        # Code source
│   ├── __init__.py
│   ├── data_loader.py          # Chargement des données
│   ├── preprocessing.py        # Prétraitement & feature engineering
│   ├── eda.py                  # Analyse exploratoire des données
│   ├── modeling.py             # Entraînement des modèles
│   ├── evaluation.py           # Évaluation & métriques
│   └── utils.py                # Fonctions utilitaires
├── models/                     # Modèles sauvegardés
├── outputs/                    # Résultats, figures, rapports
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

```bash
# 1. Cloner le repo
git clone <repo-url>
cd bank-marketing

# 2. Créer l'environnement virtuel
python -m venv venv

# 3. Activer l'environnement
source venv/Scripts/activate   # Windows (Git Bash)
# ou
venv\Scripts\activate          # Windows (cmd)

# 4. Installer les dépendances
pip install -r requirements.txt
```

---

## 📈 Approche

1. **Analyse exploratoire (EDA)** — Distribution des variables, corrélations, déséquilibre des classes
2. **Prétraitement** — Encodage des variables catégorielles, normalisation, gestion des valeurs `unknown`
3. **Modélisation** — Baseline (Dummy), Régression Logistique, Random Forest, XGBoost
4. **Évaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, matrice de confusion
5. **Optimisation** — Gestion du déséquilibre des classes (`class_weight`, `scale_pos_weight`), GridSearchCV

---

## 📚 Références

- Moro, S., Laureano, R., & Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*. ESM'2011.
- Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems.

---

## 📝 Licence

Ce projet est à but éducatif. Les données sont publiquement disponibles pour la recherche.
