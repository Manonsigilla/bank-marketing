# Document technique — Bank Marketing

> Projet de Machine Learning : prédire si un client va souscrire à un dépôt à terme bancaire.

---

## 🗂️ Structure du projet

| Fichier/Dossier | Rôle |
|---|---|
| `data/` | Données brutes (déjà importées) |
| `src/data_loader.py` | Charger les 2 datasets (Bank + Bank Additional) |
| `src/eda.py` | Analyse exploratoire (distributions, corrélations, valeurs `unknown`) |
| `src/modeling.py` | Configuration PyCaret et comparaison automatique des modèles |
| `src/evaluation.py` | Métriques manuelles, rapport de classification, guide d'interprétation |
| `src/utils.py` | Seeds, timer, sauvegarde/chargement de modèles |
| `notebooks/01_eda.ipynb` | Exploration des deux datasets |
| `notebooks/02_modeling.ipynb` | Pipeline complet de modélisation avec PyCaret (AutoML) |
| `models/` | Modèles sauvegardés |
| `outputs/` | Résultats et rapports |

---

## ⚙️ Points clés

- L'attribut `duration` est **exclu** des modèles prédictifs (inconnu avant la fin de l'appel)
- Les valeurs `unknown` sont conservées comme catégorie à part entière
- PyCaret gère automatiquement le prétraitement, l'encodage, le scaling et la gestion du déséquilibre

---

## 🚀 Lancement rapide

```bash
cd bank-marketing
conda activate bank-marketing
jupyter notebook notebooks/
```
