# Difficultés et anomalies rencontrées — Projet Bank Marketing

> Document de retour d'expérience pour la présentation orale.
> Couvre l'intégralité du projet : analyse exploratoire (EDA) + modélisation (AutoML).
> Chaque section peut être transformée en slide.

---

# PARTIE 1 — Analyse Exploratoire des Données (EDA)

## 1.1 Données : anomalies détectées dans les datasets

### `pdays = -1` pour 81.7% des clients (dataset Bank)

- **Observation :** La colonne `pdays` (jours depuis le dernier contact) vaut -1 pour la grande majorité des clients.
- **Interprétation :** `-1` signifie "jamais contacté auparavant". Ce n'est pas une erreur mais un code spécial qu'il faut comprendre. Un modèle naïf traiterait -1 comme une valeur numérique normale, ce qui fausserait tout.
- **Leçon :** Toujours lire la documentation des données (le fichier `bank-names.txt` fourni avec le dataset). Les valeurs spéciales existent et ont un sens métier.

### `balance` négatif jusqu'à -8019 (dataset Bank)

- **Observation :** Des clients ont un solde bancaire négatif.
- **Interprétation :** Comptes à découvert. C'est une information réelle et pertinente — un client à découvert n'a probablement pas d'épargne à placer.
- **Leçon :** Ne pas filtrer ou corriger les valeurs extrêmes sans comprendre leur sens métier.

### `duration = 0` pour certains appels

- **Observation :** Des appels ont une durée de 0 seconde.
- **Interprétation :** Appels non aboutis (pas de réponse, mauvais numéro). Ces lignes restent exploitables car les autres attributs sont renseignés.
- **Leçon :** Les zéros ne sont pas toujours des valeurs manquantes, ils peuvent avoir un sens.

### Le dataset Bank Additional n'a PAS de colonne `balance`

- **Observation :** Le dataset enrichi (Additional) contient 5 indicateurs socio-économiques... mais perd la colonne `balance` (solde bancaire) présente dans le dataset original.
- **Impact :** On ne peut pas utiliser les deux datasets ensemble sans perdre une information clé (le solde est très corrélé à la souscription).
- **Leçon :** "Enrichi" ne veut pas dire "strictement meilleur". Il faut comparer les variables disponibles avant de choisir son dataset.

### Distribution des `unknown` très différente entre les deux datasets

| Colonne | Bank | Bank Additional |
|---|---|---|
| `poutcome` | 81.7% unknown | nonexistent (valeur différente) |
| `contact` | 28.8% unknown | 0% unknown |
| `default` | 0% unknown | 20.9% unknown |

- **Leçon :** Les deux datasets n'ont pas été collectés de la même façon. Les `unknown` ne sont pas au même endroit. Il faut traiter chaque dataset indépendamment.

---

## 1.2 Notebook EDA : problèmes structurels

### Conclusion vides ("À compléter...")

- **Observation :** 3 sections de conclusions (A.7, B.8, Conclusions générales) sont restées vides avec la mention "À compléter...".
- **Cause :** Le notebook a été structuré avec des emplacements prévus mais jamais remplis après l'exécution.
- **Problème :** Un notebook final doit contenir les conclusions rédigées. Des sections vides donnent une impression d'inachevé.

### 18 cellules en double dans le notebook

- **Observation :** Les cellules 37 à 54 sont des doublons exacts des cellules 1 à 32. Les imports, chargements, graphiques ont été exécutés deux fois.
- **Cause :** Probablement un copier-coller accidentel lors de l'édition du notebook.
- **Impact :** Le notebook est 2x plus long que nécessaire. Les graphiques apparaissent deux fois. À l'oral, faire défiler des doublons donne une mauvaise impression.
- **Solution :** Supprimer les cellules 37 à 54 avant la remise finale.

### redondance dans les imports

- **Observation :** La cellule d'imports (cell 01) importe à la fois les modules maison (`src.data_loader`, `src.eda`, etc.) et les librairies standards (`pandas`, `numpy`). Mais à l'intérieur de `src/eda.py`, `matplotlib` et `seaborn` sont aussi importés. Double import inutile.
- **Leçon :** Structurer ses imports proprement dès le départ évite la redondance.

---

## 1.3 Enseignements clés de l'EDA

1. **Taux de souscription quasi identique :** 11.70% (Bank) vs 11.27% (Additional) — les deux datasets sont cohérents.
2. **Le dataset Bank a `balance` mais pas d'indicateurs économiques.** Le dataset Additional a les indicateurs économiques mais pas `balance`. Choix à faire.
3. **`duration` est la variable la plus corrélée à la cible** (~0.4) — mais c'est un piège (voir partie modélisation).
4. **Parmi les indicateurs économiques, `nr.employed` est le plus corrélé à la cible** (-0.35) : plus il y a d'employés dans l'économie, moins les gens souscrivent (période de croissance = moins besoin d'épargne ?).
5. **`pdays` et `previous` sont très asymétriques** : la majorité des clients n'ont jamais été contactés avant cette campagne.

---

# PARTIE 2 — Modélisation (PyCaret / AutoML)

## 2.1 Problèmes d'environnement et de compatibilité

### API change : `silent` → `verbose` (PyCaret 3.3.2)

- **Problème :** Le paramètre `silent=True` utilisé dans `src/modeling.py` n'existe pas dans PyCaret 3.3.2. Il a été renommé `verbose=False`.
- **Symptôme :** `TypeError: setup() got an unexpected keyword argument 'silent'`
- **Solution :** Remplacer par `verbose=False`.
- **Leçon :** Les API évoluent vite en ML. Un notebook écrit pour PyCaret 2.x ne fonctionnera pas en 3.x sans adaptation. Toujours vérifier la version installée.

### Python 3.11 obligatoire pour PyCaret 3.x

- **Problème :** PyCaret 3.x nécessite Python 3.11 exactement (pas 3.12, pas 3.13).
- **Impact :** Le choix de l'environnement (conda/venv) n'est pas anodin. Un mauvais Python = installation impossible.
- **Leçon :** Créer l'environnement virtuel AVANT d'installer les dépendances, avec la bonne version de Python.

### Pas de GPU, CPU uniquement

- **Constat :** L'entraînement se fait sur CPU. PyCaret et LightGBM sont bien optimisés CPU, les temps d'entraînement restent raisonnables (< 1s par modèle pour 45k lignes).
- **Leçon :** Pas besoin de GPU pour ce type de projet de classification tabulaire classique.

---

## 2.2 Problèmes de code et de logique

### `compare_models(n_select=5)` retourne une liste, pas un modèle

- **Problème :** Avec `n_select=5`, PyCaret retourne une **liste** de 5 modèles. Les fonctions `plot_model()` et `predict_model()` attendent un modèle unique → plantage.
- **Symptôme :** `ValueError: Estimator [...] does not have the required fit() method`. Message d'erreur trompeur : le vrai problème est qu'on passe une liste, pas que le modèle n'a pas de `fit()`.
- **Solution :** Extraire `best_model = top_models[0]` si le résultat est une liste.
- **Leçon :** Le comportement d'une fonction peut changer radicalement selon la valeur d'un paramètre. Lire la doc.

### `prediction_score` = probabilité de la classe PRÉDITE, pas de la classe positive

- **Problème :** `predict_model()` retourne `prediction_score` = P(classe prédite). Pour un échantillon prédit `no`, c'est P(`no`), pas P(`yes`). Or le ROC-AUC a besoin de P(`yes`).
- **Symptôme :** ROC-AUC = **0.20** (pire que le hasard !) alors que l'Accuracy est à 89%. Incohérence flagrante qui doit alerter.
- **Solution :** Recalculer manuellement :
  ```python
  y_proba = prediction_score.where(prediction_label == 'yes', 1 - prediction_score)
  ```
- **Leçon :** Ne jamais faire confiance aveuglement au nom d'une colonne. Vérifier ce qu'elle contient VRAIMENT.

### Labels string (`'yes'`/`'no'`) vs numériques (`1`/`0`)

- **Problème :** PyCaret conserve les labels string du dataset. `sklearn.metrics` attend du numérique avec `pos_label=1` par défaut.
- **Symptôme :** `ValueError: pos_label=1 is not a valid label. It should be one of ['no', 'yes']`
- **Solution :** Mapper `.map({'yes': 1, 'no': 0})`.
- **Leçon :** PyCaret et sklearn ne parlent pas le même langage. Toujours convertir explicitement.

---

## 2.3 Résultat inattendu : SMOTE inefficace

### L'hypothèse

> *"Si on rééquilibre les classes avec SMOTE, le Recall va augmenter car le modèle verra plus d'exemples de 'oui'."*

### Le test

| Métrique | Sans SMOTE | Avec SMOTE | Différence |
|---|---|---|---|
| Accuracy | 0.8956 | 0.8961 | +0.0005 |
| Precision | 0.6357 | 0.6411 | +0.0054 |
| **Recall** | **0.2524** | **0.2533** | **+0.0009** |
| F1-Score | 0.3613 | 0.3631 | +0.0018 |
| ROC-AUC | 0.8024 | 0.8070 | +0.0046 |

### La conclusion

**SMOTE n'a eu quasiment aucun effet.** Le Recall a gagné 0.09 points de pourcentage — c'est négligeable.

### L'explication

**LightGBM gère déjà nativement le déséquilibre des classes.** Il a des mécanismes internes de pondération qui rendent SMOTE redondant. Les modèles modernes de Gradient Boosting (LightGBM, XGBoost, CatBoost) sont conçus pour être robustes aux classes déséquilibrées.

### La leçon

> **Toujours tester ses hypothèses.** On pensait que SMOTE était LA solution. Ce n'est pas le cas ici. La démarche scientifique (hypothèse → test → analyse → conclusion) est au cœur du Machine Learning. Une solution "évidente" peut ne pas marcher — et c'est une information précieuse en soi.

### Pistes alternatives (non testées)

- Changer le **seuil de décision** : prédire `yes` dès que P(yes) > 0.30 au lieu de 0.50
- Utiliser `class_weight='balanced'` directement dans LightGBM
- Optimiser les hyperparamètres avec `tune_model()` en maximisant le Recall plutôt que l'Accuracy

---

## 2.4 Le piège de `duration`

### Pourquoi c'est un piège

- `duration` (durée de l'appel) est la variable la plus corrélée à la cible
- Un modèle qui l'utilise atteint facilement 90%+ d'Accuracy
- **Mais** dans la vraie vie, on ne connaît pas la durée AVANT de passer l'appel
- Utiliser `duration` = tricher = modèle inutile en production

### Comment on l'a géré

- Suppression systématique de `duration` avant toute modélisation
- Mention explicite dans le notebook et le README
- C'est un classique des projets Bank Marketing : tous les débutants tombent dans ce piège

---

# PARTIE 3 — Leçons transversales

## Pour la présentation orale (slides suggérées)

| Slide | Contenu |
|---|---|
| **Contexte** | Banque portugaise, télémarketing, prédire la souscription |
| **Données** | 2 datasets, 45k/41k lignes, déséquilibre 88/12 |
| **EDA — Découvertes** | Duration = piège, pdays=-1 = jamais contacté, balance négatif = découverts |
| **EDA — Comparaison** | Bank a `balance`, Additional a les indicateurs éco — pas les deux |
| **AutoML — PyCaret** | 14 modèles en une commande, LightGBM gagne |
| **Métriques — Le piège** | Accuracy à 89% ne veut rien dire avec 88% de `no` |
| **Difficultés techniques** | silent→verbose, liste vs modèle, prediction_score inversé, string vs int |
| **SMOTE — L'échec productif** | Hypothèse → test → résultat nul → explication → leçon |
| **Conclusion** | L'AutoML c'est puissant, mais il faut comprendre ce qu'on fait |

## Compétences développées

- Analyse exploratoire de données bancaires réelles
- Utilisation d'une librairie AutoML (PyCaret) de bout en bout
- Compréhension et interprétation des métriques de classification au-delà de l'Accuracy
- Debugging de problèmes d'intégration entre librairies (PyCaret + sklearn)
- Démarche scientifique : hypothèse → test → analyse → conclusion
- Rédaction de notebooks propres et commentés

---

*Document rédigé le 1er juillet 2026 — Projet Bank Marketing — Coaching Data IA*
