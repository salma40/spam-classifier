# SMS Spam Classifier

Projet Python de classification de SMS spam avec NLP — TF-IDF, Naive Bayes et Regression Logistique.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-97.97%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Objectif

Construire un modele capable de distinguer automatiquement les SMS spam des SMS normaux (ham), en partant du texte brut jusqu'a un classificateur evalue et pret a l'emploi.

---

## Resultats

| Modele | Accuracy | ROC-AUC |
|--------|----------|---------|
| Naive Bayes | 97.97% | 0.9874 |
| Logistic Regression | 94.97% | 0.9917 |

---

## Structure du projet

```
spam-classifier/
│
├── data_loader.py    # Telechargement et nettoyage des donnees
├── train.py          # Entrainement TF-IDF + Naive Bayes / Logistic Regression
├── analysis.py       # Visualisations
├── predict.py        # Tester le modele sur un SMS personnalise
├── requirements.txt
│
├── data/
│   └── sms_clean.csv
│
├── model/
│   ├── vectorizer.pkl
│   └── classifier.pkl
│
└── charts/
    ├── 1_distribution.png
    ├── 2_confusion_matrix.png
    ├── 3_top_words.png
    └── 4_length_analysis.png
```

---

## Tutoriel complet

### Etape 1 — Verifier que Python est installe

Ouvre un terminal et lance :

```bash
python --version
```

Si tu vois `Python 3.x.x`, c'est bon. Sinon, telecharge Python sur https://www.python.org/downloads/

---

### Etape 2 — Recuperer le projet

```bash
git clone https://github.com/salma40/spam-classifier.git
cd spam-classifier
```

Ou telecharge le ZIP depuis GitHub (bouton vert "Code" > "Download ZIP").

---

### Etape 3 — Installer les dependances

```bash
pip install -r requirements.txt
```

Cela installe : scikit-learn, pandas, matplotlib, seaborn, requests, numpy.

---

### Etape 4 — Telecharger et nettoyer les donnees

```bash
python data_loader.py
```

Ce que tu vas voir :

```
[data] Connexion a : https://raw.githubusercontent.com/...
[data] 5572 SMS telecharges
[data] 403 doublons supprimes -> 5169 SMS restants
[data] Repartition : {'ham': 4516, 'spam': 653}
[data] Sauvegarde dans 'data/sms_clean.csv'
```

---

### Etape 5 — Entrainer le modele

```bash
python train.py
```

Ce que tu vas voir :

```
[train] Train : 4132 SMS | Test : 1034 SMS

--- Naive Bayes ---
Accuracy : 0.9797
ROC-AUC  : 0.9874

--- Logistic Regression ---
Accuracy : 0.9497
ROC-AUC  : 0.9917

[train] Meilleur modele : NaiveBayes
[train] Modele sauvegarde dans model/
```

Deux fichiers sont crees dans `model/` : le vectorizer et le classificateur.

---

### Etape 6 — Generer les visualisations

```bash
python analysis.py
```

4 graphiques sont sauvegardes dans `charts/`.

---

### Etape 7 — Tester sur tes propres SMS

```bash
# Avec les exemples integres
python predict.py

# Avec ton propre texte
python predict.py "Congratulations! You have won a free prize. Call now."
```

Resultat attendu :

```
[SPAM]   (confiance : 100.0%) -> "Congratulations! You have won a free prize. Call now."
```

Tu peux tester autant de SMS que tu veux en les passant en argument.

---

### Problemes courants

**"FileNotFoundError: data/sms_clean.csv"**
Lance d'abord `python data_loader.py`.

**"FileNotFoundError: model/classifier.pkl"**
Lance d'abord `python train.py`.

**"ModuleNotFoundError"**
Lance `pip install -r requirements.txt`.

---

## Pipeline NLP

```
SMS brut
   -> Nettoyage (minuscules, ponctuation, chiffres supprimes)
   -> TF-IDF Vectorization (5000 features, unigrams + bigrams)
   -> Naive Bayes / Logistic Regression
   -> Prediction : SPAM ou HAM
```

TF-IDF (Term Frequency - Inverse Document Frequency) transforme chaque SMS en vecteur numerique en ponderant les mots selon leur frequence dans le document et leur rarete dans l'ensemble du corpus.

---

## Statistiques cles

- 5169 SMS apres nettoyage des doublons
- 87.4% ham / 12.6% spam
- Les SMS spam sont en moyenne 2x plus longs que les SMS normaux
- Mots les plus predictifs du spam : "free", "call", "txt", "claim", "prize"

---

## Source des donnees

SMS Spam Collection Dataset — UCI Machine Learning Repository
Via : https://github.com/justmarkham/pycon-2016-tutorial

---

## Licence

MIT
