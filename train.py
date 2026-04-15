"""
train.py - Entrainement du classificateur spam avec TF-IDF + Naive Bayes
"""

import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)


def load_data(path: str = "data/sms_clean.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("Lance d'abord : python data_loader.py")
    return pd.read_csv(path)


def build_pipeline(df: pd.DataFrame):
    df = df.dropna(subset=["text_clean"])
    X = df["text_clean"].astype(str)
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[train] Train : {len(X_train)} SMS | Test : {len(X_test)} SMS")

    # TF-IDF : transforme le texte en vecteurs numeriques
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=2,
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    return vectorizer, X_train_vec, X_test_vec, y_train, y_test


def evaluate(name: str, model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    return acc, auc, y_pred


def run():
    df = load_data()
    vectorizer, X_train_vec, X_test_vec, y_train, y_test = build_pipeline(df)

    # Modele 1 : Naive Bayes (classique pour le NLP)
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_vec, y_train)
    acc_nb, auc_nb, preds_nb = evaluate("Naive Bayes", nb, X_test_vec, y_test)

    # Modele 2 : Regression Logistique (souvent plus precis)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_vec, y_train)
    acc_lr, auc_lr, preds_lr = evaluate("Logistic Regression", lr, X_test_vec, y_test)

    # Sauvegarde du meilleur modele
    best_model = lr if acc_lr >= acc_nb else nb
    best_name  = "LogisticRegression" if acc_lr >= acc_nb else "NaiveBayes"
    print(f"\n[train] Meilleur modele : {best_name}")

    os.makedirs("model", exist_ok=True)
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("model/classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("[train] Modele sauvegarde dans model/")

    # Retourner les predictions pour les visualisations
    return y_test, preds_lr, best_model, vectorizer


if __name__ == "__main__":
    run()
