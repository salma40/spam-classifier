"""
analysis.py - Visualisations : distribution, matrice de confusion, mots cles spam
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

COLORS  = {"spam": "#E63946", "ham": "#457B9D"}
OUTPUT  = "charts"


def save(fig, filename):
    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[chart] {path}")
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribution des SMS : Spam vs Ham", fontsize=15, fontweight="bold")

    # Barplot
    colors = [COLORS["ham"], COLORS["spam"]]
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 20,
                     f"{val}\n({val/len(df)*100:.1f}%)",
                     ha="center", fontweight="bold")
    axes[0].set_ylabel("Nombre de SMS")
    axes[0].set_title("Nombre par classe")
    axes[0].grid(False)

    # Distribution longueur des SMS
    for label, color in COLORS.items():
        subset = df[df["label"] == label]["text_length"]
        axes[1].hist(subset, bins=40, alpha=0.6, color=color, label=label, edgecolor="white")
    axes[1].set_xlabel("Longueur du SMS (caracteres)")
    axes[1].set_ylabel("Frequence")
    axes[1].set_title("Longueur des SMS par classe")
    axes[1].legend()

    fig.tight_layout()
    save(fig, "1_distribution.png")


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Ham", "Spam"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax, cbar=False,
                annot_kws={"size": 16, "weight": "bold"})

    ax.set_xlabel("Prediction", fontsize=13)
    ax.set_ylabel("Valeur reelle", fontsize=13)
    ax.set_title("Matrice de confusion\n(Logistic Regression)", fontsize=14, fontweight="bold")

    total = cm.sum()
    ax.text(0.5, -0.12,
            f"Accuracy : {(cm[0,0]+cm[1,1])/total*100:.2f}%  |  "
            f"Faux positifs : {cm[0,1]}  |  Faux negatifs : {cm[1,0]}",
            transform=ax.transAxes, ha="center", fontsize=10, color="#555")

    fig.tight_layout()
    save(fig, "2_confusion_matrix.png")


def plot_top_spam_words(vectorizer: TfidfVectorizer, df: pd.DataFrame):
    """
    Calcule les mots avec le score TF-IDF moyen le plus eleve
    dans les SMS spam vs ham.
    """
    spam_texts = df[df["label"] == "spam"]["text_clean"]
    ham_texts  = df[df["label"] == "ham"]["text_clean"]

    tfidf_matrix_spam = vectorizer.transform(spam_texts)
    tfidf_matrix_ham  = vectorizer.transform(ham_texts)

    feature_names = vectorizer.get_feature_names_out()
    mean_spam = tfidf_matrix_spam.mean(axis=0).A1
    mean_ham  = tfidf_matrix_ham.mean(axis=0).A1

    # Top 15 mots les plus distinctifs pour chaque classe
    top_spam_idx = mean_spam.argsort()[-15:][::-1]
    top_ham_idx  = mean_ham.argsort()[-15:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Mots les plus caracteristiques par classe (TF-IDF)",
                 fontsize=14, fontweight="bold")

    for ax, idx, label, color in [
        (axes[0], top_spam_idx, "SPAM", COLORS["spam"]),
        (axes[1], top_ham_idx,  "HAM",  COLORS["ham"]),
    ]:
        words  = [feature_names[i] for i in idx]
        scores = [mean_spam[i] if label == "SPAM" else mean_ham[i] for i in idx]
        words_sorted  = words[::-1]
        scores_sorted = scores[::-1]

        ax.barh(words_sorted, scores_sorted, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(f"Top mots - {label}", fontweight="bold", color=color)
        ax.set_xlabel("Score TF-IDF moyen")

    fig.tight_layout()
    save(fig, "3_top_words.png")


def plot_sms_length_vs_label(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Longueur et nombre de mots selon la classe",
                 fontsize=14, fontweight="bold")

    # Boxplot longueur
    data_by_label = [df[df["label"] == l]["text_length"].values for l in ["ham", "spam"]]
    bp = axes[0].boxplot(data_by_label, labels=["Ham", "Spam"], patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], [COLORS["ham"], COLORS["spam"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Longueur (caracteres)")
    axes[0].set_title("Longueur des SMS")
    axes[0].grid(axis="x", alpha=0)

    # Boxplot nombre de mots
    data_words = [df[df["label"] == l]["word_count"].values for l in ["ham", "spam"]]
    bp2 = axes[1].boxplot(data_words, labels=["Ham", "Spam"], patch_artist=True,
                          medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp2["boxes"], [COLORS["ham"], COLORS["spam"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Nombre de mots")
    axes[1].set_title("Nombre de mots par SMS")
    axes[1].grid(axis="x", alpha=0)

    fig.tight_layout()
    save(fig, "4_length_analysis.png")


def run():
    print("[analysis] Chargement des donnees...")
    df = pd.read_csv("data/sms_clean.csv")

    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model/classifier.pkl", "rb") as f:
        model = pickle.load(f)

    # Reconstruire y_test et y_pred pour la matrice de confusion
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer as TV
    df = df.dropna(subset=["text_clean"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"].astype(str), df["label_num"], test_size=0.2,
        random_state=42, stratify=df["label_num"]
    )
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    print("[analysis] Generation des graphiques...")
    plot_class_distribution(df)
    plot_confusion_matrix(y_test, y_pred)
    plot_top_spam_words(vectorizer, df)
    plot_sms_length_vs_label(df)

    print(f"\n[analysis] 4 graphiques sauvegardes dans '{OUTPUT}/'")


if __name__ == "__main__":
    run()
