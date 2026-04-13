"""
data_loader.py - Telechargement et nettoyage du dataset SMS Spam Collection
"""

import requests
import pandas as pd
import os
import re

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
OUTPUT_FILE = "data/sms_clean.csv"


def fetch_dataset(url: str) -> pd.DataFrame:
    print(f"[data] Connexion a : {url}")
    r = requests.get(url, timeout=15)
    r.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(r.text), sep="\t", header=None, names=["label", "text"])
    print(f"[data] {len(df)} SMS telecharges")
    return df


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # supprime les URLs
    text = re.sub(r"\d+", "", text)              # supprime les chiffres
    text = re.sub(r"[^\w\s]", "", text)          # supprime la ponctuation
    text = re.sub(r"\s+", " ", text).strip()     # espaces multiples
    return text


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_num"] = (df["label"] == "spam").astype(int)  # spam=1, ham=0
    df["text_clean"] = df["text"].apply(clean_text)
    df["text_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))

    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"[data] {before - len(df)} doublons supprimes -> {len(df)} SMS restants")
    print(f"[data] Repartition : {df['label'].value_counts().to_dict()}")
    return df


def run():
    os.makedirs("data", exist_ok=True)
    df = fetch_dataset(DATA_URL)
    df = prepare(df)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[data] Sauvegarde dans '{OUTPUT_FILE}'\n")
    return df


if __name__ == "__main__":
    run()
