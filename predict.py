"""
predict.py - Tester le modele sur un SMS personnalise
Usage : python predict.py "Congratulations! You've won a free iPhone. Click here now!"
"""

import sys
import pickle
import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(sms: str) -> dict:
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model/classifier.pkl", "rb") as f:
        model = pickle.load(f)

    cleaned = clean_text(sms)
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    return {
        "sms":        sms,
        "prediction": "SPAM" if label == 1 else "HAM",
        "confidence": f"{max(proba) * 100:.1f}%",
        "proba_spam": f"{proba[1] * 100:.1f}%",
        "proba_ham":  f"{proba[0] * 100:.1f}%",
    }


if __name__ == "__main__":
    examples = [
        "Congratulations! You've won a free iPhone. Click here now to claim your prize!",
        "Hey, are we still on for dinner tonight?",
        "URGENT: Your account has been suspended. Call 0800 free now to reactivate.",
        "Can you pick up some milk on your way home?",
    ]

    sms_list = sys.argv[1:] if len(sys.argv) > 1 else examples

    print("\n--- Classificateur Spam SMS ---\n")
    for sms in sms_list:
        result = predict(sms)
        label_display = f"[{result['prediction']}]"
        print(f"{label_display:<8} (confiance : {result['confidence']}) -> \"{result['sms'][:70]}\"")
