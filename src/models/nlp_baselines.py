import re
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ps = PorterStemmer()

def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\d{3,}", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize_and_stem(text: str) -> List[str]:
    cleaned = _clean_text(text)
    tokens = cleaned.split()
    stems = [ps.stem(t) for t in tokens if t]
    return stems

class NaiveBayesSmishing:
    def __init__(self, ngram_range=(1,2), max_features=20000, alpha=1.0):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer(tokenizer=_tokenize_and_stem, lowercase=False, ngram_range=self.ngram_range, max_features=self.max_features)
        self.clf = MultinomialNB(alpha=self.alpha)
        self.labeler = LabelEncoder()
        self.fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        y = self.labeler.fit_transform(labels)
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, y)
        self.fitted = True
        return self

    def predict(self, texts: List[str]) -> List[str]:
        X = self.vectorizer.transform(texts)
        preds = self.clf.predict(X)
        return self.labeler.inverse_transform(preds)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)
        else:
            probs = self.clf.predict_log_proba(X)
            return np.exp(probs)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "clf": self.clf, "labeler": self.labeler}, path)

    @classmethod
    def load(cls, path: str):
        obj = cls()
        data = joblib.load(path)
        obj.vectorizer = data["vectorizer"]
        obj.clf = data["clf"]
        obj.labeler = data["labeler"]
        obj.fitted = True
        return obj

def evaluate_cv(texts: List[str], labels: List[str], ngram_range=(1,2), max_features=20000, alpha=1.0, cv=5) -> Dict:
    vec = TfidfVectorizer(tokenizer=_tokenize_and_stem, lowercase=False, ngram_range=ngram_range, max_features=max_features)
    X = vec.fit_transform(texts)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    clf = MultinomialNB(alpha=alpha)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return {"cv_scores": scores.tolist(), "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores))}

def train_and_evaluate_nb(texts: List[str], labels: List[str], output_dir: str, test_size=0.2, random_state=42, cv=5, ngram_range=(1,2), max_features=20000, alpha=1.0) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels)
    cv_res = evaluate_cv(X_train, y_train, ngram_range=ngram_range, max_features=max_features, alpha=alpha, cv=cv)
    model = NaiveBayesSmishing(ngram_range=ngram_range, max_features=max_features, alpha=alpha)
    model.fit(X_train, y_train)
    model_path = os.path.join(output_dir, "naive_bayes_smishing.joblib")
    model.save(model_path)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()
    results = {"accuracy": acc, "classification_report": report, "confusion_matrix": cm, "model_path": model_path, "cv": cv_res}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    df = pd.DataFrame({"text": X_test, "true_label": y_test, "pred_label": preds})
    try:
        prob_cols = []
        if probs is not None:
            for i in range(probs.shape[1]):
                col = f"prob_{i}"
                df[col] = probs[:, i]
                prob_cols.append(col)
    except Exception:
        pass
    df.to_csv(os.path.join(output_dir, "results_detailed.csv"), index=False)
    return results

class URLFilterStub:
    def __init__(self):
        pass
    def analyze(self, url: str) -> Dict:
        return {"url": url, "blacklist": False, "features": {"age_domain": None, "has_at": None, "has_hyphen": None, "dots": None}, "decision": "unknown"}
