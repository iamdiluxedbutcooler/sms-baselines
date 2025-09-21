import os
import json
import joblib
import random
import hashlib
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

class NaiveBayesWrapper:
    def __init__(self, vectorizer, alpha: float = 1.0):
        self.vectorizer = vectorizer
        self.clf = MultinomialNB(alpha=alpha)
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

    def predict_proba(self, texts: List[str]):
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
        data = joblib.load(path)
        obj = cls(vectorizer=data["vectorizer"], alpha=1.0)
        obj.clf = data["clf"]
        obj.labeler = data["labeler"]
        obj.fitted = True
        return obj

def evaluate_cv(texts: List[str], labels: List[str], vectorizer, alpha: float = 1.0, cv: int = 5) -> Dict:
    X = vectorizer.fit_transform(texts)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    clf = MultinomialNB(alpha=alpha)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return {"cv_scores": scores.tolist(), "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores))}

def train_and_evaluate_from_splits(train_texts: List[str], train_labels: List[str], test_texts: List[str], test_labels: List[str], output_dir: str, seed: int = 42, cv: Optional[int] = None, vectorizer=None, alpha: float = 1.0) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    cv_res = None
    if cv is not None and cv >= 2:
        try:
            cv_res = evaluate_cv(train_texts, train_labels, vectorizer=vectorizer, alpha=alpha, cv=cv)
        except Exception:
            cv_res = None
    model = NaiveBayesWrapper(vectorizer=vectorizer, alpha=alpha)
    model.fit(train_texts, train_labels)
    model_path = os.path.join(output_dir, "model.joblib")
    model.save(model_path)
    preds = model.predict(test_texts)
    probs = None
    try:
        probs = model.predict_proba(test_texts)
    except Exception:
        probs = None
    acc = float(accuracy_score(test_labels, preds))
    report = classification_report(test_labels, preds, output_dict=True)
    try:
        labels_order = list(model.labeler.classes_)
        cm = confusion_matrix(test_labels, preds, labels=labels_order).tolist()
    except Exception:
        cm = confusion_matrix(test_labels, preds).tolist()
        labels_order = []
    results = {"accuracy": acc, "classification_report": report, "confusion_matrix": cm, "model_path": model_path, "cv": cv_res, "label_names": labels_order}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    rows = []
    for i, (t, true, pred) in enumerate(zip(test_texts, test_labels, preds)):
        row = {"id": i, "text": t, "true_label": true, "pred_label": pred}
        rows.append(row)
    df = pd.DataFrame(rows)
    if probs is not None:
        try:
            label_names = list(model.labeler.classes_)
            for col_idx, label_name in enumerate(label_names):
                df[f"prob_{label_name}"] = probs[:, col_idx]
        except Exception:
            pass
    df.to_csv(os.path.join(output_dir, "results_detailed.csv"), index=False)
    return results

class URLFilterStub:
    def __init__(self):
        pass
    def analyze(self, url: str):
        return {"url": url, "blacklist": False, "features": {"age_domain": None, "has_at": None, "has_hyphen": None, "dots": None}, "decision": "unknown"}

def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def append_summary_row(summary_csv_path: str, row: Dict):
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(summary_csv_path):
        df_existing = pd.read_csv(summary_csv_path)
        df_out = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_out = df
    df_out.to_csv(summary_csv_path, index=False)
