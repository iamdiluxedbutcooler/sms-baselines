import re
import os
import json
import subprocess
import datetime
from typing import List, Dict
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from models.shared import train_and_evaluate_from_splits, URLFilterStub, set_seed, compute_sha256, append_summary_row

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

def _git_short_hash():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "nogit"

def run_bl_nlp_01(train_texts: List[str], train_labels: List[str], test_texts: List[str], test_labels: List[str], outdir: str, seed: int, config: Dict) -> Dict:
    os.makedirs(outdir, exist_ok=True)
    set_seed(seed)
    
    start_ts = datetime.datetime.utcnow().isoformat() + "Z"
    meta = {
        "baseline_id": config.get("baseline_id"), 
        "paper_id": config.get("paper_id"), 
        "seed": seed, 
        "timestamp_utc": start_ts, 
        "git_short_hash": _git_short_hash(), 
        "assumptions": ["URL behavioral modules are stubbed (URLFilterStub)."]
    }
    
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Original train labels: {set(train_labels)}")
    print(f"Original test labels: {set(test_labels)}")
    
    train_pre = [" ".join(_tokenize_and_stem(t)) for t in train_texts]
    test_pre = [" ".join(_tokenize_and_stem(t)) for t in test_texts]
    
    ngram_range = tuple(config.get("features", {}).get("ngram_range", [1,2]))
    max_features = config.get("features", {}).get("max_features", 20000)
    alpha = config.get("model", {}).get("alpha", 1.0)
    
    vec = TfidfVectorizer(analyzer="word", lowercase=False, ngram_range=ngram_range, max_features=max_features)
    
    cv_on_train = config.get("training", {}).get("cv_on_train", False)
    cv_folds = config.get("training", {}).get("cv_folds", 5)
    cv_param = cv_folds if cv_on_train else None
    
    results = train_and_evaluate_from_splits(
        train_pre, train_labels, test_pre, test_labels, 
        outdir, seed=seed, cv=cv_param, vectorizer=vec, alpha=alpha
    )
    
    url_filter = URLFilterStub()
    url_module_results = {"url_checks": []}
    results["url_module"] = url_module_results
    
    print(f"Accuracy: {results.get('accuracy')}")
    
    summary_path = os.path.join("experiments", config.get("baseline_id") or "bl_nlp_01", "summary.csv")
    run_id = f"{config.get('baseline_id')}_seed{seed}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    summary_row = {
        "run_id": run_id, 
        "seed": seed, 
        "accuracy": results.get("accuracy"), 
        "outdir": outdir, 
        "timestamp": start_ts
    }
    
    try:
        append_summary_row(summary_path, summary_row)
    except Exception as e:
        import csv
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow({k: ("" if v is None else v) for k, v in summary_row.items()})
    
    return results