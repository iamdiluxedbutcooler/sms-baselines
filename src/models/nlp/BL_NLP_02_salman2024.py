import os
import re
import json
import subprocess
import datetime
from typing import List, Dict
from models.shared import set_seed, append_summary_row, compute_sha256, URLFilterStub
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

def _git_short_hash():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None

def _safe_import(name):
    try:
        mod = __import__(name)
        return mod
    except Exception:
        return None

_langdetect = _safe_import("langdetect")
_googletrans = _safe_import("googletrans")
_pytesseract = _safe_import("pytesseract")
_nltk = _safe_import("nltk")
if _nltk:
    try:
        from nltk.corpus import stopwords
        stopwords.words("english")
    except Exception:
        try:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
        except Exception:
            stopwords = set()
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set()
else:
    stop_words = set()

def _is_english_langdetect(text: str) -> bool:
    if not _langdetect:
        return False
    try:
        from langdetect import detect
        lang = detect(text)
        return lang == "en"
    except Exception:
        return False

def _is_english_googletrans(text: str) -> bool:
    if not _googletrans:
        return False
    try:
        from googletrans import Translator
        tr = Translator()
        det = tr.detect(text)
        lang = det.lang if hasattr(det, "lang") else None
        return lang == "en"
    except Exception:
        return False

def _ocr_from_placeholder(text: str) -> str:
    if not _pytesseract:
        return ""
    try:
        import base64, io
        if text.strip().startswith("data:image") or text.strip().startswith("/9j/"):
            header, b64 = text.split(",", 1) if "," in text else ("", text)
            raw = base64.b64decode(b64)
            from PIL import Image
            img = Image.open(io.BytesIO(raw))
            import pytesseract
            return pytesseract.image_to_string(img)
    except Exception:
        return ""
    return ""

def _clean_and_tokenize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text
    ocr_text = _ocr_from_placeholder(s)
    if ocr_text:
        s = s + " " + ocr_text
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if stop_words:
        toks = [w for w in s.split() if w not in stop_words]
    else:
        toks = s.split()
    return " ".join(toks)

def _two_pass_language_filter(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        if not isinstance(t, str):
            continue
        if _langdetect:
            try:
                if _is_english_langdetect(t):
                    out.append(t)
                    continue
            except Exception:
                pass
        if _googletrans:
            try:
                if _is_english_googletrans(t):
                    out.append(t)
                    continue
            except Exception:
                pass
        if "data:image" in t or t.strip().startswith("/9j/"):
            o = _ocr_from_placeholder(t)
            if o and (_is_english_langdetect(o) or _is_english_googletrans(o)):
                out.append(o)
                continue
    return out

def run_bl_nlp_02(train_texts, train_labels, test_texts, test_labels, outdir, seed, config):
    import os, json, datetime, subprocess
    from models.shared import set_seed, append_summary_row
    set_seed(seed)
    os.makedirs(outdir, exist_ok=True)
    start_ts = datetime.datetime.utcnow().isoformat() + "Z"
    meta = {"baseline_id": config.get("baseline_id"), "paper_id": config.get("paper_id"), "seed": seed, "timestamp_utc": start_ts}
    try:
        meta["git_short_hash"] = subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        meta["git_short_hash"] = "nogit"
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    pre_train = [_clean_and_tokenize(t) for t in train_texts]
    pre_test = [_clean_and_tokenize(t) for t in test_texts]
    
    results = {"accuracy": None}
    transformer_ok = False
    
    try:
        from sklearn.preprocessing import LabelEncoder
        import pandas as _pd
        
        print(f"Original train labels: {set(train_labels)}")
        print(f"Original test labels: {set(test_labels)}")
        
        train_labels_str = [str(x) for x in train_labels]
        test_labels_str = [str(x) for x in test_labels]
        
        all_labels = list(set(train_labels_str + test_labels_str))
        le = LabelEncoder()
        le.fit(all_labels)
        
        y_train_enc = le.transform(train_labels_str)
        y_test_enc = le.transform(test_labels_str)
        
        num_labels = len(le.classes_)
        print(f"Number of unique labels: {num_labels}")
        print(f"Label classes: {le.classes_}")
        print(f"Train label range: {min(y_train_enc)} to {max(y_train_enc)}")
        print(f"Test label range: {min(y_test_enc)} to {max(y_test_enc)}")
        
        train_df = _pd.DataFrame({"text": pre_train, "labels": list(y_train_enc)})
        test_df = _pd.DataFrame({"text": pre_test, "labels": list(y_test_enc)})
        
        args = {
            "reprocess_input_data": True, 
            "overwrite_output_dir": True, 
            "silent": True, 
            "use_multiprocessing": False,
            "num_train_epochs": 1,
            "train_batch_size": 8,
            "eval_batch_size": 8,
            "max_seq_length": 128,
        }
        
        model_name = config.get("features", {}).get("model_name", "roberta-base")
        
        from simpletransformers.classification import ClassificationModel
        print(f"Creating model with {num_labels} labels")
        model = ClassificationModel("roberta", model_name, num_labels=num_labels, use_cuda=False, args=args)
        
        print("Training model...")
        model.train_model(train_df)
        
        print("Making predictions...")
        preds_raw, raw_outputs = model.predict(test_df["text"].tolist())
        
        try:
            preds_mapped = []
            for p in preds_raw:
                try:
                    idx = int(p)
                    if 0 <= idx < len(le.classes_):
                        preds_mapped.append(str(le.inverse_transform([idx])[0]))
                    else:
                        print(f"Warning: prediction {idx} out of range, using first class")
                        preds_mapped.append(str(le.classes_[0]))
                except Exception as e:
                    print(f"Error mapping prediction {p}: {e}")
                    preds_mapped.append(str(le.classes_[0]))
            preds = preds_mapped
        except Exception as e:
            print(f"Error in prediction mapping: {e}")
            preds = [str(p) for p in preds_raw]
        
        try:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            acc = float(accuracy_score(test_labels_str, preds))
            cr = classification_report(test_labels_str, preds, output_dict=True, zero_division=0)
            cm = confusion_matrix(test_labels_str, preds).tolist()
            print(f"Accuracy: {acc}")
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            acc = None
            cr = {}
            cm = []
        
        results = {"accuracy": acc, "classification_report": cr, "confusion_matrix": cm, "model_path": os.path.join(outdir, "model")}
        
        try:
            model_save_path = os.path.join(outdir, "model")
            model.save_model(model_save_path)
            print(f"Model saved to {model_save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
        
        import pandas as _pd2
        df_det = _pd2.DataFrame({
            "text": test_df["text"].tolist(), 
            "true_label": test_labels_str, 
            "pred_label": preds
        })
        
        try:
            import numpy as _np
            if raw_outputs is not None:
                try:
                    probs = _np.max(_np.asarray(raw_outputs), axis=1)
                    df_det["prob_chosen"] = probs
                except Exception as e:
                    print(f"Error adding probabilities: {e}")
        except Exception as e:
            print(f"Error processing raw outputs: {e}")
        
        df_det.to_csv(os.path.join(outdir, "results_detailed.csv"), index=False)
        transformer_ok = True
        print("RoBERTa training completed successfully")
        
    except Exception as e:
        print(f"RoBERTa training failed: {e}")
        results["error"] = "roberta training failed"
        results["exception"] = str(e)
        
        print("Falling back to logistic regression...")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder as _LE
            import joblib
            import pandas as _pd3
            
            y_train = [str(x) for x in train_labels]
            y_test = [str(x) for x in test_labels]
            
            le2 = _LE()
            y_train_enc2 = le2.fit_transform(y_train)
            
            vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
            X_train = vec.fit_transform(pre_train)
            X_test = vec.transform(pre_test)
            
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train_enc2)
            
            pred_enc = clf.predict(X_test)
            preds = le2.inverse_transform(pred_enc)
            
            try:
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                acc = float(accuracy_score(y_test, preds))
                cr = classification_report(y_test, preds, output_dict=True, zero_division=0)
                cm = confusion_matrix(y_test, preds).tolist()
            except Exception:
                acc = None
                cr = {}
                cm = []
            
            results = {"accuracy": acc, "classification_report": cr, "confusion_matrix": cm, "model_path": os.path.join(outdir, "fallback_logreg.joblib")}
            
            try:
                joblib.dump({"vec": vec, "clf": clf, "le": le2}, os.path.join(outdir, "fallback_logreg.joblib"))
            except Exception:
                pass
            
            df_det2 = _pd3.DataFrame({"text": pre_test, "true_label": y_test, "pred_label": preds})
            df_det2.to_csv(os.path.join(outdir, "results_detailed.csv"), index=False)
            print("Fallback logistic regression completed")
            
        except Exception as fallback_e:
            print(f"Fallback training also failed: {fallback_e}")
            results["fallback_error"] = str(fallback_e)
    
    try:
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    summary_path = os.path.join("experiments", config.get("baseline_id") or "bl_nlp_02", "summary.csv")
    run_id = f"{config.get('baseline_id')}_seed{seed}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    summary_row = {"run_id": run_id, "seed": seed, "accuracy": results.get("accuracy"), "outdir": outdir, "timestamp": start_ts}
    
    try:
        append_summary_row(summary_path, summary_row)
    except Exception:
        import csv
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow({k: ("" if v is None else v) for k, v in summary_row.items()})
    
    return results
