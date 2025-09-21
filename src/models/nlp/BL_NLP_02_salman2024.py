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
    from models.shared import set_seed, append_summary_row, compute_sha256
    set_seed(seed)
    start_ts = datetime.datetime.utcnow().isoformat() + "Z"
    meta = {"baseline_id": config.get("baseline_id"), "paper_id": config.get("paper_id"), "seed": seed, "timestamp_utc": start_ts, "git_short_hash": None, "assumptions": []}
    try:
        meta["git_short_hash"]=subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        meta["git_short_hash"]="nogit"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir,"run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    results = {}
    try:
        pre_train = [_clean_and_tokenize(t) for t in train_texts]
        pre_test = [_clean_and_tokenize(t) for t in test_texts]
        model_cfg = config.get("model", {})
        model_type = model_cfg.get("type", "roberta")
        if model_type.lower()=="roberta":
            try:
                from simpletransformers.classification import ClassificationModel
                import pandas as _pd
                train_df = _pd.DataFrame({"text": pre_train, "labels": train_labels})
                test_df = _pd.DataFrame({"text": pre_test, "labels": test_labels})
                args = {"reprocess_input_data": True, "overwrite_output_dir": True, "silent": True}
                model_name = config.get("features", {}).get("model_name", "roberta-base")
                use_cuda = False
                model = ClassificationModel("roberta", model_name, use_cuda=use_cuda, args=args)
                model.train_model(train_df)
                preds, raw_outputs = model.predict(test_df["text"].tolist())
                try:
                    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                    acc = float(accuracy_score(test_df["labels"].tolist(), preds))
                    cr = classification_report(test_df["labels"].tolist(), preds, output_dict=True)
                    cm = confusion_matrix(test_df["labels"].tolist(), preds).tolist()
                except Exception:
                    acc = None
                    cr = {}
                    cm = []
                results = {"accuracy": acc, "classification_report": cr, "confusion_matrix": cm, "model_path": os.path.join(outdir,"model")}
                try:
                    model.save_model(os.path.join(outdir,"model"))
                except Exception:
                    pass
                import pandas as _pd2
                df_det = _pd2.DataFrame({"text": test_df["text"].tolist(), "true_label": test_df["labels"].tolist(), "pred_label": preds})
                try:
                    import numpy as _np
                    proba = _np.max(_np.asarray(raw_outputs), axis=1) if raw_outputs is not None else None
                    if proba is not None:
                        df_det["prob_chosen"] = proba
                except Exception:
                    pass
                df_det.to_csv(os.path.join(outdir,"results_detailed.csv"), index=False)
            except Exception as e:
                results = {"accuracy": None, "error": "roberta training failed", "exception": str(e)}
                meta["assumptions"].append("roberta training failed or simpletransformers missing")
                with open(os.path.join(outdir,"results.json"), "w") as f:
                    json.dump(results, f, indent=2)
        else:
            results = {"accuracy": None, "error": "unsupported model type"}
            with open(os.path.join(outdir,"results.json"), "w") as f:
                json.dump(results, f, indent=2)
    except Exception as e:
        import traceback
        results = {"accuracy": None, "error": "unexpected exception in wrapper", "exception": str(e), "traceback": traceback.format_exc()}
        with open(os.path.join(outdir,"results.json"), "w") as f:
            json.dump(results, f, indent=2)
    if "accuracy" not in results:
        results["accuracy"] = None
    with open(os.path.join(outdir,"results.json"), "w") as f:
        json.dump(results, f, indent=2)
    baseline_id = config.get("baseline_id") or "bl_nlp_02"
    summary_path = os.path.join("experiments", baseline_id, "summary.csv")
    run_id = f"{baseline_id}_seed{seed}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    summary_row = {"run_id": run_id, "seed": seed, "accuracy": results.get("accuracy"), "outdir": outdir, "timestamp": start_ts}
    try:
        append_summary_row(summary_path, summary_row)
    except Exception:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        import csv
        with open(summary_path, "w", newline="") as f:
            writer=csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow({k: ("" if v is None else v) for k,v in summary_row.items()})
    return results

