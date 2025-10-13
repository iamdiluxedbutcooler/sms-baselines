import os
import json
import subprocess
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict
from models.shared import set_seed, append_summary_row
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def _git_short_hash():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "nogit"

def create_zero_shot_prompt(text: str) -> str:
    return f"""You are an SMS spam detection system. Classify the following SMS message as exactly one of: 'spam', 'ham', or 'smishing'.

Definitions:
- 'spam': Unwanted promotional or advertisement messages
- 'ham': Legitimate, normal messages
- 'smishing': Phishing attacks via SMS trying to steal personal information or credentials

SMS Message: {text}

Classification:"""

def extract_classification(response: str) -> str:
    response = response.strip().lower()
    if 'smishing' in response:
        return 'smishing'
    elif 'spam' in response:
        return 'spam'
    elif 'ham' in response:
        return 'ham'
    else:
        return 'ham'

def run_bl_llm_01_C(train_texts: List[str], train_labels: List[str], 
                    test_texts: List[str], test_labels: List[str], 
                    outdir: str, seed: int, config: Dict) -> Dict:
    
    os.makedirs(outdir, exist_ok=True)
    set_seed(seed)
    
    start_ts = datetime.datetime.utcnow().isoformat() + "Z"
    meta = {
        "baseline_id": config.get("baseline_id"), 
        "paper_id": config.get("paper_id"), 
        "seed": seed, 
        "timestamp_utc": start_ts,
        "git_short_hash": _git_short_hash(),
        "paper_title": "Zero-shot GPT-4o",
        "paper_year": 2025,
        "method": "zero_shot"
    }
    
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Original train labels: {set(train_labels)}")
    print(f"Original test labels: {set(test_labels)}")
    
    train_labels_str = [str(x).lower() for x in train_labels]
    test_labels_str = [str(x).lower() for x in test_labels]
    
    results = {"accuracy": None}
    
    try:
        from openai import OpenAI
        import getpass
        
        print("Setting up OpenAI authentication...")
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        client = OpenAI(api_key=api_key)
        
        print("Running zero-shot inference on test set...")
        predictions = []
        
        for i, text in enumerate(test_texts):
            if i % 100 == 0:
                print(f"Processing test sample {i}/{len(test_texts)}")
            
            prompt = create_zero_shot_prompt(text)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an SMS spam detection assistant. Respond with only one word: spam, ham, or smishing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            prediction = extract_classification(response.choices[0].message.content)
            predictions.append(prediction)
        
        accuracy = float(accuracy_score(test_labels_str, predictions))
        classification_rep = classification_report(test_labels_str, predictions, output_dict=True, zero_division=0)
        confusion_mat = confusion_matrix(test_labels_str, predictions, labels=['ham', 'smishing', 'spam']).tolist()
        
        print(f"Accuracy: {accuracy}")
        
        results = {
            "accuracy": accuracy,
            "classification_report": classification_rep,
            "confusion_matrix": confusion_mat,
            "model_path": "zero_shot_gpt4o",
            "fallback_used": False
        }
        
        detailed_results = pd.DataFrame({
            "text": test_texts,
            "true_label": test_labels_str,
            "pred_label": predictions
        })
        
        detailed_results.to_csv(os.path.join(outdir, "results_detailed.csv"), index=False)
        
        print("Zero-shot GPT-4o completed successfully")
        
    except Exception as e:
        print(f"Zero-shot GPT-4o failed: {e}")
        results["error"] = str(e)
    
    try:
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    summary_path = os.path.join("experiments", config.get("baseline_id") or "bl_llm_01_C", "summary.csv")
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