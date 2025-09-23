import os
import re
import json
import subprocess
import datetime
import string
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

def _preprocess_text_seo(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.replace('\n', ' LINEBREAK ')
    text = text.replace('\t', ' TAB ')
    text = text.replace('  ', ' DOUBLESPACE ')
    
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' LINK ', text)
    text = re.sub(r'\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}', ' CALL ', text)
    text = re.sub(r'\S+\.(pdf|doc|docx|txt|jpg|png|zip|exe|apk)', ' FILE ', text, flags=re.IGNORECASE)
    
    link_count = 0
    while 'LINK' in text and link_count < 10:
        if link_count == 0:
            text = text.replace('LINK', 'LINKA', 1)
        elif link_count == 1:
            text = text.replace('LINK', 'LINKB', 1)
        elif link_count == 2:
            text = text.replace('LINK', 'LINKC', 1)
        else:
            text = text.replace('LINK', f'LINK{link_count+1}', 1)
        link_count += 1
    
    call_count = 0
    while 'CALL' in text and call_count < 10:
        if call_count == 0:
            text = text.replace('CALL', 'CALLA', 1)
        elif call_count == 1:
            text = text.replace('CALL', 'CALLB', 1)
        elif call_count == 2:
            text = text.replace('CALL', 'CALLC', 1)
        else:
            text = text.replace('CALL', f'CALL{call_count+1}', 1)
        call_count += 1
    
    file_count = 0
    while 'FILE' in text and file_count < 10:
        if file_count == 0:
            text = text.replace('FILE', 'FILEA', 1)
        elif file_count == 1:
            text = text.replace('FILE', 'FILEB', 1)
        elif file_count == 2:
            text = text.replace('FILE', 'FILEC', 1)
        else:
            text = text.replace('FILE', f'FILE{file_count+1}', 1)
        file_count += 1
    
    return text

def create_char_vocab():
    chars = []
    
    chars.extend(list(string.ascii_lowercase))
    chars.extend(list(string.ascii_uppercase))
    chars.extend(list(string.digits))
    chars.extend(list(string.punctuation))
    chars.extend([' ', '\n', '\t'])
    
    special_tokens = ['LINKA', 'LINKB', 'LINKC', 'CALLA', 'CALLB', 'CALLC', 
                     'FILEA', 'FILEB', 'FILEC', 'LINEBREAK', 'TAB', 'DOUBLESPACE']
    chars.extend(special_tokens)
    
    for i in range(4, 11):
        chars.extend([f'LINK{i}', f'CALL{i}', f'FILE{i}'])
    
    chars.append('<UNK>')
    chars.append('<PAD>')
    
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    
    return char_to_idx, idx_to_char

def texts_to_char_sequences(texts, char_to_idx, max_length=500):
    sequences = []
    for text in texts:
        processed_text = _preprocess_text_seo(text)
        sequence = []
        i = 0
        while i < len(processed_text) and len(sequence) < max_length:
            char = processed_text[i]
            found_special = False
            
            for special_token in ['LINKA', 'LINKB', 'LINKC', 'CALLA', 'CALLB', 'CALLC', 
                                'FILEA', 'FILEB', 'FILEC', 'LINEBREAK', 'TAB', 'DOUBLESPACE'] + \
                               [f'LINK{j}' for j in range(4, 11)] + \
                               [f'CALL{j}' for j in range(4, 11)] + \
                               [f'FILE{j}' for j in range(4, 11)]:
                if processed_text[i:].startswith(special_token):
                    if special_token in char_to_idx:
                        sequence.append(char_to_idx[special_token])
                    else:
                        sequence.append(char_to_idx['<UNK>'])
                    i += len(special_token)
                    found_special = True
                    break
            
            if not found_special:
                if char in char_to_idx:
                    sequence.append(char_to_idx[char])
                else:
                    sequence.append(char_to_idx['<UNK>'])
                i += 1
        
        if len(sequence) < max_length:
            sequence.extend([char_to_idx['<PAD>']] * (max_length - len(sequence)))
        
        sequences.append(sequence[:max_length])
    
    return np.array(sequences)

def run_bl_nn_02(train_texts: List[str], train_labels: List[str], 
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
        "paper_title": "Character-level 1D CNN with mask preprocessing",
        "paper_year": 2024
    }
    
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Original train labels: {set(train_labels)}")
    print(f"Original test labels: {set(test_labels)}")
    
    train_labels_str = [str(x) for x in train_labels]
    test_labels_str = [str(x) for x in test_labels]
    
    results = {"accuracy": None}
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        tf.random.set_seed(seed)
        
        le = LabelEncoder()
        y_train = le.fit_transform(train_labels_str)
        y_test = le.transform(test_labels_str)
        
        char_to_idx, idx_to_char = create_char_vocab()
        vocab_size = len(char_to_idx)
        print(f"Character vocabulary size: {vocab_size}")
        
        max_length = 500
        X_train_char = texts_to_char_sequences(train_texts, char_to_idx, max_length)
        X_test_char = texts_to_char_sequences(test_texts, char_to_idx, max_length)
        
        print(f"Input shape: {X_train_char.shape}")
        
        embedding_dim = 48
        
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(10, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        print("Model architecture:")
        model.summary()
        
        print("Training model...")
        history = model.fit(X_train_char, y_train, 
                           batch_size=32, epochs=50, 
                           validation_split=0.1, verbose=1)
        
        print("Making predictions...")
        y_pred_proba = model.predict(X_test_char, batch_size=32)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        y_pred_labels = le.inverse_transform(y_pred)
        
        accuracy = float(accuracy_score(test_labels_str, y_pred_labels))
        classification_rep = classification_report(test_labels_str, y_pred_labels, output_dict=True, zero_division=0)
        confusion_mat = confusion_matrix(test_labels_str, y_pred_labels).tolist()
        
        print(f"Accuracy: {accuracy}")
        
        results = {
            "accuracy": accuracy,
            "classification_report": classification_rep,
            "confusion_matrix": confusion_mat,
            "model_path": os.path.join(outdir, "model.h5"),
            "vocab_size": vocab_size
        }
        
        model.save(os.path.join(outdir, "model.h5"))
        
        with open(os.path.join(outdir, "char_vocab.json"), "w") as f:
            json.dump({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}, f, indent=2)
        
        detailed_results = pd.DataFrame({
            "text": [_preprocess_text_seo(text) for text in test_texts],
            "true_label": test_labels_str,
            "pred_label": y_pred_labels,
            "pred_proba": y_pred_proba.flatten()
        })
        
        detailed_results.to_csv(os.path.join(outdir, "results_detailed.csv"), index=False)
        
        print("Character-level CNN training completed successfully")
        
    except Exception as e:
        print(f"Character-level CNN training failed: {e}")
        results["error"] = str(e)
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            import joblib
            
            print("Falling back to TF-IDF + Naive Bayes...")
            
            train_processed = [_preprocess_text_seo(text) for text in train_texts]
            test_processed = [_preprocess_text_seo(text) for text in test_texts]
            
            vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=5000, analyzer='char')
            X_train_tfidf = vectorizer.fit_transform(train_processed)
            X_test_tfidf = vectorizer.transform(test_processed)
            
            nb_model = MultinomialNB()
            nb_model.fit(X_train_tfidf, train_labels_str)
            
            y_pred_fallback = nb_model.predict(X_test_tfidf)
            
            accuracy_fallback = float(accuracy_score(test_labels_str, y_pred_fallback))
            
            results = {
                "accuracy": accuracy_fallback,
                "classification_report": classification_report(test_labels_str, y_pred_fallback, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(test_labels_str, y_pred_fallback).tolist(),
                "model_path": os.path.join(outdir, "fallback_model.joblib"),
                "fallback_used": True
            }
            
            joblib.dump({"vectorizer": vectorizer, "model": nb_model}, 
                       os.path.join(outdir, "fallback_model.joblib"))
            
            detailed_results = pd.DataFrame({
                "text": [_preprocess_text_seo(text) for text in test_texts],
                "true_label": test_labels_str,
                "pred_label": y_pred_fallback
            })
            
            detailed_results.to_csv(os.path.join(outdir, "results_detailed.csv"), index=False)
            
            print("Fallback model completed")
            
        except Exception as fallback_e:
            print(f"Fallback also failed: {fallback_e}")
            results["fallback_error"] = str(fallback_e)
    
    try:
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    summary_path = os.path.join("experiments", config.get("baseline_id") or "bl_nn_02", "summary.csv")
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