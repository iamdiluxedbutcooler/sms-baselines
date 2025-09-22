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

def _preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        return ' '.join(tokens)
    except:
        tokens = text.split()
        return ' '.join([word for word in tokens if len(word) > 1])

def run_bl_nn_01(train_texts: List[str], train_labels: List[str], 
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
        "paper_title": "CNN-BiGRU with Word2Vec embeddings",
        "paper_year": 2024
    }
    
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Original train labels: {set(train_labels)}")
    print(f"Original test labels: {set(test_labels)}")
    
    train_processed = [_preprocess_text(text) for text in train_texts]
    test_processed = [_preprocess_text(text) for text in test_texts]
    
    train_labels_str = [str(x) for x in train_labels]
    test_labels_str = [str(x) for x in test_labels]
    
    results = {"accuracy": None}
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, GRU, Dense, Dropout
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.optimizers import Adam
        from gensim.models import Word2Vec
        
        tf.random.set_seed(seed)
        
        le = LabelEncoder()
        y_train = le.fit_transform(train_labels_str)
        y_test = le.transform(test_labels_str)
        
        all_texts = train_processed + test_processed
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_texts)
        
        X_train_seq = tokenizer.texts_to_sequences(train_processed)
        X_test_seq = tokenizer.texts_to_sequences(test_processed)
        
        max_length = max(max(len(seq) for seq in X_train_seq), max(len(seq) for seq in X_test_seq))
        max_length = min(max_length, 200)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='pre')
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='pre')
        
        train_tokens = [text.split() for text in train_processed if text.strip()]
        w2v_model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=1, seed=seed)
        
        vocab_size = len(tokenizer.word_index) + 1
        embedding_dim = 100
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        for word, i in tokenizer.word_index.items():
            if word in w2v_model.wv:
                embedding_matrix[i] = w2v_model.wv[word]
            else:
                embedding_matrix[i] = np.random.normal(0, 0.1, embedding_dim)
        
        model = Sequential([
            Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], 
                     input_length=max_length, trainable=False),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Bidirectional(GRU(units=64, dropout=0.2, recurrent_dropout=0.2)),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        
        print("Training model...")
        history = model.fit(X_train_pad, y_train, 
                           batch_size=32, epochs=50, 
                           validation_split=0.1, verbose=1)
        
        print("Making predictions...")
        y_pred_proba = model.predict(X_test_pad, batch_size=32)
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
            "model_path": os.path.join(outdir, "model.h5")
        }
        
        model.save(os.path.join(outdir, "model.h5"))
        
        detailed_results = pd.DataFrame({
            "text": test_processed,
            "true_label": test_labels_str,
            "pred_label": y_pred_labels,
            "pred_proba": y_pred_proba.flatten()
        })
        
        detailed_results.to_csv(os.path.join(outdir, "results_detailed.csv"), index=False)
        
        print("Neural network training completed successfully")
        
    except Exception as e:
        print(f"Neural network training failed: {e}")
        results["error"] = str(e)
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            import joblib
            
            print("Falling back to TF-IDF + Naive Bayes...")
            
            vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
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
                "text": test_processed,
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
    
    summary_path = os.path.join("experiments", config.get("baseline_id") or "bl_nn_01", "summary.csv")
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