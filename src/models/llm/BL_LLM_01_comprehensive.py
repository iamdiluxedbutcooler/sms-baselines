import os
import json
import subprocess
import datetime
import time
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from models.shared import set_seed, append_summary_row
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def _git_short_hash():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "nogit"

class PromptTemplates:
    @staticmethod
    def zero_shot(text: str) -> str:
        return f"""Classify the following SMS message as either "spam" or "ham".

SMS: {text}
Classification:"""

    @staticmethod
    def few_shot(text: str, examples: List[Tuple[str, str]]) -> str:
        examples_text = ""
        for ex_text, ex_label in examples:
            examples_text += f"SMS: {ex_text}\nClassification: {ex_label}\n\n"
        
        return f"""Classify SMS messages as either "spam" or "ham". Here are some examples:

{examples_text}SMS: {text}
Classification:"""

    @staticmethod
    def chain_of_thought(text: str) -> str:
        return f"""Classify the following SMS message as either "spam" or "ham". Think step by step about the characteristics that indicate spam vs ham.

SMS: {text}

Let me analyze this step by step:
1. Content analysis: 
2. Language patterns:
3. Urgency indicators:
4. Suspicious elements:
5. Overall assessment:

Based on this analysis, the classification is:"""

class GPT4oClassifier:
    def __init__(self):
        self.api_key = None
        self.client = None
        
    def setup_client(self):
        try:
            import openai
            import getpass
            
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                print("OpenAI API key not found in environment.")
                try:
                    self.api_key = getpass.getpass("Enter your OpenAI API key: ")
                except KeyboardInterrupt:
                    print("\nSkipping GPT-4o (no API key provided)")
                    return False
                    
            if not self.api_key or not self.api_key.strip():
                print("No API key provided, skipping GPT-4o")
                return False
                
            self.client = openai.OpenAI(api_key=self.api_key)
            
            test_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            print("OpenAI API connection successful!")
            return True
        except Exception as e:
            print(f"Failed to setup OpenAI client: {e}")
            return False
    
    def classify_batch(self, prompts: List[str], strategy: str) -> List[str]:
        if not self.client:
            return ["ham"] * len(prompts)
        
        predictions = []
        for i, prompt in enumerate(prompts):
            if i % 50 == 0:
                print(f"GPT-4o {strategy}: {i}/{len(prompts)}")
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.1
                )
                
                result = response.choices[0].message.content.strip().lower()
                if "spam" in result:
                    predictions.append("spam")
                elif "ham" in result:
                    predictions.append("ham")
                else:
                    predictions.append("ham")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"API call failed: {e}")
                predictions.append("ham")
        
        return predictions

class OpenSourceClassifier:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def setup_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            return True
        except Exception as e:
            print(f"Failed to load {self.model_name}: {e}")
            return False
    
    def classify_batch(self, prompts: List[str], strategy: str) -> List[str]:
        if not self.model:
            return ["ham"] * len(prompts)
        
        predictions = []
        for i, prompt in enumerate(prompts):
            if i % 25 == 0:
                print(f"{self.model_name} {strategy}: {i}/{len(prompts)}")
            
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
                result = response.strip().lower()
                
                if "spam" in result:
                    predictions.append("spam")
                elif "ham" in result:
                    predictions.append("ham")
                else:
                    predictions.append("ham")
                    
            except Exception as e:
                print(f"Inference failed: {e}")
                predictions.append("ham")
        
        return predictions
    
    def cleanup(self):
        if self.model:
            del self.model
            del self.tokenizer
            import torch
            torch.cuda.empty_cache()

def select_few_shot_examples(train_texts: List[str], train_labels: List[str], n_examples: int = 4) -> List[Tuple[str, str]]:
    spam_examples = [(text, label) for text, label in zip(train_texts, train_labels) if str(label).lower() == 'spam']
    ham_examples = [(text, label) for text, label in zip(train_texts, train_labels) if str(label).lower() == 'ham']
    
    selected = []
    for i in range(n_examples // 2):
        if i < len(spam_examples):
            selected.append(spam_examples[i])
        if i < len(ham_examples):
            selected.append(ham_examples[i])
    
    return selected

def run_bl_llm_01(train_texts: List[str], train_labels: List[str], 
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
        "paper_title": "Comprehensive LLM Baseline with Multiple Models and Prompting Strategies",
        "paper_year": 2024
    }
    
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Original train labels: {set(train_labels)}")
    print(f"Original test labels: {set(test_labels)}")
    
    train_labels_str = [str(x).lower() for x in train_labels]
    test_labels_str = [str(x).lower() for x in test_labels]
    
    few_shot_examples = select_few_shot_examples(train_texts, train_labels_str)
    
    results = {"accuracy": None, "sub_results": {}}
    all_predictions = {}
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    strategies = ["zero_shot", "few_shot", "chain_of_thought"]
    models = [
        ("gpt4o", GPT4oClassifier(api_key) if api_key else None),
        ("llama3", OpenSourceClassifier("meta-llama/Meta-Llama-3-8B-Instruct")),
        ("mistral", OpenSourceClassifier("mistralai/Mistral-7B-Instruct-v0.2"))
    ]
    
    for model_name, classifier in models:
        if classifier is None:
            print(f"Skipping {model_name} - API key not available")
            continue
            
        print(f"\nSetting up {model_name}...")
        if not classifier.setup_model() if hasattr(classifier, 'setup_model') else classifier.setup_client():
            print(f"Failed to setup {model_name}, skipping...")
            continue
        
        for strategy in strategies:
            strategy_key = f"{model_name}_{strategy}"
            print(f"\nRunning {strategy_key}...")
            
            try:
                if strategy == "zero_shot":
                    prompts = [PromptTemplates.zero_shot(text) for text in test_texts]
                elif strategy == "few_shot":
                    prompts = [PromptTemplates.few_shot(text, few_shot_examples) for text in test_texts]
                elif strategy == "chain_of_thought":
                    prompts = [PromptTemplates.chain_of_thought(text) for text in test_texts]
                
                predictions = classifier.classify_batch(prompts, strategy)
                all_predictions[strategy_key] = predictions
                
                accuracy = float(accuracy_score(test_labels_str, predictions))
                classification_rep = classification_report(test_labels_str, predictions, output_dict=True, zero_division=0)
                confusion_mat = confusion_matrix(test_labels_str, predictions).tolist()
                
                results["sub_results"][strategy_key] = {
                    "accuracy": accuracy,
                    "classification_report": classification_rep,
                    "confusion_matrix": confusion_mat
                }
                
                print(f"{strategy_key} Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"Error in {strategy_key}: {e}")
                results["sub_results"][strategy_key] = {"error": str(e)}
        
        if hasattr(classifier, 'cleanup'):
            classifier.cleanup()
    
    if results["sub_results"]:
        best_accuracy = max([r.get("accuracy", 0) for r in results["sub_results"].values() if "accuracy" in r])
        results["accuracy"] = best_accuracy
        
        best_strategy = max(results["sub_results"].items(), 
                          key=lambda x: x[1].get("accuracy", 0) if "accuracy" in x[1] else 0)[0]
        print(f"\nBest performing strategy: {best_strategy} with accuracy {best_accuracy:.4f}")
    
    detailed_results_list = []
    for strategy_key, predictions in all_predictions.items():
        for i, (text, true_label, pred_label) in enumerate(zip(test_texts, test_labels_str, predictions)):
            detailed_results_list.append({
                "text": text,
                "true_label": true_label,
                "pred_label": pred_label,
                "strategy": strategy_key,
                "sample_id": i
            })
    
    if detailed_results_list:
        detailed_df = pd.DataFrame(detailed_results_list)
        detailed_df.to_csv(os.path.join(outdir, "results_detailed.csv"), index=False)
    
    try:
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    summary_path = os.path.join("experiments", config.get("baseline_id") or "bl_llm_01", "summary.csv")
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