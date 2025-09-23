import os
import json
import subprocess
import datetime
import torch
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

def create_training_prompt(text: str, label: str) -> str:
    return f"Classify this SMS message as 'Spam' or 'Ham':\n\nMessage: {text}\nClassification: {label}"

def create_inference_prompt(text: str) -> str:
    return f"Classify this SMS message as 'Spam' or 'Ham':\n\nMessage: {text}\nClassification:"

def extract_classification(response: str) -> str:
    response = response.strip().lower()
    if 'spam' in response:
        return 'spam'
    elif 'ham' in response:
        return 'ham'
    else:
        return 'ham'

def run_bl_llm_02(train_texts: List[str], train_labels: List[str], 
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
        "paper_title": "Fine-tuned Mixtral 8x7B with QLoRA",
        "paper_year": 2025
    }
    
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Original train labels: {set(train_labels)}")
    print(f"Original test labels: {set(test_labels)}")
    
    train_labels_str = [str(x).lower() for x in train_labels]
    test_labels_str = [str(x).lower() for x in test_labels]
    
    results = {"accuracy": None}
    hf_token = None
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from datasets import Dataset
        import transformers
        import getpass
        
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        
        print("Setting up HuggingFace authentication...")
        hf_token = getpass.getpass("Enter your HuggingFace access token: ")
        
        print("Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        print("Loading model with quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            low_cpu_mem_usage=True
        )
        
        print("Preparing model for QLoRA...")
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        print("Preparing training data...")
        train_prompts = [create_training_prompt(text, label) for text, label in zip(train_texts, train_labels_str)]
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
        
        train_dataset = Dataset.from_dict({"text": train_prompts})
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        
        print("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=outdir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            num_train_epochs=3,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
            optim="adamw_torch",
            fp16=True,
            dataloader_pin_memory=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )
        )
        
        print("Starting fine-tuning...")
        trainer.train()
        
        print("Saving fine-tuned model...")
        trainer.save_model(os.path.join(outdir, "fine_tuned_model"))
        
        print("Running inference on test set...")
        predictions = []
        
        for i, text in enumerate(test_texts):
            if i % 100 == 0:
                print(f"Processing test sample {i}/{len(test_texts)}")
            
            prompt = create_inference_prompt(text)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            prediction = extract_classification(response)
            predictions.append(prediction)
        
        accuracy = float(accuracy_score(test_labels_str, predictions))
        classification_rep = classification_report(test_labels_str, predictions, output_dict=True, zero_division=0)
        confusion_mat = confusion_matrix(test_labels_str, predictions).tolist()
        
        print(f"Accuracy: {accuracy}")
        
        results = {
            "accuracy": accuracy,
            "classification_report": classification_rep,
            "confusion_matrix": confusion_mat,
            "model_path": os.path.join(outdir, "fine_tuned_model")
        }
        
        detailed_results = pd.DataFrame({
            "text": test_texts,
            "true_label": test_labels_str,
            "pred_label": predictions
        })
        
        detailed_results.to_csv(os.path.join(outdir, "results_detailed.csv"), index=False)
        
        print("Mixtral fine-tuning completed successfully")
        
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Mixtral fine-tuning failed: {e}")
        results["error"] = str(e)
        
        try:
            print("Falling back to zero-shot Mixtral...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            hf_token = getpass.getpass("Enter your HuggingFace access token for fallback: ")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                token=hf_token
            )
            
            predictions_fallback = []
            for i, text in enumerate(test_texts):
                if i % 50 == 0:
                    print(f"Zero-shot inference {i}/{len(test_texts)}")
                
                prompt = create_inference_prompt(text)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
                prediction = extract_classification(response)
                predictions_fallback.append(prediction)
            
            accuracy_fallback = float(accuracy_score(test_labels_str, predictions_fallback))
            
            results = {
                "accuracy": accuracy_fallback,
                "classification_report": classification_report(test_labels_str, predictions_fallback, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(test_labels_str, predictions_fallback).tolist(),
                "model_path": "zero_shot_mixtral",
                "fallback_used": True
            }
            
            detailed_results = pd.DataFrame({
                "text": test_texts,
                "true_label": test_labels_str,
                "pred_label": predictions_fallback
            })
            
            detailed_results.to_csv(os.path.join(outdir, "results_detailed.csv"), index=False)
            
            print("Zero-shot fallback completed")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as fallback_e:
            print(f"Fallback also failed: {fallback_e}")
            results["fallback_error"] = str(fallback_e)
    
    try:
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    summary_path = os.path.join("experiments", config.get("baseline_id") or "bl_llm_02", "summary.csv")
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