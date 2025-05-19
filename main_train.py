# main_train.py

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import os
import math
from functools import partial
import pandas as pd

# --- Konfigürasyon Parametreleri ---
MODEL_NAME = "distilgpt2"
LOCAL_DATA_FILE_PATH = "C:/Users/Emirhan/Desktop/poetry_generation_project/data/PoetryFoundationData.csv"
TEXT_COLUMN_NAME_IN_CSV = "Poem"
OUTPUT_DIR = "C:/Users/Emirhan/Desktop/poetry_generation_project/fine_tuned_poetry_model_v3" 
LOGGING_DIR = 'C:/Users/Emirhan/Desktop/poetry_generation_project/logs_poetry_v3' 

# Eğitim Parametreleri (Temel Ayarlarla)
NUM_TRAIN_EPOCHS = 10 
PER_DEVICE_TRAIN_BATCH_SIZE = 4 
PER_DEVICE_EVAL_BATCH_SIZE = 4 
GRADIENT_ACCUMULATION_STEPS = 2 
LEARNING_RATE = 3e-5 
WARMUP_STEPS = 200 
SAVE_STEPS = 500 
LOGGING_STEPS = 250 
BLOCK_SIZE = 128 
FP16 = torch.cuda.is_available()
SEED = 42 

def main():
    global FP16

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Training will run on CPU (which will be very slow).")
        if FP16:
            print("Warning: FP16 is set to True but CUDA is not available. Setting FP16 to False.")
            FP16 = False

    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Loading dataset from local file: ./{LOCAL_DATA_FILE_PATH}...")
    processed_splits = {}
    text_column_name_to_use = "text"
    try:
        full_csv_path = os.path.join(os.getcwd(), LOCAL_DATA_FILE_PATH)
        if not os.path.exists(full_csv_path):
            print(f"Error: CSV file not found at {full_csv_path}")
            return

        df = pd.read_csv(full_csv_path)
        if TEXT_COLUMN_NAME_IN_CSV not in df.columns:
            print(f"Error: Column '{TEXT_COLUMN_NAME_IN_CSV}' not found in {full_csv_path}.")
            return
        poems_data = df[TEXT_COLUMN_NAME_IN_CSV].dropna().astype(str).tolist()
        poems_data = [poem for poem in poems_data if poem.strip()]
        if not poems_data:
            print("Error: No valid poems found.")
            return
        print(f"Loaded {len(poems_data)} poems from the CSV file.")
        
        dataset_dict_for_hf = { text_column_name_to_use: poems_data }
        full_dataset = Dataset.from_dict(dataset_dict_for_hf, split="train")
        
        if len(full_dataset) > 20 :
            train_test_split_dict = full_dataset.train_test_split(test_size=0.1, seed=SEED)
            processed_splits["train"] = train_test_split_dict["train"]
            processed_splits["validation"] = train_test_split_dict["test"]
            print("Created 'train' and 'validation' splits from the local CSV data.")
        else: 
            processed_splits["train"] = full_dataset
            print("Warning: Dataset too small for validation split. Using entire dataset for training. Evaluation will be skipped.")
    except FileNotFoundError:
        print(f"Error: Local data file not found at ./{LOCAL_DATA_FILE_PATH}")
        return
    except Exception as e:
        print(f"Fatal error loading or processing local dataset: {e}")
        return
            
    print("Processed dataset splits:", processed_splits)
    if processed_splits.get("train") and len(processed_splits["train"]) > 0:
        print(f"Example train data (first 200 chars): {processed_splits['train'][0][text_column_name_to_use][:200]}")

    def tokenize_function(examples, tokenizer_to_use, text_column, block_size_to_use):
        texts_to_tokenize = [str(text) if text is not None else "" for text in examples[text_column]]
        tokenized_output = tokenizer_to_use(
            texts_to_tokenize, truncation=True, padding="max_length", max_length=block_size_to_use
        )
        return tokenized_output

    print("Tokenizing datasets...")
    partial_tokenize_function = partial(
        tokenize_function, tokenizer_to_use=tokenizer, text_column=text_column_name_to_use, block_size_to_use=BLOCK_SIZE
    )
    tokenized_datasets = {}
    for split_name, dataset_content in processed_splits.items():
        if dataset_content:
            tokenized_datasets[split_name] = dataset_content.map(
                partial_tokenize_function, batched=True, num_proc=1,
                remove_columns=[text_column_name_to_use] 
            )
    
    perform_evaluation = bool(tokenized_datasets.get("validation"))

    print("Setting up Training Arguments (basic version for compatibility)...")
    training_args_dict = {
        "output_dir": OUTPUT_DIR,
        "overwrite_output_dir": True,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": 0.01,
        "logging_dir": LOGGING_DIR,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "do_eval": perform_evaluation,
        "fp16": FP16,
        "save_total_limit": 2,
        "seed": SEED,
    }
    # Sadece validation seti varsa per_device_eval_batch_size ekle
    if perform_evaluation:
        training_args_dict["per_device_eval_batch_size"] = PER_DEVICE_EVAL_BATCH_SIZE


    training_args = TrainingArguments(**training_args_dict)
    
    print("Initializing Trainer...")
    
    # *** İŞTE DÜZELTİLEN YER BURASI ***
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer, 
        data_collator=data_collator
    )

    if tokenized_datasets.get("train"):
        print("Starting 'basic compatible' training run...")
        train_result = trainer.train()
        print("Training finished.")
        metrics = train_result.metrics
        if tokenized_datasets.get("train"):
             metrics["train_samples"] = len(tokenized_datasets["train"])
        trainer.log_metrics("train_final_v3", metrics) 
        trainer.save_metrics("train_final_v3", metrics)
        
        print(f"Saving final model to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR) 
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Final model (last step) saved to {OUTPUT_DIR}.")
    else:
        print("No training dataset found. Skipping training.")

    if perform_evaluation and tokenized_datasets.get("validation"):
        print("Final evaluating on validation set...")
        eval_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
        try:
            eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
        except Exception:
            eval_metrics["perplexity"] = "N/A"
        trainer.log_metrics("eval_final_v3", eval_metrics)
        trainer.save_metrics("eval_final_v3", eval_metrics)
        print(f"Final 'Basic Compatible' Evaluation Metrics: {eval_metrics}")
    else:
        print("No validation set found or evaluation is disabled for the final run.")

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("Pandas kütüphanesi kurulu değil. Lütfen `pip install pandas` komutu ile kurun.")
        exit()
    main()