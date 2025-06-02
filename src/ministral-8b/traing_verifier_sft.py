import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import json
from typing import Dict, List, Tuple
import argparse
import datasets
import traceback # Added for the main training loop exception

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train a verifier model to check structured data based on schema and rules (no OCR).")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name for training (e.g., 'sroie').")
args = parser.parse_args()

dataset_name_arg = args.dataset

# --- Configuration: File Paths ---
FIXED_SCHEMA_PATH = "/home/bdinhlam/schema/schema.json" # This is still used in the prompt

# Paths to the PROCESSED data files which now must contain JSON to verify, and labels.
# OCR text is no longer expected or used from these files.
processed_train_path = f"/home/bdinhlam/scratch/dataset/{dataset_name_arg}/verify/train_processed_output.json"
processed_val_path = f"/home/bdinhlam/scratch/dataset/{dataset_name_arg}/verify/validation_processed_output.json"
save_model_path = f"/home/bdinhlam/scratch/weight/weight_mistral_{dataset_name_arg}_verifier_no_ocr_v1/" # Updated version name

# --- Load Schema for Prompt ---
with open(FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)



# --- Model and Tokenizer Initialization ---
model_id = "mistralai/Ministral-8B-Instruct-2410"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- LoRA Configuration ---
lora_config = LoraConfig(
        r=64, 
        lora_alpha=128,  
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

# --- Data Loading and Prompt Creation ---

def load_verifier_data_from_processed_file(processed_data_path: str) -> Tuple[List[Dict], List[str]]:
    """
    Loads JSON data to verify and verification labels directly
    from a single processed JSON file. OCR text is no longer loaded.
    Assumes the processed file contains 'input_schema', and 'labels' keys.
    The 'ocr_texts' key, if present, will be ignored.
    """
    print(f"Attempting to load verifier data (JSON and labels) from: {processed_data_path}")
    
    json_to_verify_list: List[Dict] = []
    verification_labels: List[str] = []

    with open(processed_data_path, 'r', encoding='utf-8') as f_proc:
        processed_data = json.load(f_proc)
    
    # Expected keys in the processed JSON file
    json_to_verify_list = processed_data.get("input_schema", []) 
    verification_labels = processed_data.get("labels", [])

    if not json_to_verify_list and not verification_labels:
        print(f"Warning: All essential data lists (JSON, labels) are empty after loading from {processed_data_path}.")
        return [], []

    print(f"Successfully loaded {len(json_to_verify_list)} JSON entries and {len(verification_labels)} labels from {processed_data_path}.")
    return json_to_verify_list, verification_labels


def create_verification_prompt_messages(json_to_verify: Dict, label: str) -> Dict:
    """
    Creates a structured prompt for the verifier model.
    The prompt no longer includes an OCR text section.
    The schema information is now directly part of the prompt text.
    """
    # The schema_dict is loaded globally, so we can use it here.
    # Or, you could pass fixed_schema_string as an argument if preferred.


    prompt_content = f"""
    Your task is to validate if the provided `{json_to_verify}` conforms to the `{schema_dict}`.

    Your analysis must include:
    1.  **Constraint Identification:** Automatically identify all relevant financial, arithmetic, and logical constraints based on the schema and general invoice conventions. This includes totals, taxes, discounts, and line-item aggregations.
    2.  **Intelligent Deduction:** If a value is `None` (missing), attempt to deduce it using the identified constraints and other available data. Proceed only when a unique and logical value can be determined.
    3.  **Verification:** For each constraint, compare the expected result (from calculation) with the actual value in the data. Allow for minor rounding discrepancies.

    Provide a step-by-step reasoning of the constraints you checked and their outcomes.

    Conclude your final response with a single word: `correct` or `not correct`.
    """
    assistant_response = str(label)
    return {
        'messages': [
            {'role': 'user', 'content': prompt_content.strip()},
            {'role': 'assistant', 'content': assistant_response}
        ]
    }

def create_dataset_list(json_data: List[Dict], labels: List[str]) -> List[Dict]:
    dataset_list = []
    if not (json_data and labels): # Check if lists are empty
        print("Warning: One or more input lists for dataset creation are empty. Returning empty dataset list.")
        return []
    for i in range(len(labels)):
        dataset_list.append(create_verification_prompt_messages(json_data[i], labels[i]))
    return dataset_list

def format_chat_template(example: Dict) -> Dict:
    return {
        'text': tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
    }

# --- Main Script Execution ---

print("--- Loading Training Data (JSON and Labels only) ---")
json_train, labels_train = load_verifier_data_from_processed_file(processed_train_path)
print("\n--- Loading Validation Data (JSON and Labels only) ---")
json_val, labels_val = load_verifier_data_from_processed_file(processed_val_path)

if not (json_train and labels_train and json_val and labels_val):
    print("Data loading resulted in one or more empty datasets. This might be due to issues with the processed files. Exiting.")
    exit()

print("\n--- Creating Prompt Lists ---")
train_list = create_dataset_list(json_train, labels_train)
val_list = create_dataset_list(json_val, labels_val)

if not (train_list and val_list):
    print("Critical error: Training or validation prompt list is empty. Exiting script.")
    exit()

print("\n--- Converting to Hugging Face Datasets ---")
train_dataset = datasets.Dataset.from_list(train_list).map(format_chat_template, remove_columns=['messages'])
val_dataset = datasets.Dataset.from_list(val_list).map(format_chat_template, remove_columns=['messages'])

print("\n--- Example of a formatted training sample (first 300 chars) ---")
if len(train_dataset) > 0 and 'text' in train_dataset[0]:
    print(train_dataset[0]['text'])
else:
    print("Training dataset is empty or 'text' field is missing.")
print("------------------------------------------------------------\n")

per_device_train_batch_size = 1
gradient_accumulation_steps = 1
num_epochs = 4

effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
steps_per_epoch = len(train_dataset) // effective_batch_size
eval_save_steps = 40
training_args = SFTConfig(
        output_dir=save_model_path,
        max_seq_length=6500,
        learning_rate=5e-5,  # Lower learning rate for stability
        lr_scheduler_type="cosine",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps*4,  # Increased for more stable gradients
        weight_decay=0.05,  # Increased from 0.01 for better regularization
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=40,
        save_steps=eval_save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=5,
        logging_steps=1,
        optim="adamw_torch",
        warmup_ratio=0.1,  # Reduced from 0.2 for more gradual warmup
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,  # Added gradient clipping
        )

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0.001,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    callbacks=[early_stopping_callback],
    peft_config=lora_config,
)

print("🚀 Starting model training (No OCR)...")
try:
    trainer.train()
    print("✅ Training complete!")
    print("💾 Saving final model...")
    trainer.save_model(save_model_path)
    print(f"LoRA adapter model saved to {save_model_path}")
except Exception as e: 
    print(f"An error occurred during training: {e}")
    traceback.print_exc()

print("🏁 Script execution finished.")
