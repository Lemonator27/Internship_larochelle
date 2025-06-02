import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer,SFTConfig
import json
from typing import Dict, List 
import argparse 
import datasets
parser = argparse.ArgumentParser(description="Extract structured information from images using a multimodal model.")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset name for training")
args = parser.parse_args()

dataset = args.dataset

FIXED_SCHEMA_PATH = "/home/bdinhlam/schema/schema.json" 
train_path = f"/home/bdinhlam/scratch/dataset/{dataset}/train-documents.jsonl" 
val_path  = f"/home/bdinhlam/scratch/dataset/{dataset}/validation-documents.jsonl" 
save_model_path = f"/home/bdinhlam/scratch/weight/weight_mistral_{dataset}_new/"
with open(FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)
fixed_schema_string = json.dumps(schema_dict, indent=2)
print(f"Successfully loaded fixed schema from: {FIXED_SCHEMA_PATH}")

model_id = "mistralai/Ministral-8B-Instruct-2410"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
        r=128, 
        lora_alpha=128,  
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

def load_json_lines(file_path: str) -> tuple[List[str], List[Dict]]:
    print(f"Loading labels from: {file_path}")
    targets: List[Dict] = []
    ocr_text: List[str] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            if "id" in obj and "target" in obj:
                targets.append(obj["target"])
                ocr_text.append(obj["page_texts"])
    print(f"Loaded {len(targets)} labels.")
    return targets, ocr_text

def create_instruct_messages(ocr,label):
    prompt = f"""
            Extract the information in JSON format according to the following JSON schema: {fixed_schema_string}, Additional guidelines:
            - Extract only the elements that are present verbatim in the document text. Do NOT → infer any information.
            - Extract each element EXACTLY as it appears in the document.
            - Each value in the OCR can only be used AT MOST once. If a value can correspond to multiple fields, pick the best one.
            - For each object, output all the keys from the schema even if the value is null. Empty lists should be outputted as lists with no elements.
            - If no indication of tax is given, assume the amounts to be gross amounts.
            <ocr>
            {ocr}
            </ocr>
            Please read the text carefully and follow the instructions.
            """
        
    response = f"{label}"
    return {
        'messages': [
            { 'role': 'user', 'content': prompt},
            { 'role': 'assistant', 'content': response}
        ]
    }

def create_list(label:List, ocr:List):
    list_of_prompt = []
    for i in range(len(label)):
        prompt = create_instruct_messages(ocr[i] , label[i])
        list_of_prompt.append(prompt)
        print(prompt)
    return list_of_prompt

def format_chat_template(example):
    return {
        'text': tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
    }

label_train, ocr_train = load_json_lines(train_path)
label_val, ocr_val = load_json_lines(val_path)

train_list = create_list(label_train, ocr_train)
val_list  = create_list(label_val,ocr_val)

print("Finished list")
    
train_dataset = datasets.Dataset.from_list(train_list).map(format_chat_template)
val_dataset = datasets.Dataset.from_list(val_list).map(format_chat_template)

print("Finished mapping")

print(train_dataset[0])

per_device_train_batch_size = 2
gradient_accumulation_steps = 1
num_epochs = 4

effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
steps_per_epoch = len(train_dataset) // effective_batch_size
eval_save_steps = 40
print(eval_save_steps)
early_stopping_patience = 5

early_stopping_threshold = 0.0

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=early_stopping_patience,
    early_stopping_threshold=early_stopping_threshold,
)

training_args = SFTConfig(
        output_dir=save_model_path,
        max_seq_length=16000,
        learning_rate=5e-5,  # Lower learning rate for stability
        lr_scheduler_type="cosine",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Increased for more stable gradients
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

# Initialize trainer
trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=[early_stopping_callback],
        peft_config=lora_config,
)

# Train
trainer.train()
trainer.save_model(save_model_path)
