import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import json
from typing import Dict, List 
import argparse
import os 
import datasets
from torch.utils.data import Dataset

FIXED_SCHEMA_PATH = "/home/bdinhlam/schema/schema.json" 
train_path = "/home/bdinhlam/scratch/dataset/cord/train-documents.jsonl"
val_path  = "/home/bdinhlam/scratch/dataset/cord/validation-documents.jsonl"

with open(FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)
fixed_schema_string = json.dumps(schema_dict, indent=2)
print(f"Successfully loaded fixed schema from: {FIXED_SCHEMA_PATH}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_id = "mistralai/Ministral-8B-Instruct-2410"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
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
# Apply LoRA config to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

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

label_train, ocr_train = load_json_lines(train_path)
label_val, ocr_val = load_json_lines(val_path)

def create_instruct_messages(ocr,label):
    prompt = (
            f"""Your task is to extract the information for the fields
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
        )
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
        
    return list_of_prompt

train_list = create_list(label_train, ocr_train)
val_list  = create_list(label_val,ocr_val)

print("Finished list")
    
train_dataset = datasets.Dataset.from_list(train_list)
val_dataset = datasets.Dataset.from_list(val_list)

print("Finished mapping")

per_device_train_batch_size = 2
gradient_accumulation_steps = 1
num_epochs = 4


training_args = TrainingArguments(
        output_dir="/home/bdinhlam/scratch/weight/weight_mistral_cord/",
        num_train_epochs=4,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        per_device_eval_batch_size = 1,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        optim="adamw_torch",
        report_to="wandb",
        dataloader_num_workers=4,
        greater_is_better=False,     
        eval_strategy="steps", 
        eval_steps=100,    
        do_eval = True,
        save_total_limit=5,
    )

# Initialize trainer
trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args, 
    )

# Train
trainer.train()

# Save the model
trainer.save_model("sql-assistant-final-ministral-8b") 