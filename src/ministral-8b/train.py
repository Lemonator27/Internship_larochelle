import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import json
from typing import Dict, List 
import argparse
import os 
import transformers
from torch.utils.data import Dataset

FIXED_SCHEMA_PATH = "/home/bdinhlam/schema/schema.json" 
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

# Loading json file to get OCR_text and labels 
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

class LazySupervisedDataset(Dataset):
    def __init__(self, processor: transformers.ProcessorMixin,json_path):
        super().__init__()
        self.processor = processor
        self.path_json = json_path

        self.all_labels_dict, self.ocr_text = load_json_lines(self.path_json)

    def __len__(self) -> int:
        return len(self.data['image'])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if i >= len(self.data['image']):
             raise IndexError(f"Index {i} out of bounds for dataset with length {len(self.data['image'])}")
        ocred = self.ocr_text[i]
        label_str = self.all_labels_dict[i]
        
        prompt = {"text": f"""[INST]: 
           Extract the information in JSON format according to the following JSON schema: {fixed_schema_string}
         - Extract only the elements that are present verbatim in the document text. Do NOT infer any information.
         - Extract each element EXACTLY as it appears in the document.
         - Each value in the OCR can only be used AT MOST once. If a value can correspond to multiple fields, pick the best one.
         - For each object, output all the keys from the schema even if the value is null. Empty lists should be outputted as lists with no elements.
         - If no indication of tax is given, assume the amounts to be gross amounts.
         <ocr>
         {ocred}
         </ocr>
         Please read the text carefully and follow the instructions.
         /INST]{label_str}"""}
        
        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
            max_length = 6000,
            padding="max_length",
        )

        return inputs

train_dataset = LazySupervisedDataset(
        processor=tokenizer,
        path_json="",
    )
    
val_dataset = LazySupervisedDataset(
        processor=tokenizer,
        path_json="",
    )



training_args = TrainingArguments(
        output_dir="/home/bdinhlam/scratch/weight/weight_cord/ocr_image",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        learning_rate=3e-5,
        per_device_eval_batch_size = 1,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        optim="adamw_torch",
        report_to="wandb",
        dataloader_num_workers=4,
        greater_is_better=False,
        do_eval = True,
        eval_strategy = "epoch",
    )

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset["train"],
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
)

# Train
trainer.train()

# Save the model
trainer.save_model("sql-assistant-final-ministral-8b") 