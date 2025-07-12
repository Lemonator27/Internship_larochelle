# --- 0. Setup and Imports ---
import torch
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import re
import json
import argparse
from transformers import EvalPrediction
from reward_functions import (
    check_json_syntax,
    calculate_nted_reward,
    check_exact_match,
    match_format_approximately,
    generate_random_reward,
    calculate_nted # Assuming this function exists
)

parser = argparse.ArgumentParser(description="GRPO OCR Training Script with selectable reward functions.")
parser.add_argument(
        "--reward_functions",
        type=str,
        default="all",
        choices=["nted", "exact", "syntax", "all", "random"],
        help="Which reward function(s) to use for training."
    )
reward_list = []
args = parser.parse_args()
name = "all_rewards"
if "syntax" in args.reward_functions:
    reward_list.append(check_json_syntax)
    name="syntax"
if "nted" in args.reward_functions:
    reward_list.append(calculate_nted_reward)
    name="nted"
if "exact" in args.reward_functions:
    reward_list.append(check_exact_match)
    name="exact_match"
if "random" in args.reward_functions:
    reward_list.append(generate_random_reward)
    name = "random"
if "all" in args.reward_functions:
    reward_list = [check_exact_match, calculate_nted_reward, check_json_syntax]
    name = "all_rewards"

output_dir = f"scratch/llama_weight/cord_{name}"
max_seq_length = 4982
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/bdinhlam/scratch/llama-8B", 
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# --- 2. System Prompt and Chat Template ---
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<solution>"
solution_end = "</solution>"

json_schema = {
  "type": "object",
  "properties": {
    "menu": {
      "type": "array",
      "description": "A list of all regular items purchased, based on the provided table.",
      "items": {
        "type": "object",
        "properties": {
          "nm":           { "type": "string", "description": "name of menu" },
          "num":          { "type": "string", "description": "identification # of menu" },
          "unitprice":    { "type": "string", "description": "unit price of menu" },
          "cnt":          { "type": "string", "description": "quantity of menu" },
          "discountprice":{ "type": "string", "description": "discounted price of menu" },
          "price":        { "type": "string", "description": "total price of menu" },
          "itemsubtotal": { "type": "string", "description": "price of each menu after discount applied" },
          "vatyn":        { "type": "string", "description": "whether the price includes tax or not" },
          "etc":          { "type": "string", "description": "other details for this menu item" },
          "sub_nm":       { "type": "string", "description": "name of submenu" },
          "sub_unitprice":{ "type": "string", "description": "unit price of submenu" },
          "sub_cnt":      { "type": "string", "description": "quantity of submenu" },
          "sub_price":    { "type": "string", "description": "total price of submenu" },
          "sub_etc":      { "type": "string", "description": "other details for this submenu item" }
        }
      }
    },
    "void_menu": {
      "type": "array",
      "description": "A list of all voided/cancelled items, based on the provided table.",
      "items": {
        "type": "object",
        "properties": {
          "nm":    { "type": "string", "description": "name of voided menu" },
          "price": { "type": "string", "description": "total price of voided menu" }
        }
      }
    },
    "subtotal": {
      "type": "object",
      "description": "Details of the subtotal calculation, based on the provided table.",
      "properties": {
        "subtotal_price":  { "type": "string", "description": "subtotal price" },
        "discount_price":  { "type": "string", "description": "discounted price in total" },
        "service_price":   { "type": "string", "description": "service charge" },
        "othersvc_price":  { "type": "string", "description": "added charge other than service charge" },
        "tax_price":       { "type": "string", "description": "tax amount" },
        "etc":             { "type": "string", "description": "other subtotal details" }
      }
    },
    "total": {
      "type": "object",
      "description": "Final price and payment details, based on the provided table.",
      "properties": {
        "total_price":     { "type": "string", "description": "total price" },
        "total_etc":       { "type": "string", "description": "other total details" },
        "cashprice":       { "type": "string", "description": "amount of price paid in cash" },
        "changeprice":     { "type": "string", "description": "amount of change in cash" },
        "creditcardprice": { "type": "string", "description": "amount of price paid in credit/debit card" },
        "emoneyprice":     { "type": "string", "description": "amount of price paid in emoney, point" },
        "menutype_cnt":    { "type": "string", "description": "total count of type of menu" },
        "menuqty_cnt":     { "type": "string", "description": "total count of quantity" }
      }
    }
  }
}
json_schema_str = json.dumps(json_schema, indent=2)

system_prompt = f"""Act as a meticulous data analyst. First, present your detailed reasoning in a single narrative between {reasoning_start} and {reasoning_end}.
For each extracted value, you must justify the mapping by citing the exact source text, explain assumptions or data transformations (like removing currency symbols), perform arithmetic checks to validate interdependent fields (e.g., proving that `change = cash - total`), if a field does not a value leave to be None,double check your calculations and conclusion, and explicitly declare any optional fields you could not find. Following this complete analysis, conclude by providing the final, pristine JSON object between {solution_start} and {solution_end}, ensuring it strictly and perfectly adheres to the user-provided schema.
"""

# --- 3. Data Loading and Preparation ---
data_filename = '/home/bdinhlam/cord/cord_v2_train_processed.json'
val_data_filename = '/home/bdinhlam/cord/cord_v2_validation_processed.json'

with open(data_filename, 'r', encoding='utf-8') as f:
    data_list = json.load(f)
with open(val_data_filename, 'r', encoding='utf-8') as f:
    data_val = json.load(f)

df = pd.DataFrame(data_list)
df_val = pd.DataFrame(data_val) 

df.dropna(subset=['ocr_text', 'parsed_gt'], inplace=True)
df_val.dropna(subset=['ocr_text', 'parsed_gt'], inplace=True)

def normalize_ground_truth(record):
    if isinstance(record, dict):
        if 'menu' in record and isinstance(record['menu'], dict):
            record['menu'] = [record['menu']]
        if 'total' in record and isinstance(record['total'], dict):
            if 'changeprice' in record['total']:
                record['total']['change_price'] = record['total'].pop('changeprice')
    return record

df['parsed_gt'] = df['parsed_gt'].apply(normalize_ground_truth)
df_val['parsed_gt'] = df_val['parsed_gt'].apply(normalize_ground_truth)

def standardize_ocr_text(text_entry):
    if isinstance(text_entry, list):
        return "\n".join(map(str, text_entry))
    return str(text_entry)
df['ocr_text'] = df['ocr_text'].apply(standardize_ocr_text)
df_val['ocr_text'] = df_val['ocr_text'].apply(standardize_ocr_text)

def format_dataset(example, is_eval=False):
    user_content_with_schema = f"""Please extract data from the following OCR text.
You MUST extract the data into a valid JSON object that strictly adheres to the following JSON schema.
### JSON Schema:
```json
{json_schema_str}
```
### OCR Text:
```text
{example["prompt_text"]}
```"""
    
    formatted = {
        "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content_with_schema}],
        "answer": example["solution_json_str"],
    }
    if is_eval:
        assistant_response = f"{reasoning_start}{reasoning_end}{solution_start}{example['solution_json_str']}{solution_end}"
        formatted["labels"] = assistant_response
    return formatted

train_dataset = Dataset.from_dict({
    'prompt_text': df['ocr_text'].tolist(),
    'solution_json_str': [json.dumps(gt) for gt in df['parsed_gt'].tolist()]
}).map(format_dataset, fn_kwargs={"is_eval": False})

eval_dataset = Dataset.from_dict({
    'prompt_text': df_val['ocr_text'].tolist(),
    'solution_json_str': [json.dumps(gt) for gt in df_val['parsed_gt'].tolist()]
}).map(format_dataset, fn_kwargs={"is_eval": True})


# --- 4. Evaluation Metrics and Trainer Setup ---
def extract_solution_json(text):
    match = re.search(f"{re.escape(solution_start)}(.*?){re.escape(solution_end)}", text, re.DOTALL)
    if match:
        try:
            json_str = match.group(1).strip()
            if json_str.startswith("```json"): json_str = json_str[7:]
            if json_str.endswith("```"): json_str = json_str[:-3]
            return json.loads(json_str)
        except json.JSONDecodeError: return None
    return None

def flatten_json(y, parent_key='', sep='.'):
    items = {}
    for k, v in y.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        elif isinstance(v, list):
            for i, elem in enumerate(v):
                if isinstance(elem, dict):
                    items.update(flatten_json(elem, f"{new_key}[{i}]", sep=sep))
                else:
                    items[f"{new_key}[{i}]"] = str(elem)
        else:
            items[new_key] = str(v)
    return items

class MetricCalculator:
    def __init__(self):
        self.nted_scores = []
        self.f1_scores = []

    def compute(self, eval_pred: EvalPrediction, compute_result: bool = False):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            pred_json = extract_solution_json(pred)
            label_json = extract_solution_json(label)

            if pred_json and label_json:
                nted_accuracy = calculate_nted(pred_json, label_json)
                self.nted_scores.append(1.0 - nted_accuracy)
                flat_pred = set(flatten_json(pred_json).items())
                flat_label = set(flatten_json(label_json).items())
                tp = len(flat_pred.intersection(flat_label))
                fp = len(flat_pred - flat_label)
                fn = len(flat_label - flat_pred)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                self.f1_scores.append(f1)
            else:
                self.nted_scores.append(1.0)
                self.f1_scores.append(0.0)

        if compute_result:
            final_metrics = {
                "nted_distance": np.mean(self.nted_scores) if self.nted_scores else 0.0,
                "f1_score": np.mean(self.f1_scores) if self.f1_scores else 0.0
            }
            self.nted_scores, self.f1_scores = [], []
            return final_metrics
        return {}

metric_calculator = MetricCalculator()

def get_prompt_length(example):
    prompt_str = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
    return {"prompt_len": len(tokenizer.encode(prompt_str))}

dataset_with_lengths = train_dataset.map(get_prompt_length)
max_prompt_length_for_trainer = int(np.quantile(dataset_with_lengths["prompt_len"], 1))
max_completion_length = max_seq_length - max_prompt_length_for_trainer
dataset_filtered = dataset_with_lengths.filter(lambda x: x["prompt_len"] <= max_prompt_length_for_trainer)

training_args = GRPOConfig(
    output_dir=f"{output_dir}",
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=400,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_prompt_length=max_prompt_length_for_trainer,
    max_completion_length=max_completion_length,
    num_generations=4,
    use_vllm=True,
    logging_steps=5,
    save_steps=20,
    report_to="wandb",
    seed=3407,
    save_strategy="steps",
    batch_eval_metrics=True,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset_filtered,
    reward_funcs=reward_list,
)

print("Starting GRPO training...")
trainer.train()
print("Training finished successfully.")

# --- 5. Inference ---
print("\n--- RUNNING INFERENCE WITH FINAL ADAPTER ---")
ocr_test_text = "Sample OCR text..."
user_content_for_inference = f"""Please extract data from the following OCR text according to this schema.
### JSON Schema:
```json
{json_schema_str}
```
### OCR Text:
```text
{ocr_test_text}
```"""
messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content_for_inference}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.1, pad_token_id=tokenizer.pad_token_id)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated output:\n{decoded_output}")
