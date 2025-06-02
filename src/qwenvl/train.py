from unsloth import FastVisionModel
from transformers import TextStreamer
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch
import os
import json
import argparse
from typing import Dict, List, Any
from PIL import Image
from transformers import EarlyStoppingCallback
import tqdm

parser = argparse.ArgumentParser(description="Extract structured information from images using a multimodal model.")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset name for training")
parser.add_argument("--image_in_prompt", action='store_true',
                    help="If image should be in the prompt")
parser.add_argument("--ocr_in_prompt", action='store_true',
                    help="If ocr should be in the prompt")
args = parser.parse_args()

dataset_name = args.dataset
image_bool = args.image_in_prompt
ocr_bool = args.ocr_in_prompt

if not image_bool and not ocr_bool:
    print("Warning: Neither --image_in_prompt nor --ocr_in_prompt was set. Defaulting to OCR in prompt.")
    ocr_bool = True

FIXED_SCHEMA_PATH = "/home/bdinhlam/schema/schema.json"
dataset_path = f"/home/bdinhlam/scratch/dataset/{dataset_name}/images"
train_path = f"/home/bdinhlam/scratch/dataset/{dataset_name}/train-documents.jsonl"
val_path = f"/home/bdinhlam/scratch/dataset/{dataset_name}/validation-documents.jsonl"
save_model_path_base = f"/home/bdinhlam/scratch/weight/weight_qwen_{dataset_name}_new/"

with open(FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)
fixed_schema_string = json.dumps(schema_dict, indent=2)
print(f"Successfully loaded fixed schema from: {FIXED_SCHEMA_PATH}")

data_config_path_suffix = "unknown"
if ocr_bool and image_bool:
    data_config_path_suffix = "ocr_image"
elif ocr_bool:
    data_config_path_suffix = "ocr_only"
elif image_bool:
    data_config_path_suffix = "image_only"
else:
    print("Error: No valid input combination (OCR and/or Image). Exiting.")
    exit()

save_model_path = os.path.join(save_model_path_base, data_config_path_suffix)
os.makedirs(save_model_path, exist_ok=True)

print("Saving the model to this folder: ", save_model_path)

def load_json_lines(file_path: str, base_image_path: str) -> Dict[str, List[Any]]:
    print(f"Loading labels from: {file_path}")
    targets: List[str] = []
    ocr_texts: List[Any] = []
    images: List[Any] = []
    image_ids_for_debug: List[str] = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f, desc=f"Processing {os.path.basename(file_path)}"):
            obj = json.loads(line.strip())
            if "id" in obj and "target" in obj:
                targets.append(json.dumps(obj["target"]))
                ocr_texts.append(obj.get("page_texts", ""))

                if image_bool:
                    image_filename = obj["id"] + ".jpg"
                    image_full_path = os.path.join(base_image_path, image_filename)
                    try:
                        image = Image.open(image_full_path).convert("RGB")
                        images.append(image)
                    except FileNotFoundError:
                        print(f"Warning: Image not found {image_full_path}. Storing None.")
                        images.append(None)
                    except Exception as e:
                        print(f"Warning: Could not load image {image_full_path} due to {e}. Storing None.")
                        images.append(None)
                else:
                    images.append(None)
                image_ids_for_debug.append(obj["id"])

    print(f"Loaded {len(targets)} labels.")
    dataset_dict = {"targets": targets, "ocr_text": ocr_texts, "images": images, "ids": image_ids_for_debug}
    return dataset_dict

dataset_train_raw = load_json_lines(train_path, dataset_path)
dataset_val_raw = load_json_lines(val_path, dataset_path)

ds_train = Dataset.from_dict(dataset_train_raw)
ds_val = Dataset.from_dict(dataset_val_raw)

system_message = """You are a highly capable Vision Language Model for structured data extraction.
You will be provided with OCR text, an image context, and a JSON schema.
Your primary directive is to follow all user instructions meticulously to populate the schema using only information explicitly present in the text or image.
Ensure your output is valid JSON and strictly conforms to all given constraints."""

def format_sample_to_chat_list(sample: Dict, ocr_in_prompt_flag: bool, image_in_prompt_flag: bool) -> List[Dict[str, Any]]:
    ocr_content_str = ""
    if ocr_in_prompt_flag and sample.get('ocr_text'):
        current_ocr_text = sample['ocr_text']
        if isinstance(current_ocr_text, list):
            processed_ocr_text = "\n".join(map(str, current_ocr_text))
        else:
            processed_ocr_text = str(current_ocr_text)
        ocr_content_str = f"""<ocr>
{processed_ocr_text}
</ocr>"""

    user_query = f"""
Extract the information in JSON format according to the following JSON schema: {fixed_schema_string}, Additional guidelines:
- Extract only the elements that are present verbatim in the document text or image. Do NOT infer any information.
- Extract each element EXACTLY as it appears in the document.
- Each value in the OCR can only be used AT MOST once. If a value can correspond to multiple fields, pick the best one.
- For each object, output all the keys from the schema even if the value is null. Empty lists should be outputted as lists with no elements.
- If no indication of tax is given, assume the amounts to be gross amounts.
{ocr_content_str}
Please read the text/image carefully and follow the instructions.
"""
    content_for_user_message = []
    if image_in_prompt_flag and sample.get("images") is not None:
        content_for_user_message.append(
            {"type": "image", "image": sample["images"]}
        )

    content_for_user_message.append(
        {"type": "text", "text": user_query}
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": content_for_user_message},
        {"role": "assistant", "content": [{"type": "text", "text": sample["targets"]}]},
    ]
    return messages

print("Processing training data samples into chat format using list comprehension...")
train_chat_lists = [
    format_sample_to_chat_list(sample, ocr_in_prompt_flag=ocr_bool, image_in_prompt_flag=image_bool)
    for sample in tqdm.tqdm(ds_train, desc="Formatting Train Dataset")
]
print("Processing validation data samples into chat format using list comprehension...")
val_chat_lists = [
    format_sample_to_chat_list(sample, ocr_in_prompt_flag=ocr_bool, image_in_prompt_flag=image_bool)
    for sample in tqdm.tqdm(ds_val, desc="Formatting Validation Dataset")
]

print("Sample processed chat list (training data):")
if train_chat_lists: print(train_chat_lists[0])


model, tokenizer = FastVisionModel.from_pretrained(
    'unsloth/Qwen2.5-VL-7B-Instruct',
    load_in_4bit=False,
    use_gradient_checkpointing='unsloth',
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=image_bool,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=128,
    lora_alpha=128,
    lora_dropout=0.05,
    bias='none',
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

sft_dataset_text_field = None
train_final_data = None
val_final_data = None

if data_config_path_suffix == "ocr_only":
    print("Text-only mode: Applying chat template to format chat lists into single strings.")

    _train_final_data_list = [
        {"text": tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=False)}
        for chat_list in tqdm.tqdm(train_chat_lists, desc="Applying template to Train (text-only)")
    ]
    _val_final_data_list = [
        {"text": tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=False)}
        for chat_list in tqdm.tqdm(val_chat_lists, desc="Applying template to Val (text-only)")
    ]
    train_final_data = Dataset.from_list(_train_final_data_list)
    val_final_data = Dataset.from_list(_val_final_data_list)

    sft_dataset_text_field = "text"
    print("Sample data after applying chat template (text-only):")
    if len(train_final_data) > 0: print(train_final_data[0]["text"])
else:
    print("Vision-inclusive mode: Using chat lists directly.")
    train_final_data = train_chat_lists
    val_final_data = val_chat_lists
    sft_dataset_text_field = None
    print("Sample data for vision model (direct chat list):")
    if train_final_data: print(train_final_data[0])


early_stopping_patience = 8
early_stopping_threshold = 0.0
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=early_stopping_patience,
    early_stopping_threshold=early_stopping_threshold,
)

data_collator_to_use = None

if data_config_path_suffix == "ocr_image" or data_config_path_suffix == "image_only":
    data_collator_to_use = UnslothVisionDataCollator(model=model, processor=tokenizer)
    print("Using UnslothVisionDataCollator.")
elif data_config_path_suffix == "ocr_only":
    data_collator_to_use = None
    print("Using default SFTTrainer collator for text-only (data_collator=None).")
else:
    raise ValueError(f"Unknown data configuration: {data_config_path_suffix}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator_to_use,
    train_dataset=train_final_data,
    eval_dataset=val_final_data,
    dataset_text_field=sft_dataset_text_field,
    args=SFTConfig(
        output_dir=save_model_path,
        max_seq_length=4096,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.2,
        num_train_epochs=6,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy='steps',
        eval_steps=40/4,
        save_strategy='steps',
        save_steps=40/4,
        save_total_limit=3,
        optim='adamw_torch',
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        seed=3407,
        report_to='wandb',
        load_best_model_at_end=True,
        remove_unused_columns=True,
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported() and torch.cuda.is_available(),
        dataloader_num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1),
    ),
    callbacks=[early_stopping_callback],
)

print("Starting training...")
trainer_stats = trainer.train()
print("Training finished.")

print(f"Saving model to {save_model_path}...")
trainer.save_model(save_model_path)
if tokenizer:
    tokenizer.save_pretrained(save_model_path)
print("Model and tokenizer saved.")

if 'wandb' in trainer.args.report_to and trainer.args.report_to != "none":
    import wandb
    if wandb.run:
        wandb.log({
            "train_runtime": trainer_stats.training_time if hasattr(trainer_stats, 'training_time') else None,
            "train_samples_per_second": trainer_stats.metrics.get("train_samples_per_second", 0),
            "train_steps_per_second": trainer_stats.metrics.get("train_steps_per_second", 0),
            "train_loss": trainer_stats.metrics.get("train_loss", 0),
            "epoch": trainer_stats.epoch if hasattr(trainer_stats, 'epoch') else None,
        })
        wandb.finish()

print(f"Script finished. Model artifacts are in {save_model_path}")
