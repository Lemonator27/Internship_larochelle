import os
import json
from PIL import Image
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType

# Constants
IGNORE_INDEX = -100
FIXED_SCHEMA_PATH = "/Utilisateurs/dbui/json_schema/schema.json"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# Load schema
with open(FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)
fixed_schema_string = json.dumps(schema_dict, indent=2)

system_message = f"""
            Your task is to extract the information for the fields "
            provided below from the image. Extract the information in JSON format
            according to the following JSON schema: {fixed_schema_string}, Additional guidelines:
            - Extract only the elements that are present verbatim in the document text. Do NOT → infer any information.
            - Extract each element EXACTLY as it appears in the document.
            - Each value in the OCR can only be used AT MOST once. If a value can correspond to multiple fields, pick the best one.
            - For each object, output all the keys from the schema even if the value is null. Empty lists should be outputted as lists with no elements.
            - If no indication of tax is given, assume the amounts to be gross amounts."""

# Initialize model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
processor = Qwen2VLProcessor.from_pretrained(MODEL_ID)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def format_data(sample: Dict) -> List[Dict]:
    """Format data into multimodal chat template"""
    return [
        {
            "role": "system",
            "content": [{
                "type": "text", 
                "text": f"{system_message}\n<ocr>{sample['ocr_text']}</ocr>"
            }]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": "Extract information according to schema"}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}]
        }
    ]

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Process batch with proper image/text handling"""
    formatted_batch = []
    image_batch = []

    for example in batch:
        text = processor.apply_chat_template(
            example, 
            tokenize=False, 
            add_generation_prompt=False
        )
        image = None
        for content in example[1]["content"]:
            if content["type"] == "image":
                image = content["image"]
                break
                
        if image is not None:
            formatted_batch.append(text)
            image_batch.append(image)

    inputs = processor(
        text=formatted_batch,
        images=image_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX

    image_token_ids = [151652, 151653, 151655]
    for token_id in image_token_ids:
        labels[labels == token_id] = IGNORE_INDEX
    
    inputs["labels"] = labels
    return inputs

@dataclass
class DataArguments:
    image_folder: str
    label_file: str
    ocr_in_prompt: bool = True

class VLDataset(Dataset):
    def __init__(self, processor, data_args: DataArguments):
        self.processor = processor
        self.data_args = data_args
        self.samples = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        image_ids, labels, ocr_texts = load_json_lines(self.data_args.label_file)
        samples = []
        
        for idx, (img_id, label, ocr) in enumerate(zip(image_ids, labels, ocr_texts)):
            img_path = os.path.join(self.data_args.image_folder, f"{img_id}.jpg")
                
            try:
                image = Image.open(img_path).convert("RGB")
                samples.append({
                    "image": image,
                    "label": [json.dumps(label)],
                    "ocr_text": ocr if self.data_args.ocr_in_prompt else ""
                })
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"Loaded {len(samples)} valid samples")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> List[Dict]:
        return format_data(self.samples[idx])

def load_json_lines(file_path: str) -> Tuple[List[str], List[Dict], List[str]]:
    """Load JSON lines file with validation"""
    image_ids = []
    labels = []
    ocr_texts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    image_ids.append(str(data["id"]))
                    labels.append(data["target"])
                    ocr_texts.append(data.get("page_texts", ""))
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Skipping invalid line: {e}")
    except FileNotFoundError:
        raise ValueError(f"Label file {file_path} not found")
    
    return image_ids, labels, ocr_texts

def train():
    # Initialize dataset and dataloader
    data_args = DataArguments(
        image_folder="/Utilisateurs/dbui/sroie/images",
        label_file="/Utilisateurs/dbui/sroie/train-documents.jsonl"
    )
    
    dataset = VLDataset(processor, data_args)
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Training loop
    model.train()
    for epoch in range(3):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} | Avg Loss: {total_loss/len(train_loader):.4f}")
    
    # Save model
    model.save_pretrained("./finetuned_qwen_vl")

if __name__ == "__main__":
    train()