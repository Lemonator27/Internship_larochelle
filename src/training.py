import os
import json
import copy
from PIL import Image
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoProcessor
from pydantic import Field
import torch.optim as optim
from peft import LoraConfig, get_peft_model, TaskType 

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100
LLAVA_IMAGE_TOKEN = "<image>"

FIXED_SCHEMA_PATH = "/Utilisateurs/dbui/json_schema/schema.json" 
with open(FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)
fixed_schema_string = json.dumps(schema_dict, indent=2)
print(f"Successfully loaded fixed schema from: {FIXED_SCHEMA_PATH}")


def pad_sequence(sequences: List[torch.Tensor], padding_side: str = 'right', padding_value: int = 0) -> torch.Tensor:
    if not sequences:
        return torch.empty(0)
    if not all(isinstance(s, torch.Tensor) for s in sequences):
         raise TypeError("All elements in sequences must be torch.Tensor")
    sequences = [s for s in sequences if s.nelement() > 0]
    if not sequences:
        return torch.empty(0)

    trailing_dims = sequences[0].size()[1:]
    max_len = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)

    padded_seqs = torch.full((batch_size, max_len) + trailing_dims, padding_value, dtype=sequences[0].dtype, device=sequences[0].device)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            padded_seqs[i, :length] = seq
        elif padding_side == 'left':
            padded_seqs[i, -length:] = seq
    return padded_seqs

def load_json_lines(file_path: str) -> tuple[List[str], List[Dict]]:
    print(f"Loading labels from: {file_path}")
    image_ids: List[str] = []
    targets: List[Dict] = []
    ocr_text: List[str] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            if "id" in obj and "target" in obj:
                image_ids.append(str(obj["id"]))
                targets.append(obj["target"])
                ocr_text.append(obj["page_texts"])
    print(f"Loaded {len(image_ids)} labels.")
    return image_ids, targets, ocr_text


def load_data(image_ids: List[str], targets: List[Dict],ocred: List[str], image_base_path: str) -> Dict[str, List]:
    print(f"Loading images from: {image_base_path}")
    images_files: List[Image.Image] = []
    labels_as_strings: List[str] = []
    ocr_text: List[str] = []
    valid_indices = []
    loaded_count: int = 0
    skipped_count: int = 0

    if not os.path.isdir(image_base_path):
         print(f"Warning: Image directory not found at {image_base_path}. Dataset may be empty.")
         return {'image': [], 'label': []}

    for i, img_id in enumerate(image_ids):
        possible_extensions = ['.jpg']
        img_loaded = False
        for ext in possible_extensions:
            file_path: str = os.path.join(image_base_path, f"{img_id}{ext}")
            if os.path.exists(file_path):
                img = Image.open(file_path).convert('RGB')
                images_files.append(img)
                string_data: str = json.dumps(targets[i], ensure_ascii=False)
                labels_as_strings.append(string_data)
                ocr_text.append(ocred[i])
                valid_indices.append(i)
                loaded_count += 1
                img_loaded = True
                break

        if not img_loaded:
            skipped_count += 1

    print(f"Successfully loaded and processed {loaded_count} images. Skipped {skipped_count} entries.")
    return {'image': images_files, 'label': labels_as_strings, "ocr_text": ocr_text}

@dataclass
class DataArguments:
    image_folder: str
    label_file: str

def replace_image_tokens(input_string: str, start_count: int = 1) -> tuple[str, int]:
    count = start_count
    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count -1
    while LLAVA_IMAGE_TOKEN in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN, f"<|image_{count}|>", 1)
        count += 1
    return input_string, count - 1

# --- Dataset Class ---

class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
    ):
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        self.data: Dict[str, List] = {'image': [], 'label': [], "ocr_text":[]}

        all_image_ids, all_labels_dict, ocr_text = load_json_lines(data_args.label_file)
        if all_image_ids:
             self.data = load_data(all_image_ids, all_labels_dict,ocr_text, data_args.image_folder)
             print(f"Dataset initialized with {len(self.data['image'])} valid image-label pairs.")
        else:
             print("Warning: No labels loaded, dataset will be empty.")

    def __len__(self) -> int:
        return len(self.data['image'])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if i >= len(self.data['image']):
             raise IndexError(f"Index {i} out of bounds for dataset with length {len(self.data['image'])}")

        image: Image.Image = self.data['image'][i]
        label_str: str = self.data['label'][i]
        ocr: str = self.data["ocr_text"][i]
        prompt: str = (
            f"""{LLAVA_IMAGE_TOKEN} Your task is to extract the information for the fields "
            f"provided below from the image. Extract the information in JSON format "
            f"according to the following JSON schema: {fixed_schema_string}, Additional guidelines:
            - Extract only the elements that are present verbatim in the document text. Do NOT → infer any information.
            - Extract each element EXACTLY as it appears in the document.
            - Each value in the OCR can only be used AT MOST once. If a value can correspond to multiple fields, pick the best one.
            - For each object, output all the keys from the schema even if the value is null. Empty lists should be outputted as lists with no elements.
            - If no indication of tax is given, assume the amounts to be gross amounts.
            <ocr>
            {ocr}
            </ocr>
            """
        )
        prompt, _ = replace_image_tokens(prompt)

        inputs: Dict[str, torch.Tensor] = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = inputs["input_ids"].squeeze(0)

        label_inputs = self.processor.tokenizer(
            label_str,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.processor.tokenizer.model_max_length
        )
        target_labels: torch.Tensor = label_inputs["input_ids"].squeeze(0)

        combined_ids = torch.cat([input_ids, target_labels], dim=0)
        final_labels = combined_ids.clone()
        final_labels[:len(input_ids)] = IGNORE_INDEX

        final_input_ids = combined_ids
        final_attention_mask = torch.ones_like(final_input_ids)

        data_dict: Dict[str, torch.Tensor] = {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "labels": final_labels,
        }

        if 'pixel_values' in inputs:
            data_dict['pixel_values'] = inputs['pixel_values'].squeeze(0)
        if 'image_sizes' in inputs:
             data_dict['image_sizes'] = inputs['image_sizes'].squeeze(0)

        return data_dict


class DataCollatorForSupervisedDataset(object):
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        examples = [ex for ex in examples if ex.get("input_ids") is not None and ex["input_ids"].nelement() > 0]
        if not examples:
             return {}

        input_ids = [example["input_ids"] for example in examples]
        labels = [example["labels"] for example in examples]
        attention_masks = [example["attention_mask"] for example in examples]
        pixel_values = [example.get("pixel_values") for example in examples]
        image_sizes = [example.get("image_sizes") for example in examples if example.get("image_sizes") is not None]

        input_ids = pad_sequence(input_ids, padding_side='left', padding_value=self.pad_token_id)
        labels = pad_sequence(labels, padding_side='left', padding_value=IGNORE_INDEX)
        attention_masks = pad_sequence(attention_masks, padding_side='left', padding_value=0)

        batch_dict: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
        }

        if any(p is not None for p in pixel_values):
            dummy_shape = (3, 224, 224)
            first_valid_pv = next((p for p in pixel_values if p is not None), None)
            if first_valid_pv is not None:
                dummy_shape = first_valid_pv.shape
            valid_pixel_values = [p if p is not None else torch.zeros(dummy_shape) for p in pixel_values]
            batch_dict["pixel_values"] = torch.stack(valid_pixel_values)

            if image_sizes and len(image_sizes) == len([p for p in pixel_values if p is not None]):
                 batch_dict["image_sizes"] = torch.stack(image_sizes)

        return batch_dict

if __name__ == "__main__":
    model_id = "microsoft/Phi-3-vision-128k-instruct"
    image_dir = r"/Utilisateurs/dbui/sroie/images"
    label_path = r"/Utilisateurs/dbui/sroie/train-documents.jsonl"
    num_epochs = 3
    batch_size = 1
    learning_rate = 3e-5
    lora_rank = 128
    lora_alpha = 32
    lora_dropout = 0.05

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model and Processor ---
    print(f"Loading base model and processor: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=None,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='flash_attention_2',
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Base model and processor loaded successfully.")

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # --- Configure LoRA ---
    print("Configuring LoRA...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # --- Wrap Model with PEFT ---
    print("Applying LoRA to the model...")
    model = get_peft_model(model, lora_config)
    print("LoRA applied successfully.")
    model.print_trainable_parameters()
    model.to(device)
    print(f"PEFT model moved to {device}.")

    # --- Prepare Data ---
    data_args = DataArguments(
        image_folder=image_dir,
        label_file=label_path,
    )

    print("Creating dataset...")
    train_dataset = LazySupervisedDataset(
        processor=processor,
        data_args=data_args,
    )
    print(f"Dataset created with {len(train_dataset)} samples.")

    if len(train_dataset) == 0:
        print("FATAL ERROR: Dataset is empty. Exiting.")
        exit()

    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )
    print("Data collator created.")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )
    print(f"DataLoader created with batch size {batch_size}.")

    # --- Setup Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"Optimizer AdamW created with learning rate {learning_rate}.")

    # --- Simple Training Loop ---
    print("\n--- Starting LoRA Training ---")
    model.train()

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        total_loss = []
        num_batches = 0

        for step, batch in enumerate(train_dataloader):
            if not batch:
                continue

            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')} 

            if 'pixel_values' not in batch:
                 continue # Skip if no image data

            outputs = model(**batch)
            loss = outputs.loss
            
            if loss is None or not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss.append(loss.item())
            num_batches += 1
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{len(train_dataloader)}, Batch Loss: {loss.item():.4f}")

        avg_epoch_loss = sum(total_loss) / num_batches if num_batches > 0 else 0
        print(f"--- End of Epoch {epoch+1} ---")
        print(f"Average Training Loss: {avg_epoch_loss:.4f}")

    print("\n--- Training Finished ---")

    # --- (Optional) Save LoRA Adapters ---
    save_path = "./my_lora_adapters_simplified"
    print(f"Saving model adapters and tokenizer to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    processor.tokenizer.save_pretrained(save_path)

    print("LoRA adapters and tokenizer saved.")

