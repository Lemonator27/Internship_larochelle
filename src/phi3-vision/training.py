import os
import json
import copy
from PIL import Image
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoProcessor,Trainer,TrainingArguments
from pydantic import Field
import torch.optim as optim
from peft import LoraConfig, get_peft_model, TaskType 
import argparse

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100
LLAVA_IMAGE_TOKEN = "<image>"

FIXED_SCHEMA_PATH = "/home/bdinhlam/schema/schema.json" 
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

    for i, img_id in enumerate(image_ids):
        possible_extensions = ['.jpg']
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

    print(f"Successfully loaded and processed {loaded_count} images. Skipped {skipped_count} entries.")
    return {'image': images_files, 'label': labels_as_strings, "ocr_text": ocr_text}

@dataclass
class DataArguments:
    image_folder: str
    label_file: str
    ocr_in_promt: bool = True
    image_in_prompt: bool = True

def replace_image_tokens(input_string: str, start_count: int = 1) -> tuple[str, int]:
    count = start_count
    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count -1
    while LLAVA_IMAGE_TOKEN in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN, f"<|image_{count}|>", 1)
        count += 1
    return input_string, count - 1


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
        
        if data_args.ocr_in_promt == True:
            ocr: str = self.data["ocr_text"][i]
        else:
            ocr: str = None
        
        if self.data_args.image_in_prompt == False:
            image_token_for_prompt: str = "" 
            image: Image.Image = None
        else:
            image_token_for_prompt = LLAVA_IMAGE_TOKEN
            image: Image.Image = self.data['image'][i]
        
        prompt: str = (
            f"""<|user|>\n{image_token_for_prompt} 
            Your task is to extract the information for the fields provided below from the image. Extract the information in JSON format
            according to the following JSON schema: {fixed_schema_string}, Additional guidelines:
            - Extract only the elements that are present verbatim in the document text. Do NOT → infer any information.
            - Extract each element EXACTLY as it appears in the document.
            - Each value in the OCR can only be used AT MOST once. If a value can correspond to multiple fields, pick the best one.
            - For each object, output all the keys from the schema even if the value is null. Empty lists should be outputted as lists with no elements.
            - If no indication of tax is given, assume the amounts to be gross amounts.
            <ocr>
            {ocr}
            </ocr>
            Please read the text carefully and follow the instructions.
            <|end|><|assistant|>
            """
        )
        prompt, _ = replace_image_tokens(prompt)

        inputs = self.processor(
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
            max_length=16000
        )
        target_labels = label_inputs["input_ids"].squeeze(0)

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
    image_dir = r"/home/bdinhlam/scratch/dataset/cord/images"
    label_path = r"/home/bdinhlam/scratch/dataset/cord/train-documents.jsonl"
    val_path = r"/home/bdinhlam/scratch/dataset/cord/validation-documents.jsonl"
    num_epochs = 3
    batch_size = 1
    learning_rate = 3e-5
    lora_rank = 2
    lora_alpha = 32
    lora_dropout = 0.05
    save_path = "/home/bdinhlam/scratch/weight/weight_cord/image_only"
    
    parser = argparse.ArgumentParser(description="Extract structured information from images using a multimodal model.")
    
    parser.add_argument("--image_in_prompt", action='store_true', 
                    help="If image should be in the prompt")
    parser.add_argument("--ocr_in_prompt", action='store_true', 
                    help="If ocr should be in the prompt")
    args = parser.parse_args()
    

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='flash_attention_2',
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        
    processor.tokenizer.model_max_length = 6000
    processor.tokenizer.pad_token = processor.tokenizer.unk_token  
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
    processor.tokenizer.padding_side = 'right'

    # --- Configure LoRA ---
    print("Configuring LoRA...")
    target_modules = ["qkv_proj","gate_up_proj","down_proj",]

    lora_config = LoraConfig(
        r=128,
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

    #Unfreezing the vision encoder since the LoRA_config would freeze all the weights automatically
    for layers in model.base_model.model.model.vision_embed_tokens.img_processor.vision_model.encoder.layers:
        for param in layers.self_attn.parameters():
            param.requires_grad = True
        for param in layers.mlp.parameters():
            param.requires_grad = True
    model.print_trainable_parameters()
    
    # --- Prepare Data ---
    
    ocr_in_prompt = args.ocr_in_prompt
    image_in_prompt = args.image_in_prompt

    if ocr_in_prompt and image_in_prompt:
        data_config_path = "ocr_image"
    elif ocr_in_prompt:
        data_config_path = "ocr_only"
    elif image_in_prompt:
        data_config_path = "image_only"
    if ocr_in_prompt and image_in_prompt:
        data_config_path = "ocr_image"
    print(data_config_path)
         
    
    data_args = DataArguments(
        image_folder=image_dir,
        label_file=label_path,
        image_in_prompt = image_in_prompt,
        ocr_in_promt = ocr_in_prompt,
    )

    val_args = DataArguments(
        image_folder=image_dir,
        label_file=val_path,
        image_in_prompt= image_in_prompt,
        ocr_in_promt = ocr_in_prompt,
    )


    train_dataset = LazySupervisedDataset(
        processor=processor,
        data_args=data_args,
    )
    
    val_dataset = LazySupervisedDataset(
        processor=processor,
        data_args=val_args,
    )
    
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )
    model.train()
    # --- Configure Training Arguments ---
    training_args = TrainingArguments(
        output_dir=f"/home/bdinhlam/scratch/weight/weight_cord/{data_config_path}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",         
        eval_strategy="steps", 
        eval_steps=100,    
        remove_unused_columns=False,
        optim="adamw_torch",
        report_to="wandb",
        dataloader_num_workers=4,
        greater_is_better=False,
        do_eval = True,
    )

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = val_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )

    # --- Training ---
    print("\n--- Starting LoRA Training with HF Trainer ---")
    trainer.train()
    
    # --- Save Model ---
    print("\n--- Saving Final Model ---")
    model.save_pretrained("/home/bdinhlam/scratch/weight/weight_cord/{path}")
    processor.tokenizer.save_pretrained(save_path)
    print("Training complete and model saved.")

