print("Starting......")
import os
print("os")
import json
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
print("transformers")
from peft import PeftModel
import logging
import argparse # Added argparse

# --- Argument Parsing Setup ---
parser = argparse.ArgumentParser(description="Extract structured information from images using a multimodal model.")
parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-vision-128k-instruct",
                    help="Hugging Face model ID for the base model.")
parser.add_argument("--image_folder", type=str, required=True,
                    help="Path to the folder containing the images.")
parser.add_argument("--schema_path",default="/home/bdinhlam/schema/schema.json", type=str,
                    help="Path to the JSON schema file.")
parser.add_argument("--input_jsonl", type=str, required=True,
                    help="Path to the input .jsonl file containing image IDs and OCR text.")
parser.add_argument("--lora_adapter", type=str, default=None,
                    help="Path to the LoRA adapter weights directory (optional).")
parser.add_argument("--output_json", type=str, default=None,
                    help="Path to save the extracted results as a JSON file (optional, prints to stdout if not provided).")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device to run the model on (e.g., 'cuda', 'cpu').")
parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                    help="Attention implementation ('flash_attention_2', 'sdpa', 'eager'). Use 'None' for default.")
parser.add_argument("--max_new_tokens", type=int, default=None, # Default to None, will use processor default later
                    help="Maximum number of new tokens to generate.")
parser.add_argument("--do_sample", action='store_true', 
                    help="Enable sampling during generation.")
parser.add_argument("--image_in_prompt", action='store_true', 
                    help="If image should be in the prompt")
parser.add_argument("--ocr_in_prompt", action='store_true', 
                    help="If ocr should be in the prompt")


args = parser.parse_args()

model_id = args.model_id
IMAGE_FOLDER_PATH = args.image_folder
FIXED_SCHEMA_PATH = args.schema_path
lora_adapter_path = args.lora_adapter
input_jsonl_path = args.input_jsonl
output_json_path = args.output_json
device = args.device
attn_implementation = args.attn_implementation if args.attn_implementation != 'None' else None
image_in_prompt = args.image_in_prompt
ocr_in_prompt = args.ocr_in_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"Script arguments: {args}")

with open(FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)
fixed_schema_string = json.dumps(schema_dict, indent=2)
logging.info(f"Successfully loaded JSON schema from: {FIXED_SCHEMA_PATH}")


logging.info(f"Loading base model: {model_id}")
# Direct model loading without try-except
load_kwargs = {
    "device_map": device, # Use device argument
    "trust_remote_code": True,
    "torch_dtype": "auto",
}
if attn_implementation:
    load_kwargs["_attn_implementation"] = attn_implementation

base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
logging.info(f"Base model loaded successfully onto device: {device}.") # More specific logging


logging.info(f"Attempting to load LoRA adapter from: {lora_adapter_path}")
if lora_adapter_path: 
    if not os.path.isdir(lora_adapter_path):
        logging.info(f"LoRA adapter path specified but not found: {lora_adapter_path}. Proceeding without merging.")
        model = base_model
    else:
        logging.info("Loading and merging LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        model = model.merge_and_unload()
        logging.info("Finished merging LoRA adapter.")
else:
    logging.info("No LoRA adapter path specified. Using the base model.")
    model = base_model


logging.info(f"Loading processor for: {model_id}")
# Direct processor loading without try-except
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
logging.info("Processor loaded successfully.")


# --- Setup Generation Arguments ---
effective_max_new_tokens = 7000
generation_args = {
    "max_new_tokens": effective_max_new_tokens,
}


logging.info(f"Generation arguments set: {generation_args}")

SUPPORTED_EXTENSIONS = ('.jpg')

def load_test_image(file_path: str):
    logging.info(f"Loading image paths and OCR data from: {file_path}")
    image_paths = []
    ocr_results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line.strip())
            base_name = obj["id"]
            found_image = False
            image_filename = base_name + ".jpg"
            full_path = os.path.join(IMAGE_FOLDER_PATH, image_filename)
            image_paths.append(full_path)
            ocr_results.append(obj["page_texts"])
    return image_paths, ocr_results

image_paths, ocr_text = load_test_image(input_jsonl_path)

logging.info(f"Starting image processing...")
results = {}
print("Inferencing.....")
for i, filename in enumerate(image_paths):
    logging.info(f"Processing image ({i+1}/{len(image_paths)}): {filename}")

    # Direct image opening, processing, and inference without try-except
    if image_in_prompt:
        image = Image.open(filename).convert('RGB')
        image_token = "<|image_1|> Your task is to extract the information for the fields provided below from the image"
    else:
        image = None
        image_token = None
    
    if ocr_in_prompt:
        ocred = ocr_text[i]
    else: 
        ocred = None

    messages = [
        {"role": "user", "content": f"""
           {image_token}
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
         <|end|><|assistant|>"""},
    ]
    print(messages)
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    input_token_len = inputs['input_ids'].shape[1]
    generated_part_ids = generate_ids[:, input_token_len:]

    response = processor.batch_decode(generated_part_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if not response:
        print("The script faileds")

    response = response.strip()
    if response.startswith("```json"):
        response = response[len("```json"):].strip()
    if response.startswith("```"):
        response = response[len("```"):].strip()
    if response.endswith("```"):
        response = response[:-len("```")].strip()

    try:
        parsed_json = json.loads(response)
        results[filename] = parsed_json
    except json.JSONDecodeError as json_e:
        print(json_e)
    del inputs
    del generate_ids
    del image
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Output Results ---
if output_json_path:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logging.info("Results saved successfully.")
else:
    print("\n\n=== All Extracted Results (stdout) ===")
    print(json.dumps(results, indent=4, ensure_ascii=False))