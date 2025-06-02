from transformers import TextStreamer
import torch
from unsloth import FastVisionModel
import json
from typing import Dict, List, Any
import argparse
from PIL import Image
import os
from datasets import Dataset
from qwen_vl_utils import process_vision_info
import re
# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Extract structured information from images using a multimodal model.")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset name (e.g., 'mydataset' if path is /home/user/scratch/dataset/mydataset)")
parser.add_argument("--image_in_prompt", action='store_true',
                    help="If image should be in the prompt (currently always included by create_chat_messages)")
parser.add_argument("--ocr_in_prompt", action='store_true',
                    help="If ocr should be in the prompt (currently always included by create_chat_messages)")
args = parser.parse_args()

dataset_name = args.dataset
USER_HOME_DIR =  "/home/bdinhlam"
BASE_SCRATCH_PATH = os.path.join(USER_HOME_DIR, "scratch") 

FIXED_SCHEMA_PATH = os.path.join(USER_HOME_DIR, "schema", "schema.json") 
dataset_base_dir = os.path.join(BASE_SCRATCH_PATH, "dataset", dataset_name) 
image_files_dir = os.path.join(dataset_base_dir, "images")
jsonl_file_path = os.path.join(dataset_base_dir, "test-documents.jsonl")

output_dir = os.path.join(BASE_SCRATCH_PATH, "output", dataset_name) 
os.makedirs(output_dir, exist_ok=True)
output_json_path = os.path.join(output_dir, f"results_qwen_vl_{dataset_name}.json")

# save_model_path = f"/home/bdinhlam/scratch/weight/weight_qwen_{dataset_name}_new/" # Defined but not used in this inference script


with open(FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)
fixed_schema_string = json.dumps(schema_dict, indent=2)
print(f"Successfully loaded fixed schema from: {FIXED_SCHEMA_PATH}")
# --- Data Loading Function ---
def load_json_lines(file_path: str, images_dir: str) -> Dict[str, List[Any]]:
    print(f"Loading data from: {file_path}")
    processed_entries = []

    if not os.path.exists(file_path):
        print(f"Error: Data file not found {file_path}")
        return {"targets": [], "ocr_text": [], "images": [], "ids": []}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            obj = json.loads(line.strip())
            image_doc_id = str(obj["id"]) 
            image_filename = image_doc_id + ".jpg"  
            image_full_path = os.path.join(images_dir, image_filename)
            image_as_object = Image.open(image_full_path).convert("RGB")
            current_ocr = obj["page_texts"]
            processed_entries.append({
                "target": json.dumps(obj["target"]), 
                "ocr_text": current_ocr,
                "images": image_as_object,
                "ids": image_doc_id     
            })
    
    dataset_dict: Dict[str, List[Any]] = {}
    dataset_dict["targets"] = [entry["target"] for entry in processed_entries]
    dataset_dict["ocr_text"] = [entry["ocr_text"] for entry in processed_entries]
    dataset_dict["images"] = [entry["images"] for entry in processed_entries]
    dataset_dict["ids"] = [entry["ids"] for entry in processed_entries]
    print(f"Successfully loaded {len(dataset_dict['ids'])} entries from {file_path}.")
        
    return dataset_dict

print(f"Attempting to load data from: {jsonl_file_path} with images from {image_files_dir}")
dataset_loaded_dict = load_json_lines(jsonl_file_path, image_files_dir)

if not dataset_loaded_dict["ids"]:
    print("No data to process after loading. Exiting.")
    exit()

dataset_to_process = Dataset.from_dict(dataset_loaded_dict)

# --- System Message and Prompt Formatting ---
system_message_content = """You are a highly capable Vision Language Model for structured data extraction.
                     You will be provided with OCR text, an image context, and a JSON schema.
                     Your primary directive is to follow all user instructions meticulously to populate the schema using only information explicitly present in the text.
                     Ensure your output is valid JSON and strictly conforms to all given constraints."""

def create_chat_messages_for_sample(sample: Dict) -> List[Dict]:
    user_query_text = f"""
    Extract the information in JSON format according to the following JSON schema: {fixed_schema_string}, Additional guidelines:
    - Extract only the elements that are present verbatim in the document text. Do NOT infer any information.
    - Extract each element EXACTLY as it appears in the document.
    - Each value in the OCR can only be used AT MOST once. If a value can correspond to multiple fields, pick the best one.
    - For each object, output all the keys from the schema even if the value is null. Empty lists should be outputted as lists with no elements.
    - If no indication of tax is given, assume the amounts to be gross amounts.
    <ocr>
    {sample['ocr_text']}
    </ocr>
    Please read the text carefully and follow the instructions.
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message_content}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample["images"]}, # PIL.Image object
            {"type": "text", "text": user_query_text},
        ]} 
    ]

all_chat_formatted_samples = [create_chat_messages_for_sample(sample) for sample in dataset_to_process]

print(f"Loading model ...")
model, tokenizer = FastVisionModel.from_pretrained(
    f'/home/bdinhlam/scratch/weight/weight_qwen_{dataset_name}_new/ocr_image/',
    load_in_4bit=False,
)
print("Model and tokenizer loaded successfully.")
FastVisionModel.for_inference(model) # Unsloth specific optimization for inference
print("Model prepared for inference.")

# --- (Optional) describe_image function ---
# This function was in your original script. It's kept here if you need it as a separate utility.
# It is not used in the main JSON extraction loop below.
text_streamer_for_describe = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
results_dict = {}
document_ids_list = dataset_to_process["ids"] # Get the list of document IDs
amount_valid = 0
total = 0
print(f"\nStarting JSON extraction for {len(all_chat_formatted_samples)} documents...")
for i, full_chat_sample in enumerate(all_chat_formatted_samples):
    current_doc_id = document_ids_list[i]
    print(f"Processing document ID: {current_doc_id} ({i+1}/{len(all_chat_formatted_samples)})")
    messages_for_inference_input = [
        full_chat_sample[1]  
    ]
    image_inputs, _ = process_vision_info(messages_for_inference_input)
    
    input_text_prompt_string = tokenizer.apply_chat_template(
        messages_for_inference_input, 
        add_generation_prompt=True, 
        tokenize=False
    )
    inputs = tokenizer(
        text=[input_text_prompt_string],
        images = image_inputs, 
        return_tensors='pt',
    ).to(model.device)
    print(inputs)

    content_to_parse = model.generate( 
        **inputs, 
        streamer=text_streamer_for_describe, 
        use_cache=True,
        do_sample=False, 
    )
    actual_output_tokens = content_to_parse[:, inputs["input_ids"].shape[1]:]
    eval_output = tokenizer.decode(actual_output_tokens[0], skip_special_tokens=True)
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*|(?:\[[^\[\]]*\]))*\}'
    if eval_output.startswith("```json"):
        eval_output = eval_output[len("```json"):].strip()
    if eval_output.startswith("```"):
        eval_output = eval_output[len("```"):].strip()
    if eval_output.endswith("```"):
        eval_output = eval_output[:-len("```")].strip()
    if eval_output.endswith("</json>"):
        eval_output = eval_output[:-len("</json>")].strip()
    eval_output = re.findall(json_pattern, eval_output)
    print("Post-filtering: ", eval_output)
    total+=1
    try:
        parsed_json_output = json.loads(eval_output[0])
        results_dict[current_doc_id] = parsed_json_output
        print(f"  Successfully extracted and parsed JSON for {current_doc_id}.")
        amount_valid+=1
        print(f"  Amount of valid document being {amount_valid/total}")
    except json.JSONDecodeError as e:
        print(f"  JSONDecodeError for {current_doc_id}: {e}")
        print(f"  Failed content was: {content_to_parse}")
        results_dict[current_doc_id] = {"error": str(eval_output[0])}

    # --- Incremental Save ---
    if (i + 1) % 10 == 0 or (i + 1) == len(all_chat_formatted_samples): # Save every 10 or at the end
        print(f"Saving intermediate results ({i+1}/{len(all_chat_formatted_samples)}) to {output_json_path}...")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f_out:
                json.dump(results_dict, f_out, indent=4, ensure_ascii=False)
            print(f"  Results successfully saved.")
        except Exception as e_save:
            print(f"  Error saving results: {e_save}")


print("\n----------------------------------------------------")
print(f"All {len(all_chat_formatted_samples)} documents processed.")
print(f"Final results are saved to: {output_json_path}")
print("Script finished.")
