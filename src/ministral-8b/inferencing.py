import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import json
from typing import Dict, List
import argparse
from peft import PeftModel
import ast
parser = argparse.ArgumentParser(description="Extract structured information from images using a multimodal model.")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset name for training")
parser.add_argument("--adapter", type=str, required=True,
                    help="Folder name of the LoRA adapter")
args = parser.parse_args()

dataset = args.dataset
adapter_folder_name = args.adapter

FIXED_SCHEMA_PATH = "/home/bdinhlam/schema/schema.json"
test_path = f"/home/bdinhlam/scratch/dataset/{dataset}/test-documents.jsonl"
lora_adapter_path = f"/home/bdinhlam/scratch/weight/{adapter_folder_name}"
output_json_path = f"/home/bdinhlam/scratch/weight/weight_mistral_cord/output_{dataset}.json"


with open(FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)
fixed_schema_string = json.dumps(schema_dict, indent=2)
print(f"Successfully loaded fixed schema from: {FIXED_SCHEMA_PATH}")

print(f"Loading the adapter from the {lora_adapter_path} folder ...")

model_id = "mistralai/Ministral-8B-Instruct-2410"
print(f"Using base model ID: {model_id}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer.pad_token to tokenizer.eos_token")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, lora_adapter_path)
model = model.merge_and_unload()
model.eval()

print("Finished merging weights and set model to evaluation mode.")

def load_json_lines(file_path: str) -> tuple[List[str], List[str]]:
    print(f"Loading data from: {file_path}")
    ocr_text_list: List[str] = []
    id_image_list: List[str] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                if "page_texts" in obj and "id" in obj:
                    ocr_text_list.append(obj["page_texts"])
                    id_image_list.append(obj["id"])
                else:
                    print(f"Warning: Line {line_number} missing 'page_texts' or 'id'. Skipping.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}. Skipping line.")

    print(f"Loaded {len(ocr_text_list)} OCR texts and {len(id_image_list)} IDs.")
    return ocr_text_list, id_image_list

def create_inference_prompt_content(ocr_text: str, schema_str: str) -> str:
    return f"""
Extract the information in JSON format according to the following JSON schema: {schema_str}, Additional guidelines:
- Extract only the elements that are present verbatim in the document text. Do NOT infer any information.
- Extract each element EXACTLY as it appears in the document.
- Each value in the OCR can only be used AT MOST once. If a value can correspond to multiple fields, pick the best one.
- For each object, output all the keys from the schema even if the value is null. Empty lists should be outputted as lists with no elements.
- If no indication of tax is given, assume the amounts to be gross amounts.
<ocr>
{ocr_text}
</ocr>
"""

def get_eval_results(ocr_list: List[str], id_list: List[str], schema_str: str) -> Dict[str, Dict]:
    eval_results: Dict[str, Dict] = {}
    if not ocr_list:
        print("No OCR texts to process.")
        return eval_results

    for i, ocr_content in enumerate(ocr_list):
        print(f"Processing sample {i+1}/{len(ocr_list)}, ID: {id_list[i]}")

        user_prompt_content = create_inference_prompt_content(ocr_content, schema_str)

        messages = [
            {"role": "user", "content": user_prompt_content}
        ]
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                tokenized_chat,
                max_new_tokens=16000,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )

        response_ids = output_ids[:, tokenized_chat.shape[1]:]
        eval_output = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        response = eval_output.strip()

        if response.startswith("```json"):
            response = response[len("```json"):].strip()
        if response.startswith("```"):
            response = response[len("```"):].strip()
        if response.endswith("```"):
            response = response[:-len("```")].strip()

        print(f"ID {id_list[i]} - Raw Response: '{response}'")

        if not response:
            print(f"Warning: Empty response for ID {id_list[i]}.")
            eval_results[id_list[i]] = {"error": "empty response from model"}
            continue

        try:
            response = ast.literal_eval(response)
            json_file = json.dumps(response)
            parsed_json = json.loads(json_file)
            eval_results[id_list[i]] = parsed_json
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError for ID {id_list[i]}: {e}")
            print(f"Problematic response was: '{response}'")
            eval_results[id_list[i]] = {"error": "JSONDecodeError", "raw_output": response}

    print(f"Finished processing {len(eval_results)} samples")
    return eval_results

if __name__ == "__main__":
    print("Loading test JSONL file...")
    ocr_test_list, id_image_list = load_json_lines(test_path)

    if ocr_test_list:
        print("Starting inferencing...")
        eval_results = get_eval_results(ocr_test_list, id_image_list, fixed_schema_string)

        print(f"Saving all results to: {output_json_path}")
        import os
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=4, ensure_ascii=False)
        print("All results saved.")
    else:
        print("No data loaded from test file. Exiting.")