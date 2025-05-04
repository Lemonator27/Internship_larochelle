import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import json
from typing import Dict, List 
import argparse
from peft import PeftModel
import json


parser = argparse.ArgumentParser(description="Extract structured information from images using a multimodal model.")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset name for training")
args = parser.parse_args()

dataset = args.dataset
FIXED_SCHEMA_PATH = "/home/bdinhlam/schema/schema.json" 
test_path = f"/home/bdinhlam/scratch/dataset/{dataset}/test-documents.jsonl"
lora_adapter_path = f"/home/bdinhlam/scratch/weight/weight_mistral_{dataset}"
output_json_path = f"/home/bdinhlam/scratch/weight/home/bdinhlam/scratch/weight/weight_mistral_{dataset}/output_{dataset}.json"

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

model = PeftModel.from_pretrained(model, lora_adapter_path)
model = model.merge_and_unload()

print("Finished mergin weights")
def load_json_lines(file_path: str) -> tuple[List[Dict]]:
    print(f"Loading labels from: {file_path}")
    ocr_text: List[str] = []
    id_image: List[str] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            if "id" in obj and "target" in obj:
                ocr_text.append(obj["page_texts"])
                id_image.append(obj["id"])
                
    return ocr_text,id_image

def create_instruct_messages(ocr):
    prompt = f"""
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

    return prompt

def create_list(ocr:List):
    list_of_prompt = []
    for i in range(len(ocr)):
        prompt = create_instruct_messages(ocr[i])
        list_of_prompt.append(prompt)
    return list_of_prompt

def get_eval_results(test_list,id_list):
    eval_results: Dict[str, Dict] = {}
    for i, prompt in enumerate(test_list):
        with torch.no_grad():
            print(f"Processing sample of id {id_list[i]}")
            input_index = tokenizer(prompt,return_tensors="pt").to("cuda")
            print(prompt)
            output_ids = model.generate(**input_index,max_new_tokens = 16000)
            actual_output_tokens = output_ids[:, input_index["input_ids"].shape[1]:]
            eval_output = tokenizer.decode(actual_output_tokens[0], skip_special_tokens=True)
            print(f"Raw model output for sample {id_list[i]}:\n{eval_output}")
            response = eval_output.strip()
            if response.startswith("```json"):
                response = response[len("```json"):].strip()
            if response.startswith("```"):
                response = response[len("```"):].strip()
            if response.endswith("```"):
                response = response[:-len("```")].strip()
            if response == None:
                print("The script failed somewhere")
            try:
                schemas = json.loads(response)
                eval_results[id_list[i]] = schemas
            except json.JSONDecodeError as e:   
                print(e)
    print(f"Finished processing {eval_results} samples")
    return eval_results

if __name__ == "__main__":
    print("Loading json files")
    ocr_test,id_image = load_json_lines(test_path)
    print("Loading prompt")
    prompt_list = create_list(ocr_test)
    print("Start inferencing...")
    eval_results = get_eval_results(prompt_list,id_image)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)
            



