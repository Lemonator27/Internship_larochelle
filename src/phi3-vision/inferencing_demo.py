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

print("Starting...")
model_id = "microsoft/Phi-3-vision-128k-instruct"
IMAGE_FOLDER_PATH = "/Utilisateurs/dbui/sroie/images/" 
FIXED_SCHEMA_PATH = "/Utilisateurs/dbui/json_schema/schema.json" 
lora_adapter_path = "/Utilisateurs/dbui/my_lora_adapters_simplified" 


if not os.path.isdir(IMAGE_FOLDER_PATH):
    logging.error(f"Image folder not found: {IMAGE_FOLDER_PATH}")
    exit()

with open(FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)
fixed_schema_string = json.dumps(schema_dict, indent=2)
logging.info(f"Successfully loaded JSON schema from: {FIXED_SCHEMA_PATH}")


logging.info(f"Loading base model: {model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'
)
print("Finished merging models")


logging.info(f"Merging LoRA adapter from: {lora_adapter_path}")
if not os.path.isdir(lora_adapter_path):
     logging.warning(f"LoRA adapter path not found: {lora_adapter_path}. Proceeding without merging.")
     model = base_model
else:
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model = model.merge_and_unload()
    logging.info("Finished merging LoRA adapter.")


logging.info(f"Loading processor for: {model_id}")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
logging.info("Processor loaded successfully.")


generation_args = {
    "max_new_tokens": processor.tokenizer.model_max_length,
    "temperature": 0.0,
    "do_sample": False,
}
logging.info(f"Generation arguments set: {generation_args}")

SUPPORTED_EXTENSIONS = ('.jpg')

logging.info(f"Starting image processing in folder: {IMAGE_FOLDER_PATH}")
results = {}
print("Inferencing.....")
for filename in os.listdir(IMAGE_FOLDER_PATH):
    if filename.lower().endswith(SUPPORTED_EXTENSIONS):
        image_path = os.path.join(IMAGE_FOLDER_PATH, filename)
        logging.info(f"Processing image: {filename}")

        image = Image.open(image_path).convert('RGB')

        messages = [
            {"role": "user", "content": f"""<|image_1|>
             Your task is to extract the information for the fields provided below from the image. Extract the information in JSON format according to the following JSON schema: {fixed_schema_string}"""},
        ]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(prompt, [image], return_tensors="pt").to(model.device)

        with torch.no_grad():
             generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

        input_token_len = inputs['input_ids'].shape[1]
        generate_ids = generate_ids[:, input_token_len:]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if response == None:
            print("failed scripts")
        print(f"Respones: {response}")
        response = response.strip()
        if response.endswith("```"):
             response = response[:-3].strip()
        parsed_json = json.loads(response)
        logging.info(f"Successfully extracted JSON for {filename}")
        results[filename] = parsed_json


        del inputs
        del generate_ids
        del image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


logging.info("Finished processing all images.")

print("\n\n=== All Extracted Results ===")
print(json.dumps(results, indent=4))
