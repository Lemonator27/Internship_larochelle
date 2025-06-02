from unsloth import FastLanguageModel
import torch
import json # Added
import re
from typing import Dict, List, Any, Tuple # Added Tuple
from vllm import SamplingParams # Added for GRPOTrainer
from trl import GRPOConfig, GRPOTrainer # Added for GRPOTrainer
import numpy as np # Added
import datasets # Added, assuming dataset is loaded later

# For Verifier Model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # Removed Aliases
from peft import PeftModel # Removed Alias

# --- Configuration for the Main Model ---
max_seq_length = 2048
lora_rank = 32

# --- Configuration for the Verifier Model ---
VERIFIER_MODEL_BASE_ID = "mistralai/Mistral-7B-Instruct-v0.2" 
VERIFIER_ADAPTER_PATH = "/path/to/your/mistral_verifier_lora_adapter/" # <<<< UPDATE THIS
VERIFIER_SCHEMA_PATH = "/path/to/your/verifier_schema.json" # <<<< UPDATE THIS 

# Global variables for the verifier model and its components
verifier_model_instance = None
verifier_tokenizer_instance = None
verifier_schema_dict_global = None # This is the schema for the VERIFIER'S prompt
VERIFIER_DEVICE = "cuda:0" 

# --- Configuration for SFT-style Dataset Loading ---
# This schema is used for constructing the prompt for the MAIN Qwen3 model
SFT_FIXED_SCHEMA_PATH = "/home/bdinhlam/schema/schema.json" # Path to schema used in SFT prompt
SFT_DATASET_NAME = "sroie" # <<<< UPDATE THIS if not using argparse from SFT script
SFT_TRAIN_JSONL_PATH = f"/home/bdinhlam/scratch/dataset/{SFT_DATASET_NAME}/train-documents.jsonl" # <<<< UPDATE THIS if path structure is different or SFT_DATASET_NAME is not set

sft_fixed_schema_string_global = None # Schema string for the MAIN Qwen3 model's prompt

# --- Function to Load Verifier Model (Call this once) ---
def load_verifier_model_globally():
    global verifier_model_instance, verifier_tokenizer_instance, verifier_schema_dict_global
    if verifier_model_instance is not None:
        print("Verifier model already loaded.")
        return

    print(f"Loading verifier model: {VERIFIER_MODEL_BASE_ID} with adapter from {VERIFIER_ADAPTER_PATH}")
    print("NOTE: Verifier model quantization is DISABLED. This will use more GPU memory.")
    
    # Removed broad try-except. Errors here will halt the script.
    base_model = AutoModelForCausalLM.from_pretrained(
        VERIFIER_MODEL_BASE_ID,
        device_map=VERIFIER_DEVICE, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 
    )
    verifier_model_instance = PeftModel.from_pretrained(base_model, VERIFIER_ADAPTER_PATH)
    verifier_model_instance = verifier_model_instance.eval() 

    verifier_tokenizer_instance = AutoTokenizer.from_pretrained(VERIFIER_MODEL_BASE_ID, trust_remote_code=True)
    if verifier_tokenizer_instance.pad_token is None:
        verifier_tokenizer_instance.pad_token = verifier_tokenizer_instance.eos_token
    verifier_tokenizer_instance.padding_side = "right"

    # Kept try-except for critical file loading
    try:
        with open(VERIFIER_SCHEMA_PATH, 'r', encoding='utf-8') as f:
            verifier_schema_dict_global = json.load(f)
        print("Verifier model (non-quantized), tokenizer, and schema loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Verifier schema file not found at {VERIFIER_SCHEMA_PATH}. Reward function will fail.")
        verifier_schema_dict_global = None # Ensure it's None if loading failed
        # Optionally, set verifier_model_instance to None to disable verifier reward
        # verifier_model_instance = None 
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from verifier schema file: {VERIFIER_SCHEMA_PATH}. Reward function will fail.")
        verifier_schema_dict_global = None
        # verifier_model_instance = None


# --- Function to Load SFT Schema (for main model's prompt) ---
def load_sft_schema_globally():
    global sft_fixed_schema_string_global
    with open(SFT_FIXED_SCHEMA_PATH, 'r', encoding='utf-8') as f:
        sft_schema_dict = json.load(f)
    sft_fixed_schema_string_global = json.dumps(sft_schema_dict, indent=2)
    print(f"Successfully loaded SFT schema string from: {SFT_FIXED_SCHEMA_PATH}")

# --- Main Model Loading ---
print("Loading main model (Qwen3-4B-Base)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False, 
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = lora_rank*2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# --- Prompting and Chat Template Setup for Main Model (GRPO-style) ---
reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>" 
solution_end    = "</SOLUTION>"
grpo_system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{grpo_system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"
chat_template = chat_template\
    .replace("'{grpo_system_prompt}'",   f"'{grpo_system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

# --- Load Verifier Model and SFT Schema (call these once before training) ---
print("Preparing to load verifier model...")
load_verifier_model_globally()
print("Preparing to load SFT schema for main model prompts...")
load_sft_schema_globally()

# --- Regex for Extracting Solution from Main Model Output ---
solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
    "(?:" + re.escape(tokenizer.eos_token) + ")?"
match_format = re.compile(
    rf"{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end_regex}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

# --- Existing Reward Functions (for format checking) ---
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0; response = completion[0]["content"]
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0; response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

# --- Helper to Create Prompt for Verifier Model ---
def create_prompt_for_verifier_model(json_to_verify_str: str, schema_dict: Dict) -> str:
    # Kept try-except here as main model's output might be invalid JSON
    try: 
        parsed_json_to_verify = json.loads(json_to_verify_str)
    except json.JSONDecodeError: 
        print(f"Warning: Could not parse solution as JSON for verifier: {json_to_verify_str[:100]}...")
        return None 
    
    if schema_dict is None: # Check if verifier schema failed to load
        print("Warning: Verifier schema is not available. Cannot create verifier prompt.")
        return None

    schema_for_prompt = json.dumps(schema_dict, indent=2)
    rules_text = """- `taxable_amount` should be `base_taxable_amount - net_discounts + net_service_charge`
- `net_total` should be `taxable_amount + non_taxable_amount`
- `tax_amount` should be `taxable_amount * tax_rate`
- `gross_total` should be `net_total + tax_amount - gross_discounts + gross_service_charge`
- `paid_amount` should equal `net_due_amount`
- All monetary values should be numerically consistent according to the rules.
- Data types should adhere to the schema."""
    prompt_content = f"""You are an expert data verifier for invoices. Your task is to determine if the provided 'Extracted JSON' is accurate and consistent with the general invoice 'Schema' and the given financial 'Rules'.
**Schema:**
```json
{schema_for_prompt}
```
**Rules for Verification:**
{rules_text}
**Extracted JSON to Verify:**
```json
{json.dumps(parsed_json_to_verify, indent=2)}
```
Based on the Schema and the Rules, is the Extracted JSON correct?
Respond with only one word: "Correct" or "Incorrect".
"""
    messages = [{'role': 'user', 'content': prompt_content.strip()}]
    if verifier_tokenizer_instance:
        return verifier_tokenizer_instance.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return None

# --- Reward Function using the Verifier Model ---
global VERIFIER_PRINTED_TIMES; VERIFIER_PRINTED_TIMES = 0
global VERIFIER_PRINT_EVERY_STEPS; VERIFIER_PRINT_EVERY_STEPS = 10

def check_completion_with_verifier(prompts, completions, answer, **kwargs):
    global VERIFIER_PRINTED_TIMES, VERIFIER_PRINT_EVERY_STEPS
    if verifier_model_instance is None or verifier_tokenizer_instance is None or verifier_schema_dict_global is None:
        # print("Verifier model/tokenizer/schema not available. Assigning low reward.")
        return [-5.0] * len(completions)
    scores = []
    for i, completion_obj in enumerate(completions):
        completion_text = completion_obj[0]["content"]; extracted_solution_match = match_format.search(completion_text)
        score = -3.0 
        if extracted_solution_match:
            solution_json_str = extracted_solution_match.group(1).strip()
            verifier_prompt = create_prompt_for_verifier_model(solution_json_str, verifier_schema_dict_global)
            if verifier_prompt:
                # Kept try-except for the verifier inference call itself
                try:
                    inputs = verifier_tokenizer_instance(verifier_prompt, return_tensors="pt").to(VERIFIER_DEVICE)
                    with torch.no_grad():
                        outputs = verifier_model_instance.generate(**inputs, max_new_tokens=10, pad_token_id=verifier_tokenizer_instance.eos_token_id, eos_token_id=verifier_tokenizer_instance.eos_token_id, do_sample=False)
                    verifier_response_text = verifier_tokenizer_instance.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
                    if "correct" in verifier_response_text: score = 5.0
                    elif "incorrect" in verifier_response_text: score = -2.0
                    else: score = -2.5 
                    if VERIFIER_PRINTED_TIMES % VERIFIER_PRINT_EVERY_STEPS == 0: print(f"\n--- Verifier Check ---\nProblem: {prompts[i][-1]['content'][:100]}...\nMain Model Solution JSON: {solution_json_str[:100]}...\nVerifier Response: {verifier_response_text}\nAssigned Score: {score}\n----------------------")
                except Exception as e: 
                    print(f"Error during verifier inference: {e}"); score = -4.0 
            else: 
                score = -3.5
                if VERIFIER_PRINTED_TIMES % VERIFIER_PRINT_EVERY_STEPS == 0: print(f"\n--- Verifier Check ---\nFailed to create verifier prompt (likely invalid JSON from main model): {solution_json_str[:100]}...\nAssigned Score: {score}\n----------------------")
        elif VERIFIER_PRINTED_TIMES % VERIFIER_PRINT_EVERY_STEPS == 0: 
             print(f"\n--- Verifier Check ---\nSolution format not matched in completion: {completion_text[:100]}...\nAssigned Score: {score}\n----------------------")
        scores.append(score)
        if i == 0: VERIFIER_PRINTED_TIMES +=1 
    return scores

# --- SFT-style Dataset Loading and Preprocessing for GRPO ---
def sft_load_json_lines(file_path: str) -> tuple[List[Dict], List[str]]:
    print(f"Loading SFT-style data from: {file_path}")
    target_json_dicts: List[Dict] = []
    ocr_texts_list: List[str] = []
    # Kept try-except for critical file loading
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Removed inner try-except for individual line parsing.
                # Errors here will halt the script.
                obj = json.loads(line.strip()) 
                if "target" in obj and "page_texts" in obj:
                    target_json_dicts.append(obj["target"]) 
                    ocr_texts_list.append(obj["page_texts"])
                else:
                    print(f"Warning: Line {line_num} in {file_path} is missing 'target' or 'page_texts'. Skipping.")
        print(f"Loaded {len(target_json_dicts)} entries from {file_path}.")
    except FileNotFoundError:
        print(f"ERROR: SFT data file not found at {file_path}. Exiting.")
        exit()
    except json.JSONDecodeError as e: # Catch if the whole file is not JSON, or first error line
        print(f"ERROR: Could not decode JSON in {file_path} (likely malformed file or line): {e}. Exiting.")
        exit()
    return target_json_dicts, ocr_texts_list

def sft_create_user_prompt_for_main_model(ocr_text: Any, schema_str: str) -> str:
    if isinstance(ocr_text, list):
        ocr_content = "\n".join(ocr_text)
    else:
        ocr_content = str(ocr_text)
    user_prompt = f"""Extract the information in JSON format according to the following JSON schema: {schema_str}, Additional guidelines:
- Extract only the elements that are present verbatim in the document text. Do NOT infer any information.
- Extract each element EXACTLY as it appears in the document.
- Each value in the OCR can only be used AT MOST once. If a value can correspond to multiple fields, pick the best one.
- For each object, output all the keys from the schema even if the value is null. Empty lists should be outputted as lists with no elements.
- If no indication of tax is given, assume the amounts to be gross amounts.
<ocr>
{ocr_content}
</ocr>
Please read the text carefully and follow the instructions to produce the JSON output.
"""
    return user_prompt.strip()

print("--- Preparing GRPO Dataset using SFT-style data ---")
if sft_fixed_schema_string_global is None:
    print("CRITICAL: SFT Schema string not loaded. Cannot create main model prompts. Exiting.")
    exit()

sft_target_jsons, sft_ocr_texts = sft_load_json_lines(SFT_TRAIN_JSONL_PATH)

grpo_dataset_list = []
if not sft_target_jsons or not sft_ocr_texts or len(sft_target_jsons) != len(sft_ocr_texts):
    print("CRITICAL: Problem with loaded SFT data (empty or misaligned). Exiting.")
    exit()

for i in range(len(sft_target_jsons)):
    target_json_dict = sft_target_jsons[i]
    ocr = sft_ocr_texts[i]
    user_content_for_main_model = sft_create_user_prompt_for_main_model(ocr, sft_fixed_schema_string_global)
    grpo_prompt_messages = [
        {"role": "system", "content": grpo_system_prompt},
        {"role": "user", "content": user_content_for_main_model}
    ]
    answer_json_string = json.dumps(target_json_dict)
    grpo_dataset_list.append({"prompt": grpo_prompt_messages, "answer": answer_json_string})

if not grpo_dataset_list:
    print("CRITICAL: No data processed for GRPO training. Exiting.")
    exit()

dataset = datasets.Dataset.from_list(grpo_dataset_list)
print(f"Created GRPO dataset with {len(dataset)} examples from SFT data.")

print("Tokenizing GRPO dataset for length calculation...")
tokenized_grpo_dataset = dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
)
tokenized_grpo_dataset = tokenized_grpo_dataset.map(lambda x: {"L" : len(x["tokens"])})

maximum_prompt_length_calculated = int(np.quantile(tokenized_grpo_dataset["L"], 0.9)) if len(tokenized_grpo_dataset["L"]) > 0 else max_seq_length
print("Max GRPO Prompt Length (90th percentile based on Qwen3 tokenizer) = ", maximum_prompt_length_calculated)

indices_to_keep = np.where(np.array(tokenized_grpo_dataset["L"]) <= maximum_prompt_length_calculated)[0]
dataset = dataset.select(indices_to_keep)
print(f"GRPO dataset filtered. New number of examples: {len(dataset)}")

max_prompt_length_for_trainer = maximum_prompt_length_calculated + 1 
max_completion_length_for_trainer = max_seq_length - max_prompt_length_for_trainer
if max_completion_length_for_trainer <= 50:
    print(f"Warning: max_completion_length is very low ({max_completion_length_for_trainer}). Adjusting.")
    max_completion_length_for_trainer = max(128, max_seq_length // 4) 
    max_prompt_length_for_trainer = max_seq_length - max_completion_length_for_trainer
print(f"Using max_prompt_length: {max_prompt_length_for_trainer}, max_completion_length: {max_completion_length_for_trainer}")

vllm_sampling_params = SamplingParams(
    min_p = 0.1, top_p = 1.0, top_k = -1, seed = 3407,
    stop = [tokenizer.eos_token, solution_end], 
    include_stop_str_in_output = True,
    max_tokens = max_completion_length_for_trainer,
)
training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params, temperature = 1.0, learning_rate = 5e-6,
    weight_decay = 0.01, warmup_ratio = 0.1, lr_scheduler_type = "linear", optim = "adamw_8bit",
    logging_steps = 1, per_device_train_batch_size = 1, gradient_accumulation_steps = 1,
    num_generations = 4, max_prompt_length = max_prompt_length_for_trainer,
    max_completion_length = max_completion_length_for_trainer, max_steps = 100, save_steps = 100,
    report_to = "none", output_dir = "outputs_grpo_verifier_sft_data_no_quant_min_try", # Updated output dir
    remove_unused_columns=False,
)

print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model = model, processing_class = tokenizer, 
    reward_funcs = [match_format_exactly, match_format_approximately, check_completion_with_verifier],
    args = training_args, train_dataset = dataset,
)

print("Starting GRPO training...")
# Kept try-except for the main training loop
try:
    trainer.train()
    print("GRPO training finished.")
except Exception as e:
    print(f"An error occurred during GRPOTrainer.train(): {e}")
    import traceback
    traceback.print_exc()


print("Saving LoRA adapter for the main model...")
model.save_lora("grpo_qwen_verified_sft_data_no_quant_min_try_lora") # Updated save name

# Kept try-except for this post-training verification step
from safetensors import safe_open
try:
    with safe_open("grpo_qwen_verified_sft_data_no_quant_min_try_lora/adapter_model.safetensors", framework = "pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.numel() > 0: assert(not torch.all(tensor == 0).item()), f"Tensor {key} is all zeros."
    print("LoRA adapter for main model seems to be trained (non-zero weights found).")
except Exception as e: print(f"Could not verify LoRA adapter: {e}")

print("\n--- Inference Example with Trained Main Model ---")
example_ocr_for_inference = "Sample OCR text for a widget sale: Item: SuperWidget, Qty: 3, Unit Price: $50.00, Subtotal: $150.00, Tax (10%): $15.00, Total: $165.00. Invoice #INV-2025-001. Date: 2025-06-01."
inference_user_content = sft_create_user_prompt_for_main_model(example_ocr_for_inference, sft_fixed_schema_string_global)
messages_for_inference = [
    {"role": "system", "content": grpo_system_prompt},
    {"role": "user",   "content": inference_user_content},
]
inference_prompt_text = tokenizer.apply_chat_template(messages_for_inference, add_generation_prompt = True, tokenize = False)
inference_sampling_params = SamplingParams(
    temperature = 0.1, top_p = 0.9, max_tokens = max_completion_length_for_trainer,
    stop = [tokenizer.eos_token, solution_end], include_stop_str_in_output = True,
)
print(f"\nGenerating completion for prompt (user part): {inference_user_content[:150]}...")
generated_outputs = model.fast_generate(
    inference_prompt_text, sampling_params = inference_sampling_params,
)[0].outputs[0].text
print("\nGenerated Output:")
print(generated_outputs)

print("\nScript finished.")
