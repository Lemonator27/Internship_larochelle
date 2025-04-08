import os
import json
import math
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model

# --- Configuration ---
model_id = "microsoft/Phi-3-vision-128k-instruct"
# Define target size for image resizing to save memory
IMAGE_TARGET_SIZE = (512, 512)
# Define writer batch size for map function to potentially reduce peak RAM
MAP_WRITER_BATCH_SIZE = 100

# --- Model and Processor Loading ---
print(f"Loading model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='eager'
)

print(f"Loading processor for: {model_id}")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# --- Prompt Definition ---
messages = [
    {"role": "user", "content": """ <|image_1|> Your task is to extract the information for the fields provided below. Extract the information in JSON format according to the following JSON schema:{
  "$defs": {
    "InvoiceLineItem": {
      "properties": {
        "name": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The name of the menu item",
          "title": "Name"
        },
        "net_unit_price": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The unit price before tax",
          "title": "Net Unit Price"
        },
        "unit_tax": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Tax amount per unit",
          "title": "Unit Tax"
        },
        "gross_unit_price": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Unit price including tax",
          "title": "Gross Unit Price"
        },
        "quantity": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Quantity ordered (can be decimal for weights/volumes/litres)",
          "title": "Quantity"
        },
        "net_price": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Total price before tax (quantity \\u00d7 net_unit_price)",
          "title": "Net Price"
        },
        "tax_amount": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Total tax amount (quantity \\u00d7 unit_tax)",
          "title": "Tax Amount"
        },
        "gross_price": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Total price including tax",
          "title": "Gross Price"
        },
        "sub_items": {
          "anyOf": [
            {
              "items": {
                "$ref": "#/$defs/InvoiceSubLineItem"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Additional components or modifications",
          "identifier_field_name": "nm",
          "title": "Sub Items"
        },
        "net_sub_items_total": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Total price of all sub-items before tax",
          "title": "Net Sub Items Total"
        },
        "gross_sub_items_total": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Total price of all sub-items including tax",
          "title": "Gross Sub Items Total"
        },
        "net_total": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Combined net price of item and sub-items before discounts",
          "title": "Net Total"
        },
        "net_discounts": {
          "anyOf": [
            {
              "items": {
                "type": "string"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Discounts applied to net total of this item",
          "title": "Net Discounts",
          "unordered": true
        },
        "total_tax": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Combined tax amount for item and sub-items",
          "title": "Total Tax"
        },
        "gross_discounts": {
          "anyOf": [
            {
              "items": {
                "type": "string"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Discounts applied to the gross total of this item",
          "title": "Gross Discounts",
          "unordered": true
        },
        "gross_total": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Final price including tax and after discounts",
          "title": "Gross Total"
        }
      },
      "title": "InvoiceLineItem",
      "type": "object"
    },
    "InvoiceSubLineItem": {
      "properties": {
        "name": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The name of the sub-item or modification",
          "title": "Name"
        },
        "net_unit_price": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The unit price of the sub-item before tax",
          "title": "Net Unit Price"
        },
        "unit_tax": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The tax amount per unit of the sub-item",
          "title": "Unit Tax"
        },
        "gross_unit_price": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The unit price of the sub-item including tax",
          "title": "Gross Unit Price"
        },
        "quantity": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The quantity of the sub-item (can be a decimal for items sold by weight or volume)",
          "title": "Quantity"
        },
        "net_price": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The total price of the sub-item before tax",
          "title": "Net Price"
        },
        "tax_amount": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The total tax amount for the sub-item",
          "title": "Tax Amount"
        },
        "gross_price": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The total price of the sub-item including tax",
          "title": "Gross Price"
        }
      },
      "title": "InvoiceSubLineItem",
      "type": "object"
    }
  },
  "properties": {
    "base_taxable_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The base amount that is subject to tax",
      "title": "Base Taxable Amount"
    },
    "net_discounts": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Discounts applied to taxable amount before tax calculation",
      "title": "Net Discounts",
      "unordered": true
    },
    "net_service_charge": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Service charge applied to taxable amount before tax calculation",
      "title": "Net Service Charge"
    },
    "taxable_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The amount that is subject to tax. This is the base amount plus net discounts and net service charges",
      "title": "Taxable Amount"
    },
    "non_taxable_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The base amount that is not subject to tax",
      "title": "Non Taxable Amount"
    },
    "net_total": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Sum of taxable and non-taxable amounts with their modifiers",
      "title": "Net Total"
    },
    "tax_rate": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Tax rate percentage applied to taxable amount",
      "title": "Tax Rate"
    },
    "tax_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Total amount of tax on the invoice",
      "title": "Tax Amount"
    },
    "base_gross_total": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The base amount that is subject to gross discounts and service charges",
      "title": "Base Gross Total"
    },
    "gross_discounts": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Discounts applied to entire net total after tax",
      "title": "Gross Discounts",
      "unordered": true
    },
    "gross_service_charge": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Service charge applied to entire net total after tax",
      "title": "Gross Service Charge"
    },
    "gross_total": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Final amount after all taxes and modifications",
      "title": "Gross Total"
    },
    "rounding_adjustment": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Amount added/subtracted to round to desired precision",
      "title": "Rounding Adjustment"
    },
    "commission_fee": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Commission amount deducted from total",
      "title": "Commission Fee"
    },
    "due_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The amount due for the transaction before considering prior balance",
      "title": "Due Amount"
    },
    "prior_balance": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Previous balance or credit applied to the current transaction",
      "title": "Prior Balance"
    },
    "net_due_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The final amount due after applying prior balance",
      "title": "Net Due Amount"
    },
    "paid_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The total amount paid by the customer",
      "title": "Paid Amount"
    },
    "change_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The amount returned to the customer if overpayment occurred",
      "title": "Change Amount"
    },
    "cash_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The amount paid in cash",
      "title": "Cash Amount"
    },
    "creditcard_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The amount paid by credit card",
      "title": "Creditcard Amount"
    },
    "emoney_amount": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The amount paid using electronic money",
      "title": "Emoney Amount"
    },
    "other_payments": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Amounts paid using other methods (e.g., coupons, vouchers)",
      "title": "Other Payments",
      "unordered": true
    },
    "menutype_count": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The number of distinct menu item types in the order",
      "title": "Menutype Count"
    },
    "menuquantity_sum": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "The total quantity of all menu items ordered",
      "title": "Menuquantity Sum"
    },
    "line_items": {
      "anyOf": [
        {
          "items": {
            "$ref": "#/$defs/InvoiceLineItem"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Detailed list of individual items in the order",
      "identifier_field_name": "nm",
      "title": "Line Items"
    }
  },
  "title": "Invoice",
  "type": "object"
}"""}
]

# --- Data Loading Functions ---

def load_json_lines(file_path):
    print(f"Loading labels from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Label file not found at {file_path}")
        return [], []
    except Exception as e:
        print(f"Error reading label file {file_path}: {e}")
        return [], []

    json_objects = []
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line.strip())
            json_objects.append(obj)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON on line {i+1} in {file_path}")
            continue

    image_ids = [obj["id"] for obj in json_objects if "id" in obj and "target" in obj]
    targets = [obj["target"] for obj in json_objects if "id" in obj and "target" in obj]

    print(f"Loaded {len(image_ids)} labels.")
    return image_ids, targets

def load_data(image_ids, targets, image_base_path, target_size=None):
    """Loads images, resizes them (if target_size is provided), and pairs with labels."""
    print(f"Loading images from: {image_base_path}")
    images_files = []
    labels_as_strings = []
    loaded_count = 0
    skipped_count = 0

    for i, img_id in enumerate(image_ids):
        image_with_extension = f"{img_id}.jpg"
        file_path = os.path.join(image_base_path, image_with_extension)

        try:
            img = Image.open(file_path).convert("RGB") # Ensure RGB

            # *** Add resizing step ***
            if target_size:
                # print(f"Resizing image {img_id} to {target_size}") # Optional: uncomment for verbose logging
                img = img.resize(target_size, Image.Resampling.LANCZOS) # High quality downsampling

            images_files.append(img)
            string_data = json.dumps(targets[i])
            labels_as_strings.append(string_data)
            loaded_count += 1
        except FileNotFoundError:
            print(f"Warning: Image file not found: {file_path}. Skipping entry.")
            skipped_count += 1
            continue
        except Exception as e:
            print(f"Warning: Error loading/resizing image {file_path}: {e}. Skipping entry.")
            skipped_count += 1
            continue

    print(f"Successfully loaded and processed {loaded_count} images. Skipped {skipped_count} entries.")
    return {'image': images_files, 'label': labels_as_strings}

# --- Specify Data Paths ---
base_data_dir = r'/kaggle/input/testing/sroie' # Modify if your base path is different
label_path_train = os.path.join(base_data_dir, 'train-documents.jsonl')
label_path_val = os.path.join(base_data_dir, 'train-documents.jsonl')
label_path_test = os.path.join(base_data_dir, 'test-documents.jsonl')
image_dir_path = os.path.join(base_data_dir, "images")

# --- Load Labels ---
image_ids_train, labels_train = load_json_lines(label_path_train)
image_ids_val, labels_val = load_json_lines(label_path_val)
image_ids_test, labels_test = load_json_lines(label_path_test)

# --- Load Image Data (with resizing) ---
data_dict_train = load_data(image_ids_train, labels_train, image_dir_path, target_size=IMAGE_TARGET_SIZE) if image_ids_train else {'image': [], 'label': []}
data_dict_val = load_data(image_ids_val, labels_val, image_dir_path, target_size=IMAGE_TARGET_SIZE) if image_ids_val else {'image': [], 'label': []}
data_dict_test = load_data(image_ids_test, labels_test, image_dir_path, target_size=IMAGE_TARGET_SIZE) if image_ids_test else {'image': [], 'label': []}

# --- Create Datasets ---
dataset_train = Dataset.from_dict(data_dict_train) if data_dict_train.get('image') else None
dataset_val = Dataset.from_dict(data_dict_val) if data_dict_val.get('image') else None
dataset_test = Dataset.from_dict(data_dict_test) if data_dict_test.get('image') else None

datasets_to_include = {}
if dataset_train:
    datasets_to_include['train'] = dataset_train
    print("\nTraining Dataset Info:")
    print(dataset_train)
if dataset_val:
    datasets_to_include['validation'] = dataset_val
    print("\nValidation Dataset Info:")
    print(dataset_val)
if dataset_test:
    datasets_to_include['test'] = dataset_test
    print("\nTest Dataset Info:")
    print(dataset_test)

if datasets_to_include:
    full_dataset = DatasetDict(datasets_to_include)
    print("\nFull Dataset Structure:")
    print(full_dataset)
else:
    full_dataset = None
    print("\nNo datasets were created successfully.")

def tokenize_function(examples):
    images = examples['image']
    labels = examples['label']
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=images, 
        return_tensors="pt",
        padding="longest",
        truncation=True
    )
    try:
        label_inputs = processor.tokenizer(
            labels, 
            return_tensors="pt",
            padding="longest",
            truncation=True
        )
        model_labels = label_inputs["input_ids"]
        inputs["labels"] = model_labels
    except Exception as e:
        return {}
    return inputs

tokenized_dataset = full_dataset.map(tokenize_function,batched=True,num_proc=1, batch_size=1,
            remove_columns=["image", "label"], 
            writer_batch_size=100, 
            load_from_cache_file=False,
        )
