import json
import datasets

def create_sequential_text(valid_line_data: list) -> str:
    """
    Extracts all text from the 'valid_line' entries in the exact order
    they appear in the file and joins them together.
    """
    all_words_in_order = []
    for line_group in valid_line_data:
        for word_info in line_group.get("words", []):
            if "text" in word_info:
                all_words_in_order.append(word_info["text"])
    return " ".join(all_words_in_order)

def main():
    """
    Main function to load, process, and save the CORD dataset using the
    original gt_parse ground truth.
    """
    # 1. Load the dataset from Hugging Face Hub
    print("Loading 'naver-clova-ix/cord-v2' dataset from Hugging Face...")
    try:
        cord_dataset = datasets.load_dataset("naver-clova-ix/cord-v2")
        print("✅ Dataset loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load dataset. Please check your internet connection. Error: {e}")
        return

    # 2. Process each split (train, validation, test)
    for split_name, dataset_split in cord_dataset.items():
        print(f"\nProcessing '{split_name}' split...")
        
        processed_data = []
        
        for i, example in enumerate(dataset_split):
            try:
                # The 'ground_truth' field is a JSON string, so we need to parse it
                ground_truth_data = json.loads(example['ground_truth'])
                valid_line_data = ground_truth_data.get("valid_line", [])

                # --- Perform the two processing tasks ---

                # Task 1: Create the simple, sequential OCR text
                ocr_text = create_sequential_text(valid_line_data)
                
                # Task 2: Use the original ground truth parse directly
                # This is the key change from the previous script.
                parsed_gt = ground_truth_data.get("gt_parse", {})

                # Create the final record with an incremental ID
                processed_data.append({
                    "id": i + 1,
                    "ocr_text": ocr_text,
                    "parsed_gt": parsed_gt
                })

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping record {i} in '{split_name}' split due to a parsing error: {e}")
                continue
        
        # 3. Save the processed data to a new JSON file
        output_filename = f"cord_v2_{split_name}_processed.json"
        print(f"Saving processed data to '{output_filename}'...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(processed_data)} records to '{output_filename}'.")

if __name__ == "__main__":
    # You might need to install the 'datasets' library first:
    # pip install datasets
    main()