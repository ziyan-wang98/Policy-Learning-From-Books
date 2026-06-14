import json
import os
from tqdm import tqdm


input_file_path = "redpajama/book.jsonl"
cleaned_file_path = "redpajama/book_cleaned.jsonl"
error_file_path = "redpajama/book_error.jsonl"

# Read and validate each line with a progress bar

print("Cleaning the dataset")
with open(input_file_path, 'r') as input_file, \
     open(cleaned_file_path, 'w') as output_file, \
     open(error_file_path, 'w') as error_file:

    for line in input_file:
        try:
            # Try to parse the JSON line
            json_data = json.loads(line)
            # Write the valid line to the output file
            output_file.write(line)
        except json.JSONDecodeError as e:
            # Write lines that cause JSON decoding errors to the error file
            error_file.write(line)

# Now use the datasets library to load the cleaned dataset

print("Finished cleaning the dataset")

from datasets import load_dataset
dataset = load_dataset("json", data_files=cleaned_file_path, cache_dir=".cache/huggingface")

# Rest of your processing
