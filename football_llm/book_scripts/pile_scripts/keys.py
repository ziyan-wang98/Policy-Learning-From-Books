import json
from collections import defaultdict

def infer_data_structure(file_path, num_lines=1000):
    structure = defaultdict(set)
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= num_lines:
                break
            try:
                data = json.loads(line)
                for key, value in data.items():
                    structure[key].add(type(value).__name__)
            except json.JSONDecodeError:
                continue
    return {key: list(value_types) for key, value_types in structure.items()}

file_path = "redpajama/book_cleaned.jsonl"
inferred_structure = infer_data_structure(file_path)

print("Inferred Data Structure:")
for key, types in inferred_structure.items():
    print(f"{key}: {types}")
