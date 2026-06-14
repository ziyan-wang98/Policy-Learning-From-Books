import os
import re
from datasets import load_dataset

# regex
regex = re.compile('football', re.IGNORECASE)

dataset_dir = os.environ.get('PLFB_WIKI_DUMP_DIR', './data/wiki_dump/07')
data_file = os.environ.get('PLFB_WIKI_DATA_FILE', 'wiki20230701.json')
dataset = load_dataset(path=dataset_dir, data_files=data_file, split="train")
filtered_dataset = dataset.filter(lambda example: regex.search(example['text'], re.IGNORECASE), batch_size=10000, num_proc=50)
output_path = os.environ.get('PLFB_WIKI_FOOTBALL_OUTPUT', f"{data_file.split('.')[0]}_football.jsonl")
filtered_dataset.to_json(output_path)
