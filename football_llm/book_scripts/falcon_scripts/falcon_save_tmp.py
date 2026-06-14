import os
import re
from datasets import load_dataset

os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(os.environ['HF_HOME'], 'datasets'))

# regex
pattern_terms = '|'.join([i.strip() for i in open("football_terms_regex.txt")])
pattern_terms = pattern_terms[:-1] if pattern_terms[-1] == '|' else pattern_terms
regex_terms = re.compile(pattern_terms, re.IGNORECASE)

# load dataset
dataset = load_dataset("tiiuae/falcon-refinedweb", num_proc=50, cache_dir=os.environ['HF_DATASETS_CACHE'], ignore_verifications=True)
filtered_dataset = dataset.filter(lambda example: len(set(regex_terms.findall(example['content'])))>=10, batch_size=10000, num_proc=100)
output_dir = os.environ.get('PLFB_FALCON_OUTPUT_DIR', './runs/falcon')
os.makedirs(output_dir, exist_ok=True)
filtered_dataset.to_json(os.path.join(output_dir, 'falcon_football_terms10_tmp.jsonl'))
print(f'filtered data: {len(filtered_dataset)}')
