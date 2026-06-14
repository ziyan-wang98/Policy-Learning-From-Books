import os
import re
from datasets import load_dataset, concatenate_datasets

os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(os.environ['HF_HOME'], 'datasets'))

QUOTA = 0.005
# load dataset
dataset_dir = os.environ.get('PLFB_PILE_TERMS10_DIR', './runs/pile_football_terms10')
output_dir = os.environ.get('PLFB_PILE_QUOTA_OUTPUT_DIR', './runs/pile_football_terms_quota')
os.makedirs(output_dir, exist_ok=True)
dataset = concatenate_datasets([load_dataset(path=dataset_dir, data_files=f'pile_train_deduped_football{i:0>2d}.jsonl', split="train") for i in range(20)])
# filter
filtered_dataset = dataset.filter(lambda example: sum(eval(example['football_terms']).values())/len(example['text'].split())>QUOTA, batch_size=2000, num_proc=20)
# save dataset to jsonl
filtered_dataset.to_json(os.path.join(output_dir, f"pile_football_terms_quota{QUOTA}.jsonl"))
print(f'filtered data: {len(filtered_dataset)}')

