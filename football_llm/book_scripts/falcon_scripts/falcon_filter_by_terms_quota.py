import os
from datasets import load_dataset

os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(os.environ['HF_HOME'], 'datasets'))

QUOTA = 0.005
# load dataset
dataset_dir = os.environ.get('PLFB_FALCON_DIR', './runs/falcon')
output_dir = os.environ.get('PLFB_FALCON_OUTPUT_DIR', './runs/falcon')
os.makedirs(output_dir, exist_ok=True)
dataset = load_dataset(path=dataset_dir, data_files=f'falcon_football_terms10.jsonl', split="train")
# filter
filtered_dataset = dataset.filter(lambda example: sum(eval(example['football_terms']).values())/len(example['content'].split())>QUOTA, batch_size=2000, num_proc=20)
# save dataset to jsonl
filtered_dataset.to_json(os.path.join(output_dir, f"falcon_football_quota{QUOTA}.jsonl"))
print(f'filtered data: {len(filtered_dataset)}')
