import os
import re
from datasets import load_from_disk

# regex
regex = re.compile('football', re.IGNORECASE)

# list datasets dir
dataset_dir = os.environ.get('PLFB_PILE_RAW_DIR', './data/pile_raw/pile_dedup')
output_dir = os.environ.get('PLFB_PILE_FOOTBALL_ONLY_DIR', './data/pile_football_only')
os.makedirs(output_dir, exist_ok=True)

files = os.listdir(dataset_dir)
for p in files:
    x = re.findall('football\d{2}$', p)
    if len(x)>0 :
        dpath = os.path.join(dataset_dir, p)
        print(f"loading dataset: {dpath}")
        dataset = load_from_disk(dataset_path=dpath)
        filtered_dataset = dataset.filter(lambda example: regex.search(str(example['text']), re.IGNORECASE), batch_size=10000, num_proc=100)
        if len(dataset)!=len(filtered_dataset) and len(filtered_dataset)>0:
            filtered_dataset.to_json(os.path.join(output_dir, f"pile_train_deduped_football{p[-2:]}.jsonl"))
