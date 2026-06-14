import os
import re
from datasets import load_dataset

# list datasets dir
dataset_dir = os.environ.get('PLFB_PILE_TERMS2_DIR', './runs/pile_football_terms2')
output_dir = os.environ.get('PLFB_PILE_TERMS10_DIR', './runs/pile_football_terms10')
os.makedirs(output_dir, exist_ok=True)
files = os.listdir(dataset_dir)
for p in files:
    x = re.findall('football\d{2}.jsonl$', p)
    if len(x)>0 :
        dpath = os.path.join(dataset_dir, p)
        # load dataset
        print(f"loading dataset: {dpath}")
        dataset = load_dataset(path=dataset_dir, data_files=dpath, split="train")
        # filter
        filtered_dataset = dataset.filter(lambda example: len(list(eval(example['football_terms']).keys()))>=10, batch_size=2000, num_proc=30)
        # save dataset to jsonl
        if len(dataset)!=len(filtered_dataset) and len(filtered_dataset)>0:
            filtered_dataset.to_json(os.path.join(output_dir, f"pile_train_deduped_football{p[-8:-6]}.jsonl"))
            print(f'{p} filtered data: {len(filtered_dataset)}')

