import os
import re
from datasets import load_dataset

components = ["Pile",
"Books3",
"OpenWebText2",
"ArXiv",
"StackExchange",
"Gutenberg",
"Wikipedia",
"BookCorpus2",
"YoutubeSubtitles"]
dataset_dir = os.environ.get('PLFB_PILE_RAW_DIR', './data/pile_raw/pile_dedup')
output_dir = os.environ.get('PLFB_PILE_META_OUTPUT_DIR', './runs/pile_meta')
os.makedirs(output_dir, exist_ok=True)

files = os.listdir(dataset_dir)
pat = '|'.join([i for i in components])
print(pat)

for p in files:
    x = re.findall('.jsonl$', p)
    if len(x)>0 :
        dpath = os.path.join(dataset_dir, p)
        print(f"loading dataset: {dpath}")
        dataset = load_dataset(path=dataset_dir, data_files=dpath, split="train")
        filtered_dataset = dataset.filter(lambda example: re.search(pat, str(example['meta']), re.IGNORECASE), batch_size=10000, num_proc=50)
        if len(dataset) != len(filtered_dataset):
            filtered_dataset.save_to_disk(os.path.join(output_dir, f"pile_train_deduped_sub{p[-10:]}"))
