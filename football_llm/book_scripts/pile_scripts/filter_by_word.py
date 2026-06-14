import os
import re
from datasets import load_dataset

def match_info(example):
    match = re.findall(pat, example['text'], re.IGNORECASE)
    word_count = {}
    for m in match:
        word = m.lower()
        if word in word_count:
            word_count[word]+=1
        else:
            word_count[word]=1
    return {"football_words": str(word_count)}

dataset_dir = os.environ.get('PLFB_PILE_RAW_DIR', './data/pile_raw/pile_dedup')
output_dir = os.environ.get('PLFB_PILE_WORD_OUTPUT_DIR', './runs/pile_word')
os.makedirs(output_dir, exist_ok=True)
key_words_cfg = 'football_hypo.txt'
with open(key_words_cfg, "r") as f:
    word_list = f.readlines()
pat = '|'.join([i.strip() for i in word_list])
if pat[-1] == '|':
    pat = pat[:-1]
print(pat)

files = os.listdir(dataset_dir)
for p in files:
    x = re.findall('.jsonl$', p)
    if len(x)>0 :
        dpath = os.path.join(dataset_dir, p)
        print(f"loading dataset: {dpath}")
        dataset = load_dataset(path=dataset_dir, data_files=dpath, split="train")
        filtered_dataset = dataset.filter(lambda example: re.search(pat, str(example['text']), re.IGNORECASE), batch_size=10000, num_proc=100)
        if len(dataset)!=len(filtered_dataset) and len(filtered_dataset)>0:
            filtered_dataset = filtered_dataset.map(match_info, batch_size=10000, num_proc=100)
            filtered_dataset.save_to_disk(os.path.join(output_dir, f"pile_train_deduped_football{p[-8:-6]}"))
