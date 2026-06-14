import os
import re
from datasets import load_dataset

# regex
key_words_cfg = 'football_terms_regex.txt'
pattern_terms = '|'.join([i.strip() for i in open(key_words_cfg)])
if pattern_terms[-1] == '|':
    pattern_terms = pattern_terms[:-1]
regex_terms = re.compile(pattern_terms, re.IGNORECASE)

# word frequency
def match_info(example):
    match = regex_terms.findall(example['text'])
    word_count = {}
    for m in match:
        word = m.lower()
        if word in word_count:
            word_count[word]+=1
        else:
            word_count[word]=1
    return {"football_terms": str(word_count)}

# list datasets dir
dataset_dir = os.environ.get('PLFB_PILE_FOOTBALL_ONLY_DIR', './data/pile_football_only')
output_dir = os.environ.get('PLFB_PILE_TERMS_DIR', './runs/pile_football_terms')
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
        filtered_dataset = dataset.filter(lambda example: regex_terms.search(str(example['text'])), batch_size=10000, num_proc=100)
        # save dataset to jsonl
        if len(dataset)!=len(filtered_dataset) and len(filtered_dataset)>0:
            filtered_dataset = filtered_dataset.map(match_info, batch_size=10000, num_proc=100)
            filtered_dataset.to_json(os.path.join(output_dir, f"pile_train_deduped_football{p[-8:-6]}.jsonl"))
            print(f'{p} filtered data: {len(filtered_dataset)}')

