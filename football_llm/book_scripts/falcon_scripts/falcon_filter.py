import os
import re
from datasets import load_dataset

os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(os.environ['HF_HOME'], 'datasets'))

# regex
pattern_terms = '|'.join([i.strip() for i in open("football_terms_regex.txt")])
pattern_terms = pattern_terms[:-1] if pattern_terms[-1] == '|' else pattern_terms
regex_terms = re.compile(pattern_terms, re.IGNORECASE)

pattern_words = '|'.join([i.strip() for i in open("football_regex3.txt")])
pattern_words = pattern_words[:-1] if pattern_words[-1] == '|' else pattern_words
regex_words = re.compile(pattern_words, re.IGNORECASE)

# word frequency
def match_info(example):
    football_word_count = {}
    match = regex_words.findall(example['content'])
    for m in match:
        word = m.lower()
        if word in football_word_count:
            football_word_count[word]+=1
        else:
            football_word_count[word]=1

    football_terms_count = {}
    match = regex_terms.findall(example['content'])
    for m in match:
        terms = m.lower()
        if terms in football_terms_count:
            football_terms_count[terms]+=1
        else:
            football_terms_count[terms]=1

    return {"football_words": str(football_word_count), "football_terms": str(football_terms_count)}

# load dataset
dataset = load_dataset("tiiuae/falcon-refinedweb", num_proc=50, cache_dir=os.environ['HF_DATASETS_CACHE'], ignore_verifications=True)
filtered_dataset = dataset.filter(lambda example: len(set(regex_terms.findall(example['content'])))>=10, batch_size=10000, num_proc=100)
filtered_dataset = filtered_dataset.map(match_info, batch_size=10000, num_proc=100)
output_dir = os.environ.get('PLFB_FALCON_OUTPUT_DIR', './runs/falcon')
os.makedirs(output_dir, exist_ok=True)
filtered_dataset['train'].to_json(os.path.join(output_dir, 'falcon_football_terms10.jsonl'))
print(f'filtered data: {len(filtered_dataset)}')
