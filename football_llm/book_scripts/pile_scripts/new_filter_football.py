import os
import re
import argparse
import pickle as pkl
import json
from datasets import load_dataset, Dataset
from transformers import pipeline

## Data Frame:
# {'meta': {'short_book_title','publication_date','url'},'text'}
# {'meta': {'title'},'text'}



def check_meta_and_log_football(books_path, output_file_path):
    
    classifier = pipeline("zero-shot-classification",
                      model="sileod/deberta-v3-base-tasksource-nli", device=0)
    
    # labels=['soccer-related', 'not soccer-related']
    labels=["Football Novel", "Footballer Biography", "Football History", "Football Tactics", "Tutorial"]
    
    expected_keys = {'short_book_title', 'publication_date', 'url'}
    
    output_books_num = 0
    
    with open(books_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            data = json.loads(line)
            if 'meta' in data and isinstance(data['meta'], dict):
                # Get the set of keys in 'meta'
                meta_keys = set(data['meta'].keys())
                # # Check if there are any unexpected keys
                # if not meta_keys.issubset(expected_keys):
                #     prediction = classifier(data['meta']['title'], labels, multi_label=False)
                # else:
                #     prediction = classifier(data['meta']['short_book_title'], labels, multi_label=False)
                
                # Step2: Check with the text
                sample_text = data['text'][0:2000]
                try:
                    # prediction = classifier(sample_text, labels, multi_label=False)
                    prediction = classifier(sample_text, labels)
    
                    if prediction['labels'][0] == "Football Tactics"  or prediction['labels'][1] == "Football Tactics":
                        output_books_num += 1
                        with open(output_file_path, 'a') as output_file:
                            output_file.write(json.dumps(data) + '\n')
                except:
                    print("Error in prediction")
                    import pdb; pdb.set_trace()
                    continue
                
                
                    
                # if prediction['labels'][0] == "soccer-related" and prediction['scores'][0] >= 0.65 or prediction['labels'][1] == "soccer-related" and prediction['scores'][1] >= 0.65:
                #     output_books_num += 1
                #     with open(output_file_path, 'a') as output_file:
                #         output_file.write(json.dumps(data) + '\n')
            print("Processing Books", line_number, "Output Books:", output_books_num, end='\r')
                    
book_dir = "redpajama/football_soccer_filtered.jsonl"

football_dir = "redpajama/football_tactics_soccer.jsonl"


check_meta_and_log_football(book_dir, football_dir)













# print(len(results))
# pkl.dump(results, open(f'./football_list_{args.gpu_id}.pkl', 'wb'))