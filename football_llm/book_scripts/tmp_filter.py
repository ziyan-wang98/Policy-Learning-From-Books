import json
import os
from pathlib import Path
import tqdm

REDPAJAMA_ROOT = Path(os.environ.get("PLFB_REDJPAJAMA_ROOT", Path(__file__).resolve().parents[2] / "redpajama"))
books_path = ori_book_dir = str(Path(os.environ.get("PLFB_BOOK_JSONL", REDPAJAMA_ROOT / "how_to_be_a_footballer.jsonl")))
expected_keys = {'short_book_title', 'publication_date', 'url'}
books_path_text = ori_book_dir = str(Path(os.environ.get("PLFB_BOOK_TEXT", REDPAJAMA_ROOT / "how_to_be_a_footballer.text")))
with open(books_path_text, 'w') as save_file:
    with open(books_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            
            data = json.loads(line)
            if 'meta' in data and isinstance(data['meta'], dict):
                meta_keys = set(data['meta'].keys())
                # Check if there are any unexpected keys
                if not meta_keys.issubset(expected_keys):
                    title = data['meta']['title']
                else:
                    title = data['meta']['short_book_title']
                
                
                text = data['text']
                text = text.replace('\n\n', '||')
                text = text.replace('\n', ' ')
                paragraphs = text.split('||')
                
                ## The longest chapter is have 11757 characters, which is 2569 words.
                current_chapter = ""
                current_chapter_title = ""
                res_list = []
                
                
                for paragraph in tqdm.tqdm(paragraphs, desc="Processing Paragraphs"):
                    if paragraph.startswith('#') or paragraph.startswith('##'):
                        save_file.write("------\n\n") 
                    save_file.write(paragraph + "\n\n")
