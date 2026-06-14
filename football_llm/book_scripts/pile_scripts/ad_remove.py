import sys
import re
import argparse
import difflib
import json

def remove_common_parts(text1, text2, min_length=50):
    common_text = []
    s = difflib.SequenceMatcher(None, text1, text2)
    matching_blocks = s.get_matching_blocks()
    for block in matching_blocks[:-1]:
        if block[2]-block[0] > min_length:
            start = block[0]
            end = start + block[2]
            common = text1[start:end]
            common_text.append(common)
            text1 = text1[:start] + text1[end:]
            text2 = re.sub(common, ' ', text2)
    return common_text, text1, text2

def main():
    
    parser = argparse.ArgumentParser(description='A common string removal script for advertisement filtering.')
    parser.add_argument('-l', '--length', default=30, type=str, help='max length of the detected common string')
    parser.add_argument('-i', '--input', default=None, type=str, help='input file path, e.g., "batch1/dedup-md5-pile-openwebtext2.jsonl"')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file path')
    args = parser.parse_args()
        
    if args.input:
        
        print('<=== Start: Loading Raw Text Data ===>')
        
        texts = []
        with open(args.input, 'r', encoding='utf-8') as fin:
            for line in fin:
                data = json.loads(line)
                texts.append(data)
            
        print('<=== Done: Loading Raw Text Data ===>')
        
        print('<=== Start: Processing Raw Text ===>')
        
        sentences = []
        common_text = []
        for sample in texts:
            next = sample['text']
            for common in common_text:
                next = re.sub(common, ' ', next)
            for i, sent in enumerate(sentences):
                common, text1, text2 = remove_common_parts(next, sent, min_length=args.length)
                if len(common) > 0:
                    # print(common)
                    next = text1
                    sentences[i] = text2
                    common_text.extend(common)
            sentences.append(next)
            
        for i, sent in enumerate(sentences):
            texts[i]['text'] = sent
            
        print('<=== Done: Processing Raw Text ===>')
        
        print('<=== Start: Writing Output Text ===>')
        
        outfile = args.output if args.output is not None else args.input.split('.')[0] + '_removed.jsonl'
        with open(outfile, 'w', encoding='utf-8') as json_file:
            json.dump(texts, json_file)
            
        print('<=== Done: Writing Output Text ===>')
    else:
        raise Exception('Please specify input files or directories.')

if __name__ == '__main__':
    sys.exit(main())
