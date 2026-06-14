from openai import OpenAI
import openai
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
DATA_ROOT = Path(os.environ.get("PLFB_DATASET_PATH", PROJECT_ROOT / "data")).resolve()
REDPAJAMA_ROOT = Path(os.environ.get("PLFB_REDJPAJAMA_ROOT", PROJECT_ROOT.parent / "redpajama")).resolve()
LLAMA_INDEX_DATA_ROOT = Path(os.environ.get("PLFB_LLAMA_INDEX_DATA", DATA_ROOT / "llama_index")).resolve()
import re
import argparse
import pickle as pkl
import json
from langdetect import detect
from transformers import pipeline
from tqdm import tqdm
import numpy as np
import pprint
from utils import *
from prompt_templete import *
filter_stringency_level = 'strict'
derive_element_type = 'Policy'
model_name = os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")
model_name_stage_1 = os.environ.get("PLFB_OPENAI_FILTER_MODEL", model_name)
asked_types = ["Policy", "Dynamics", "Reward", "None"]

expected_keys = {'short_book_title', 'publication_date', 'url'}
agg_num = 5

# step1: enough information to derive Model, Reward, and Policy.
# step2: summarize the specific theorem, principle, rule, or law, from the paragraph.
def two_stage_query(user_prompt, model_name_stage_1, model_name_stage_2):

    global_stage1_prompt = stage1_prompt(filter_stringency_level, derive_element_type)
    global_stage2_prompt = stage2_prompt(filter_stringency_level, derive_element_type)
    print("============STAGE 1==========")
    user_prompt_stage1 = "The paragraph to analysis: \n" + user_prompt
    query_res = query(global_stage1_prompt, user_prompt_stage1, model_name_stage_1)
    if query_res is None or query_res['answer'] == 0:
        return 0, None
    elif query_res['answer'] == 1:
        print("============STAGE 2==========")
        explanation = query_res['explanation']
        user_prompt_stage2 = "The paragraph to analysis: \n" + user_prompt + "\n\n" + "Some information that why the paragraph is recommended for you to analyze: \n" + explanation
        query_res = query(global_stage2_prompt, user_prompt_stage2, model_name)
        return 1, query_res
    elif query_res['answer'] == 2:
        return 2, None
    else:
        return 0, None

def one_stage_query(user_prompt, model_name_stage_1):
    user_prompt_stage1 = "The paragraph to analysis: \n" + user_prompt
    global_prompt = stage12_prompt(filter_stringency_level, derive_element_type)
    query_res = query(global_prompt, user_prompt_stage1, model_name_stage_1)
    if query_res is None or query_res['answer'] == 0:
        return 0, None
    elif query_res['answer'] == 1:
        return 1, query_res
    elif query_res['answer'] == 2:
        return 2, None
    else:
        return 0, None


def is_low_quality_book(query_times, accept_times):
    if query_times > 20 and accept_times/ query_times < 1/20:
        print("low quality book, skip", accept_times, "/", query_times)
        return True
    else:
        return False


def query_divide_chapter(text, current_title):
    res = []
    filter_text = text.split('\n\n')
    current_p = ''
    all_content  = ''
    query_times = 0
    accept_times = 0
    current_title = current_title + ' '
    for t in tqdm(filter_text, desc="Processing Paragraph"):
        new_chapter_flag = t.startswith('#')
        user_prompt = current_p
        content_list = custom_split(user_prompt, '. ', 50)
        if len(current_p) > 200 or len(content_list) > 5:
            # print(f"content_list: ({len(content_list)})")
            # pprint.pprint(content_list, width=400, compact=True)
            answer, query_res = one_stage_query( all_content, model_name)
            query_times += 1
            if answer == 1:
                query_res['original_text'] = all_content
                if query_res is not None:
                    res.append(query_res)
                    accept_times += 1
            if is_low_quality_book(query_times, accept_times):
                break
            if answer == 2:
                current_p = ''
                continue
            else:
                all_content  = ''
                current_p = ''
        else:
            current_p += t + '\n\n'
            all_content += t + '\n\n'

    if current_p:
        user_prompt = current_p
        content_list = custom_split(user_prompt, '. ', 50)
        if len(current_p) > 200 or len(content_list) > 5:
            answer, query_res = one_stage_query(all_content, model_name)
            if answer == 1:
                query_res['original_text'] = all_content
                if query_res is not None:
                    res.append(query_res)
                    accept_times += 1
    print("original text: ", filter_text)
    print("res: ", pprint.pprint(res))
    return res, query_times, accept_times


def filte_one_book_obj(data):
    text = data['text']
    text = text.replace('\n\n', '||')
    text = text.replace('\n', ' ')
    paragraphs = text.split('||')

    ## The longest chapter is have 11757 characters, which is 2569 words.
    current_chapter = ""
    current_chapter_title = ""
    res_list = []
    all_query_times = 0
    all_accept_times = 0
    for paragraph in tqdm(paragraphs, desc="Processing Chapter"):
        if paragraph.startswith('#'):
            if current_chapter: # finish the previous chapter
                res_chapter, query_times, accept_times = query_divide_chapter(current_chapter,current_chapter_title)
                res_list.extend(res_chapter)
                print(current_chapter)
                print("------")
                current_chapter = ""
                current_chapter_title = ""
                all_query_times += query_times
                all_accept_times += accept_times
            if is_low_quality_book(all_query_times, all_accept_times):
                break

            current_chapter_title = paragraph
            current_chapter = paragraph + '\n'
        else:
            current_chapter += '\n\n' + paragraph

    if current_chapter:
        res_chapter, query_times, accept_times = query_divide_chapter(current_chapter,current_chapter_title)
        res_list.extend(res_chapter)
        current_chapter= ""
        # print(current_chapter)
    print("Total number of paragraphs: ", len(res_list))
    return res_list

def filer_pi_t_obj(books_path, classifier_flag):


    if classifier_flag:
        classifier = pipeline("zero-shot-classification",
                        model="sileod/deberta-v3-base-tasksource-nli", device=0)
        # labels=['soccer-related', 'not soccer-related']
        # labels=["Football Novel", "Footballer Biography", "Football History", "Football Tactics", "Tutorial"]
        labels=["Football Tactics", "Football's Objective", "Football Rules", "Not Football Game related"]

    output_books_num = 0
    all_res_list = []
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

                print("[No.", line_number, "] Processing Book: ", title)
                text = data['text']
                text = text.replace('\n\n', '||')
                text = text.replace('\n', ' ')
                paragraphs = text.split('||')

                ## The longest chapter is have 11757 characters, which is 2569 words.
                current_chapter = ""
                current_chapter_title = ""
                res_list = []

                for paragraph in tqdm(paragraphs, desc="Processing Chapter"):
                    if paragraph.startswith('#'):
                        if current_chapter: # finish the previous chapter
                            res_chapter = query_divide_chapter(current_chapter,current_chapter_title)
                            res_list.extend(res_chapter)
                            print(current_chapter)
                            print("------")
                            current_chapter = ""
                            current_chapter_title = ""
                        current_chapter_title = paragraph
                        current_chapter = paragraph + '\n'
                    else:
                        current_chapter += '\n\n' + paragraph

                if current_chapter:
                    res_chapter = query_divide_chapter(current_chapter,current_chapter_title)
                    res_list.extend(res_chapter)
                    current_chapter= ""
                    # print(current_chapter)

            output_books_num += 1
            all_res_list.extend(res_list)
    print("Finish processing all books")
    print("Total number of books: ", output_books_num)
    print("Total number of paragraphs: ", len(all_res_list))
    np.savez(output_dir + '/raw_results.npz', res=np.array(all_res_list))
    return all_res_list

def agg_code(all_res_list, use_text_flag, agg_num=5, threshold=0.9):
    if use_text_flag:
        global_prompt = agg_prompt(filter_stringency_level, derive_element_type)
    else:
        global_prompt = code_agg_prompt(filter_stringency_level, derive_element_type)
    all_res_code_list = []
    embedding_list = []
    agg_res = []
    for i in tqdm(range(len(all_res_list)), desc="Processing Embeddings"):
        for k in all_res_list[i].keys():
            if k in ['answer', 'explanation', 'original_text', 'embedding', 'text']:
                continue
            code_str = f"TITLE: {k}" + ". \n\n".join(all_res_list[i][k])
            try:
                embedding = get_embedding(code_str)
            except openai.BadRequestError as e:
                e.response.content
                if str(json.loads(e.response.content)['error']['message']).find("maximum context"):
                    if use_text_flag:
                        agg_res.append({'code': all_res_list[i][k], "original_text": all_res_list[i]['original_text'], "code_title": k})
                    else:
                        agg_res.append({'code': all_res_list[i][k], "code_title": k})
                    continue
                else:
                    raise e
            if use_text_flag:
                all_res_code_list.append({'code': all_res_list[i][k], "text": all_res_list[i]['original_text'], "embedding": embedding, "code_title": k})
            else:
                all_res_code_list.append({'code': all_res_list[i][k], "embedding": embedding, "code_title": k})
            embedding_list.append(embedding)
    num_of_code_before_agg = len(embedding_list) + len(agg_res)
    print("num skip embedding", len(agg_res))
    idx_for_agg = np.array(range(len(all_res_code_list)))
    embedding_list = np.array(embedding_list)
    all_res_code_list = np.array(all_res_code_list)

    num_of_code_after_agg = len(agg_res)
    while True:
        if idx_for_agg is None or len(idx_for_agg) == 0:
            break
        else:
            print("left embedding: ", idx_for_agg.shape)
        # find the most similar embedding
        left_embedding_list = embedding_list[idx_for_agg]
        left_res_code_list = all_res_code_list[idx_for_agg]
        embedding = left_embedding_list[np.random.randint(0, left_embedding_list.shape[0])]
        sim_coef = get_cosine_similarity(left_embedding_list, embedding)
        match_res_list = []
        match_idx = np.argsort(sim_coef)[-agg_num:]
        match_idx_threshold = match_idx[np.where(sim_coef[match_idx] > threshold)]
        print(sim_coef[match_idx_threshold])
        if len(agg_res) == 22:
            print("debug")
        if match_idx_threshold.shape[0] == 1:
            # wihout similar code, just add the code to agg_res
            if use_text_flag:
                agg_res.append({left_res_code_list[match_idx_threshold[0]]["code_title"]: left_res_code_list[match_idx_threshold[0]]["code"], "text": left_res_code_list[match_idx_threshold[0]]['text']} )
            else:
                agg_res.append({left_res_code_list[match_idx_threshold[0]]["code_title"]: left_res_code_list[match_idx_threshold[0]]["code"]})
            idx_for_agg = np.delete(idx_for_agg, match_idx_threshold)
            num_of_code_after_agg += 1
            continue
        else:
            for id in match_idx_threshold:
                if use_text_flag:
                    match_res_list.append({"code": left_res_code_list[id]["code"], "text": left_res_code_list[id]['text'], "code_title": left_res_code_list[id]['code_title']})
                else:
                     match_res_list.append({"code": left_res_code_list[id]["code"], "code_title": left_res_code_list[id]['code_title']})

        user_prompt = ''
        for match_id in range(len(match_res_list)):
            if type(match_res_list[match_id]['code']) == list:
                code_to_str = f"TITLE: {match_res_list[match_id]['code_title']}" + "\n".join(match_res_list[match_id]['code'])
            else:
                code_to_str = match_res_list[match_id]['code']
            if use_text_flag:
                user_prompt += f"\n\n\n ({match_id}) original text: \n\n" + match_res_list[0]['text'] + '\n\n' + f" ({match_id}) coarse-grained presudocode [just for reference. It is unnecessary to follow the code]:\n\n" + code_to_str
            else:
                user_prompt += f"\n\n\n ({match_id}) presudocode [you should keep the logic of this code in your aggregation code]:\n\n" + code_to_str

        res = query(global_prompt, user_prompt, os.environ.get('PLFB_OPENAI_AGG_MODEL', model_name))
        if res is None:
            # failed to query, just add the code to agg_res
            agg_res.append({match_res_list[-1]['code_title']: match_res_list[-1]['code']})
            idx_for_agg = np.delete(idx_for_agg, match_idx_threshold[-1])
            num_of_code_after_agg += 1
        else:
            agg_res.append(res)
            num_of_code_after_agg += len(res.keys())
            # remove matched idx from idx_for_agg
            idx_for_agg = np.delete(idx_for_agg, match_idx_threshold)
        if 'forward' in str(res):
            print("debug")
    return agg_res, num_of_code_before_agg, num_of_code_after_agg

def write_res_text(res_list, output_file):
    str_data = pprint.pformat(res_list, width=140)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(str_data)

if __name__ == "__main__":
    """
    This is a precomputed results for the  books with many paragraphs that related to MDPs
    ['The Soccer Goalkeeping Handbook (3rd Edition) - Alex Welsh', 182]
    ['How to Watch Soccer - Ruud Gullit', 197]
    ['Mike Smith - The Road to Glory', 47]
    ["Duncan Adams - Football Grounds a Fans' Guide [Retail 9781782814207]", 50]
    ['Football Intelligence', 39]
    """
    # cached_related_book = ["Duncan Adams - Football Grounds a Fans' Guide [Retail 9781782814207]"] #[ 'Football Intelligence']

    cached_related_book = ["Duncan Adams - Football Grounds a Fans' Guide [Retail 9781782814207]", 'The Soccer Goalkeeping Handbook (3rd Edition) - Alex Welsh', 'How to Watch Soccer - Ruud Gullit', 'Mike Smith - The Road to Glory', 'Football Intelligence']
    # Override PLFB_BOOK_JSONL to select a different local book extract.
    book_jsonl = Path(os.environ.get("PLFB_BOOK_JSONL", REDPAJAMA_ROOT / "football_soccer_90.jsonl"))
    all_books_data = load_jsonl_list(str(book_jsonl))
    loading = True
    use_cached_books = True
    all_pre_agg_res = []
    output_dir = os.environ.get(
        "PLFB_BOOK_FILTER_OUTPUT",
        str(DATA_ROOT / f"v4-{model_name}-level-{filter_stringency_level}" / derive_element_type),
    )
    valid_book_num = 0
    valid_book_list = []
    all_related_texts = []
    for data in all_books_data:
        if 'meta' in data and isinstance(data['meta'], dict):
            meta_keys = set(data['meta'].keys())
            # Check if there are any unexpected keys
            if not meta_keys.issubset(expected_keys):
                title = data['meta']['title']
            else:
                title = data['meta']['short_book_title']
            book_name = title
            if use_cached_books and book_name not in cached_related_book:
                continue
            print(" Processing Book: ", title)
            if detect(title) != 'en':
                print("skip non-english book")
                continue
            if 'history' in title or "History" in title:
                print("skip history book")
                continue

            ori_book_dir = str(LLAMA_INDEX_DATA_ROOT / f"{book_name}.json")
            book_level_output_dir = os.path.join(output_dir, book_name)
            os.makedirs(book_level_output_dir, exist_ok=True)
            raw_result_path = os.path.join(book_level_output_dir, 'raw_results.npz')
            if loading and os.path.exists(raw_result_path):
                all_res_list = np.load(raw_result_path, allow_pickle=True)['res']
            else:
                all_res_list = filte_one_book_obj(data)
                np.savez(raw_result_path, res=np.array(all_res_list))
                readable_file = os.path.join(book_level_output_dir, 'text', f'raw.txt')
                write_res_text(all_res_list, readable_file)
            for res in all_res_list:
                if len(res['original_text']) > 200:
                    all_related_texts.append({'text': res['original_text']})
            b_theshold = 10
            if len(all_res_list) < b_theshold:
                print(f"skip book with less than {b_theshold} paragraphs")
                continue
            # else:
            code_review_file = os.path.join(book_level_output_dir, f'review.jsonl')
            if loading and not os.path.exists(code_review_file):
                review_code_res, num_of_code_before_agg, num_of_code_after_agg = agg_code(all_res_list, True, agg_num=2)
                output_jsonl(review_code_res, code_review_file)
                write_res_text(review_code_res, os.path.join(book_level_output_dir, 'text',  f'review.txt'))
            else:
                review_code_res = load_jsonl_list(code_review_file)
            first_agg_b_threshold = 20
            first_agg_code_num = np.sum([len(code_dict.keys()) for code_dict in review_code_res])
            valid_book_list.append([book_name, first_agg_code_num])
            valid_book_num += 1
            all_pre_agg_res.extend(review_code_res)

    totoal_code_num = np.sum([res[1]  for res in valid_book_list])
    for v in valid_book_list:
        if v[1] <35:
            continue
        print(v)
    print("all_books_data", len(all_books_data), "valid_book_num", valid_book_num, "all_pre_agg_res", totoal_code_num, "all_related_texts", len(all_related_texts))
    output_jsonl(all_related_texts, os.path.join(output_dir, 'multi', 'related_texts.jsonl'))
    output_jsonl(all_pre_agg_res, os.path.join(output_dir, 'multi', 'all_pre_agg_res.jsonl'))
    # TODO: get all the code from multiple files.
    level = 0
    num_of_code_before_agg = totoal_code_num
    while True:
        print("++++++++++++++++++ agg level: ", level, " +++++++++++++++++++++++++++++")
        saved_agg_file = os.path.join(output_dir, 'multi', f'v2-agg-level-{level}.jsonl')
        if loading and os.path.exists(saved_agg_file):
            agg_res_new = load_jsonl_list(saved_agg_file)
        else:
            agg_res_new, _, _ = agg_code(all_pre_agg_res, False, agg_num=5, threshold=0.95)
            write_res_text(agg_res_new, os.path.join(output_dir, 'multi', 'text', f'agg-level-{level}.txt'))
            output_jsonl(agg_res_new, saved_agg_file)
            # break
        num_of_code_after_agg = np.sum([len(code_dict.keys()) for code_dict in agg_res_new])
        print("num_of_code_after_agg", num_of_code_after_agg, "num_of_code_before_agg", num_of_code_before_agg)
        if num_of_code_after_agg >= num_of_code_before_agg * 0.9:
            print("--- agg finished ---")
            output_jsonl(agg_res_new, os.path.join(output_dir, 'multi', 'best', f'agg-best.jsonl'))
            single_code_list = []
            for agg_res in agg_res_new:
                for k in agg_res.keys():
                    single_code_list.append({"code": agg_res[k], "code_title": k})
            output_jsonl(single_code_list, os.path.join(output_dir, 'multi', 'best_single', f'agg-best.jsonl'))
            write_res_text(agg_res_new, os.path.join(output_dir, 'multi', 'text', 'best', f'agg-best.txt'))
            break
        all_pre_agg_res = agg_res_new
        num_of_code_before_agg = num_of_code_after_agg
        level += 1


    # print("valid_book_num", valid_book_num, "all_pre_agg_res", len(all_pre_agg_res))
    # print("agg_res", len(agg_res), "level", level)
