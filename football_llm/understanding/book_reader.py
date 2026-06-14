import os
import pprint
from tqdm import tqdm
import numpy as np
import openai
import json

import random

from funcs import load_jsonl_list
from langdetect import detect
from understanding.utils import custom_split, output_jsonl, query, get_embedding, get_cosine_similarity
from CONFIG import *

filter_stringency_level = 'strict'
filter_stringency_level_prompt_dict = {
    'loose': 'You can put all related content that is helpful to derive {0} element/function here',
    'medium': 'You should only put the content that can derive {0} element/function directly here',
    'strict': 'You should only put the content that can derive {0} element/function directly here, and the content should be very close to the element/function',
}

def is_low_quality_book(query_times, accept_times):
    if query_times > 20 and accept_times/ query_times < 1/20:
        print("low quality book, skip", accept_times, "/", query_times)
        return True
    else:
        return False


def write_res_text(res_list, output_file):
    str_data = pprint.pformat(res_list, width=140)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(str_data)


class BaseBooks2Knowledge(object):
    def __init__(self, book_path, book_res_path, cached_book_name, 
                 model_name, agg_model_name, knowledge_type, 
                 task_info, element_type_prompt_dict, element_type_code_example_dict,
                 query_knowledge_str_len_threshold=200) -> None:
        self.book_path = book_path
        self.cached_book_name = cached_book_name
        self.book_res_path = book_res_path
        self.model_name = model_name
        self.agg_model_name = agg_model_name
        self.query_knowledge_str_len_threshold = query_knowledge_str_len_threshold
        self.knowledge_type = knowledge_type
        self.task_info = task_info
        self.element_type_prompt_dict = element_type_prompt_dict
        self.element_type_code_example_dict = element_type_code_example_dict
        pass

    def _get_book_name(self, book_data):
        raise NotImplementedError
    
    def paragraph_generator(self, book_data):
        raise NotImplementedError

    def get_code_agg_prompt(self, use_text_flag):
        if use_text_flag:
            global_prompt = self.agg_prompt()
        else:
            global_prompt = self.code_agg_prompt()
        return global_prompt

    def get_book_info_extract_prompt(self):
        filter_stringency_prompt = filter_stringency_level_prompt_dict[filter_stringency_level]
        global_prompt = """
                I want you to act {2}. 
                You need to analyze the given paragraph step-by-step from a related context to derive the specific theorem, principle, rule, and law of the related elements or concepts: 
                {0}. {1}.
                """.format(self.element_type_prompt_dict[self.knowledge_type], filter_stringency_prompt.format(self.knowledge_type), self.task_info) + \
                """
                Requirements:

                About the answer:
                - If you think the paragraph contains the above elements, please answer 1. The answer is 1 only when you can write the specific theorem, principle, rule, and law of the related elements into presudocode snippet.
                - If you think the paragraph does not contain the above elements, please answer 0.
                - If you think the given paragraph is not clear enough to answer, please answer 2. Then I will give you the following paragraph to help you answer.
                - If you think the paragraph contains the above elements but the content is not clear enough to derive the specific theorem, principle, rule, and law of the related elements, please answer 2. Then I will give you the following paragraph to help you answer.
                NOTE: {0}.

                About the anylysis:
                - If the answer is 1, you should give the specific theorem, principle, rule, and law of the related elements.
                - You should write the specific theorem, principle, rule, and law of the related elements via presudo code.
                - Please provide the PYTHON-tyle presudo code as detailed as you can to cover the most information of the original content.
                
                About the presudocode snippet: 
                - {1}
                About the format:
                    - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.
                """.format(filter_stringency_prompt.format(self.knowledge_type), self.element_type_code_example_dict[self.knowledge_type]) + \
                """
                
                The response example:
                {
                    "answer": 1,
                    "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
                    "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
                    "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
                }
                """
        return global_prompt

    def main_process(self, reuse_cache=True):
        self.load_book_data()
        all_related_texts, all_pre_agg_res = self.book_level_knowledge_extraction(reuse_cache)
        self.code_aggregation(all_pre_agg_res, reuse_cache)
        
    def load_book_data(self):
        self.all_books_data = load_jsonl_list(self.book_path)
    
    def book_level_knowledge_extraction(self, reuse_cache=True):
        all_related_texts = []
        all_pre_agg_res = []
        for book_data in self.all_books_data:
            book_name = self._get_book_name(book_data)
            query_times = 0
            accept_times = 0
            if book_name is None:
                continue
            all_content  = ''
            current_p = ''
            book_res = []
            prev_chapter_index = 0
            raw_result_path = os.path.join(self.book_res_path, book_name, 'one_book_knowledge.npz')
            readable_file =  os.path.join(self.book_res_path, book_name, 'text', 'one_book_knowledge.txt')
            # mkdir 
            os.makedirs(os.path.dirname(raw_result_path), exist_ok=True)
            os.makedirs(os.path.dirname(readable_file), exist_ok=True)

            if reuse_cache and os.path.exists(raw_result_path):
                book_res = np.load(raw_result_path, allow_pickle=True)['res']
            else:
                print("Processing Book: ", book_name)
                for chapter_index, paragraph in self.paragraph_generator(book_data):
                    if prev_chapter_index != chapter_index:
                        if current_p: # finish the previous chapter
                            content_list = custom_split(current_p, '. ', 50)
                            if len(current_p) > self.query_knowledge_str_len_threshold or len(content_list) > 5:
                                answer, query_res = self.book_info_extract(all_content)
                                if answer == 1:
                                    query_res['original_text'] = all_content
                                    if query_res is not None:
                                        book_res.append(query_res)
                                        accept_times += 1
                        all_content  = ''
                        current_p = ''
                        prev_chapter_index = chapter_index
                    content_list = custom_split(all_content, '. ', 50)
                    # handle when the paragraph is long enough
                    if len(current_p) > self.query_knowledge_str_len_threshold or len(content_list) > 5:
                        answer, query_res = self.book_info_extract(all_content)
                        query_times += 1
                        if answer == 1: # accept
                            query_res['original_text'] = all_content
                            if query_res is not None:
                                book_res.append(query_res)
                                accept_times += 1
                        if is_low_quality_book(query_times, accept_times):
                            print("low quality book, skip", accept_times, "/", query_times)
                            break
                        if answer == 2: # useless contents
                            current_p = ''
                            continue
                        else: # answer == 0, need more information
                            all_content  = ''
                            current_p = ''
                    else:
                        current_p += paragraph + '\n\n'
                        all_content += paragraph + '\n\n'
                np.savez(raw_result_path, res=np.array(book_res))
                write_res_text(book_res, readable_file)
            for res in book_res:
                if len(res['original_text']) > 200:
                    all_related_texts.append({'text': res['original_text']})
            # code-knowledge review
            code_review_file = os.path.join(self.book_res_path, book_name, 'review.jsonl')
            if reuse_cache and not os.path.exists(code_review_file):
                review_code_res, _, _ = self._agg_code(book_res, True, agg_num=2)
                output_jsonl(review_code_res, code_review_file)
                write_res_text(review_code_res, os.path.join(self.book_res_path, book_name, 'text', f'review.txt'))
            else:
                review_code_res = load_jsonl_list(code_review_file)
            all_pre_agg_res.extend(review_code_res)
        os.makedirs(os.path.join(self.book_res_path, 'multi'), exist_ok=True)
        output_jsonl(all_related_texts, os.path.join(self.book_res_path, 'multi', 'related_texts.jsonl'))
        output_jsonl(all_pre_agg_res, os.path.join(self.book_res_path, 'multi', 'all_pre_agg_res.jsonl'))
        return all_related_texts, all_pre_agg_res
    
    def code_aggregation(self, all_pre_agg_res, reuse_cache=True):
        os.makedirs(os.path.join(self.book_res_path, 'multi'), exist_ok=True)
        level = 0
        num_of_code_before_agg = np.sum([len(code_dict.keys()) for code_dict in all_pre_agg_res])
        while True:
            print("++++++++++++++++++ agg level: ", level, " +++++++++++++++++++++++++++++")
            saved_agg_file = os.path.join(self.book_res_path, 'multi', f'v2-agg-level-{level}.jsonl')
            if reuse_cache and os.path.exists(saved_agg_file):
                agg_res_new = load_jsonl_list(saved_agg_file)
            else:
                agg_res_new, _, _ = self._agg_code(all_pre_agg_res, False, agg_num=5, threshold=0.95) 
                write_res_text(agg_res_new, os.path.join(self.book_res_path, 'multi', 'text', f'agg-level-{level}.txt'))
                output_jsonl(agg_res_new, saved_agg_file)
                # break
            num_of_code_after_agg = np.sum([len(code_dict.keys()) for code_dict in agg_res_new])
            print("num_of_code_after_agg", num_of_code_after_agg, "num_of_code_before_agg", num_of_code_before_agg)
            if num_of_code_after_agg >= num_of_code_before_agg * 0.9:
                print("--- agg finished ---")
                output_jsonl(agg_res_new, os.path.join(self.book_res_path, 'multi', 'best', f'agg-best.jsonl'))
                single_code_list = []
                for agg_res in agg_res_new:
                    for k in agg_res.keys():
                        single_code_list.append({"code": agg_res[k], "code_title": k})
                output_jsonl(single_code_list, os.path.join(self.book_res_path, 'multi', 'best_single', f'agg-best.jsonl'))
                write_res_text(agg_res_new, os.path.join(self.book_res_path, 'multi', 'text', 'best', f'agg-best.txt'))
                break
            all_pre_agg_res = agg_res_new
            num_of_code_before_agg = num_of_code_after_agg
            level += 1
        
    def _agg_code(self, all_res_list, use_text_flag, agg_num=5, threshold=0.9):
        global_prompt = self.get_code_agg_prompt(use_text_flag)
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
                
            res = query(global_prompt, user_prompt, self.agg_model_name)
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
        return agg_res, num_of_code_before_agg, num_of_code_after_agg

    def book_info_extract(self, user_prompt):
        user_prompt = "The paragraph to analysis: \n" + user_prompt + "Output: \n"
        global_prompt = self.get_book_info_extract_prompt()
        query_res = query(global_prompt, user_prompt, self.model_name)
        if query_res is None or query_res['answer'] == 0:
            return 0, None
        elif query_res['answer'] == 1:
            return 1, query_res
        elif query_res['answer'] == 2:
            return 2, None
        else:
            return 0, None

    def code_agg_prompt(self):
        filter_stringency_prompt = filter_stringency_level_prompt_dict[filter_stringency_level]
        global_prompt = """
            I want you to act like {2}. 
            
            I will give you several presudocode snippets to define the specific theorem, principle, rule, and law of the related elements or concepts: 
                {0}. {1}.
            
            These codes might share some common logic. Your tasks is to aggregate the common logic of the given codes into a single code snippet, while still covering all of the given codes.
            
            """.format(self.element_type_prompt_dict[self.knowledge_type], filter_stringency_prompt.format(self.knowledge_type), self.task_info) + \
            """
            Requirements:

            About the aggregated code:
            - Please provide the PYTHON-tyle presudo code as detailed as you can to cover the most information of the original content.
            - Using the least number of presudocode items.
            - Covering all of the code. However, please feel free to add more presudocode items if needed.
            - Since I will delete the original code after getting your aggregated code,  you cannot call the presudocodes that I provided in the prompt. If it is necessary to call the presudocode, please still return the presudocode as an individual item in the answer.
            
                
            About the presudocode snippet: 
            - {0}
            About the format:
                - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.
            """.format(self.element_type_code_example_dict[self.knowledge_type]) + \
            """
            
            The response example:
            
            {
                "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
                "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
                "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
            }
            """
        return global_prompt

    def agg_prompt(self):

        filter_stringency_prompt = filter_stringency_level_prompt_dict[filter_stringency_level]
        global_prompt = """
                
            I want you to act like {2}.  
                
                I will give you several paragraphs and the corresponding summary written by code from several related books.
                
                You need to analyze the given paragraph step-by-step from a related context to aggregate the specific theorem, principle, rule, and law of the related elements or concepts: 
                {0}. {1}.
                """.format(self.element_type_prompt_dict[self.knowledge_type], filter_stringency_prompt.format(self.knowledge_type), self.task_info) + \
                """
                Requirements:

                About the anylysis:
                - You should write the specific theorem, principle, rule, and law of the related elements via presudo code.
                - Please provide the PYTHON-tyle presudo code as detailed as you can to cover the most information of the original content.
                - You should aggregate the given information as much as you can that 
                    1. using the least number of presudocode items.
                    2. covering all of the code and most of the original texts. However, please feel free to add more presudocode items if needed.
                    3. Since I will delete the original code after getting your aggregated code,  you cannot call the presudocodes that I provided in the prompt. If it is necessary to call the presudocode, please still return the presudocode as an individual item in the answer.

                
                About the presudocode snippet: 
                - {0}
                About the format:
                    - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.
                """.format(self.element_type_code_example_dict[self.knowledge_type]) + \
                """
                
                The response example:
                {
                    "aggregated_pseudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
                    "aggregated_pseudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
                }
                """
        return global_prompt



class FootballBooks2Knowledge(BaseBooks2Knowledge):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        pass

    def _get_book_name(self, book_data):

        expected_keys = {'short_book_title', 'publication_date', 'url'}
        data = book_data
        if 'meta' in data and isinstance(data['meta'], dict):
            meta_keys = set(data['meta'].keys())
            # Check if there are any unexpected keys
            if not meta_keys.issubset(expected_keys):
                title = data['meta']['title']
            else:
                title = data['meta']['short_book_title']
            book_name = title
            if self.cached_related_book and book_name not in self.cached_related_book:
                return None
            print(" Processing Book: ", title)
            if detect(title) != 'en':
                print("skip non-english book")
                return None
            if 'history' in title or "History" in title:
                print("skip history book")
                return None
            return book_name
        else:
            return None

    def paragraph_generator(self, book_data):
        data = book_data
        if 'text' in data:
            text = data['text']
            for paragraph in text.split('\n'):
                yield paragraph
        else:
            return None

        text = data['text']
        text = text.replace('\n\n', '||')
        text = text.replace('\n', ' ')
        paragraphs = text.split('||')
        
        ## The longest chapter is have 11757 characters, which is 2569 words.
        current_chapter = ""
        chapter_index = 0
        for paragraph in tqdm(paragraphs, desc="Processing Chapter"):
            if paragraph.startswith('#'):
                if current_chapter: # finish the previous chapter
                    print(current_chapter)
                    filter_text = current_chapter.split('\n\n')
                    for text in filter_text:
                        yield chapter_index, text
                    print("------") 
                    yield current_chapter
                    current_chapter = "" 
                    chapter_index += 1
                current_chapter = paragraph + '\n' 
            else:
                current_chapter += '\n\n' + paragraph 

        if current_chapter:
            filter_text = current_chapter.split('\n\n')
            for text in filter_text:
                yield chapter_index, text

    def get_code_agg_prompt(self, use_text_flag):
        if use_text_flag:
            global_prompt = agg_prompt(self.filter_stringency_level, self.knowledge_type)
        else:
            global_prompt = code_agg_prompt(self.filter_stringency_level, self.knowledge_type)
        return global_prompt

class TicTacToeBooks2Knowledge(BaseBooks2Knowledge):
    def __init__(self, skip_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.skip_rate = skip_rate
        pass

    def _get_book_name(self, book_data):
        return 'tic-tac-toe'
    
    def paragraph_generator(self, book_data):
        data = book_data
        random.shuffle(data)
        for paragraph in data:
            if np.random.rand() < self.skip_rate:
                continue
            paragraph = paragraph["policy_knowledge"] + "\n\n=====================\n\n"
            yield 0, paragraph

    def load_book_data(self):
        self.all_books_data = [load_jsonl_list(self.book_path)]



