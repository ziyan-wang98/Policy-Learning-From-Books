from openai import OpenAI
import os
from pathlib import Path
import re
import argparse
import pickle as pkl
import json
from datasets import load_dataset, Dataset
from transformers import pipeline
from tqdm import tqdm
import numpy as np
import pprint
## Data Frame:
# {'meta': {'short_book_title','publication_date','url'},'text'}
# {'meta': {'title'},'text'}

# global_prompt = """I want you to act like a football manager. Analyze the given paragraph from a football-related context to
#                    identify if it contains any of the following elements:
#                    1. 'Policy': This refers to the tactics, strategies or guidelines that govern decisions in the team or club management, such as how players should be used in the frontcourt during a match, or when forwards should take shots.
#                    2. "Dynamics": This represents how the rules of the football game operate, such as when a corner kick occurs, which team should score when the ball enters the goal, and represents the dynamics of the football game.
#                    3. "Objective": The goal in a football game. For example, when we lose the ball, the goal should be to regain the ball. The final goal of the game should be that the team's score is greater than the opponent's score, so as to win.
#                    4. "None": A scene that does not belong to any of the above descriptions, this is just an ordinary text description without any value.
#                    Each element will be evaluated to determine its presence in the paragraph. The analysis results will be returned in a structured JSON with keys: type (the type of this context chosen from the above four types)"""

filter_stringency_level = 'strict'

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("PLFB_DATASET_PATH", PROJECT_ROOT / "data")).resolve()
REDPAJAMA_ROOT = Path(os.environ.get("PLFB_REDJPAJAMA_ROOT", PROJECT_ROOT.parent / "redpajama")).resolve()
LLAMA_INDEX_DATA_ROOT = Path(os.environ.get("PLFB_LLAMA_INDEX_DATA", DATA_ROOT / "llama_index")).resolve()

filter_stringency_level_prompt_dict = {
    'loose': 'You can put all related content that is helpful to derive {0} element/function here',
    'medium': 'You should only put the content that can derive {0} element/function directly here',
    'strict': 'You should only put the content that can derive {0} element/function directly here, and the content should be very close to the element/function',
}

filter_stringency_prompt = filter_stringency_level_prompt_dict['strict']
global_prompt = """
        I want you to act like a football manager and also an expert in Reinforcement Learning that want to learn a football manager policy in a football simulator.

        You need to analyze the given content list from a football-related context to identify if it contains any of the related elements or concepts in Reinforcement Learning or Markov decision process (MDP):
        1. "Policy function": The football manager policy is to give the tactics and strategies for all players in the team, such as how players should be used in the frontcourt during a match, or when forwards should take shots. For example: "When watching defenders you have to assess how they respond to their opponents as well as the ball." {0}.
        2. "Environment Dynamics function": Dynamics is to give the dynamics function or related rules of the football game under the football manager policy's action, such as after shotting, the ball will be in the goal or not. For example: "When the direction of shotting is vertical to the goal, the ball will be easy to the goal." {1}.
        3. "Rewards function": Reward is to give the reward or punishment of the football manager policy. For example: "When the forwards are restricted, the midfielder can support and take away the defenders, which is a very encouraging behavior." {2}.
        5. "None": A scene that does not belong to any of the above descriptions, this is just an ordinary text description without any value. For example: "Cruijff knew where to place players and was always talking, on and off the pitch."

        REQ:
        1. You should match one of the above types for each item. If you cannot match any of the above types, you should set the item to "None" type. All the items should be mentioned in at least one of the types in your results.
        2. One item can belong to multiple types. For example, "When the forwards are restricted, the midfielder can support and take away the defenders, which is a very encouraging behavior." can be both "Policy" and "Rewards".
        3. NOTE: response should be in pure json format.
        """.format(filter_stringency_prompt.format('Policy'),  filter_stringency_prompt.format('Dynamics'), filter_stringency_prompt.format('Rewards')) + \
        """
        response example (you should use the following format to construct your response):

        [
            {"items": [0,1], "Policy": "Why it is related to the policy", "Rewards": "Why  it is related to the Reward"},
            {"items": [2], "Policy": "Why it is related to the policy", "Dynamics": "Why it is related to the Dynamics"},
            {"items": [3,4,5], "Rewards": "Why  it is related to the Reward"},
            ...
        ]

        provided python item list:

        """

model_name = os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")
asked_types = ["Policy", "Dynamics", "Rewards", "None"]


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set before calling OpenAI.")
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def query(user_prompt):

    response = _openai_client().chat.completions.create(
        messages=[
        {"role": "system", "content": "You are a football manager and reinforcement-learning expert deriving a football policy for a simulator. "},
        {"role": "user", "content": global_prompt + user_prompt},
        ],
        model=model_name,
        temperature=0,
        max_tokens=1000,
        # response_format={"type": "json_object"}
    )

    try:
        query_res = json.loads(response.choices[0].message.content)
        pprint.pprint(query_res, width=400, compact=True)
    except:
        print("-------------------error json ----------")
        print(response.choices[0].message.content)
        return None
    return query_res


context_is_sentence = False
def result_construction_util(query_res, content_list, current_title, res, new_chapter_flag, original_text):
    if type(query_res) == dict:
        query_res = [query_res]
    for qr in query_res:
        for cur_type in asked_types:
            if cur_type in qr.keys():
                new_query_res = {}
                if new_chapter_flag:
                    try:
                        new_query_res['context'] = ". ".join(np.array(content_list)[qr['items']].tolist())
                    except IndexError:
                        print ("++++ IndexError ERROR ++++")
                        print("content_list: ", content_list)
                        new_query_res['context']  =  original_text
                        print ("---- IndexError ERROR ----")
                    if 'context_paragraph' not in new_query_res.keys():
                        new_query_res['context_paragraph'] = original_text
                    new_query_res['summary'] = qr[cur_type]
                else:
                    try:
                        new_query_res['context'] = current_title + "\n\n" + ". ".join(np.array(content_list)[qr['items']].tolist())
                    except IndexError:
                        print("++++ IndexError ERROR ++++")
                        print("content_list: ", content_list)
                        new_query_res['context']  = current_title + "\n\n" + original_text
                        print ("---- IndexError ERROR ----")
                    if 'context_paragraph' not in new_query_res.keys():

                        new_query_res['context_paragraph'] = current_title + "\n\n" + original_text
                    new_query_res['summary'] = qr[cur_type]
                new_query_res['type'] = cur_type
                res.append(new_query_res)

def custom_split(s, delimiter, threshold):
    s = s.replace('.\n', '. ')
    parts = s.split(delimiter)
    result = []

    current_part = ''
    for part in parts:
        if len(current_part) + len(part) < threshold:
            current_part += part
        else:
            if current_part:
                result.append(current_part)
            current_part = part

    if current_part:
        result.append(current_part)

    return result

def query_divide_chapter(text, current_title):
    res = []
    filter_text = text.split('\n\n')
    current_p = ''
    current_title = current_title + ' '
    for t in tqdm(filter_text, desc="Processing Paragraph"):
        new_chapter_flag = t.startswith('#')

        user_prompt = current_p
        content_list = custom_split(user_prompt, '. ', 50)
        if len(current_p) > 200 or len(content_list) > 5:
            # print(current_p)
            print(f"content_list: ({len(content_list)})")
            pprint.pprint(content_list, width=400, compact=True)
            query_res = query(f"NOTE: we have {len(content_list)} items, the index of items should be from 0 to {len(content_list)-1}. \n" + str(content_list))
            if query_res is not None:
                result_construction_util(query_res, content_list, current_title, res, new_chapter_flag, user_prompt)
            current_p = ''
        else:
            current_p += t + '\n\n'

    if current_p:
        user_prompt = current_p
        content_list = custom_split(user_prompt, '. ', 50)
        print(f"content_list: ({len(content_list)})")

        pprint.pprint(content_list, width=400)
        query_res = query(f"NOTE: we have {len(content_list)} items, the index of items should be from 0 to {len(content_list)-1}. \n" + str(content_list))
        if query_res is not None:
            result_construction_util(query_res, content_list, current_title, res, new_chapter_flag, user_prompt)
    print("original text: ", filter_text)
    print("res: ", pprint.pprint(res))
    return res


def pick_book(books_path, output_file_path, book_name):
    expected_keys = {'short_book_title', 'publication_date', 'url'}
    with open(books_path, 'r') as file:
        for _, line in enumerate(file, start=1):
            data = json.loads(line)
            if 'meta' in data and isinstance(data['meta'], dict):
                meta_keys = set(data['meta'].keys())
                # Check if there are any unexpected keys
                if not meta_keys.issubset(expected_keys):
                    prediction = (data['meta']['title'] == book_name)
                else:
                    prediction = (data['meta']['short_book_title'] == book_name)

                if prediction:
                    with open(output_file_path, 'a') as output_file:
                        output_file.write(json.dumps(data))
                    break


def save_as_jsonl(file_name, data_list):
    with open(file_name, 'w', encoding='utf-8') as file:
        for item in data_list:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

def filer_pi_t_obj(books_path, classifier_flag):

    expected_keys = {'short_book_title', 'publication_date', 'url'}

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

def output_jsonl(all_res_list, use_paragraph, output_path):
    policy_list = []
    transition_list = []
    rewards_list = []
    none_list = []

    for item in tqdm(all_res_list, desc="Processing Jsons"):
        item_type = item.get('type')
        if use_paragraph:
            item_context = item.get('context_paragraph')
        else:
            item_context = item.get('context')
        if item_type == 'Policy':
            policy_list.append(item_context)
        elif item_type == 'Dynamics':
            transition_list.append(item_context)
        elif item_type == 'Rewards':
            rewards_list.append(item_context)
        elif item_type == 'None':
            none_list.append(item_context)
    if use_paragraph:
        output_path += '_paragraph/'
    else:
        output_path += '_sentence/'
    policy_list = list(set(policy_list))
    transition_list = list(set(transition_list))
    rewards_list = list(set(rewards_list))




    reward_json_path = output_path + 'reward/'
    policy_json_path = output_path + 'policy/'
    transition_json_path = output_path + 'transition/'

    if not os.path.exists(reward_json_path):
        os.makedirs(reward_json_path)
    if not os.path.exists(policy_json_path):
        os.makedirs(policy_json_path)
    if not os.path.exists(transition_json_path):
        os.makedirs(transition_json_path)


    save_as_jsonl(output_path + 'all.jsonl', all_res_list)
    save_as_jsonl(reward_json_path + 'rewards.jsonl', rewards_list)
    save_as_jsonl(policy_json_path + 'policy.jsonl', policy_list)
    save_as_jsonl(transition_json_path + 'transition.jsonl', transition_list)
    save_as_jsonl(output_path + 'none.jsonl', none_list)



if __name__ == "__main__":

    # Override PLFB_BOOK_JSON to select a different local book extract.
    ori_book_dir = os.environ.get("PLFB_BOOK_JSON", str(LLAMA_INDEX_DATA_ROOT / "football_intelligence.json"))

    # Override PLFB_BOOK_FILTER_OUTPUT to select a different output directory.
    output_dir = os.environ.get(
        "PLFB_BOOK_FILTER_OUTPUT",
        str(REDPAJAMA_ROOT / f"football_intelligence_res-{model_name}-level-{filter_stringency_level}"),
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("---Phase1: Query all books-----")
    # filer_pi_t_obj(ori_book_dir, False)

    print("---Phase2: Save to Jsonl-----")
    all_res_list =  np.load(output_dir + '/raw_results.npz', allow_pickle=True)['res']
    output_jsonl(all_res_list, True, output_dir)
    output_jsonl(all_res_list, False, output_dir)
    # Example: use pick_book(...) to export one book for debugging.
    # pick_book(ori_book_dir, output_dir, "How to Watch Soccer - Ruud Gullit")
