from openai import OpenAI
import os
import json
import pprint
import pandas as pd
import numpy as np
from funcs import load_jsonl_list


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set before calling OpenAI.")
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def get_cosine_similarity(embedding_list, embedding2):
    product = np.dot(embedding_list, embedding2)
    norm = np.linalg.norm(embedding_list, axis=1) * np.linalg.norm(embedding2)
    return product / norm


def get_embedding(text, model=None):
   model = model or os.environ.get("PLFB_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
   text = text.replace("\n", " ")
   return _openai_client().embeddings.create(input = [text], model=model).data[0].embedding


def output_jsonl(all_res_list, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        for all_res_dict in all_res_list:
            json.dump(all_res_dict, file, ensure_ascii=False)
            file.write('\n')

def query(global_prompt, user_prompt, model_name, print_user_prompt=True, req_json=False, print_global_prompt=True):
    if req_json:
        other_kwargs = {"response_format": {"type": "json_object"}}
    else:
        other_kwargs = {}
    try:
        response = _openai_client().chat.completions.create(
            messages=[
            {"role": "system", "content": "You are a concise domain reasoning assistant. "},
            {"role": "user", "content": global_prompt + user_prompt},
            ],
            model=model_name,
            temperature=0,
            **other_kwargs,
            # max_tokens=1000,
            # response_format={"type": "json_object"}
        )
    except Exception as e:
        print("error in query", e)
        print("global_prompt: ", global_prompt)
        print("user_prompt: ", user_prompt)
        return None
    if print_global_prompt:
        print("global_prompt: ", global_prompt)
    if print_user_prompt:
        print("user_prompt: ", user_prompt)
    try:
        res = response.choices[0].message.content
        res = res.replace("```json", '').replace("```", '')
        query_res = json.loads(res)
        pprint.pprint(query_res, width=400)
    except:
        print("-------------------error json ----------")
        print(res)
        return None
    return query_res



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

