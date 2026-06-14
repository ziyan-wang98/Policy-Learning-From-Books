import sys
sys.path.append("football_llm/TiZero/submission/tizero_agent")
sys.path.append('football_llm')
from learning.utils import npz_extractor
from llm.utils.openai_compat import openai_chat_query
from CONFIG import *
from retrieval.prompt import *
from book_scripts.utils import query
from llm.openai_server import OpenAIServer
from funcs import load_jsonl_list, npz_extractor, json_str_clean
import json
import tqdm
import pprint
ONE_MINITE_STEP = 60/1.5
ANCHOR_STEPS = 10

CODE_NUM = 10000 / ONE_MINITE_STEP
CONCEPT_TYPE = 'Policy'
model_name = os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini')
KNOWLEDGE_REP = 'language'

CODE_FOLDER_NAME = f'{KNOWLEDGE_REP}_label-v2'
openai_server = OpenAIServer(model=model_name, max_token=1500)


if __name__=='__main__':
    # load offline dataset
    code_data_dict = []
    num = 0
    for file in tqdm.tqdm(os.listdir(OFFLINE_DATASET_PATH)):
        if file.endswith(".npz") and 'rule_based' in file:
            data = npz_extractor(os.path.join(OFFLINE_DATASET_PATH, file))
            if data is None:
                continue
            if max(data['reward']) == 0:
                continue
            saved_file = os.path.join(OFFLINE_DATASET_PATH, CODE_FOLDER_NAME, file.split('.npz')[0] + '.json')
            if os.path.exists(saved_file):
                code_data_dict.append(json.loads(open(saved_file).read()))
                num += int(len(code_data_dict[-1]['action']) / ONE_MINITE_STEP)
                continue
            anchor_obs_list = []
            code_data_dict.append({'file_name': file, 'texted_obs': [], 'action': [], 'reward': [], 'label': []})
            all_obs_list = []
            all_acs_list = []
            
            for idx, obs in enumerate(data['texted_obs']):
                all_obs_list.append(obs)
                all_acs_list.append(data['action'][idx])
                
                if (idx + 1) % int(ONE_MINITE_STEP/ANCHOR_STEPS) == 0:
                    anchor_obs_list.append(str(obs) + " ACTION:" + str(data['action'][idx])  + " Reward:" + str(data['dense_rewards'][idx]) + '\n\n')
                if len(anchor_obs_list) == ANCHOR_STEPS:
                    if KNOWLEDGE_REP == 'code':
                        res_prompt = code_gen_prompt('strict', CONCEPT_TYPE, "\n ".join([f"step: {idx}. observation: {obs}\n" for idx, obs in enumerate(anchor_obs_list)]))
                    elif KNOWLEDGE_REP == 'language':
                        res_prompt = language_gen_prompt('strict', CONCEPT_TYPE, "\n ".join([f"step: {idx}. observation: {obs}\n" for idx, obs in enumerate(anchor_obs_list)]))
                    else:
                        raise ValueError("Knowledge representation not supported")
                    # query_res = query(res_prompt, '', model_name, req_json=True, print_global_prompt=True)

                    # query_res = openai_server.chat(res_prompt)
                    query_res = openai_chat_query('', res_prompt, model_name=model_name, req_json=True)
                    query_res = json.loads(query_res)["choices"][0]["message"]["content"]
                    code_response_clean = json_str_clean(query_res, single_line=True)
                    query_res = json.loads(code_response_clean)
                    if query_res is not None:
                        pprint.pprint(query_res, width=400)
                        print(" number", num)
                        for j in range(int(ONE_MINITE_STEP)):
                            code_data_dict[-1]['label'].append(query_res)
                            code_data_dict[-1]['texted_obs'].append(str(all_obs_list[j]))
                            code_data_dict[-1]['action'].append(int(all_acs_list[j]))

                        num += 1

                    anchor_obs_list = []
            # save code data for each file
            os.makedirs(os.path.join(OFFLINE_DATASET_PATH, CODE_FOLDER_NAME), exist_ok=True)
            with open(saved_file, 'w') as f:
                json.dump(code_data_dict[-1], f, indent=4)
            if num > CODE_NUM:
                break
    # save the code data dict to json
    with open(os.path.join(OFFLINE_DATASET_PATH, CODE_FOLDER_NAME, 'code_data_dict.json'), 'w') as f:
        json.dump(code_data_dict, f, indent=4)
    
            
                
    # query gpt and get code
    # save the dataset