from generate_main import get_policy_code
from utils.obs2text import npz_extractor, format_code
from utils.obs2text import imaginary_data_observation_v2,get_raw_obs_to_llm_response
import json
import numpy as np
import os.path as osp
from tqdm import tqdm
import config.gen_main_parser as imaginary_data_parser
from utils.rewarder import Rewarder
import time
import os   
from datasets import load_dataset
from pathlib import Path
import json
import os.path as osp
import io
from utils.llama_index_compat import GradientBaseModelLLM, GradientFinetuneEngine, require_optional_llama_index_symbol

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("PLFB_DATASET_PATH", PROJECT_ROOT / "data")).resolve()
SAMPLED_DATA_ROOT = Path(os.environ.get("PLFB_SAMPLED_DATA_PATH", PROJECT_ROOT / "sampled_data")).resolve()
FINETUNE_DATA_ROOT = Path(os.environ.get("PLFB_FINETUNE_DATA_PATH", DATA_ROOT / "finetune_dataset")).resolve()
FILTER_DATA_ROOT = Path(os.environ.get("PLFB_FILTER_PATH", DATA_ROOT / "test-llama-index-finetuning")).resolve()

rewarder = Rewarder()

args = imaginary_data_parser.parse_args(return_parser=False)


def load_jsonl(data_dir):
    data_path = Path(data_dir).as_posix()
    data = load_dataset("json", data_files=data_path)
    return data


def save_jsonl(data_dicts, out_path):
    with open(out_path, "w") as fp:
        for data_dict in data_dicts:
            fp.write(json.dumps(data_dict) + "\n")


def load_data_sql(data_dir: str = "data_sql"):
    dataset = load_dataset("b-mc2/sql-create-context")

    dataset_splits = {"train": dataset["train"]}
    out_path = Path(data_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for key, ds in dataset_splits.items():
        with open(out_path, "w") as f:
            for item in ds:
                newitem = {
                    "input": item["question"],
                    "context": item["context"],
                    "output": item["answer"],
                }
                f.write(json.dumps(newitem) + "\n")
                
def load_inner_npz_by_index(data, index):
    keys = list(data.keys())  # Convert the keys to a list
    if index < len(keys):
        inner_file_key = keys[index]  # Get the key at the specified index
        npz_binary = data[inner_file_key]
        npz_data = np.load(io.BytesIO(npz_binary), allow_pickle=True)
        return npz_data
    else:
        raise ValueError("index out of range")


def generate_dataset(prompt_lag,gen_choose, load_num):

    if prompt_lag == "old":
        fine_tune_prompt = """<s>### Instruction:\n{system_message}{user_message}\n\n### Response:\n{response}</s>"""
    else:
        fine_tune_prompt = """<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message}[/INST]{response}</s>"""
    
    # read data
    file_name = os.environ.get("PLFB_FINETUNE_SAMPLE_NPZ", str(SAMPLED_DATA_ROOT / "sample_50_20240127-225653.npz"))
    
    offline_data = np.load(file_name, allow_pickle=True)
    
    data_list = []
    
    for k in tqdm(range(load_num)):
        res = load_inner_npz_by_index(offline_data, k)
        raw_obs = res['obs_before_modified_by_acs']
        actions = res['action']
        rewards = res['reward']

        
    
        for i in range(1,len(raw_obs)-2):
            obs = raw_obs[i][0]
            act = actions[i]
            obs = imaginary_data_observation_v2(obs, i)
            
            # For the action code
            if gen_choose == "action_dataset":
                global_prompt = ""
                user_prompt = args.raw_policy_prompt_step2_shorter.format(policy_code = " ", text_obs=obs)
                response = act
            elif gen_choose == "img_obs_dataset":
                global_prompt =  ""
                user_prompt = args.imaginary_obv_prompt_v2_shorter.format(transition_code = " ", reward_code = " ", current_texted_obs=obs , action = act)
                next_obs = get_raw_obs_to_llm_response(raw_obs[i+1][0], i)
                next_obs['dense_reward'] = rewarder.calc_reward(rewards[i+1], raw_obs[i+1][0], raw_obs[i][0], actions[i+1])
                response = next_obs
                
            data = fine_tune_prompt.format(system_message=global_prompt, user_message=user_prompt, response=response)
            data_list.append(data)

    json_list = []
    for i in range(len(data_list)):
        json_res = {"inputs": data_list[i]}
        json_list.append(json_res)
    
    save_path = str(FINETUNE_DATA_ROOT)
    if not osp.exists(save_path):
        os.makedirs(save_path)
        
    time_str = time.strftime("%Y%m%d-%H%M%S")
    save_jsonl(json_list, osp.join(save_path , f"finetune_dataset_{load_num}_{gen_choose}_{time_str}.jsonl"))
        
        
def fine_tune(data_path, type ,name = "football_finetune_action"):
        
    if not os.environ.get("GRADIENT_ACCESS_TOKEN") or not os.environ.get("GRADIENT_WORKSPACE_ID"):
        raise RuntimeError("Set GRADIENT_ACCESS_TOKEN and GRADIENT_WORKSPACE_ID before fine-tuning.")
    
    base_model_slug = "llama2-7b-chat"
    # base_llm = GradientBaseModelLLM(
    #     base_model_slug=base_model_slug, max_tokens=8
    # )
    
    model_adapter_id_action = os.environ.get("GRADIENT_ACTION_MODEL_ADAPTER_ID")
    model_adapter_id_obs = os.environ.get("GRADIENT_OBS_MODEL_ADAPTER_ID")

    if type == "action":
        model_adapter_id = model_adapter_id_action
    elif type == "obs":
        model_adapter_id = model_adapter_id_obs
    else:
        raise ValueError(f"unsupported Gradient fine-tune type: {type}")
    if not model_adapter_id:
        raise RuntimeError("Set GRADIENT_ACTION_MODEL_ADAPTER_ID or GRADIENT_OBS_MODEL_ADAPTER_ID before fine-tuning.")

    finetune_engine_cls = require_optional_llama_index_symbol(GradientFinetuneEngine, 'GradientFinetuneEngine', 'gradient finetuning')
    finetune_engine = finetune_engine_cls(
        # base_model_slug=base_model_slug,
        model_adapter_id=model_adapter_id,
        name=name,
        data_path=data_path,
        verbose=True,
        max_steps=300,
        batch_size=4,
    )
    
    print(finetune_engine.model_adapter_id)
    
    epochs = 1
    for i in range(epochs):
        print(f"** EPOCH {i} **")
        finetune_engine.finetune()
        
    llm = finetune_engine.get_finetuned_model(max_tokens=10)
        
    return llm

def get_engine(json_value,json_schema):
    from utils.llama_index_compat import JSONQueryEngine, OpenAI, make_service_context

    llm = OpenAI(model=os.environ.get("PLFB_OPENAI_JSON_MODEL", os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")))
    service_context = make_service_context(llm=llm)
    kwargs = {"json_value": json_value, "json_schema": json_schema}
    if service_context is not None:
        kwargs["service_context"] = service_context
    nl_query_engine = JSONQueryEngine(**kwargs)
    
    return nl_query_engine

def action_gradient(sample_query):

    from gradientai import Gradient

    adapter_id = os.environ.get("GRADIENT_ACTION_MODEL_ADAPTER_ID")
    if not os.environ.get("GRADIENT_ACCESS_TOKEN") or not os.environ.get("GRADIENT_WORKSPACE_ID") or not adapter_id:
        raise RuntimeError(
            "Set GRADIENT_ACCESS_TOKEN, GRADIENT_WORKSPACE_ID, and "
            "GRADIENT_ACTION_MODEL_ADAPTER_ID before using Gradient."
        )
    gradient = Gradient()

    new_model_adapter = gradient.get_model_adapter(model_adapter_id=adapter_id)

    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
    print(f"Generated: {completion}")




if __name__ == "__main__":
    
        
    if not os.environ.get("GRADIENT_ACCESS_TOKEN") or not os.environ.get("GRADIENT_WORKSPACE_ID"):
        raise RuntimeError("Set GRADIENT_ACCESS_TOKEN and GRADIENT_WORKSPACE_ID before using Gradient.")
    #img_obs_dataset
    #action_dataset
    # generate_dataset(prompt_lag="old",gen_choose="img_obs_dataset", load_num=50)
    
    type = "action"
    #type = "obs"
    
    if type=="action":
        data_path = os.environ.get(
            "PLFB_ACTION_FINETUNE_JSONL",
            str(FINETUNE_DATA_ROOT / "finetune_dataset_50_action_dataset_20240130-035827.jsonl"),
        )
    elif type=="obs":
    
        data_path = os.environ.get(
            "PLFB_OBS_FINETUNE_JSONL",
            str(FINETUNE_DATA_ROOT / "finetune_dataset_50_img_obs_dataset_20240130-044338.jsonl"),
        )
    
    
    # llm = fine_tune(data_path,type ,name = "football_finetune_obs")
    
    file_folder_path = str(FILTER_DATA_ROOT)
    
    from utils.index import get_docs, get_index, index_llm
    policy_doc, reward_doc, transition_doc = get_docs(file_folder_path)
    
    json_value = reward_doc
    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "description": "Schema for an array of code snippets, each with a title",
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "code": {
                    "description": "Array of strings representing lines of code",
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "code_title": {
                    "description": "Title of the code snippet",
                    "type": "string"
                }
            },
            "required": ["code", "code_title"]
        }
    }
    

    
    
    
    finetune_engine_cls = require_optional_llama_index_symbol(GradientFinetuneEngine, 'GradientFinetuneEngine', 'gradient finetuning')
    finetune_engine = finetune_engine_cls(
        # base_model_slug=base_model_slug,
        model_adapter_id=os.environ.get('GRADIENT_ACTION_MODEL_ADAPTER_ID'),
        name='action',
        data_path=data_path,
        verbose=True,
        max_steps=300,
        batch_size=4,
    )
    
    fine_tuned_model = finetune_engine.get_finetuned_model(max_tokens=100)
    
    query = "### Instruction:\n\n      \n    The pseudocode for the policy you want this active player to implement is as follows:\n\n  'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1,    \n    ---------------------\n\n    The texted observation for the current state:\n\n    {'sticky_actions': ['Sprint'], 'game_mode': 'Noramal', 'score': [0, 0], 'step': 1, 'time': '0 minutes 1 seconds', 'active_player': 1, 'active_player_role': 'Forward', 'ball_ownership': 0, 'ball_ownership_player': -1, 'ball_zone': [10, 7], 'ball_direction': 'east', 'player_0': {'team': 'Left', 'role': 'Goalkeeper', 'zone': [0, 7], 'direction': 'east'}, 'player_1': {'team': 'Left', 'role': 'Forward', 'zone': [11, 6], 'direction': 'north'}, 'player_2': {'team': 'Left', 'role': 'Forward', 'zone': [11, 7], 'direction': 'northeast'}, 'player_3': {'team': 'Left', 'role': 'Defender', 'zone': [6, 9], 'direction': 'east'}, 'player_4': {'team': 'Left', 'role': 'Defender', 'zone': [5, 7], 'direction': 'east'}, 'player_5': {'team': 'Left', 'role': 'Defender', 'zone': [5, 6], 'direction': 'east'}, 'player_6': {'team': 'Left', 'role': 'Defender', 'zone': [6, 4], 'direction': 'northeast'}, 'player_7': {'team': 'Left', 'role': 'Midfielder', 'zone': [9, 8], 'direction': 'east'}, 'player_8': {'team': 'Left', 'role': 'Midfielder', 'zone': [8, 7], 'direction': 'east'}, 'player_9': {'team': 'Left', 'role': 'Midfielders', 'zone': [9, 5], 'direction': 'east'}, 'player_10': {'team': 'Left', 'role': 'Forward', 'zone': [10, 10], 'direction': 'east'}, 'player_11': {'team': 'Right', 'role': 'Goalkeeper', 'zone': [21, 7], 'direction': 'west'}, 'player_12': {'team': 'Right', 'role': 'Forward', 'zone': [11, 7], 'direction': 'west'}, 'player_13': {'team': 'Right', 'role': 'Forward', 'zone': [11, 10], 'direction': 'west'}, 'player_14': {'team': 'Right', 'role': 'Defender', 'zone': [15, 4], 'direction': 'west'}, 'player_15': {'team': 'Right', 'role': 'Defender', 'zone': [16, 6], 'direction': 'northwest'}, 'player_16': {'team': 'Right', 'role': 'Defender', 'zone': [16, 7], 'direction': 'west'}, 'player_17': {'team': 'Right', 'role': 'Defender', 'zone': [15, 9], 'direction': 'southwest'}, 'player_18': {'team': 'Right', 'role': 'Midfielder', 'zone': [12, 5], 'direction': 'northwest'}, 'player_19': {'team': 'Right', 'role': 'Midfielder', 'zone': [13, 6], 'direction': 'southwest'}, 'player_20': {'team': 'Right', 'role': 'Midfielders', 'zone': [12, 8], 'direction': 'west'}, 'player_21': {'team': 'Right', 'role': 'Forward', 'zone': [11, 3], 'direction': 'west'}}\n    ---------------------\n\n    Possible Action information is below,\n    action set = (\n        \"0\": \"action_idle, a no-op action, sticky actions are not affected, player maintains his directional movement.\",\n        \"1\": \"action_left, sticky action, player will continue to move left until another action is taken.\",\n        \"2\": \"action_top_left, sticky action, player will continue to move top left until another action is taken.\",\n        \"3\": \"action_top, sticky action, player will continue to move top until another action is taken.\",\n        \"4\": \"action_top_right, sticky action, player will continue to move top right until another action is taken.\",\n        \"5\": \"action_right, sticky action, player will continue to move right until another action is taken.\",\n        \"6\": \"action_bottom_right, sticky action, player will continue to move bottom right until another action is taken.\",\n        \"7\": \"action_bottom, sticky action, player will continue to move bottom until another action is taken.\",\n        \"8\": \"action_bottom_left, sticky action, player will continue to move bottom left until another action is taken.\",\n        \"9\": \"action_long_pass, player will try to pass the ball to a teammate.\",\n        \"10\": \"action_high_pass, player will try to pass the ball to a teammate.\",\n        \"11\": \"action_short_pass, player will try to pass the ball to a teammate.\",\n        \"12\": \"action_shot, player will try to shoot the ball.\",\n        \"13\": \"action_sprint, player will sprint.\",\n        \"14\": \"action_release_direction, player will stop moving in the current direction.\",\n        \"15\": \"action_release_sprint, player will stop sprinting.\",\n        \"17\": \"action_dribble, player will try to dribble.\",\n        \"18\": \"action_release_dribble, player will stop dribbling.\"\n    )\n    ---------------------\n\n      \n    Requirements:\n\n    About the action choosing:\n    - Please choose the action that best fits the code logic.\n\n    About the format:\n        - you should answer in pure JSON format with the key: 'action': a int number from 0 to 18.\n            \n    Response example (you should resposne in the following order):        \n    {\n        \"action\": 0\n    }\n\n    ### Response:\n    "
    
    responese = fine_tuned_model.complete(query)
    print(responese.text)
    
