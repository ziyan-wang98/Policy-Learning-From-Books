import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()
TIZERO_AGENT_PATH = Path(
    os.environ.get("PLFB_TIZERO_AGENT_PATH", PROJECT_ROOT / "setup" / "TiZero" / "submission" / "tizero_agent")
)
for path in (TIZERO_AGENT_PATH, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from llm.imaginary_data_generation import change_format
from llm.utils.rewarder import Rewarder
from llm.utils.obs2text import npz_extractor

import re
from tqdm import tqdm
import os.path as osp
import io
from llm.utils.openai_compat import openai_chat_query
import json
from termcolor import colored
from llm.utils.obs2text import imaginary_data_observation_v2, imaginary_data_observation, imaginary_data_to_vector
import llm.config.gen_main_parser as imaginary_data_parser
from llm.utils.buildin_ai import obs_to_robot_acs, update_stack_obs
from llm.utils.index import get_docs, get_index, index_llm, find_first_code_block
try:
    from gradientai import Gradient
except ImportError:
    Gradient = None


DATA_ROOT = Path(os.environ.get("PLFB_DATASET_PATH", PROJECT_ROOT / "data")).resolve()
FILTER_DATA_ROOT = Path(os.environ.get("PLFB_FILTER_PATH", DATA_ROOT / "test-llama-index-finetuning")).resolve()
SAMPLED_DATA_ROOT = Path(os.environ.get("PLFB_SAMPLED_DATA_PATH", PROJECT_ROOT / "llm" / "sampled_data")).resolve()
OFFLINE_DATASET_PATH = Path(os.environ.get("PLFB_OFFLINE_DATASET_PATH", DATA_ROOT / "offline_dataset-v4")).resolve()
BC_MODEL_ROOT = Path(os.environ.get("PLFB_BC_MODEL_ROOT", PROJECT_ROOT / "ORL_LOG_BC-v4" / "d3rlpy_logs")).resolve()

gradient = None
if Gradient is not None and os.environ.get("GRADIENT_ACCESS_TOKEN") and os.environ.get("GRADIENT_WORKSPACE_ID"):
    gradient = Gradient()
# Optional Gradient adapters are configured through GRADIENT_*_MODEL_ADAPTER_ID environment variables.



MAX_ATTAMPTS = 500


args = imaginary_data_parser.parse_args(return_parser=False)

ACTION_P_STEP_1 = args.raw_policy_prompt_step1
ACTION_P_STEP_1_all_code = args.raw_policy_prompt_step1_all_code
ACTION_P_STEP_2 = args.raw_policy_prompt_step2
ACTION_P_STEP_2_SHORT = args.raw_policy_prompt_step2_shorter
ACTION_GP_STEP_2 = args.gloabl_prompt_step2
IMA_P = args.imaginary_obv_prompt
IMA_P_V2 = args.imaginary_obv_prompt_v2
IMA_P_V2_SHORT = args.imaginary_obv_prompt_v2_shorter

LLM_AGENT_PROMPT = args.llm_agent_prompt
LLM_RAG_PROMPT = args.llm_rag_prompt

def load_inner_npz(data, inner_file_key):
    if inner_file_key in data:
        npz_binary = data[inner_file_key]
        npz_data = np.load(io.BytesIO(npz_binary), allow_pickle=True)
        return npz_data
    else:
        raise KeyError(f"{inner_file_key} not found in {file_name}")

def load_inner_npz_by_index(data, index):
    keys = list(data.keys())  # Convert the keys to a list
    if index < len(keys):
        inner_file_key = keys[index]  # Get the key at the specified index
        npz_binary = data[inner_file_key]
        npz_data = np.load(io.BytesIO(npz_binary), allow_pickle=True)
        return npz_data
    else:
        raise IndexError(f"Index {index} is out of bounds for the number of files in {file_name}.")


def sample_data(dir_path):
    """
    Randomly pick a file from dir_path and load the data.
    Keep trying until a file with the key 'wrapper_obs' is found.
    """

    # for file in tqdm(os.listdir(dir_path), desc='loading data'):

    load_num = 50

    save_dir = str(SAMPLED_DATA_ROOT)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # check if already exists a file named Sample_{load_num}_{time_str}.npz
    for file in os.listdir(save_dir):
        if file.startswith(f'sample_{load_num}'):
            print("Found existing sample file, skip sampling")
            return os.path.join(save_dir, file)

    npz_files = {}
    counter = 0

    for file in os.listdir(dir_path):

        if file.endswith('.npz'):
            res = npz_extractor(osp.join(dir_path, file))
            if res is None:
                continue
            if 'obs_before_modified_by_acs' not in res.keys():
                print("skip this file", file)
                continue
            with open(os.path.join(dir_path, file), 'rb') as f:
                npz_files[file] = f.read()

            counter += 1
            if counter >= load_num:
                break

    # save the log
    import time
    time_str = time.strftime("%Y%m%d-%H%M%S")
    file_name = os.path.join(save_dir, f'sample_{load_num}_{time_str}.npz')
    np.savez_compressed(file_name, **npz_files)
    return file_name

def generate_combined_data(start_point, original_data, imaginary_data_gen, sample_length, gen_length=10):

    sampled_data = original_data[start_point:start_point + sample_length]

    for i in range(gen_length):
        current_obs = sampled_data['obs_before_modified_by_acs_list'][-1][0]
        next_obv, next_obs, action = imaginary_data_gen(current_obs)

    combined_data = {}
    return combined_data


def get_jsonl_length(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)

def read_nth_jsonl(file_path, n):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i == n:
                return json.loads(line)
    return None

def start_point_picker_v2(offline_data, n, interval=10):

    start_point_list = []

    data = load_inner_npz_by_index(offline_data, n)
    for j in range(int(len(data['steps']) / interval)):
        num = np.random.randint(0, interval-1)
        start_point = j * interval + num
        start_point_list.append([n, start_point])

    return start_point_list


def start_point_picker(offline_data, interval=10):

    start_point_list = []

    for i in range(len(offline_data)):
        data = load_inner_npz_by_index(offline_data, i)
        for j in range(int(len(data['steps']) / interval)):
            num = np.random.randint(0, interval-1)
            start_point = j * interval + num
            start_point_list.append([i, start_point])

    return start_point_list

def get_all_code(obs, policy_retriever, reward_retriever, transition_retriever, gpt_model=None):

    gpt_model = gpt_model or os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini')
    obs = json.dumps(obs)

    policy_nodes = policy_retriever.retrieve(obs)
    policy_context_str = "\n\n".join([n.node.get_content() for n in policy_nodes])
    policy_context_str = find_first_code_block(policy_context_str)


    reward_nodes = reward_retriever.retrieve(obs)
    reward_context_str = "\n\n".join([n.node.get_content() for n in reward_nodes])
    reward_context_str = find_first_code_block(reward_context_str)
    reward_path = str(FILTER_DATA_ROOT / "reward" / "reward.jsonl")
    reward_length = get_jsonl_length(reward_path)
    reward_random_index = random.randint(0, reward_length - 1)
    reward_context_str = json.dumps(read_nth_jsonl(reward_path, reward_random_index))

    # transition_nodes = transition_retriever.retrieve(obs)
    # transition_context_str = "\n\n".join([n.node.get_content() for n in transition_nodes])
    # transition_context_str = find_first_code_block(transition_context_str)
    transition_json_path = str(FILTER_DATA_ROOT / "transition" / "transition.jsonl")
    transition_length = get_jsonl_length(transition_json_path)
    transition_random_index = random.randint(0, transition_length - 1)
    transition_context_str = json.dumps(read_nth_jsonl(transition_json_path, transition_random_index))

    current_obs_prompt = ACTION_P_STEP_1_all_code.format(transition_string = transition_context_str, reward_string = reward_context_str, policy_string = policy_context_str, text_obs=obs)
    print("start to get code")
    for attempt in range(MAX_ATTAMPTS):
        try:
            code_response = openai_chat_query("", current_obs_prompt, model_name=gpt_model, req_json=True)
            code_json_load = json.loads(code_response)
            code = json.loads(code_json_load["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
            policy_code = code["policy_code"]
            reward_code = code["reward_code"]
            transition_code= code["transition_code"]
            code_thought = ""
            break
        except Exception as e:
            print(colored(f"Get Code Error: {e}, retrying...", "red"))
            print("code_json_load: ", code_json_load)
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue
    print("end to get code")
    return policy_code,reward_code, transition_code, code_thought




def get_policy_code(obs, gpt_model=None):

    gpt_model = gpt_model or os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini')
    context_path = os.environ.get(
        "PLFB_POLICY_CONTEXT_PATH",
        str(DATA_ROOT / "v4-gpt-3.5-turbo-1106-level-strict" / "best" / "policy" / "agg_postprocess.jsonl"),
    )
    context_list = []

    with open(context_path, 'r',encoding='utf-8') as file:
        for line in file:
            context_list.append(line)
    context_str = "\n\n".join(context_list)
    current_obs_prompt = ACTION_P_STEP_1.format(context_str=context_str, text_obs=obs)

    for attempt in range(MAX_ATTAMPTS):
        try:
            policy_code_response = openai_chat_query("", current_obs_prompt, model_name=gpt_model, req_json=True)
            policy_code = json.loads(policy_code_response)
            policy_code = json.loads(policy_code["choices"][0]["message"]["content"])
            policy_code_answer = policy_code["code"]
            policy_code_thought = policy_code["analyze"]
            break
        except Exception as e:
            print(colored(f"Error: {e}, retrying...", "red"))
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue

    return policy_code_answer, policy_code_thought


def action_gradient(sample_query):
    adapter_id = os.environ.get("GRADIENT_ACTION_MODEL_ADAPTER_ID")
    if gradient is None or not adapter_id:
        raise RuntimeError(
            "Set GRADIENT_ACCESS_TOKEN, GRADIENT_WORKSPACE_ID, and "
            "GRADIENT_ACTION_MODEL_ADAPTER_ID to use Gradient adapters."
        )
    new_model_adapter = gradient.get_model_adapter(model_adapter_id=adapter_id)
    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output

    return completion


# def llm_agent(current_dict_obs, model=None):
#     action_prompt = LLM_AGENT_PROMPT.format(text_obs=current_dict_obs)
#     for attempt in range(MAX_ATTAMPTS):
#         try:
#             action_response = openai_chat_query(ACTION_GP_STEP_2, action_prompt, model_name=model, req_json=True)
#             action_r = json.loads(action_response)
#             action_rj = json.loads(action_r["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
#             action = action_rj["action"]
#             # action_thought = action_rj["thought"]
#             break
#         except Exception as e:
#             print(colored(f"Action Error: {e}, retrying...", "red"))
#             print("action_response: ", action_response)
#             if attempt == MAX_ATTAMPTS - 1:
#                 raise e
#             continue
#     return action

# def llm_rag(current_dict_obs, policy_code, model=None):
#     action_prompt = LLM_RAG_PROMPT.format(text_obs=current_dict_obs, policy_code=policy_code)
#     for attempt in range(MAX_ATTAMPTS):
#         try:
#             action_response = openai_chat_query(ACTION_GP_STEP_2, action_prompt, model_name=model, req_json=True)
#             action_r = json.loads(action_response)
#             action_rj = json.loads(action_r["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
#             action = action_rj["action"]
#             # action_thought = action_rj["thought"]
#             break
#         except Exception as e:
#             print(colored(f"Action Error: {e}, retrying...", "red"))
#             print("action_response: ", action_response)
#             if attempt == MAX_ATTAMPTS - 1:
#                 raise e
#             continue
#     return action



def imaginary_data_gen_v2(current_dict_obs, policy_code, reward_code, transition_code, other_player_action, model=None):
    model = model or os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini')

    gradient_flag = False

    if gradient_flag:
        for attempt in range(MAX_ATTAMPTS):
            try:
                fine_tune_prompt = """<s>### Instruction:\n{system_message}{user_message}\n\n### Response:\n{response}</s>"""
                system_message = ""
                user_message = ACTION_P_STEP_2_SHORT.format(policy_code=policy_code, text_obs=current_dict_obs)
                response = ""
                fine_tune_prompt = fine_tune_prompt.format(system_message=system_message, user_message=user_message, response=response)

                action_response = action_adapter.complete(query=fine_tune_prompt, max_generated_token_count=100).generated_output

                first_number = re.search(r'\d+', action_response).group()
                action = int(first_number)

                break
            except Exception as e:
                print(colored(f"Get Action: Error: {e}, retrying...", "red"))
                print("action_response: ", action_response)
                if attempt == MAX_ATTAMPTS - 1:
                    raise e
                continue
    else:
        action_prompt = ACTION_P_STEP_2.format(text_obs=current_dict_obs, policy_code=policy_code)
        for attempt in range(MAX_ATTAMPTS):
            try:
                action_response = openai_chat_query(ACTION_GP_STEP_2, action_prompt, model_name=model, req_json=True)
                action_r = json.loads(action_response)
                action_rj = json.loads(action_r["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
                action = action_rj["action"]
                # action_thought = action_rj["thought"]
                break
            except Exception as e:
                print(colored(f"Action Error: {e}, retrying...", "red"))
                print("action_response: ", action_response)
                if attempt == MAX_ATTAMPTS - 1:
                    raise e
                continue



    next_obs_prompt = IMA_P_V2.format(transition_code = transition_code, reward_code = reward_code, current_texted_obs = current_dict_obs, action = action)

    for attempt in range(MAX_ATTAMPTS):
        try:
            next_obs_response = openai_chat_query("", next_obs_prompt, model_name=model, req_json=True)
            next_obs = json.loads(next_obs_response)
            next_obs = json.loads(next_obs["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
            dense_rewards = next_obs['dense_reward']
            break
        except Exception as e:
            print(colored(f"Generate obs Error: {e}, retrying...", "red"))
            print("next_obs_response: ", next_obs_response)
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue


    next_obv = change_format(current_dict_obs, next_obs, action, other_player_action)

    return next_obv, action, dense_rewards

def imaginary_data_gen(current_dict_obs, policy_code, other_player_action, model=None):
    model = model or os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini')

    action_prompt = ACTION_P_STEP_2.format(text_obs=current_dict_obs, policy_code=policy_code)

    for attempt in range(MAX_ATTAMPTS):
        try:
            action_response = openai_chat_query(ACTION_GP_STEP_2, action_prompt, model_name=model, req_json=True)
            action_r = json.loads(action_response)
            action_rj = json.loads(action_r["choices"][0]["message"]["content"])
            action = action_rj["action"]
            action_thought = action_rj["thought"]
            break
        except Exception as e:
            print(colored(f"Error: {e}, retrying...", "red"))
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue


    next_obs_prompt = IMA_P.format(current_texted_obs = current_dict_obs, action = action)

    for attempt in range(MAX_ATTAMPTS):
        try:
            next_obs_response = openai_chat_query("", next_obs_prompt, model_name=model, req_json=True)
            next_obs = json.loads(next_obs_response)
            next_obs = json.loads(next_obs["choices"][0]["message"]["content"])
            break
        except Exception as e:
            print(colored(f"Error: {e}, retrying...", "red"))
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue


    # step 6: transfer the output's format
    next_obv = change_format(current_dict_obs, next_obs, action, other_player_action)

    return next_obv, next_obs, action, action_thought

if __name__ == "__main__":


    file_folder_path = str(FILTER_DATA_ROOT)
    # policy_doc, reward_doc, transition_doc = (file_folder_path)
    encorder_type = "defult"
    llm_type = "gpt-4"
    filter_path = str(FILTER_DATA_ROOT)

    policy_index, reward_index, transition_index = get_index(encorder_type, llm_type, filter_path)

    policy_retriever = policy_index.as_retriever()
    # reward_retriever = None
    # transition_retriever = None
    reward_retriever = reward_index.as_retriever()
    transition_retriever = transition_index.as_retriever()

    offline_games = 50
    imaginary_games = 50
    gen_length = 10
    time_steps = 1500
    n = 0.2

    imaginary_gen_time = (imaginary_games * time_steps) / gen_length
    sample_sp_from_offline_times = int(imaginary_gen_time * n)
    sample_sp_from_offline_length = int((offline_games * time_steps) / sample_sp_from_offline_times)


    # print("Number of sampling data from the offline dataset: ", sample_sp_from_offline_times)
    # print("Length of data sampling",sample_sp_from_offline_length)


    file_name = sample_data(str(OFFLINE_DATASET_PATH))

    offline_data = np.load(file_name, allow_pickle=True)

    # start_point_list = start_point_picker(offline_data, interval=10)
    assert args.number < len(offline_data) , "number should be smaller than the number of files in the offline dataset"
    start_point_list = start_point_picker_v2(offline_data, args.number,interval=10)

    print("Number of start points: ", len(start_point_list))

    img_dataset_all = {}

    img_dataset_start_point = []
    img_dataset_current_obs = []

    img_dataset_gt_next_obs = []
    img_dataset_gt_next_actions = []
    img_dataset_gt_next_rewards = []
    img_dataset_gt_next_dense_rewards = []
    img_dataset_gt_next_done = []

    img_dataset_im_next_obs = []
    img_dataset_im_next_actions = []
    img_datset_im_reward = []
    img_datset_im_dense_reward = []
    img_dataset_im_done = []


    img_dataset_policy_code = []
    img_dataset_reward_code = []
    img_dataset_transition_code = []
    img_dataset_gen_times = []

    try:
        rewarder = Rewarder()
    except Exception as e:
        print("Error: ", e)
        print("rewarder init failed, retrying...")
        rewarder = Rewarder()

    bc_root_path = str(BC_MODEL_ROOT)

    bc_models = {}

    import d3rlpy

    for i in range(22):
        file_start = f"comment=final-nz-10x-fix-feat&stack_hist=10&stack_obs=3&batch_size=4096&pl={i}&"
        # find the file start with file_start
        for file in os.listdir(bc_root_path):
            if file.startswith(file_start):
                exp_path = os.path.join(bc_root_path, file)
                train_episodes = 500000
                path = model_path =  os.path.join(exp_path, f'model_{train_episodes}.d3')
                player_id = path.split('pl=')[1].split('_')[0].split('&')[0]
                stack_hist = path.split('stack_hist=')[1].split('&')[0]
                stack_obs_num = path.split('stack_obs=')[1].split('&')[0]
                bc = d3rlpy.load_learnable(path)
                bc_models[i] = bc

    one_debug = True

    import time
    time_str = time.strftime("%Y%m%d-%H%M%S")

    randomized_start_points = random.sample(start_point_list, len(start_point_list))

    for i in range(len(start_point_list)):
    #for i in range(5):

        start_point = randomized_start_points[i]


        print("Current start point: ", start_point, "Current index: ", i , "Total: ", len(start_point_list))

        offline_d = load_inner_npz_by_index(offline_data, start_point[0])

        current_obs = offline_d['obs_before_modified_by_acs'][start_point[1]][0]

        current_obs = imaginary_data_observation_v2(current_obs, start_point[1], ret_type='dict')

        gt_next_gen_length_obs = offline_d['obs_before_modified_by_acs'][start_point[1] :start_point[1] + 1 + gen_length]
        gt_next_gen_length_actions = offline_d['action'][start_point[1] :start_point[1] + 1 + gen_length]
        gt_next_gen_length_rewards = offline_d['reward'][start_point[1] :start_point[1] + 1 + gen_length]
        gt_next_gen_length_dense_rewards = offline_d['dense_rewards'][start_point[1] :start_point[1] + 1 + gen_length]
        gt_next_gen_length_done = offline_d['done'][start_point[1] :start_point[1] + 1 + gen_length]

        # im_next_obs = gt_next_gen_length_obs
        fine_tuning_all_flag = True
        if fine_tuning_all_flag:
            policy_code, reward_code, transition_code, code_thought = get_all_code(current_obs,policy_retriever, reward_retriever, transition_retriever)
        else:
            policy_code_answer, policy_code_thought = get_policy_code(current_obs)
            policy_code = "Code: " + json.dumps(policy_code_answer) + "\n" + "Thought: " + json.dumps(policy_code_thought) + "\n"


        im_next_obs_list = []
        im_action_list = []
        im_reward_list = []
        im_dense_reward_list = []
        im_done_list = []

        previous_score_diff = current_obs['score'][0] - current_obs['score'][1]
        prev_img_obs = current_obs


        acs_shape = 4 # ([zone_diff, [own_the_ball_state_diff], [ball_direction_diff]])
        stack_obs_len = imaginary_data_to_vector(current_obs, TODO_missing = True).shape[0] * 3 + acs_shape * 10 #3 is the last 3 timestep obs, 4 is the action length , 10 is the last 10 time steps action

        sp1 = start_point[1]


        if sp1 < 10:
            assert gen_length == 10,  "gen_length should be 10"
            stack_obs = np.zeros(stack_obs_len)
            acs_list = [np.zeros(acs_shape)for i in range(22)]
        else:
            stack_obs = np.zeros(stack_obs_len)
            acs_list = [np.zeros(acs_shape)for i in range(22)]

            past_10_obs = offline_d['obs_before_modified_by_acs'][start_point[1]-10 : start_point[1]+1]

            acs_pred_error_list = [[] for i in range(22)]


            for i in range(len(past_10_obs)-1):

                current_obs_stack = past_10_obs[i][0]
                current_obs_stack = imaginary_data_observation_v2(current_obs_stack, start_point[1] - 10 +  i, ret_type='dict')


                for j in range(22):

                    acs = acs_list[j]
                    bc = bc_models[j]
                    obs_vector = imaginary_data_to_vector(current_obs_stack, TODO_missing = True)
                    stack_obs = update_stack_obs(obs_vector, acs, stack_obs, state_stack_num=3)

                next_obs = past_10_obs[i+1][0]
                next_obs = imaginary_data_observation_v2(next_obs, start_point[1] -10 + i + 1, ret_type='dict')


                for k in range(22):
                    acs = obs_to_robot_acs(next_obs, current_obs_stack, k)
                    acs_list.append(acs)
                    if one_debug:
                        acs_pred_error_list[k].append((np.abs(bc_models[k].predict(np.array([stack_obs])) - acs)))

            if one_debug:
                for i in range(22):
                    player_error = np.mean(acs_pred_error_list[i], axis=0)
                    print("Player: ", i, "Error: ", player_error)

                error_all = np.mean(acs_pred_error_list)
                print("ALL Error: ", error_all)
                one_debug = False

        log_loss_list = []

        im_next_obs_list.append(current_obs)

        for i in tqdm(range(gen_length)):


            # stack obs
            other_player_action = []
            for k in range(22):
                acs = acs_list[k]
                bc = bc_models[k]
                obs_vector = imaginary_data_to_vector(current_obs, TODO_missing = True)
                stack_obs = update_stack_obs(obs_vector, acs, stack_obs, state_stack_num=3)
                acs_pred = bc.sample_action(np.array([stack_obs]))
                other_player_action.append(acs_pred)


            #next_obv, next_obs, action, action_thought

            if fine_tuning_all_flag:
                im_next_obs, im_action, im_dense_reward = imaginary_data_gen_v2(current_obs, policy_code, reward_code, transition_code, other_player_action)
            else:
                im_next_obs, _ , im_action, _ = imaginary_data_gen(current_obs, policy_code, other_player_action)

            log_vector_next_obs = imaginary_data_to_vector(im_next_obs , TODO_missing = True)
            log_vector_gt_next_obs = imaginary_data_observation_v2(gt_next_gen_length_obs[i][0], start_point[1] + i , ret_type='vector')
            log_loss = np.abs(log_vector_next_obs - log_vector_gt_next_obs)
            log_loss_list.append(log_loss)
            im_next_obs_list.append(im_next_obs)
            im_action_list.append(im_action)


            # Reward:
            score_diff = im_next_obs['score'][0] - im_next_obs['score'][1]
            reward = score_diff - previous_score_diff
            previous_score_diff = score_diff

            im_reward_list.append(reward)

            # Dense Reward:
            # dense_rewards = rewarder.calc_reward_v2(reward, im_next_obs, prev_img_obs, im_action)
            # prev_img_obs = im_next_obs

            dense_rewards = im_dense_reward

            im_dense_reward_list.append(dense_rewards)

            # Done:
            if reward !=0:
                done = True
            else:
                done = False

            im_done_list.append(done)

            # update
            # im_next_obs ， current_obs


            for j in range(22):
                acs = obs_to_robot_acs(im_next_obs, current_obs, j)
                acs_list.append(acs)


            current_obs = im_next_obs

        print("-- 10 steps log loss: ", np.mean(log_loss_list, axis = 0))

        img_dataset_start_point.append(start_point)
        img_dataset_current_obs.append(current_obs)
        img_dataset_gt_next_obs.append(gt_next_gen_length_obs)
        img_dataset_gt_next_actions.append(gt_next_gen_length_actions)
        img_dataset_gt_next_rewards.append(gt_next_gen_length_rewards)
        img_dataset_gt_next_dense_rewards.append(gt_next_gen_length_dense_rewards)
        img_dataset_gt_next_done.append(gt_next_gen_length_done)


        img_dataset_im_next_obs.append(im_next_obs_list)
        img_dataset_im_next_actions.append(im_action_list)
        img_datset_im_reward.append(im_reward_list)
        img_datset_im_dense_reward.append(im_dense_reward_list)
        img_dataset_im_done.append(im_done_list)

        img_dataset_policy_code.append(policy_code)
        img_dataset_reward_code.append(reward_code)
        img_dataset_transition_code.append(transition_code)

        img_dataset_gen_times.append(1)

        current_wapper_obs = load_inner_npz_by_index(offline_data, start_point[0])['wrapper_obs'][start_point[1]][0]


        img_dataset_all['start_point'] = img_dataset_start_point
        img_dataset_all['current_obs'] = img_dataset_current_obs
        img_dataset_all['gt_next_obs'] = img_dataset_gt_next_obs
        img_dataset_all['gt_next_actions'] = img_dataset_gt_next_actions
        img_dataset_all['gt_reward'] = img_dataset_gt_next_rewards
        img_dataset_all['gt_dense_reward'] = img_dataset_gt_next_dense_rewards
        img_dataset_all['gt_done'] = img_dataset_gt_next_done


        img_dataset_all['im_next_obs'] = img_dataset_im_next_obs
        img_dataset_all['im_next_actions'] = img_dataset_im_next_actions
        img_dataset_all['im_reward'] = img_datset_im_reward
        img_dataset_all['im_dense_reward'] = img_datset_im_dense_reward
        img_dataset_all['im_done'] = img_dataset_im_done

        img_dataset_all['im_done'] = img_dataset_im_done
        img_dataset_all['policy_code'] = img_dataset_policy_code
        img_dataset_all['reward_code'] = img_dataset_reward_code
        img_dataset_all['transition_code'] = img_dataset_transition_code

        img_dataset_all['gen_times'] = img_dataset_gen_times

        print("Number of imaginary dataset: ", len(img_dataset_all['im_next_obs']))

        # save this to a npz file

        save_dir = os.environ.get('PLFB_IMAGINARY_DATASET_PATH', './imaginary_dataset_0520')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'no_{args.number}_imaginary_dataset_{len(start_point_list)}x{gen_length}_{time_str}.npz')
        try:
            np.savez_compressed(file_name, **img_dataset_all)
        except Exception as e:
            print("Error: ", e)
            for key in img_dataset_all:
                print("Key: ", key, "Shape: ", np.array(img_dataset_all[key]).shape)
            print("Saving to npz file failed, saving to pkl file")
            import pdb; pdb.set_trace()
            import pickle
            with open(file_name, 'wb') as f:
                pickle.dump(img_dataset_all, f)

