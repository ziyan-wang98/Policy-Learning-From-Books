import os
from pathlib import Path
import sys
sys.path.append("football_llm/TiZero/submission/tizero_agent")
sys.path.append('football_llm')

import os.path as osp
from pydantic import BaseModel, Field
from typing import List
from tqdm import tqdm
import json
import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from llm.utils.obs2text import directions
# from ...rule_base_1 import agent as rule_base_1_agent
# from rule_base_1.agent import agent as rule_base_1_agent
from llm.algo.rule_base_2.gfootball import agent as rule_base_2_agent
from llm.algo.rule_base_2.gfootball import agent_dict as rule_base_2_agent_dict
from llm.algo.rule_base_2.gfootball import get_memory_patterns_new as rule_base_2_get_memory_patterns
from enum import Enum

from llm.utils.obs2text import observation_to_text_human, observation_to_text_raw, get_zons_240, format_code, imaginary_data_observation, imaginary_data_to_vector
from llm.utils.index import get_engine, get_index, index_llm
from llm.utils.openai_query import query as openai_query

import llm.config.imaginary_data_parser as imaginary_data_parser
from llm.utils.openai_query import query as openai_query
from llm.utils.openai_compat import openai_chat_query
from llm.utils.obs2text import get_zons_240_list, vector_to_direction,vector_to_direction

import numpy as np

MAX_ATTAMPTS = 5

class Action(Enum):
    Idle = 0
    Left = 1
    TopLeft = 2
    Top = 3
    TopRight = 4
    Right = 5
    BottomRight = 6
    Bottom = 7
    BottomLeft = 8
    LongPass= 9
    HighPass = 10
    ShortPass = 11
    Shot = 12
    Sprint = 13
    ReleaseDirection = 14
    ReleaseSprint = 15
    Slide = 16
    Dribble = 17
    ReleaseDribble = 18


sticky_index_to_action = [
    Action.Left,
    Action.TopLeft,
    Action.Top,
    Action.TopRight,
    Action.Right,
    Action.BottomRight,
    Action.Bottom,
    Action.BottomLeft,
    Action.Sprint,
    Action.Dribble
]


ACTION_TEXT = [
    "idle",
    "go left",
    "go top left",
    "go top",
    "go top right",
    "go right",
    "go bottom right",
    "go bottom",
    "go bottom left",
    "do long_pass",
    "do high_pass",
    "do short_pass",
    "shot",
    "do sprint",
    "release_direction",
    "release_sprint",
    "sliding",
    "dribble",
    "release_dribble",
]

GPT_MODEL = os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")


def load_data(root_path):

    dataset = []
    for file in tqdm.tqdm(os.listdir(root_path), desc='loading data'):
        if file.endswith('.npz'):
            res = np.load(osp.join(root_path, file), allow_pickle=True)
            if 'obs_before_modified_by_acs' not in res.keys():
                print("skip this file", file)
                continue
            dataset.append(res)
    return dataset

def imaginary_data_log(len_data_from_offline_dataset, generate_len, steps_list, action_list, reward_list, dense_rewards_list, done_list, imaginary_obs_list):

    default_imaginary_path = os.path.join(os.environ.get("PLFB_WORK_ROOT", "plfb_work"), "imaginary_data-v3", os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    imaginary_dataset_path = os.environ.get("PLFB_IMAGINARY_DATASET_PATH", default_imaginary_path)
    if not os.path.exists(imaginary_dataset_path):
        os.makedirs(imaginary_dataset_path)
    index = 0
    unique_file_name = f"imaginary_offlinelen{len_data_from_offline_dataset}+{generate_len}_{index}.npz"
    while os.path.exists(os.path.join(imaginary_dataset_path, unique_file_name)):
        index += 1
        unique_file_name = f"imaginary_offlinelen{len_data_from_offline_dataset}+{generate_len}_{index}.npz"

    offline_dataset_file_path = os.path.join(imaginary_dataset_path, unique_file_name)

    np.savez(offline_dataset_file_path, steps=steps_list, action=action_list, reward=reward_list, dense_rewards=dense_rewards_list, done=done_list, imaginary_obs=imaginary_obs_list)

def get_direction_from_text(current_zone, current_direction, next_zone):
    if current_zone == next_zone:
        return current_direction
    else:
        delta_x = next_zone[0] - current_zone[0]
        delta_y = next_zone[1] - current_zone[1]
        return vector_to_direction(delta_x, delta_y)


def change_format(current_obs, next_obs, action, other_player_action=None):
    res = {}

    sticky_list = ["Left","TopLeft","Top","TopRight","Right","BottomRight","Bottom","BottomLeft","Sprint","Dribble"]
    game_modes = ["Noramal", "KickOff", "GoalKick", "FreeKick", "Corner", "hrowIn", "Penalty"]  # Replace with actual game modes
    role_list = ["Goalkeeper", "Forward", "Forward", "Defender", "Defender", "Defender", "Defender", "Midfielder", "Midfielder", "Midfielders", "Forward"]


    #sticky_actions

    sticky_action_list = [0] * len(sticky_list)
    for a in current_obs["sticky_actions"]:
        sticky_action_list[sticky_list.index(a)] = 1


    # Sticky action
    if current_obs["active_player"] == next_obs["active_left_player"]:
        if action in range(1,9):
            # sticky_action_list[0:8] should be 0
            sticky_action_list[0:8] = [0] * 8
            sticky_action_list[action-1] = 1

        elif action == 13:
            sticky_action_list[8] = 1
        elif action == 17:
            sticky_action_list[9] = 1
    else:
        sticky_action_list = [0 for _ in range(10)]

    sticky_action_list = np.array(sticky_action_list)
    if len(sticky_action_list) > 0 and np.any(sticky_action_list != 0):
        indices = np.where(sticky_action_list == 1)[0]
        selected_actions = [sticky_list[idx] for idx in indices]
        res["sticky_actions"] = selected_actions
    else:
        res["sticky_actions"] = []

    # Game mode
    res["game_mode"] = "Noramal"


    # Score
    res["score"] = next_obs["score"]

    # Step
    res["step"] = next_obs["step"]

    # Time
    time =  next_obs["step"] * 1.8 # 3000 timpstep = 90 mins
    min = int(time // 60)
    sec = int(time % 60)
    time_text = f"The current time is {min} minutes {sec} seconds. "
    res["time"] = time_text



    # Ball direction
    current_ball_zone = current_obs["ball_zone"]
    current_ball_direction = current_obs["ball_direction"]
    next_obs["ball_direction"] = get_direction_from_text(current_ball_zone, current_ball_direction, next_obs["ball_zone"])


    bc_active_player_list = []
    ball_change_flag_list = []
    # Player
    for i in range(22):

        player = {}
        if i < 11:
            player["team"] = "Left"
        else:
            player["team"] = "Right"
        player["role"] = role_list[i%11]
        if i == next_obs["active_left_player"]:
            player["zone"] = next_obs["left_active_player_zone"]
            current_player_zone = current_obs[f"player_{i}"]["zone"]
            current_player_directioin = current_obs[f"player_{i}"]["direction"]
            player["direction"] = get_direction_from_text(current_player_zone, current_player_directioin, player["zone"])
            ball_change_flag_list.append(False)
        else:
            zone_diff_x  = np.round(other_player_action[i][0][0])
            zone_diff_y = np.round(other_player_action[i][0][1])
            own_the_ball_state_diff = np.clip(np.round(other_player_action[i][0][2]), -1, 1)
            ball_direction_diff = np.round(other_player_action[i][0][3])
            ball_change_flag = (ball_direction_diff != 0)

            # for players
            player["zone"] = [current_obs[f"player_{i}"]["zone"][0] + zone_diff_x, current_obs[f"player_{i}"]["zone"][1] + zone_diff_y]
            current_player_zone = current_obs[f"player_{i}"]["zone"]
            current_player_directioin = current_obs[f"player_{i}"]["direction"]
            player["direction"] = get_direction_from_text(current_player_zone, current_player_directioin, player["zone"])

            # active player
            own_the_ball_state = current_obs["ball_ownership_player"]  == i
            next_own_the_ball_state = own_the_ball_state + own_the_ball_state_diff

            if next_own_the_ball_state == 1:
                bc_active_player_list.append(i)

            # ball direction
            ball_change_flag_list.append(ball_change_flag)

        res[f"player_{i}"] = player

    if len(bc_active_player_list) == 0:
        # no one control the ball or the human player is playing
        if current_obs["ball_ownership_player"] != current_obs['active_player']:
            next_obs["ball_ownership_player"] = -1
            next_obs["ball_ownership"] = 0
        else:
            if action in (9,10,11,12):
                next_obs["ball_ownership_player"] = -1
                next_obs["ball_ownership"] = 0
            else:
                next_obs["ball_ownership_player"] = current_obs["ball_ownership_player"]
                next_obs["ball_ownership"] = 1
    else:
        if len(bc_active_player_list) > 1:
            next_obs["ball_ownership_player"] = np.random.choice(bc_active_player_list)
        else:
            next_obs["ball_ownership_player"] = bc_active_player_list[0]
        next_obs['ball_zone'] = res[f"player_{next_obs['ball_ownership_player']}"]['zone']

        if next_obs["ball_ownership_player"] < 11:
            next_obs["ball_ownership"] = 1
            next_obs["active_left_player"] = next_obs["ball_ownership_player"]
        else:
            next_obs["ball_ownership"] = 2
            # find the closet player by using zones and let it be the left active player
            left_active_player = 0
            left_active_player_zone = res[f"player_{left_active_player}"]["zone"]
            left_active_player_zone_diff = np.sum(np.abs(np.array(left_active_player_zone) - np.array(next_obs["ball_zone"])))
            for i in range(1, 11):
                current_zone = res[f"player_{i}"]["zone"]
                current_zone_diff = np.sum(np.abs(np.array(current_zone) - np.array(next_obs["ball_zone"])))
                if current_zone_diff < left_active_player_zone_diff:
                    left_active_player = i
                    left_active_player_zone = current_zone
                    left_active_player_zone_diff = current_zone_diff
            next_obs["active_left_player"] = left_active_player

    for i in range(22):
        if i == next_obs["ball_ownership_player"]:
            if ball_change_flag_list[i]:
                ball_direction_diff = np.round(other_player_action[i][0][3])
                ball_current_direction = current_obs["ball_direction"]
                next_obs["ball_direction"] = directions[int(directions.index(ball_current_direction) + ball_direction_diff) % len(directions)]
            else:
                next_obs["ball_direction"] = res[f"player_{i}"]["direction"]

    # Active player
    res["active_player"] = next_obs["active_left_player"]

    # Active player role
    res["active_player_role"] = role_list[next_obs["active_left_player"]]

    # Ball ownership
    res["ball_ownership"] = next_obs["ball_ownership"]

    # Ball ownership player
    res["ball_ownership_player"] = next_obs["ball_ownership_player"]

    # Ball zone
    res["ball_zone"] = next_obs["ball_zone"]

    # Ball direction
    res["ball_direction"] = next_obs["ball_direction"]

    return res



def imaginary_data_gen(current_dict_obs):

    ## TODO: Change here to finetune the model
    context_path = os.environ.get("PLFB_POLICY_CONTEXT_PATH", str(Path(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts")) / "book_derived" / "v4-gpt-3.5-turbo-1106-level-strict" / "best" / "policy" / "agg_postprocess.jsonl"))
    context_list = []
    with open(context_path, 'r',encoding='utf-8') as file:
        for line in file:
            context_list.append(line)
    context_str = "\n\n".join(context_list)

    action_prompt = raw_policy_prompt.format(context_str=context_str, text_obs=current_dict_obs)

    if not args.use_openai_compat_client:
        for attempt in range(MAX_ATTAMPTS):
            try:
                action_response = openai_query("", action_prompt, GPT_MODEL, req_json=True)
                action = json.loads(action_response.choices[0].message.content)["action"]
                action_thought = json.loads(action_response.choices[0].message.content)["thought"]
                break
            except Exception as e:
                print(colored(f"Error: {e}, retrying...", "red"))
                if attempt == MAX_ATTAMPTS - 1:
                    raise e
                continue
    else:
        for attempt in range(MAX_ATTAMPTS):
            try:
                action_response = openai_chat_query("", action_prompt, model_name=GPT_MODEL, req_json=True)
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

    print(f"The football coach choose action: {Action(action)}, thought: {action_thought}")


    # step 5: get the next obs
    next_obs_prompt = imaginary_data_prompt.format(current_texted_obs = current_dict_obs, action = action)

    if not args.use_openai_compat_client:
        for attempt in range(MAX_ATTAMPTS):
            try:
                next_obs_response = openai_query("", next_obs_prompt, GPT_MODEL, req_json=True)
                next_obs = json.loads(next_obs_response.choices[0].message.content)
                break
            except Exception as e:
                print(colored(f"Error: {e}, retrying...", "red"))
                if attempt == MAX_ATTAMPTS - 1:
                    raise e
                continue
    else:
        for attempt in range(MAX_ATTAMPTS):
            try:
                next_obs_response = openai_chat_query("", next_obs_prompt, model_name=GPT_MODEL, req_json=True)
                next_obs = json.loads(next_obs_response)
                next_obs = json.loads(next_obs["choices"][0]["message"]["content"])
                break
            except Exception as e:
                print(colored(f"Error: {e}, retrying...", "red"))
                if attempt == MAX_ATTAMPTS - 1:
                    raise e
                continue


    # step 6: transfer the output's format
    next_obv = change_format(current_dict_obs, next_obs, action)

    return next_obv, next_obs, action


def imaginary_data_gen_v2(current_dict_obs, policy_code,raw_policy_prompt_step2_globel, raw_policy_prompt_step2, imaginary_data_prompt, model_name):

    choose_action_prompt = raw_policy_prompt_step2.format(text_obs=current_dict_obs, policy_code=policy_code)

    for attempt in range(MAX_ATTAMPTS):
        try:
            next_obs_response = openai_chat_query(raw_policy_prompt_step2_globel, choose_action_prompt, model_name=model_name, req_json=True)
            next_obs = json.loads(next_obs_response)
            next_obs = json.loads(next_obs["choices"][0]["message"]["content"])
            break
        except Exception as e:
            print(colored(f"Error: {e}, retrying...", "red"))
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue


    next_obs_prompt = imaginary_data_prompt.format(current_texted_obs = current_dict_obs, action = action)

    for attempt in range(MAX_ATTAMPTS):
        try:
            next_obs_response = openai_chat_query("", next_obs_prompt, model_name=GPT_MODEL, req_json=True)
            next_obs = json.loads(next_obs_response)
            next_obs = json.loads(next_obs["choices"][0]["message"]["content"])
            break
        except Exception as e:
            print(colored(f"Error: {e}, retrying...", "red"))
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue


    # step 6: transfer the output's format
    next_obv = change_format(current_dict_obs, next_obs, action)

    return next_obv, next_obs, action




if __name__ == '__main__':

    args = imaginary_data_parser.parse_args(return_parser=False)

    # get the all prompt and setup
    imaginary_data_prompt = args.imaginary_obv_prompt
    raw_policy_prompt = args.raw_policy_prompt
    len_data_from_offline_dataset = 10

    generate_len = 10


    #['steps', 'obs', 'wrapper_obs', 'texted_obs', 'action', 'reward', 'dense_rewards', 'done', 'obs_before_modified_by_acs', 'imaginary_obs']


    # step 1: get the dataset:
    test =  True
    offline_dataset_path = os.environ.get("PLFB_OFFLINE_DATASET_PATH", os.path.join(os.environ.get("PLFB_DATASET_PATH", os.path.join(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts"), "football")), "offline_dataset-v4"))

    if not test:
        dataset = load_data(offline_dataset_path)
    else:
        dataset = np.load(osp.join(offline_dataset_path, 'rule_based_2_11_vs_11_easy_stochastic_200.npz'), allow_pickle=True)

    print(list(dataset.keys()))

    # step 2: save the offline dataset part to imaginary dataset

    res_list = [[] for _ in range(10)]

    keys = ['steps', 'obs', 'wrapper_obs', 'texted_obs', 'action', 'reward', 'dense_rewards', 'done', 'obs_before_modified_by_acs', 'imaginary_obs']
    for list_, key in zip(res_list, keys):
        if key == 'imaginary_obs':
            for i in range(len_data_from_offline_dataset):
                current_obs = dataset['obs_before_modified_by_acs'][i][0]
                current_warpper_obs = dataset['wrapper_obs'][i][0]
                current_dict_obs, current_vec_obs = imaginary_data_observation(current_warpper_obs, current_obs, i, ret_type='both')
                list_.append([current_dict_obs])
        else:
            list_.extend(dataset[key][0:len_data_from_offline_dataset])


    for _ in tqdm(range(generate_len), desc='generating imaginary data'):

        # step 3: get the current data
        current_list = [list_[-1] for list_ in res_list]

        current_dict_obs = current_list[9][0]

        next_obv, next_obs, action = imaginary_data_gen(current_dict_obs)

        # next_victor_obs = imaginary_data_to_vector(next_obv)

        # step 7: add the next obs to the dataset
        #keys = ['steps', 'obs', 'wrapper_obs', 'texted_obs', 'action', 'reward', 'dense_rewards', 'done', 'obs_before_modified_by_acs', 'imaginary_obs']
        res_list[0].append(current_list[0]+1)
        action_array = np.array([action])
        res_list[4].append(action_array)

        res_list[5].append(float(next_obs["score"][0])) # TODO: check reward
        res_list[6].append(float(next_obs["score"][0])) # TODO: check dense reward

        if res_list[0][-1] < 1500:
            res_list[7].append(False)
        else:
            res_list[7].append(True)
        res_list[9].append([next_obv])


    # Save the imaginary dataset
    imaginary_data_log(len_data_from_offline_dataset, generate_len, res_list[0], res_list[4], res_list[5], res_list[6], res_list[7], res_list[9])
