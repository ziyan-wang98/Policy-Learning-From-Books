import sys
sys.path.append("football_llm/TiZero/submission/tizero_agent")
sys.path.append('football_llm')
import time
import os
import copy
import json
import config.baseline_agent_parser as baseline_agent_parser
from utils.obs2text import observation_to_text_human, observation_to_text_raw, long_obs_list_to_text
from algo.baseline_v1_single_agent import BaselineV1SingleAgent # , Baseline_v1_multi_agent
from algo.baseline_v1_multi_agent import BaselineV1MultiAgent
import gfootball.env as football_env
import numpy as np
from enum import Enum
import time
from algo.rule_base_2.gfootball import agent as rule_based_agent_2
from algo.tizero_agent.submission import my_controller, my_controller_multi
from utils.rewarder import Rewarder

from gfootball.env.wrappers import Simple115StateWrapper_ball_owned_player
import datetime

from utils.obs2text import observation_to_text_human
# Set your API keys here
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# os.environ["REPLICATE_API_TOKEN"] = "YOUR_REPLICATE_API_TOKEN"
# assert os.environ["OPENAI_API_KEY"] != None, "Please set your OPENAI_API_KEY in the environment variable"
# assert os.environ["REPLICATE_API_TOKEN"] != None, "Please set your REPLICATE_API_TOKEN in the environment variable"


def setup_directories(args):
    """
    Creates necessary directories for saving outputs and returns their paths.

    :param args: ArgumentParser object containing necessary attributes.
    :return: A tuple containing the paths to the video and JSON directories.
    """
    # Create the base result directory
    result_dir = './result/new_rule_based_2_offline_dataset'
    os.makedirs(result_dir, exist_ok=True)

    # Create a unique subdirectory
    index = 0
    unique_dir_name = f"{args.algo}_{args.environment}_{index}"
    unique_dir_path = os.path.join(result_dir, unique_dir_name)
    while os.path.exists(unique_dir_path):
        index += 1
        unique_dir_name = f"{args.algo}_{args.environment}_{index}"
        unique_dir_path = os.path.join(result_dir, unique_dir_name)
    
    os.makedirs(unique_dir_path, exist_ok=True)

    # Create subdirectories for videos and JSON files
    video_dir = os.path.join(unique_dir_path, 'video')
    json_dir = os.path.join(unique_dir_path, 'json')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    return video_dir, json_dir

def convert_to_serializable(obj):
    """
    Recursively convert various non-serializable objects (like Enums, numpy arrays, sets, numpy numerical types) 
    to serializable forms (like strings, lists, int).
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays to lists and handle numpy numerical types
        return [convert_to_serializable(elem) for elem in obj.tolist()]
    elif isinstance(obj, Enum):
        return obj.name  # Convert Enums (including Action) to their name representation
    elif isinstance(obj, set):
        return list(obj)  # Convert sets to lists
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()  # Convert numpy numerical types to Python scalar types
    else:
        return obj
    
def save_json_to_file(json_outputs_list, json_dir):
    """
    Saves a list of JSON serializable objects to a file. Converts any NumPy arrays to lists.

    :param json_outputs_list: List of objects to be saved as JSON.
    :param json_dir: Directory and filename where the JSON file will be saved.
    """
    # Convert any ndarray to list
    json_outputs_list = convert_to_serializable(json_outputs_list)
    
    try:
        with open(json_dir+'/log_{}.json'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), 'w') as file:
            json.dump(json_outputs_list, file, indent=4)
        return True, f"Data successfully saved to {json_dir}"
    except Exception as e:
        return False, f"An error occurred: {e}"


if __name__ == "__main__":
    args = baseline_agent_parser.parse_args()
    wrapper = None
    wrapper_func = None
    
    
    if args.logall:
        video_dir, json_dir = setup_directories(args)
        args.logdir = video_dir
        args.write_full_episode_dumps = True
    
    if args.offline_dataset_collection:
        args.write_full_episode_dumps = False
        
        offline_dataset_path = os.environ.get("PLFB_OFFLINE_DATASET_PATH", os.path.join(os.environ.get("PLFB_DATASET_PATH", os.path.join(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts"), "football")), "offline_dataset-v4"))
        if not os.path.exists(offline_dataset_path):
            os.makedirs(offline_dataset_path)
 
        wrapper_func = Simple115StateWrapper_ball_owned_player
        
        rewarder = Rewarder()
        steps_list = []
        obs_list = []
        wrapper_obs_list = []
        texted_obs_list = []
        action_list = []
        reward_list = []
        dense_rewards_list = [] 
        obs_before_modified_by_acs_list = []
        done_list = []
            
        
        
    if args.algo == "baseline_v1_single_agent":
        policy = BaselineV1SingleAgent(args)
        wrapper_func = Simple115StateWrapper_ball_owned_player
    elif args.algo == "baseline_v1_multi_agent":
        args.num_players = 11
        policy = BaselineV1MultiAgent(args)
        wrapper_func = Simple115StateWrapper_ball_owned_player
    elif args.algo == "baseline_v2_raw_policy":
        raise NotImplementedError
    elif args.algo == "baseline_v2_MPC":
        raise NotImplementedError
    elif args.algo == "baseline_v2_Offline":
        raise NotImplementedError
    elif args.algo == "rule_based_2":
        policy = rule_based_agent_2
    elif args.algo == "tizero_agent":
        policy = my_controller
    elif args.algo == "tizero_agent_multi": # TODO: fix this
        args.num_players = 11
        policy = my_controller_multi
    else:
        raise NotImplementedError
    
    env = football_env.create_environment(env_name=args.environment,representation='raw', \
                                    stacked=False, logdir=args.logdir, write_goal_dumps=args.write_goal_dumps, \
                                    write_full_episode_dumps=args.write_full_episode_dumps, render=args.render,\
                                    number_of_left_players_agent_controls=args.num_players)
    
    print("args.representation: ", args.representation)
    

    if wrapper_func or args.offline_dataset_collection:
        wrapper = wrapper_func(env)
    
    steps = 0
    obs = env.reset()
    if wrapper or args.offline_dataset_collection:
        wrapper_obs = wrapper.observation(obs)
        text_obs = observation_to_text_human(wrapper_obs[0], obs[0], steps, block_mode=args.block_mode)

    json_outputs_list = []
    
    if args.offline_dataset_collection:
        rand = np.clip(np.random.rand(), 0, 0.4)
    else:
        rand = 0
    while True:
        obs_before_modified_by_acs = copy.deepcopy(obs)
        if "baseline" in args.algo:
            action, json_output = policy.get_action(obs[0], wrapper_obs[0], steps)
            if args.logall:
                json_output = convert_to_serializable(json_output)
                json_outputs_list.append(json_output)
        elif "rule_based" in args.algo:
            action = policy(obs[0])
        elif args.algo ==  "tizero_agent":
            action = policy(obs[0])
        elif args.algo ==  "tizero_agent_multi":
            # obs_list = [obs[0] for _ in range(10)]
            action = policy(obs)
        else:
            raise NotImplementedError
        if np.random.rand() < rand:
            action = np.random.randint(19)
        if type(action) == list:
            action = action[0]
        next_obs, rew, done, info = env.step(action)
        if args.offline_dataset_collection:
            # calculate dense rewards from https://github.com/Shanghai-Digital-Brain-Laboratory/DB-Football/blob/ca1bc3826ddceb7f5c3fc3d3b1685a915054ca14/light_malib/envs/gr_football/rewarder_basic.py
            prev_obs = obs_before_modified_by_acs_list[-1] if len(obs_before_modified_by_acs_list) > 0 else obs_before_modified_by_acs
            dense_rewards = rewarder.calc_reward(rew, obs_before_modified_by_acs[0], prev_obs[0], action)
            # Save data
            steps_list.append(steps)
            obs_list.append(obs)
            obs_before_modified_by_acs_list.append(obs_before_modified_by_acs)
            wrapper_obs_list.append(wrapper_obs)
            texted_obs_list.append(text_obs)
            action_list.append(action)
            reward_list.append(rew)
            dense_rewards_list.append(dense_rewards)
            done_list.append(done)
            
        obs = next_obs
        if wrapper or args.offline_dataset_collection:
            wrapper_obs = wrapper.observation(copy.deepcopy(obs))
            text_obs = observation_to_text_human(copy.deepcopy(wrapper_obs[0]), copy.deepcopy(obs[0]), steps, block_mode=args.block_mode)
        steps += 1
        if steps==args.num_timesteps or done:
            if args.offline_dataset_collection:
                rand = np.clip(np.random.rand(), 0, 0.4)
            break
    # print("Steps: %d Reward: %.2f" % (steps, rew))
    env.close()
    
    
    if args.logall:
        try:
            save_json_to_file(json_outputs_list, json_dir)
        except Exception as e:
            print("An error occurred: ", e)
            
    if args.offline_dataset_collection:
        index = 0
        unique_file_name = f"{args.algo}_{args.environment}_{index}.npz"
        while os.path.exists(os.path.join(offline_dataset_path, unique_file_name)):
            index += 1
            unique_file_name = f"{args.algo}_{args.environment}_{index}.npz"
        offline_dataset_file_path = os.path.join(offline_dataset_path, unique_file_name)
        np.savez(offline_dataset_file_path, steps=steps_list, obs=obs_list, \
            wrapper_obs=wrapper_obs_list, texted_obs=texted_obs_list, action=action_list, \
                reward=reward_list, dense_rewards=dense_rewards_list, done=done_list, \
                    obs_before_modified_by_acs=obs_before_modified_by_acs_list)