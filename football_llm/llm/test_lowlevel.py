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

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
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
    result_dir = './low_level_results'
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
    os.makedirs(video_dir, exist_ok=True)
    return video_dir

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


def run_env(algo, map):
    args = baseline_agent_parser.parse_args()
    wrapper = None
    wrapper_func = None
    

    args.algo = algo
    args.environment = map
    args.num_timesteps = 1500
    
    if args.logall:
        video_dir = setup_directories(args)
        args.logdir = video_dir
        args.write_full_episode_dumps = True
    
        
    if args.algo == "baseline_v1_single_agent_l0":
        policy = BaselineV1SingleAgent(args, level=0)
        wrapper_func = Simple115StateWrapper_ball_owned_player
    elif args.algo == "baseline_v1_single_agent_l1":
        policy = BaselineV1SingleAgent(args, level=1)
        wrapper_func = Simple115StateWrapper_ball_owned_player
    elif args.algo == "baseline_v1_single_agent_l2":
        policy = BaselineV1SingleAgent(args, level=2)
        wrapper_func = Simple115StateWrapper_ball_owned_player
    elif args.algo == "baseline_v1_single_agent_l3":
        policy = BaselineV1SingleAgent(args, level=3)
        wrapper_func = Simple115StateWrapper_ball_owned_player
    elif args.algo == "baseline_v1_single_agent_l4":
        policy = BaselineV1SingleAgent(args, level = 4)
        wrapper_func = Simple115StateWrapper_ball_owned_player
    elif args.algo == "baseline_v1_single_agent_l5":
        policy = BaselineV1SingleAgent(args, level = 5)
        wrapper_func = Simple115StateWrapper_ball_owned_player
    elif args.algo == "rule_based_2":
        policy = rule_based_agent_2
    elif args.algo == "tizero_agent":
        policy = my_controller
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

    
    while True:
        obs_before_modified_by_acs = copy.deepcopy(obs)
        if "baseline" in args.algo:
            action, json_output = policy.get_action(obs[0], wrapper_obs[0], steps)
        elif "rule_based" in args.algo:
            action = policy(obs[0])
        elif args.algo ==  "tizero_agent":
            action = policy(obs[0])
        elif args.algo ==  "tizero_agent_multi":
            # obs_list = [obs[0] for _ in range(10)]
            action = policy(obs)
        else:
            raise NotImplementedError
        if type(action) == list:
            action = action[0]
        next_obs, rew, done, info = env.step(action)

            
        obs = next_obs
        if wrapper or args.offline_dataset_collection:
            wrapper_obs = wrapper.observation(copy.deepcopy(obs))
            text_obs = observation_to_text_human(copy.deepcopy(wrapper_obs[0]), copy.deepcopy(obs[0]), steps, block_mode=args.block_mode)
        steps += 1
        if steps==args.num_timesteps or done:
            break
    # print("Steps: %d Reward: %.2f" % (steps, rew))
    score = next_obs[0]['score']
    env.close()
    return score
    

if __name__ == "__main__":
    
    
    
    # algos = ["baseline_v1_single_agent_l0", "baseline_v1_single_agent_l1", "baseline_v1_single_agent_l2", "baseline_v1_single_agent_l3", "baseline_v1_single_agent_l4", "baseline_v1_single_agent_l5","rule_based_2"]
    
    algos = ['baseline_v1_single_agent_l2']
    maps = ["11_vs_11_level_4","11_vs_11_level_2"]
    for map in maps:
        for algo in algos:
            print("Current algo: ", algo)
            result = []
            for i in range(10):
                print("Current iteration: ", i)
                test = run_env(algo, map)
                if test[0] > test[1]:
                    result.append(1)
                elif test[0] < test[1]:
                    result.append(-1)
                else:
                    result.append(0)
            
        
            win_count = result.count(1)
            lose_count = result.count(-1)
            draw_count = result.count(0)


            total_tests = len(result)


            win_probability = win_count / total_tests
            lose_probability = lose_count / total_tests
            draw_probability = draw_count / total_tests

            print("win:", win_probability)
            print("lose", lose_probability)
            print("draw:", draw_probability)
            
            results_str = f"{algo}, win: {win_probability}\lose: {lose_probability}\draw: {draw_probability} in {total_tests} tests, map: {map}"
            time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_path = 'football_llm/llm/' + algo + '_low_level_test' + time +'.txt'
            with open(file_path, 'w') as file:
                file.write(results_str)

        
            
        

