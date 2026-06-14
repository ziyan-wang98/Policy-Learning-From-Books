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
from tqdm import tqdm

from gfootball.env.wrappers import Simple115StateWrapper_ball_owned_player
import datetime

from concurrent.futures import ProcessPoolExecutor
import copy

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
    result_dir = './result'
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


def run_env(env_name):
    win, lose, draw = 0, 0, 0
    env = football_env.create_environment(env_name=env_name,representation='raw', \
                                stacked=False, logdir=args.logdir, write_goal_dumps=args.write_goal_dumps, \
                                write_full_episode_dumps=args.write_full_episode_dumps, render=args.render,\
                                number_of_left_players_agent_controls=args.num_players)
    
    steps = 0
    obs = env.reset()
    json_outputs_list = []
    rand = 0
    while True:
        obs_before_modified_by_acs = copy.deepcopy(obs)
        if "baseline" in args.algo:
            wrapper_obs = wrapper.observation(obs)
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
        
        if done:
            print("Done")
        
        score = next_obs[0]['score']
        obs = next_obs
        steps += 1
        if steps==args.num_timesteps or done:
            if score[0] > score[1]:
                win += 1
                res  = 1
                print("Win")
            elif score[0] < score[1]:
                lose += 1
                res = -1
                print("Lose")
            else:
                draw += 1
                res = 0
                print("Draw")
            break
    env.close()   
    return env_name, res, win, lose, draw

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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
    elif args.algo == "rule_based_2":
        policy = rule_based_agent_2
    elif args.algo == "tizero_agent":
        policy = my_controller
    else:
        raise NotImplementedError
    
    args.eval_type = "11v11"
    
    
    if args.eval_type == "11v1":
        args.env_number = 100
        
        base_env = "11_vs_1_level_1"
        
        args.eval_envs = [base_env+"_mid", base_env+"_right", base_env+"_left"]
    elif args.eval_type == "11v11":
        args.env_number = 10
        args.eval_envs = ["11_vs_11_level_0", "11_vs_11_level_1", "11_vs_11_level_2"]
        
    total_win, total_lose, total_draw = 0, 0, 0
    res_dict = {}
    with ProcessPoolExecutor(max_workers=10) as executor:
        for chunk in chunks(args.eval_envs, 10):
            for env_name, res, win, lose, draw in executor.map(run_env, chunk):
                if env_name not in res_dict:
                    res_dict[env_name] = []
                res_dict[env_name].append(res)
                total_win += win
                total_lose += lose
                total_draw += draw

    for env_name, res_list in res_dict.items():
        print(f"Rewards for {env_name}: {res_list}")
        print(f"Mean reward for {env_name}: {np.mean(res_list)}")

    all_rewards = [rew for res_list in res_dict.values() for rew in res_list]
    print("All rewards: ", all_rewards)
    print("Overall mean: ", np.mean(all_rewards))
    
    print("win rate", total_win/(total_win+total_lose+ total_draw))
    print("lose rate", total_lose/(total_win+total_lose+total_draw))
    print("draw rate", total_draw/(total_win+total_lose+total_draw))