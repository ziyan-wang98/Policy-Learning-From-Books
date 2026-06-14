import numpy as np
from llm.utils.obs2text import directions
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()
GFOOTBALL_ROOT = Path(os.environ.get("PLFB_GFOOTBALL_ROOT", PROJECT_ROOT / "setup" / "football")).resolve()
GFOOTBALL_DATA_DIR = Path(
    os.environ.get("PLFB_GFOOTBALL_DATA_DIR", GFOOTBALL_ROOT / "gfootball_engine" / "data")
).resolve()
EVAL_OUTPUT_DIR = Path(os.environ.get("PLFB_EVAL_OUTPUT_DIR", PROJECT_ROOT / "learning" / "eval_result")).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from funcs import npz_extractor

import gfootball.env as football_env

def _resolve_gfootball_data_dir():
    if GFOOTBALL_DATA_DIR.exists():
        return GFOOTBALL_DATA_DIR
    fallback = GFOOTBALL_ROOT / "third_party" / "gfootball_engine" / "data"
    if fallback.exists():
        return fallback.resolve()
    return GFOOTBALL_DATA_DIR

def ensure_gfootball_working_dir():
    if GFOOTBALL_ROOT.exists():
        os.chdir(GFOOTBALL_ROOT)
    data_dir = _resolve_gfootball_data_dir()
    current_data_dir = os.environ.get("GFOOTBALL_DATA_DIR")
    if data_dir.exists() and (not current_data_dir or not Path(current_data_dir).exists()):
        os.environ["GFOOTBALL_DATA_DIR"] = str(data_dir)
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

def obs_to_robot_acs(next_obs_dict, obs_dict, player_id):
    next_zone =  np.array(next_obs_dict[f"player_{player_id}"]['zone'])
    zone = np.array(obs_dict[f"player_{player_id}"]['zone'])
    zone_diff = next_zone - zone
    next_own_the_ball = int(next_obs_dict['ball_ownership_player'] == player_id)
    own_the_ball = int(obs_dict['ball_ownership_player'] == player_id)
    own_the_ball_state_diff = next_own_the_ball - own_the_ball
    next_ball_direction = directions.index(next_obs_dict['ball_direction'])
    ball_direction = directions.index(obs_dict['ball_direction'])
    ball_direction_diff = ((next_ball_direction - ball_direction) + len(directions)) % len(directions)
    assert (ball_direction + ball_direction_diff) % len(directions) == next_ball_direction, "reconstruct the ball direction failed"
    return np.concatenate([zone_diff, [own_the_ball_state_diff], [ball_direction_diff]])

def update_stack_obs(obs_vector, acs, stack_obs, state_stack_num):
    stack_obs = stack_obs.copy()
    stack_obs[:obs_vector.shape[0]*(state_stack_num-1)] = stack_obs[obs_vector.shape[0]:obs_vector.shape[0]*(state_stack_num)]
    stack_obs[obs_vector.shape[0]*(state_stack_num-1):obs_vector.shape[0]*(state_stack_num)] = obs_vector
    hist_acs = stack_obs[obs_vector.shape[0]*state_stack_num:]
    hist_acs[:acs.shape[0]] = hist_acs[-acs.shape[0]:]
    hist_acs[-acs.shape[0]:] = acs
    stack_obs[obs_vector.shape[0]*state_stack_num:] = hist_acs
    return stack_obs

def update_stack_img_obs(obs_vector, stack_obs, obs_stack_num):
    stack_obs = stack_obs.copy()
    stack_obs[:obs_vector.shape[0]*(obs_stack_num-1)] = stack_obs[obs_vector.shape[0]:obs_vector.shape[0]*(obs_stack_num)]
    stack_obs[obs_vector.shape[0]*(obs_stack_num-1):obs_vector.shape[0]*(obs_stack_num)] = obs_vector
    return stack_obs


def env_creator(env_name, args):
    # Create the base result directory
    result_dir = str(Path(getattr(args, "eval_output_dir", str(EVAL_OUTPUT_DIR))).resolve())
    os.makedirs(result_dir, exist_ok=True)

    # Create a unique subdirectory
    index = 0
    unique_dir_name = f"{args.eval_algo}_{env_name}_{index}"
    unique_dir_path = os.path.join(result_dir, unique_dir_name)
    while os.path.exists(unique_dir_path):
        index += 1
        unique_dir_name = f"{args.eval_algo}_{env_name}_{index}"
        unique_dir_path = os.path.join(result_dir, unique_dir_name)
    
    os.makedirs(unique_dir_path, exist_ok=True)

    # Create subdirectories for videos and JSON files
    video_dir = os.path.join(unique_dir_path, 'dump_file')
    os.makedirs(video_dir, exist_ok=True)

    args.write_full_episode_dumps = True
    args.render = False
    args.write_goal_dumps = False
    args.num_players = 1
    
    ensure_gfootball_working_dir()
    eval_env = football_env.create_environment(env_name=env_name,representation='raw', \
                                                stacked=False, logdir=video_dir, write_goal_dumps=args.write_goal_dumps, \
                                                write_full_episode_dumps=args.write_full_episode_dumps, render=args.render,\
                                                write_video=False, \
                                                number_of_left_players_agent_controls=args.num_players)
    eval_env.env_name = env_name
    return eval_env

def env_list_creator(env_name_list, args):
    return [env_creator(env_name, args) for env_name in env_name_list]
