import numpy as np
from llm.utils.obs2text import directions
import zipfile
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

def npz_extractor(npz_file_path):
    file = npz_file_path
    try:
        res = np.load(file, allow_pickle=True)
        if 'obs_before_modified_by_acs' not in res.keys() or 'action' not in res.keys():
            print("skip this file", file)
            return None     
    except zipfile.BadZipFile as e:
        print("BadZipFile, skip this file", file)
        print("error", e)
        return None 
    return res