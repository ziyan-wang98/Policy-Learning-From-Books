
import sys
sys.path.append("football_llm/TiZero/submission/tizero_agent")

sys.path.append('football_llm')
import tqdm
import numpy as np
import os.path as osp
import os
import argparse
import random
import llm.config.baseline_agent_parser as baseline_agent_parser
from llm.utils.obs2text import imaginary_data_observation
import numpy as np
import torch
import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.datasets import get_atari
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
from d3rlpy.algos import DiscreteBC, DiscreteBCConfig, BCConfig

from d3rlpy.metrics import EnvironmentEvaluator
# import ActionSpace
from d3rlpy.constants import ActionSpace
from gym.spaces import Box

from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset.replay_buffer import Signature
from d3rlpy.datasets import ReplayBuffer
from d3rlpy.metrics.evaluators import DatasetErrorEvaluator

from learning.utils import obs_to_robot_acs, update_stack_obs

DATA_ROOT = os.environ.get("PLFB_DATASET_PATH", os.path.join(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts"), "football"))
OFFLINE_DATA_ROOT = os.environ.get("PLFB_OFFLINE_DATASET_PATH", os.path.join(DATA_ROOT, "offline_dataset"))
BC_MODEL_ROOT = os.environ.get("PLFB_BC_MODEL_ROOT", os.path.join(os.environ.get("PLFB_WORK_ROOT", "plfb_work"), "ORL_LOG_BC", "d3rlpy_logs"))


if __name__ == '__main__':
    # load
    root_path = BC_MODEL_ROOT
    exp_path = os.path.join(root_path, 'comment=stack-10-obs-3&stack_hist=10&stack_obs=3&pl=1_20240120181538')
    path = model_path =  os.path.join(exp_path, 'model_160000.d3')
    player_id = path.split('pl=')[1].split('_')[0].split('&')[0]
    stack_hist = path.split('stack_hist=')[1].split('&')[0]
    stack_obs_num = path.split('stack_obs=')[1].split('&')[0]
    bc = d3rlpy.load_learnable(path)
    # call
    # bc.predict(batch.observations)

    check_param_dict = {
        'pl': int(player_id),
        'stack_hist': int(stack_hist),
        'stack_obs': int(stack_obs_num),
    }
    # test
    root_path = OFFLINE_DATA_ROOT
    res = np.load(osp.join(root_path, 'rule_based_2_11_vs_11_easy_stochastic_100.npz'), allow_pickle=True)
    obs_dict, obs_vector = imaginary_data_observation(res['wrapper_obs'][1, 0], res['obs_before_modified_by_acs'][1, 0], 1, ret_type='both')
    prev_obs_dict, prev_obs_vector = imaginary_data_observation(res['wrapper_obs'][0, 0], res['obs_before_modified_by_acs'][0, 0], 0, ret_type='both')
    acs = obs_to_robot_acs(obs_dict, prev_obs_dict, player_id)
    obs_len = obs_vector.shape[-1] * check_param_dict['stack_obs'] + acs.shape[-1] * check_param_dict['stack_hist']


    steps = res['steps']
    obs = res['obs']
    wrapper_obs = res['wrapper_obs']
    texted_obs = res['texted_obs']
    action = res['action']
    reward = res['reward']
    dense_rewards = res['dense_rewards']
    done = res['done']
    obs_before_modified_by_acs = res['obs_before_modified_by_acs']
    stack_obs = np.zeros(obs_len)
    acs = np.zeros(acs.shape)
    error = []
    for i in range(len(steps) - 1):
        obs_dict, obs_vector = imaginary_data_observation(wrapper_obs[i, 0], obs_before_modified_by_acs[i, 0], steps[i], ret_type='both')
        stack_obs = update_stack_obs(obs_vector, acs, stack_obs, state_stack_num=check_param_dict['stack_obs'])
        next_obs_dict, next_obs_vector = imaginary_data_observation(wrapper_obs[i+1, 0], obs_before_modified_by_acs[i+1, 0], steps[i+1], ret_type='both')
        acs = obs_to_robot_acs(next_obs_dict, obs_dict, check_param_dict['pl'])
        acs_pred = bc.predict(np.array([stack_obs]))
        acs_sample = bc.sample_action(np.array([stack_obs]))
        if np.mean(np.abs(acs)) != 0:
            print(np.round(acs_pred), acs, np.round(acs_sample))
        error.append(np.mean(np.abs(acs_pred - acs)))
        prev_obs_dict = obs_dict
    print(np.mean(error))
