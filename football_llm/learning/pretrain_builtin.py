
import sys
sys.path.append("football_llm/TiZero/submission/tizero_agent")
import zipfile
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
from d3rlpy.preprocessing import MinMaxActionScaler, MinMaxObservationScaler

from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset.replay_buffer import Signature
from d3rlpy.datasets import ReplayBuffer
from d3rlpy.metrics.evaluators import DatasetErrorEvaluator
from learning.utils import obs_to_robot_acs, update_stack_obs, npz_extractor

DATA_ROOT = os.environ.get("PLFB_DATASET_PATH", os.path.join(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts"), "football"))
OFFLINE_DATA_ROOT = os.environ.get("PLFB_OFFLINE_DATASET_PATH", os.path.join(DATA_ROOT, "offline_dataset-v4"))


def get_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=int, default=3)
    parser.add_argument("--comment", type=str, default='none')
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--policy_type", type=str, default='stochastic_weighted_multi_head')
    parser.add_argument("--regen_data", default=False, action='store_true')
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


if __name__ == '__main__':
    args_parser = baseline_agent_parser.parse_args(return_parser=True)
    args = get_args(args_parser)
    device = args.device
    root_path = OFFLINE_DATA_ROOT
    res = np.load(osp.join(root_path, 'tizero_agent_11_vs_11_hard_stochastic_0.npz'), allow_pickle=True)

    check_param_dict = {
        'comment': args.comment,
        'stack_hist': 10,
        'stack_obs': 3,
        "batch_size": args.batch_size,
        'pl': args.player,
        'policy_type': args.policy_type
    }
    if args.debug:
        log_root = 'football_llm/ORL_LOG_DEBUG'
    else:
        log_root = 'football_llm/ORL_LOG_BC-v4'
    exp_name = '&'.join([f'{k}={v}'.replace(" ", "--") for k, v in check_param_dict.items()])
    saved_data_dir = os.path.join(root_path, "data_buffer-bc", '&'.join(f"{k}={check_param_dict[k]}" for k in ['pl', 'stack_hist', 'stack_obs']))
    os.makedirs(saved_data_dir, exist_ok=True)
    # find all file in root_path
    logger_adapter = d3rlpy.logging.CombineAdapterFactory([
    d3rlpy.logging.FileAdapterFactory(root_dir=os.path.join(log_root, "d3rlpy_logs")),
    d3rlpy.logging.TensorboardAdapterFactory(root_dir=os.path.join(log_root, "tensorboard_logs")),
    ])
    obs_dict, obs_vector = imaginary_data_observation(res['wrapper_obs'][1, 0], res['obs_before_modified_by_acs'][1, 0], 1, ret_type='both', TODO_missing=True)
    prev_obs_dict, prev_obs_vector = imaginary_data_observation(res['wrapper_obs'][0, 0], res['obs_before_modified_by_acs'][0, 0], 0, ret_type='both', TODO_missing=True)
    acs = obs_to_robot_acs(obs_dict, prev_obs_dict, check_param_dict['pl'])
    observation_signature = Signature(
        dtype=[np.float32],
        shape=[obs_vector.shape[-1] * check_param_dict['stack_obs'] + acs.shape[-1] * check_param_dict['stack_hist']],  # type: ignore
    )
    action_signature = Signature(
        dtype=[np.float32],
        shape=[acs.shape[-1]],  # type: ignore
    )
    reward_signature = Signature(
        dtype=[np.dtype(np.float32)],
        shape=[[1]],
    )
    train_buffer = InfiniteBuffer()
    train_dataset = ReplayBuffer(
        train_buffer,
        observation_signature=observation_signature,
        action_signature=action_signature,
        reward_signature=reward_signature,
        action_space=ActionSpace.CONTINUOUS,
        action_size=acs.shape[-1],
        cache_size=2000,
    )
    test_buffer = InfiniteBuffer()
    test_dataset = ReplayBuffer(test_buffer,
        observation_signature=observation_signature,
        action_signature=action_signature,
        reward_signature=reward_signature,
        action_space=ActionSpace.CONTINUOUS,
        action_size=acs.shape[-1],
        cache_size=2000,
    )
    train_ratio = 0.8
    if not args.regen_data and os.path.exists(os.path.join(saved_data_dir, 'train.pkl')):
        with open(os.path.join(saved_data_dir, 'train.pkl'), 'rb') as f:
            train_dataset.load(f, buffer=train_buffer)
        with open(os.path.join(saved_data_dir, 'test.pkl'), 'rb') as f:
            test_dataset.load(f, buffer=test_buffer)
    else:
        for file in tqdm.tqdm(os.listdir(root_path), desc='loading data'):
            if file.endswith('.npz'):
                res = npz_extractor(osp.join(root_path, file))
                if res is None:
                    continue
                steps = res['steps']
                obs = res['obs']
                wrapper_obs = res['wrapper_obs']
                texted_obs = res['texted_obs']
                reward = res['reward']
                dense_rewards = res['dense_rewards']
                done = res['done']
                action = res['action']
                obs_before_modified_by_acs = res['obs_before_modified_by_acs']
                stack_obs = np.zeros(observation_signature.shape)
                acs = np.zeros(action_signature.shape)
                if  np.random.rand() <= train_ratio:
                    selected_dataset = train_dataset
                else:
                    selected_dataset = test_dataset
                try:
                    for i in range(len(steps) - 1):
                        obs_dict, obs_vector = imaginary_data_observation(wrapper_obs[i, 0], obs_before_modified_by_acs[i, 0], steps[i], ret_type='both', TODO_missing=True)
                        stack_obs = update_stack_obs(obs_vector, acs, stack_obs, state_stack_num=check_param_dict['stack_obs'])
                        next_obs_dict, next_obs_vector = imaginary_data_observation(wrapper_obs[i+1, 0], obs_before_modified_by_acs[i+1, 0], steps[i+1], ret_type='both', TODO_missing=True)
                        acs = obs_to_robot_acs(next_obs_dict, obs_dict, check_param_dict['pl'])
                        selected_dataset.append(stack_obs, acs, dense_rewards[i] + reward[i] * 5)
                        prev_obs_dict = obs_dict
                except AttributeError as e:
                    print("error", e)
                    print("skip this file", file)
                    continue
                selected_dataset.clip_episode(True)
                # break
        train_dataset.dump(os.path.join(saved_data_dir, 'train.pkl'))
        test_dataset.dump(os.path.join(saved_data_dir, 'test.pkl'))

    encoder_factory = VectorEncoderFactory(hidden_units=[512, 256, 256, 128], activation='relu')
    # rescale and keep zero as the default action for weighting computation in the downstream
    # zone is rescaled to [-1, 1] cuz the differences larger than 1 is not common.
    action_scaler = MinMaxActionScaler(minimum=[-1, -1, -1, -7], maximum=[1, 1, 1, 7])
    obs_samples = train_dataset.sample_transition_batch(102400).observations
    min_obs_val = obs_samples.min(axis=0)
    max_obs_val = obs_samples.max(axis=0)
    max_obs_val[(max_obs_val - min_obs_val) < 1] = min_obs_val[(max_obs_val - min_obs_val) < 1] + 1
    obs_scaler = MinMaxObservationScaler(minimum=min_obs_val, maximum=max_obs_val)
    bc = BCConfig(batch_size=args.batch_size, learning_rate=1e-5, policy_type=args.policy_type,
                  encoder_factory=encoder_factory, observation_scaler=obs_scaler, action_scaler=action_scaler).create(device=device)
    # start training

    bc.fit(train_dataset, n_steps=int(1e6), n_steps_per_epoch=1000, save_interval=10,
           evaluators={'test_error': DatasetErrorEvaluator(test_dataset)},
           logger_adapter=logger_adapter, experiment_name=exp_name)
    # Tuning notes:
    # The conservative value reflects one-step performance change from behavior policy to target policy; it should relate to the one-step reward gap and stay below the maximum one-step reward.
