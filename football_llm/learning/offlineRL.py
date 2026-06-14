
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
from d3rlpy.algos import DiscreteBC, DiscreteBCConfig
from d3rlpy.metrics import EnvironmentEvaluator
# import ActionSpace
from d3rlpy.constants import ActionSpace
from gym.spaces import Box

from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset.replay_buffer import Signature
from d3rlpy.datasets import ReplayBuffer

from d3rlpy.preprocessing import MinMaxActionScaler, MinMaxObservationScaler

# from d3rlpy.metrics.scorer import evaluate_on_environment
# from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from sklearn.model_selection import train_test_split

import gfootball.env as football_env
from learning.utils import npz_extractor

DATA_ROOT = os.environ.get("PLFB_DATASET_PATH", os.path.join(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts"), "football"))
OFFLINE_DATA_ROOT = os.environ.get("PLFB_OFFLINE_DATASET_PATH", os.path.join(DATA_ROOT, "offline_dataset-v4"))

def get_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--regen_data", default=False, action='store_true')
    parser.add_argument("--data_num", type=int, default=400)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()



if __name__ == '__main__':
    root_path = OFFLINE_DATA_ROOT
    res = np.load(osp.join(root_path, 'rule_based_2_11_vs_11_hard_stochastic_0.npz'), allow_pickle=True)

    args_parser = baseline_agent_parser.parse_args(return_parser=True)
    args = get_args(args_parser)
    check_param_dict = {
        'comment': 'none',
        'data_num': args.data_num,
        'alpha': 1.0,
    }
    if args.debug:
        log_root = 'football_llm/ORL_LOG_DEBUG'
    else:
        log_root = 'football_llm/ORL_LOG'
    exp_name = '&'.join([f'{k}={check_param_dict[k]}'.replace(" ", "--") for k in ['data_num']])
    saved_data_dir = os.path.join(root_path, f"data_buffer-orl-{exp_name}")
    os.makedirs(saved_data_dir, exist_ok=True)
    # find all file in root_path
    logger_adapter = d3rlpy.logging.CombineAdapterFactory([
    d3rlpy.logging.FileAdapterFactory(root_dir=os.path.join(log_root, "d3rlpy_logs")),
    d3rlpy.logging.TensorboardAdapterFactory(root_dir=os.path.join(log_root, "tensorboard_logs")),
    ])
    video_dir = os.path.join(log_root, "video_logs", exp_name)
    device = args.device
    env = football_env.create_environment(env_name=args.environment,representation='raw', \
                                    stacked=False, logdir=video_dir, write_goal_dumps=args.write_goal_dumps, \
                                    write_full_episode_dumps=args.write_full_episode_dumps, render=args.render,\
                                    number_of_left_players_agent_controls=args.num_players)

    env_name_list = [f'11_vs_11_level_{i}' for i in [1,3,4, 5]]
    def env_creator(env_name):
        eval_env = football_env.create_environment(env_name=env_name,representation='raw', \
                                    stacked=False, logdir=video_dir, write_goal_dumps=args.write_goal_dumps, \
                                    write_full_episode_dumps=args.write_full_episode_dumps, render=args.render,\
                                    number_of_left_players_agent_controls=args.num_players)
        eval_env.env_name = env_name
        return eval_env
    eval_env_list = [env_creator(env_name) for env_name in env_name_list]

    env_evaluator = EnvironmentEvaluator(eval_env_list)

    observation_signature = Signature(
        dtype=[np.float32],
        shape=[res['wrapper_obs'].shape[-1]],  # type: ignore
    )
    action_signature = Signature(
        dtype=[np.int32],
        shape=[1],  # type: ignore
    )
    reward_signature = Signature(
        dtype=[np.dtype(np.float32)],
        shape=[[1]],
    )
    buffer = InfiniteBuffer()
    dataset = ReplayBuffer(
        buffer,
        env=env,
        observation_signature=observation_signature,
        action_signature=action_signature,
        reward_signature=reward_signature,
        cache_size=2000,
    )
    accepted_data_num = 0
    if not args.regen_data and os.path.exists(os.path.join(saved_data_dir, 'offline_data.pkl')):
        with open(os.path.join(saved_data_dir, 'offline_data.pkl'), 'rb') as f:
            dataset.load(f, buffer=buffer)
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
                action = res['action']
                reward = res['reward']

                dense_rewards = res['dense_rewards']
                done = res['done']
                if 'obs_before_modified_by_acs' not in res.keys():
                    print("skip this file", file)
                    continue
                obs_before_modified_by_acs = res['obs_before_modified_by_acs']
                try:
                    for i in range(len(steps)):
                        dataset.append(wrapper_obs[i, 0], action[i], dense_rewards[i] + reward[i] * 5)
                    dataset.clip_episode(True)
                    accepted_data_num += 1
                except AttributeError as e:
                    print("error", e)
                    print("skip this file", file)
                    continue
                if accepted_data_num > check_param_dict['data_num']:
                    break
                # break
        dataset.dump(os.path.join(saved_data_dir, 'offline_data.pkl'))

    # dataset.save_snapshot('dataset.pkl')
    encoder_factory = VectorEncoderFactory(hidden_units=[256, 256, 256, 256, 256], activation='relu')
    # obs_samples = dataset.sample_transition_batch(102400).observations
    # min_obs_val = obs_samples.min(axis=0)
    # max_obs_val = obs_samples.max(axis=0)
    # max_obs_val[(max_obs_val - min_obs_val) < 1] = min_obs_val[(max_obs_val - min_obs_val) < 1] + 1
    # obs_scaler = MinMaxObservationScaler(minimum=min_obs_val, maximum=max_obs_val)
    # TODO: reward scale
    # select_eps = np.random.randint(0, len(dataset.episodes), size=check_param_dict['data_num'])
    # dataset._buffer._episodes = [dataset._buffer._episodes[eps_id] for eps_id in select_eps]
    cql = DiscreteCQLConfig(alpha=check_param_dict['alpha'], n_critics=2, batch_size=256,
                            encoder_factory=encoder_factory).create(device=device)
    # start training
    if args.debug:
        n_steps_per_epoch = 100
    else:
        n_steps_per_epoch = 20000
    cql.fit(dataset, n_steps=int(1e7), n_steps_per_epoch=n_steps_per_epoch, save_interval=1000, evaluators={
                'environment': env_evaluator, }, logger_adapter=logger_adapter, experiment_name=exp_name)
    # Tuning notes:
    # The conservative value reflects one-step performance change from behavior policy to target policy; it should relate to the one-step reward gap and stay below the maximum one-step reward.
