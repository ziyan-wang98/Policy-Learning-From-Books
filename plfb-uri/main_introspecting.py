import os
import sys
from pathlib import Path

URI_ROOT = Path(os.environ.get("PLFB_URI_ROOT", Path(__file__).resolve().parent)).resolve()
FOOTBALL_LLM_ROOT = Path(os.environ.get("PLFB_ROOT", URI_ROOT.parent / "football_llm")).resolve()
TIZERO_AGENT_PATH = Path(
    os.environ.get("PLFB_TIZERO_AGENT_PATH", FOOTBALL_LLM_ROOT / "setup" / "TiZero" / "submission" / "tizero_agent")
)
for path in (TIZERO_AGENT_PATH, FOOTBALL_LLM_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import tqdm
import numpy as np
import os.path as osp
import argparse
import gym
import numpy as np

import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.datasets import get_atari
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig, DiscreteCQLSACConfig, DiscreteCQLSAC
from d3rlpy.algos import DiscreteBC, DiscreteBCConfig, EnsembleBC, EnsembleBCConfig
from d3rlpy.preprocessing import MinMaxActionScaler, MinMaxObservationScaler
from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset.replay_buffer import Signature
from d3rlpy.datasets import ReplayBuffer
from d3rlpy.metrics import EnvironmentEvaluator, EnsembleDatasetErrorEvaluator, TTTEnvironmentEvaluator
# import ActionSpace
from d3rlpy.constants import ActionSpace

from introspecting.uncertainty_predictor import UncertaintyPredictor
from introspecting.data_loader import get_img_dataset
from const import *
from funcs import *

DATA_ROOT = Path(os.environ.get("PLFB_DATASET_PATH", FOOTBALL_LLM_ROOT / "data")).resolve()
TTT_DATA_ROOT = Path(os.environ.get("PLFB_TTT_DATA_PATH", DATA_ROOT / "tic_tac_toe_data")).resolve()
IRL_LOG_ROOT = Path(os.environ.get("PLFB_IRL_LOG_ROOT", FOOTBALL_LLM_ROOT / "IRL_LOG")).resolve()
MODEL_LOG_ROOT = Path(os.environ.get("PLFB_MODEL_LOG_ROOT", FOOTBALL_LLM_ROOT / "IRL_LOG_MODEL")).resolve()
DEBUG_LOG_ROOT = Path(os.environ.get("PLFB_DEBUG_LOG_ROOT", FOOTBALL_LLM_ROOT / "ORL_LOG_DEBUG")).resolve()

# Algorithm parameters can still be tuned further.
# Target values in the 0.5-1.0 range produced similar results.
# Larger batches performed slightly worse but improved numerical stability.
# Coefficient-style parameters were not extensively tuned.
# The generated data scale may need to be roughly tripled relative to offline data.
def get_policy_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--target_value", type=float, default=0.9)
    parser.add_argument("--offline_algo", type=str, default="cql")
    parser.add_argument("--ent_target_coef", type=float, default=0.8) # Values below 0.4 become too deterministic and unstable; historical default was 0.98.
    parser.add_argument("--coef_r", type=float, default=0.05)
    parser.add_argument("--coef_t", type=float, default=0.05)
    parser.add_argument("--strategy", type=str, default='all_replaced')
    parser.add_argument("--lr_rescale", type=float, default=0.5)
    parser.add_argument("--keep_rate", type=float, default=1.0)
    parser.add_argument("--alg_type", type=str, default=AlgType.CIQL)
    parser.add_argument("--neg_keep_rate", type=float, default=1.0)
    parser.add_argument("--just_evaluation", default=False, action='store_true')
    parser.add_argument("--batch_size", type=int, default=256)
    # experiment config
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--device", type=str, default='cuda:2')
    parser.add_argument("--comment", type=str, default='none')
    parser.add_argument("--model_tag", type=str, default='with-mmdata-v2')
    parser.add_argument("--seed", type=int, default=0)
    # data_args
    parser.add_argument("--obs_stack_num", type=int, default=1)
    parser.add_argument("--env_name", type=str, default='tictactoe-v0')
    parser.add_argument("--fake_rollout_num", type=int, default=0)
    parser.add_argument("--data_version", type=str, default='v0')
    return parser.parse_args()



if __name__=="__main__":
    args = get_policy_args()
    root_path = str(TTT_DATA_ROOT)
    # root_path = os.path.join(root_path, args.env_name)
    model_root_path = str(MODEL_LOG_ROOT / args.env_name)
    check_param_dict = {
        'comment': args.comment, 'alpha': args.alpha, 
        'keep_rate': args.keep_rate, 'neg_keep_rate': args.neg_keep_rate, 
        'coef_r': args.coef_r, 'coef_t': args.coef_t, 'target_value': args.target_value}
    # mkdir
    os.makedirs(root_path, exist_ok=True)
    os.makedirs(model_root_path, exist_ok=True)
    # 
    model_param_dict = {'model_tag': args.model_tag, 'obs_stack_num': args.obs_stack_num}
    img_merge_saved_path = osp.join(root_path, 'merged_data', args.data_version + '&'.join(f"{k}={v}" for k, v in model_param_dict.items()) + '.npz')
    if args.alg_type==AlgType.CQL or args.coef_r == 0 and args.coef_t == 0:
        skip_uncertainty = True
    else:
        skip_uncertainty = False
    if args.env_name == 'tictactoe-v0':
        from envs.tictactoe import TicTacToeEnv
        env = TicTacToeEnv()
        gamma = 0.9
        n_steps = 8_000
        hidden_units = [512, 512, 256, 128, 128]
    else:
        env = gym.make(args.env_name)
        gamma = 0.99
        n_steps = 250_000
        hidden_units = [512, 512, 256, 128, 128]
    uncertainty_predictor = UncertaintyPredictor(env=env, model_root_path=model_root_path, 
                                                 model_param_dict=model_param_dict, skip_uncertainty=skip_uncertainty, 
                                                 coef_r=args.coef_r, coef_t=args.coef_t, device=args.device)
    if args.debug:
        log_root = str(DEBUG_LOG_ROOT / args.env_name)
    else:
        log_root = str(IRL_LOG_ROOT / args.env_name)
    logger_adapter = d3rlpy.logging.CombineAdapterFactory([
        d3rlpy.logging.FileAdapterFactory(root_dir=os.path.join(log_root, "d3rlpy_logs")),
        d3rlpy.logging.TensorboardAdapterFactory(root_dir=os.path.join(log_root, "tensorboard_logs"))])
    img_dataset = get_img_dataset(img_merge_saved_path, env, uncertainty_predictor, args)
    obs_list, action_list, next_obs_list, reward_list, end_epi_list, done_list = img_dataset.load_raw_data(name_filter='seed-')

    
    if not skip_uncertainty:
        if not uncertainty_predictor.load_model(): 
             uncertainty_predictor.train_model(obs_list, action_list, next_obs_list, reward_list, 
                                               end_epi_list, done_list)

    img_dataset.construction(obs_list, action_list, reward_list, end_epi_list, done_list)

    dataset = img_dataset.dataset
    samples = dataset.sample_transition_batch(dataset.transition_count)
    obs_samples = samples.observations
    rew_samples = samples.rewards

    ret_list = np.array(img_dataset.data_info['return_list'])
    original_epi_num = img_dataset.data_info['original_epi_num']
    print("reward mean", rew_samples.mean(), "std", rew_samples.std(), "max", rew_samples.max(), "min", rew_samples.min())
    print("ret mean", ret_list.mean(), "std", ret_list.std(), "max", ret_list.max(), "min", ret_list.min())
    print("episodes", len(dataset.episodes), "original episodes", original_epi_num)
    min_obs_val = obs_samples.min(axis=0)
    max_obs_val = obs_samples.max(axis=0)
    max_obs_val[(max_obs_val - min_obs_val) < 1] = min_obs_val[(max_obs_val - min_obs_val) < 1] + 1
    obs_scaler = MinMaxObservationScaler(minimum=min_obs_val, maximum=max_obs_val)
    
    encoder_factory = VectorEncoderFactory(hidden_units=hidden_units, activation='gelu', use_layer_norm=True, dropout_rate=0.0)
    # start training
    if args.alpha > 0:
        alpha_learning_rate = -1
    else:
        alpha_learning_rate = 1e-3
    args.eval_algo = args.alg_type
    args.num_players = 1

    n_steps_per_epoch = 100


    exp_name = formulate_param_to_name(check_param_dict)
    env_evaluator = TTTEnvironmentEvaluator(env, oppo_players=['random', 'minimax'], n_trials=20)
    if args.offline_algo == 'sac':
        sac = DiscreteCQLSACConfig(n_critics=4, batch_size=256, 
                                actor_encoder_factory=encoder_factory, 
                                critic_encoder_factory=encoder_factory,
                                ent_target_coef=args.ent_target_coef,
                                alpha_learning_rate=alpha_learning_rate,
                                conservative_weight=args.alpha, 
                                target_value=args.target_value, 
                                observation_scaler=obs_scaler, 
                                gamma=gamma,
                                actor_learning_rate=1e-4 * args.lr_rescale, 
                                critic_learning_rate=3e-4 * args.lr_rescale).create(device=args.device)
        sac.fit(dataset, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch, 
                save_interval=1, evaluators={'environment': env_evaluator}, logger_adapter=logger_adapter, experiment_name=exp_name)
    else:
        cql = DiscreteCQLConfig(alpha=args.alpha, n_critics=2, 
                        batch_size=args.batch_size, learning_rate=1e-4, alpha_learning_rate=alpha_learning_rate,
                        encoder_factory=encoder_factory, gamma=gamma, target_value=args.target_value,
                        observation_scaler=obs_scaler).create(device=args.device)
        cql.fit(dataset, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch, 
                save_interval=1, evaluators={'environment': env_evaluator, }, 
                logger_adapter=logger_adapter, experiment_name=exp_name)
    
    env.close()
