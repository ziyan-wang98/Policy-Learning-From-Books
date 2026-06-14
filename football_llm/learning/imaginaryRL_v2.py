import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()
TIZERO_AGENT_PATH = Path(
    os.environ.get("PLFB_TIZERO_AGENT_PATH", PROJECT_ROOT / "setup" / "TiZero" / "submission" / "tizero_agent")
)
for path in (TIZERO_AGENT_PATH, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import tqdm
import numpy as np
import os.path as osp
import argparse
import gym
import numpy as np
import torch

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

from learning.utils import update_stack_img_obs, ensure_gfootball_working_dir
from learning.uncertainty_predictor import UncertaintyPredictor
from learning.data_loader import get_img_dataset
from learning.utils import env_list_creator
from CONFIG import *
from funcs import *

DATA_ROOT = Path(os.environ.get("PLFB_DATASET_PATH", PROJECT_ROOT / "data")).resolve()
OFFLINE_DATA_ROOT = Path(os.environ.get("PLFB_OFFLINE_DATASET_PATH", DATA_ROOT)).resolve()
DEFAULT_GEN_MODEL = os.environ.get("PLFB_GEN_MODEL", "gpt-4o-mini")
IMAGINARY_DATA_ROOT = Path(
    os.environ.get("PLFB_IMAGINARY_DATASET_PATH", DATA_ROOT / "imaginary_dataset_0204")
).resolve()
TTT_DATA_ROOT = Path(os.environ.get("PLFB_TTT_DATA_PATH", DATA_ROOT / "tic_tac_toe_data")).resolve()
IRL_LOG_ROOT = Path(os.environ.get("PLFB_IRL_LOG_ROOT", PROJECT_ROOT / "IRL_LOG")).resolve()
MODEL_LOG_ROOT = Path(os.environ.get("PLFB_MODEL_LOG_ROOT", PROJECT_ROOT / "IRL_LOG_MODEL")).resolve()
DEBUG_LOG_ROOT = Path(os.environ.get("PLFB_DEBUG_LOG_ROOT", PROJECT_ROOT / "ORL_LOG_DEBUG")).resolve()
MERGED_DATA_CACHE_ROOT = Path(
    os.environ.get("PLFB_MERGED_DATA_CACHE_ROOT", IRL_LOG_ROOT / "_merged_data_cache")
).resolve()


def set_reproducibility_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_manual_seed_all = getattr(getattr(torch, "cuda", None), "manual_seed_all", None)
    cuda_seeded = callable(cuda_manual_seed_all)
    if cuda_seeded:
        cuda_manual_seed_all(seed)

    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False
    cudnn_deterministic = getattr(cudnn, "deterministic", None)
    cudnn_benchmark = getattr(cudnn, "benchmark", None)

    d3rlpy_seed_status = "unavailable"
    d3rlpy_seed = getattr(d3rlpy, "seed", None)
    if callable(d3rlpy_seed):
        try:
            d3rlpy_seed(seed)
            d3rlpy_seed_status = "called"
        except Exception as exc:
            d3rlpy_seed_status = f"failed:{type(exc).__name__}"

    print(
        f"reproducibility seed active: seed={seed}, "
        f"python_random=True, numpy=True, torch=True, "
        f"torch_cuda_manual_seed_all={cuda_seeded}, "
        f"cudnn_deterministic={cudnn_deterministic}, "
        f"cudnn_benchmark={cudnn_benchmark}, "
        f"d3rlpy_seed={d3rlpy_seed_status}"
    )


# Algorithm parameters can still be tuned further.
# Target values in the 0.5-1.0 range produced similar results.
# Larger batches performed slightly worse but improved numerical stability.
# Coefficient-style parameters were not extensively tuned.
# The generated data scale may need to be roughly tripled relative to offline data.
def get_policy_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--target_value", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--offline_algo", type=str, default="cql")
    parser.add_argument("--ent_target_coef", type=float, default=0.8) # Values below 0.4 become too deterministic and unstable; historical default was 0.98.
    parser.add_argument("--coef_r", type=float, default=0.05)
    parser.add_argument("--coef_t", type=float, default=0.05)
    parser.add_argument("--strategy", type=str, default='all_replaced')
    parser.add_argument("--keep_rate", type=float, default=1.0)
    parser.add_argument("--alg_type", type=str, default=AlgType.CIQL)
    parser.add_argument("--neg_keep_rate", type=float, default=1.0)
    parser.add_argument("--just_evaluation", default=False, action='store_true')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=None, help="Override total d3rlpy training steps.")
    parser.add_argument("--n_steps_per_epoch", type=int, default=None, help="Override d3rlpy steps per epoch.")
    parser.add_argument("--uncertainty_n_steps", type=int, default=None, help="Override uncertainty predictor training steps.")
    parser.add_argument("--uncertainty_n_steps_per_epoch", type=int, default=None, help="Override uncertainty predictor steps per epoch.")
    parser.add_argument("--uncertainty_batch_size", type=int, default=None, help="Override uncertainty predictor batch size.")
    parser.add_argument("--eval_trials", type=int, default=10, help="Training-time evaluation trials per football level.")
    parser.add_argument("--eval_env_names", type=str, default=None, help="Comma-separated football env names for evaluation.")
    parser.add_argument("--observation_scaler", type=str, default="min_max", choices=("min_max", "none"),
                        help="Observation scaler for the d3rlpy policy. Use none for historical final CIQL reproduction.")
    # experiment config
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--device", type=str, default='cuda:2')
    parser.add_argument("--comment", type=str, default='none')
    parser.add_argument("--model_tag", type=str, default='small-rew')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--uncertainty_model_path", "--pretrained_uncertainty_model_path",
        dest="uncertainty_model_path", type=str, default="",
        help="Exact .d3 uncertainty checkpoint to load; disables model_tag auto-selection when set.")
    # data_args
    parser.add_argument("--obs_stack_num", type=int, default=1)
    parser.add_argument("--env_name", type=str, default='football')
    parser.add_argument("--fake_rollout_num", type=int, default=10)
    parser.add_argument("--skip_gen_data", default=False, action='store_true')
    parser.add_argument("--data_version", type=str, default='v1')
    parser.add_argument(
        "--merged_data_cache_file",
        type=str,
        default=os.environ.get("PLFB_MERGED_DATA_CACHE_FILE", ""),
        help="Exact merged .npz cache path to load/save; overrides generated cache naming when set.",
    )
    return parser.parse_args()



if __name__=="__main__":
    args = get_policy_args()
    set_reproducibility_seed(args.seed)
    if args.env_name == 'football':
        gen_model = DEFAULT_GEN_MODEL
        root_path = str(IMAGINARY_DATA_ROOT)
        args.obs_stack_num = 4
        args.neg_keep_rate = 0.1
        load_img_data_params = {"fake_rollout_num": args.fake_rollout_num, 'skip_gen_data': args.skip_gen_data,  'obs_stack_num': args.obs_stack_num}
        model_param_dict = {'model_tag': args.model_tag}

    else:
        root_path = str(TTT_DATA_ROOT)
        load_img_data_params = {}
        model_param_dict = {'model_tag': args.model_tag, 'obs_stack_num': args.obs_stack_num}
    model_param_dict.update(load_img_data_params)
    # root_path = os.path.join(root_path, args.env_name)
    model_root_path = str(MODEL_LOG_ROOT / args.env_name)
    if args.env_name == 'football':
        check_param_dict = {
            'comment': args.comment,
            'pi_alg_type': args.alg_type,
            'alpha': args.alpha,
            'target_value': args.target_value,
            'ent_target_coef': args.ent_target_coef,
            'coef_t': args.coef_t,
            'coef_r': args.coef_r,
            'trace_real_num': 0,
            'extra_real_traj_num': 0,
            'skip_gen_data': args.skip_gen_data,
            'rollout_num': 0,
            'fake_rollout_num': args.fake_rollout_num,
            'obs_stack_num': args.obs_stack_num,
        }
    else:
        check_param_dict = {
            'comment': args.comment, 'alpha': args.alpha,
            'neg_keep_rate': args.neg_keep_rate, 'target_value': args.target_value,
            'coef_r': args.coef_r, 'coef_t': args.coef_t}
    # mkdir
    os.makedirs(root_path, exist_ok=True)
    os.makedirs(model_root_path, exist_ok=True)

    def build_cache_name(params):
        return args.data_version + '&'.join(f"{k}={str(v).replace('/', '-')}" for k, v in params.items()) + '.npz'

    merged_data_cache_file = args.merged_data_cache_file.strip()
    if merged_data_cache_file:
        img_merge_saved_path = str(Path(merged_data_cache_file).expanduser())
        print('using merged data cache override', img_merge_saved_path)
    else:
        cache_param_dict = dict(load_img_data_params)
        if args.env_name == 'football':
            cache_param_dict["gen_model"] = gen_model
        cache_name = build_cache_name(cache_param_dict)
        img_merge_saved_path = str(MERGED_DATA_CACHE_ROOT / args.env_name / cache_name)
        if args.env_name == 'football' and not Path(img_merge_saved_path).exists():
            legacy_cache_params = [dict(cache_param_dict), dict(load_img_data_params)]
            if args.fake_rollout_num < 0:
                inferred_rollout_num = None
                for sample_file in sorted(os.listdir(root_path)):
                    if not sample_file.endswith('.npz'):
                        continue
                    try:
                        inferred_rollout_num = int(sample_file.split('-')[1])
                        break
                    except (IndexError, ValueError):
                        continue
                if inferred_rollout_num is not None:
                    inferred_with_model = dict(cache_param_dict)
                    inferred_with_model['fake_rollout_num'] = inferred_rollout_num
                    inferred_without_model = dict(load_img_data_params)
                    inferred_without_model['fake_rollout_num'] = inferred_rollout_num
                    legacy_cache_params.extend([inferred_with_model, inferred_without_model])
            for legacy_params in legacy_cache_params:
                legacy_cache_path = Path(root_path) / 'merged_data' / build_cache_name(legacy_params)
                if legacy_cache_path.is_file():
                    img_merge_saved_path = str(legacy_cache_path)
                    print('using retained merged data cache', img_merge_saved_path)
                    break
    print(img_merge_saved_path)
    img_merge_saved_dir = osp.dirname(img_merge_saved_path)
    if img_merge_saved_dir:
        os.makedirs(img_merge_saved_dir, exist_ok=True)
    if args.alg_type==AlgType.CQL or args.coef_r == 0 and args.coef_t == 0:
        skip_uncertainty = True
    else:
        skip_uncertainty = False
    if args.debug:
        log_root = str(DEBUG_LOG_ROOT / args.env_name)
    else:
        log_root = str(IRL_LOG_ROOT / args.env_name)
    name_filter = None
    if args.env_name == 'tictactoe-v0':
        from envs.tictactoe import TicTacToeEnv
        env = TicTacToeEnv()
        gamma = 0.9
        n_steps = 8_000
        hidden_units = [512, 512, 256, 128, 128]
        name_filter = 'seed-'
        data_path = str(TTT_DATA_ROOT / "imaginary_data-v2")
    elif args.env_name == 'football':
        import gfootball.env as football_env
        ensure_gfootball_working_dir()
        video_dir = os.path.join(log_root, "video_logs", 'None')
        env = football_env.create_environment(env_name='11_vs_11_hard_stochastic',representation='raw', \
                                        stacked=False, logdir=video_dir, write_goal_dumps=False, \
                                        write_full_episode_dumps=False, render=False, write_video=False,\
                                        number_of_left_players_agent_controls=1)
        env.env_name = 'football-11_vs_11'
        gamma = 0.99
        n_steps = 200_000
        hidden_units = [512, 512, 256, 128, 128]
        data_path = root_path

    else:
        env = gym.make(args.env_name)
        gamma = 0.99
        n_steps = 100_000
        hidden_units = [512, 512, 256, 128, 128]
        data_path = None
    uncertainty_predictor = UncertaintyPredictor(env=env, model_root_path=model_root_path,
                                                 model_param_dict=model_param_dict, skip_uncertainty=skip_uncertainty,
                                                 coef_r=args.coef_r, coef_t=args.coef_t, device=args.device)

    logger_adapter = d3rlpy.logging.CombineAdapterFactory([
        d3rlpy.logging.FileAdapterFactory(root_dir=os.path.join(log_root, "d3rlpy_logs")),
        d3rlpy.logging.TensorboardAdapterFactory(root_dir=os.path.join(log_root, "tensorboard_logs"))])
    img_dataset = get_img_dataset(data_path, img_merge_saved_path, env, uncertainty_predictor, args)
    obs_list, action_list, next_obs_list, reward_list, end_epi_list, done_list = img_dataset.load_raw_data(name_filter='seed-', **load_img_data_params)

    if not skip_uncertainty:
        if not uncertainty_predictor.load_model(args.uncertainty_model_path):
            uncertainty_train_kwargs = {}
            if args.uncertainty_n_steps is not None:
                uncertainty_train_kwargs["n_steps"] = args.uncertainty_n_steps
            if args.uncertainty_n_steps_per_epoch is not None:
                uncertainty_train_kwargs["n_steps_per_epoch"] = args.uncertainty_n_steps_per_epoch
            if args.uncertainty_batch_size is not None:
                uncertainty_train_kwargs["batch_size"] = args.uncertainty_batch_size
            uncertainty_predictor.train_model(obs_list, action_list, next_obs_list, reward_list,
                                              end_epi_list, done_list, **uncertainty_train_kwargs)

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
    if args.observation_scaler == "min_max":
        min_obs_val = obs_samples.min(axis=0)
        max_obs_val = obs_samples.max(axis=0)
        max_obs_val[(max_obs_val - min_obs_val) < 1] = min_obs_val[(max_obs_val - min_obs_val) < 1] + 1
        obs_scaler = MinMaxObservationScaler(minimum=min_obs_val, maximum=max_obs_val)
    elif args.observation_scaler == "none":
        obs_scaler = None
    else:
        raise ValueError(f"unsupported observation_scaler: {args.observation_scaler}")
    print("observation_scaler", args.observation_scaler)

    encoder_factory = VectorEncoderFactory(hidden_units=hidden_units, activation='relu', use_layer_norm=True, dropout_rate=0.1)
    # start training
    if args.alpha > 0:
        alpha_learning_rate = -1
    else:
        alpha_learning_rate = 1e-3
    args.eval_algo = args.alg_type
    args.num_players = 1

    n_steps_per_epoch = 100


    exp_name = formulate_param_to_name(check_param_dict)
    if args.env_name == 'football':
        if args.debug:
            n_steps_per_epoch = 10
            env_name_list = ['11_vs_11_level_0']
        else:
            n_steps_per_epoch = 5000
            env_name_list = [f'11_vs_11_level_{i}' for i in [0, 1, 2]] # + [f'11_vs_1_level_{i}_left' for i in [0, 1, 2]]
        if args.eval_env_names:
            env_name_list = [name.strip() for name in args.eval_env_names.split(",") if name.strip()]
        eval_env_list = env_list_creator(env_name_list, args)
        stack_obs_len = obs_list[0].shape[-1]
        # Duplicates code in the data loader.
        update_stack_obs = lambda obs, stack_obs: update_stack_img_obs(obs, stack_obs, args.obs_stack_num)
        env_evaluator = EnvironmentEvaluator(eval_env_list, obs_type='imginary_obs',
                                             update_stack_obs=update_stack_obs,
                                             stack_obs_len=stack_obs_len, n_trials=args.eval_trials, acs_replace_strategy=args.strategy)
    elif args.env_name == 'tictactoe-v0':
        env_evaluator = TTTEnvironmentEvaluator(env, oppo_players=['random', 'minimax'], n_trials=args.eval_trials)
    if args.n_steps is not None:
        n_steps = args.n_steps
    if args.n_steps_per_epoch is not None:
        n_steps_per_epoch = args.n_steps_per_epoch
    if args.alg_type == AlgType.CIQL:
        cql = DiscreteCQLSACConfig(
            n_critics=4,
            batch_size=args.batch_size,
            actor_encoder_factory=encoder_factory,
            critic_encoder_factory=encoder_factory,
            ent_target_coef=args.ent_target_coef,
            alpha_learning_rate=alpha_learning_rate,
            conservative_weight=args.alpha,
            target_value=args.target_value,
            observation_scaler=obs_scaler,
            gamma=gamma,
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
        ).create(device=args.device)
    elif args.alg_type == AlgType.CQL:
        cql = DiscreteCQLConfig(alpha=args.alpha, n_critics=2,
                        batch_size=args.batch_size, learning_rate=1e-4, alpha_learning_rate=alpha_learning_rate,
                        encoder_factory=encoder_factory, gamma=gamma, target_value=args.target_value,
                        observation_scaler=obs_scaler).create(device=args.device)
    else:
        raise ValueError(f"unsupported alg_type: {args.alg_type}")
    cql.fit(dataset, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch,
            save_interval=1, evaluators={'environment': env_evaluator, },
            logger_adapter=logger_adapter, experiment_name=exp_name)

    env.close()
