

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
from llm.utils.imaginary import load_inner_npz_by_index
from llm.utils.rewarder import Rewarder
import numpy as np
import torch
import json
import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.datasets import get_atari
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig, DiscreteCQLSACConfig, DiscreteCQLSAC
from d3rlpy.algos import DiscreteBC, DiscreteBCConfig, EnsembleBC, EnsembleBCConfig
from d3rlpy.preprocessing import MinMaxActionScaler, MinMaxObservationScaler
from llm.utils.obs2text import img_obs_to_text, imaginary_data_observation_stacked_info

from d3rlpy.metrics import EnvironmentEvaluator, EnsembleDatasetErrorEvaluator
# import ActionSpace
from d3rlpy.constants import ActionSpace
from gym.spaces import Box

from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset.replay_buffer import Signature
from d3rlpy.datasets import ReplayBuffer

# from d3rlpy.metrics.scorer import evaluate_on_environment
# from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from sklearn.model_selection import train_test_split
import gfootball.env as football_env
from llm.utils.obs2text import imaginary_data_observation, imaginary_data_to_vector, imaginary_data_observation_v2
from learning.utils import npz_extractor, update_stack_img_obs, env_list_creator
from learning.prompt import llm_rt_policy_prompt
STAGE = 'policy'


from CONFIG import *


DATA_ROOT = os.environ.get("PLFB_DATASET_PATH", os.path.join(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts"), "football"))
OFFLINE_DATA_ROOT = os.environ.get("PLFB_OFFLINE_DATASET_PATH", os.path.join(DATA_ROOT, "offline_dataset-v4"))
IMAGINARY_DATA_ROOT = os.environ.get("PLFB_IMAGINARY_DATASET_PATH", os.path.join(OFFLINE_DATA_ROOT, "imaginary_data-v3", "gpt-4o-mini"))


def formulate_param_to_name(params_dict):
    return  '&'.join([f'{k}={v}'.replace(" ", "--") for k, v in params_dict.items()])

def get_args(parser=None, return_parser=False):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--comment", type=str, default="none")
    parser.add_argument("--obs_stack_num", type=int, default=4)
    parser.add_argument("--extra_real_traj_num", type=int, default=0)
    parser.add_argument("--trace_real_num", type=int, default=0)
    parser.add_argument("--rollout_num", type=int, default=0)
    parser.add_argument("--fake_rollout_num", type=int, default=-1)
    parser.add_argument("--skip_gen_data", default=False, action='store_true')
    parser.add_argument("--alg_type", type=str, default=AlgType.CIQL)
    parser.add_argument("--human_reward_weight", type=float, default=1.0) # TODO: conduct ablation study but with low priority.
    # parser.add_argument("--stage", type=str, default="model")
    # ent_target_coef
    if return_parser:
        return parser
    else:
        return parser.parse_args()

def get_model_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=float, default=4096)
    return parser.parse_args()

def get_policy_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=60)
    parser.add_argument("--target_value", type=float, default=0.02)
    parser.add_argument("--offline_algo", type=str, default="sac")
    parser.add_argument("--ent_target_coef", type=float, default=0.8) # Values below 0.4 become too deterministic and unstable; historical default was 0.98.
    parser.add_argument("--coef_r", type=float, default=0.5)
    parser.add_argument("--coef_t", type=float, default=0.5)
    parser.add_argument("--strategy", type=str, default='all_replaced')
    parser.add_argument("--lr_rescale", type=float, default=0.5)
    parser.add_argument("--keep_rate", type=float, default=1.0)
    parser.add_argument("--neg_keep_rate", type=float, default=0.1)
    parser.add_argument("--just_evaluation", default=False, action='store_true')

    return parser.parse_args()

def real_data_append_func(s, e, found_real_data, stack_obs, update_stack_obs, obs_list, action_list, reward_list, dense_reward_lsit, next_obs_list, done_list, img_obs_dict_list, end_epi_list):
    # wrapper_obs = found_real_data['wrapper_obs']
    obs_before_modified_by_acs = found_real_data['obs_before_modified_by_acs']
    action = found_real_data['action']
    reward = found_real_data['reward']
    dense_rewards = found_real_data['dense_rewards']

    for i in range(s, e):
        dict_obs, vec_obs = imaginary_data_observation_v2(obs_before_modified_by_acs[i, 0], i, ret_type='both')
        stack_obs = update_stack_obs(vec_obs, stack_obs)
        acs = action[i]
        rew = reward[i]
        dense_rew = dense_rewards[i]
        next_vec_obs = imaginary_data_observation_v2(obs_before_modified_by_acs[i+1, 0], i+1, ret_type='vector')
        if rew != 0 or e == 1499:
            done = 1
        else:
            done = 0
        obs_list.append(stack_obs)
        action_list.append(acs)
        reward_list.append(rew)
        dense_reward_lsit.append(dense_rew)
        next_obs_list.append(next_vec_obs)
        done_list.append(done)
        img_obs_dict_list.append(dict_obs)
        if i == e - 1:
            end_epi_list.append(True)
        else:
            end_epi_list.append(False)

if __name__ == '__main__':
    legacy_imaginary_root = os.environ.get("PLFB_LEGACY_IMAGINARY_DATASET_PATH", os.path.join(os.environ.get("PLFB_WORK_ROOT", "plfb_work"), "legacy_imaginary_dataset"))
    legacy_seed_file = os.environ.get("PLFB_LEGACY_IMAGINARY_SEED_NPZ", os.path.join(legacy_imaginary_root, "no_1_imaginary_dataset_150x10_20240126-214157.npz"))
    if os.path.exists(legacy_seed_file):
        res = np.load(legacy_seed_file, allow_pickle=True)
    real_data_root_path = os.environ.get("PLFB_SAMPLED_DATA_PATH", os.path.join(OFFLINE_DATA_ROOT, "sample_data"))
    offline_data_path = OFFLINE_DATA_ROOT

    root_path = IMAGINARY_DATA_ROOT
    seed_imaginary_npz = os.environ.get("PLFB_IMAGINARY_SEED_NPZ", os.path.join(root_path, "no_0_imaginary_dataset_11-10-42.npz"))
    if os.path.exists(seed_imaginary_npz):
        res = np.load(seed_imaginary_npz, allow_pickle=True)

    args_parser = baseline_agent_parser.parse_args(return_parser=True)
    parser = get_args(args_parser, return_parser=True)
    if STAGE == 'model':
        args = get_model_args(parser)
        task_param_dict = {
            'comment': args.comment,
        }
    elif STAGE == 'policy':
        args = get_policy_args(parser)
        task_param_dict = {
            'comment': args.comment,
            'pi_alg_type': args.alg_type,
            'strategy': args.strategy,
            # 'algo': args.offline_algo,
            'alpha': args.alpha,
            "target_value": args.target_value,
            # "ent_target_coef": args.ent_target_coef,
            "coef_t": args.coef_t,
            "coef_r": args.coef_r,
            # 'lr_rescale': args.lr_rescale,
            'keep_rate': args.keep_rate,
            # "human_reward_weight": args.human_reward_weight,
        }
    else:
        raise ValueError("stage should be model or policy")
    if args.alg_type == AlgType.CIQL:
        args.extra_real_traj_num = 0
        args.trace_real_num = 0
        args.skip_gen_data = False
        args.rollout_num = 0
        args.fake_rollout_num = -1 # use all
    elif args.alg_type ==AlgType.CQL:
        args.extra_real_traj_num = 0
        args.trace_real_num = 0
        args.skip_gen_data = True
        args.rollout_num = 1
        args.fake_rollout_num = -1
    elif args.alg_type == AlgType.CQL_REAL:
        args.extra_real_traj_num = 0
        args.trace_real_num = 0
        args.skip_gen_data = True
        args.rollout_num = 10
        args.fake_rollout_num = -1
    elif args.alg_type == AlgType.RT:
        args.extra_real_traj_num = 0
        args.trace_real_num = 0
        args.skip_gen_data = True
        args.rollout_num = 1
        args.fake_rollout_num = -1
        args.neg_keep_rate = 0.0
        task_param_dict['coef_r'] = 0.0
        task_param_dict['coef_t'] = 0.0
    else:
        raise NotImplementedError
    data_param_dict = {
            "trace_real_num": args.trace_real_num,
            "extra_real_traj_num": args.extra_real_traj_num,
            "skip_gen_data": args.skip_gen_data,
            "rollout_num": args.rollout_num,
            "fake_rollout_num": args.fake_rollout_num,
            'obs_stack_num': args.obs_stack_num,
    }
    check_param_dict = task_param_dict.copy()
    check_param_dict.update(data_param_dict)

    args.stage = STAGE
    if args.stage == 'policy' and args.coef_r == 0 and args.coef_t == 0:
        skip_uncertainty = True
    else:
        skip_uncertainty = False
    data_version= 'v5data'
    if args.skip_gen_data:
        if args.rollout_num == 10:
            if args.fake_rollout_num < 0:
                img_merge_saved_path = osp.join(root_path, 'merged_data', data_version + '&'.join(f"{k}={check_param_dict[k]}" for k in ['trace_real_num', 'extra_real_traj_num', 'obs_stack_num', 'skip_gen_data']) + '.npz')
            else:
                img_merge_saved_path = osp.join(root_path, 'merged_data', data_version + '&'.join(f"{k}={check_param_dict[k]}" for k in ['trace_real_num', 'extra_real_traj_num', 'obs_stack_num', 'skip_gen_data', 'fake_rollout_num']) + '.npz')
        else:
            if args.fake_rollout_num < 0:
                img_merge_saved_path = osp.join(root_path, 'merged_data', data_version + '&'.join(f"{k}={check_param_dict[k]}" for k in ['trace_real_num', 'extra_real_traj_num', 'obs_stack_num', 'rollout_num', 'skip_gen_data']) + '.npz')
            else:
                img_merge_saved_path = osp.join(root_path, 'merged_data', data_version + '&'.join(f"{k}={check_param_dict[k]}" for k in ['trace_real_num', 'extra_real_traj_num', 'obs_stack_num', 'rollout_num', 'skip_gen_data', 'fake_rollout_num']) + '.npz')
    else:
        # Version compatibility.
        if args.rollout_num == 10:
            if args.fake_rollout_num < 0:
                img_merge_saved_path = osp.join(root_path, 'merged_data', data_version + '&'.join(f"{k}={check_param_dict[k]}" for k in ['trace_real_num', 'extra_real_traj_num', 'obs_stack_num']) + '.npz')
            else:
                img_merge_saved_path = osp.join(root_path, 'merged_data', data_version + '&'.join(f"{k}={check_param_dict[k]}" for k in ['trace_real_num', 'extra_real_traj_num', 'obs_stack_num', 'fake_rollout_num']) + '.npz')
        else:
            if args.fake_rollout_num < 0:
                img_merge_saved_path = osp.join(root_path, 'merged_data', data_version + '&'.join(f"{k}={check_param_dict[k]}" for k in ['trace_real_num', 'extra_real_traj_num', 'obs_stack_num', 'rollout_num']) + '.npz')
            else:

                img_merge_saved_path = osp.join(root_path, 'merged_data', data_version + '&'.join(f"{k}={check_param_dict[k]}" for k in ['trace_real_num', 'extra_real_traj_num', 'obs_stack_num', 'rollout_num', 'fake_rollout_num']) + '.npz')

    os.makedirs(osp.dirname(img_merge_saved_path), exist_ok=True)
    if args.debug:
        log_root = 'football_llm/ORL_LOG_DEBUG'
    else:
        if args.stage == 'model':
            log_root = 'football_llm/IRL_LOG_MODEL'
        else:
            log_root = 'football_llm/IRL_LOG'
    exp_name = formulate_param_to_name(check_param_dict)
    # find all file in root_path
    logger_adapter = d3rlpy.logging.CombineAdapterFactory([
    d3rlpy.logging.FileAdapterFactory(root_dir=os.path.join(log_root, "d3rlpy_logs")),
    d3rlpy.logging.TensorboardAdapterFactory(root_dir=os.path.join(log_root, "tensorboard_logs")),
    ])
    video_dir = os.path.join(log_root, "video_logs", exp_name)
    args.video_dir = video_dir


    device = args.device
    env = football_env.create_environment(env_name=args.environment,representation='raw', \
                                    stacked=False, logdir=video_dir, write_goal_dumps=args.write_goal_dumps, \
                                    write_full_episode_dumps=args.write_full_episode_dumps, render=args.render,\
                                    number_of_left_players_agent_controls=args.num_players)

    obs_sample = imaginary_data_to_vector(res['current_obs'][0], TODO_missing=True)
    action_key_name = 'im_next_actions'
    stack_obs_len = args.obs_stack_num * obs_sample.shape[-1]
    # steps = res['steps']
    # obs_dict = res['imaginary_obs']
    # action = res['action']
    # reward = res['reward']
    # dense_rewards = res['dense_rewards']
    # done = res['done']
    obs_list = []
    img_obs_dict_list = []
    action_list = []
    reward_list = []
    dense_reward_lsit = []
    next_obs_list = []
    done_list = []
    end_epi_list = []
    real_data_res = np.load(osp.join(real_data_root_path, 'sample_50_20240126-135504.npz'), allow_pickle=True)
    real_data_res_key_list = list(real_data_res.keys())
    update_stack_obs = lambda obs, stack_obs: update_stack_img_obs(obs, stack_obs, check_param_dict['obs_stack_num'])
    def init_stack_obs(start_step, end_step, datanpz):
        stack_obs = np.zeros(stack_obs_len)
        # for i in range(max(start_point-args.trace_real_num - args.obs_stack_num, 0), start_point-args.trace_real_num ):
        for i in range(start_step, end_step):
            vec_obs = imaginary_data_observation_v2(datanpz['obs_before_modified_by_acs'][i, 0], i, ret_type='vector')
            stack_obs = update_stack_obs(vec_obs, stack_obs)
        return stack_obs
    ########  --- load dataset ---
    if img_merge_saved_path is not None and os.path.exists(img_merge_saved_path):
        print("load saved data")
        res = np.load(img_merge_saved_path, allow_pickle=True)
        obs_list = res['obs']
        action_list = res['action']
        reward_list = res['reward']
        dense_reward_lsit = res['dense_rewards']
        next_obs_list = res['next_obs']
        done_list = res['done']
        img_obs_dict_list = res['img_obs_dict']
        end_epi_list = res['end_epi_list']
    else:
        for file in tqdm.tqdm(os.listdir(root_path), desc='loading data'):
            if file.endswith('.npz'):
                res = npz_extractor(osp.join(root_path, file), key_check_list=['current_obs', 'im_next_obs'])
                if res is None:
                    continue
                if args.fake_rollout_num < 0:
                    fake_rollout_num = int(file.split('-')[1])
                else:
                    fake_rollout_num = args.fake_rollout_num
                print(file, len(res['start_point']))

                start_points = res['start_point']
                for idx, sp in tqdm.tqdm(enumerate(start_points), desc='find real data'):
                    data_index =sp[0]
                    found_real_data = load_inner_npz_by_index(real_data_res, data_index)
                    start_point = sp[1]
                    if args.trace_real_num > 0 and start_point - args.trace_real_num >=0:
                        # load real data
                        print("trace back real data from", start_point, "to", start_point-args.trace_real_num)
                        # init stack_obs
                        stack_obs = init_stack_obs(max(start_point-args.trace_real_num - args.obs_stack_num, 0), start_point-args.trace_real_num, found_real_data)
                        real_data_append_func(start_point-args.trace_real_num, start_point, found_real_data, stack_obs, update_stack_obs, obs_list, action_list, reward_list, dense_reward_lsit, next_obs_list, done_list, img_obs_dict_list, end_epi_list)
                    forward_real_num = args.rollout_num
                    stack_obs = init_stack_obs(max(start_point - args.obs_stack_num, 0), start_point, found_real_data)
                    if forward_real_num > 0 and start_point + forward_real_num < 1500:
                        print("forward real data from", start_point, "to", start_point+forward_real_num)
                        real_data_append_func(start_point, start_point + forward_real_num, found_real_data, stack_obs, update_stack_obs, obs_list, action_list, reward_list, dense_reward_lsit, next_obs_list, done_list, img_obs_dict_list, end_epi_list)
                    if args.skip_gen_data:
                        continue
                    print("generate imaginary data from", start_point, "to", start_point+fake_rollout_num)
                    current_obs = res['current_obs'][idx]
                    im_next_obs = res['im_next_obs'][idx]
                    im_actions = res[action_key_name][idx]
                    im_reward = res['im_reward'][idx]
                    im_llm_reward = res['im_dense_reward'][idx]
                    for step in range(fake_rollout_num):
                        if step == 0:
                            cur_dict_obs = current_obs
                            img_vec_obs = imaginary_data_to_vector(current_obs, TODO_missing=True)
                        else:
                            cur_dict_obs = im_next_obs[step-1]
                            img_vec_obs = imaginary_data_to_vector(im_next_obs[step-1], TODO_missing=True)
                        img_action = im_actions[step]
                        img_reward = im_reward[step]
                        img_llm_reward = im_llm_reward[step]
                        # img_rule_based_reward = 0
                        img_next_obs = imaginary_data_to_vector(im_next_obs[step], TODO_missing=True)
                        if img_reward != 0 or start_point + step == 1499:
                            done = 1
                        else:
                            done = 0
                        stack_obs = update_stack_obs(img_vec_obs, stack_obs)
                        obs_list.append(stack_obs)
                        img_obs_dict_list.append(cur_dict_obs)
                        action_list.append(img_action)
                        reward_list.append(img_reward)
                        dense_reward_lsit.append(img_llm_reward)
                        next_obs_list.append(img_next_obs)
                        done_list.append(done)
                        if step ==fake_rollout_num - 1:
                            end_epi_list.append(True)
                        else:
                            end_epi_list.append(False)
        data_num = 0
        for file in tqdm.tqdm(os.listdir(offline_data_path), desc='loading offline data'):
            if file.endswith('.npz'):
                res = npz_extractor(osp.join(offline_data_path, file))
                if res is None:
                    continue
                stack_obs = np.zeros(stack_obs_len)
                if data_num >= args.extra_real_traj_num:
                    break
                real_data_append_func(0, 1499, res, stack_obs, update_stack_obs, obs_list, action_list, reward_list, dense_reward_lsit, next_obs_list, done_list, img_obs_dict_list, end_epi_list)
                data_num += 1

        obs_list = np.array(obs_list)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        dense_reward_lsit = np.array(dense_reward_lsit)
        next_obs_list = np.array(next_obs_list)
        done_list = np.array(done_list)
        img_obs_dict_list = np.array(img_obs_dict_list)
        end_epi_list = np.array(end_epi_list)
        np.savez_compressed(img_merge_saved_path, obs=obs_list, action=action_list, reward=reward_list,
                            dense_rewards=dense_reward_lsit, next_obs=next_obs_list, done=done_list, img_obs_dict=img_obs_dict_list, end_epi_list=end_epi_list)
    ########  --- load dataset ---
    if args.alg_type == AlgType.RT:
        if not os.environ.get("GRADIENT_ACCESS_TOKEN") or not os.environ.get("GRADIENT_WORKSPACE_ID"):
            raise RuntimeError("Set GRADIENT_ACCESS_TOKEN and GRADIENT_WORKSPACE_ID before RT fine-tuning.")
        rt_data_path = img_merge_saved_path.split('.npz')[0] + '-rt.jsonl'
        out_fp = open(rt_data_path, "w")
        for i in range(len(obs_list)):
            text_obs = img_obs_to_text(img_obs_dict_list[i])
            action = action_list[i]
            instruction = llm_rt_policy_prompt(text_obs)
            out_dict = {
                "inputs": f"<s>[INST] <<SYS>>\nYou are an instruction-following model. \n<</SYS>>\n\n{instruction} [/INST] {action} </s>"
            }
            out_fp.write(json.dumps(out_dict) + "\n")
        print("save rt data to", rt_data_path)
        from llm.utils.llama_index_compat import GradientFinetuneEngine, require_optional_llama_index_symbol
        GradientFinetuneEngine = require_optional_llama_index_symbol(GradientFinetuneEngine, 'GradientFinetuneEngine', 'gradient finetuning')
        base_model_slug = "llama2-7b-chat"
        # NOTE: can only specify one of base_model_slug or model_adapter_id
        finetune_engine = GradientFinetuneEngine(
            base_model_slug=base_model_slug,
            # model_adapter_id=os.environ.get('GRADIENT_MODEL_ADAPTER_ID'),
            name="RT-code-2",
            data_path=rt_data_path,
            verbose=True,
            batch_size=4,
            rank=32,
        )
        epochs = 1
        for i in range(epochs):
            print(f"** EPOCH {i} **")
            finetune_engine.finetune()
        print(finetune_engine.model_adapter_id)
        print(finetune_engine.get_finetuned_model(max_tokens=300))
        exit()


    ######## init some reuseable functions
    print("data loaded")
    print("obs shape", obs_list.shape)
    rewarder = Rewarder(yaml_name='reward2.yaml')
    def compute_dense_reward_from_data(index):
        if index > 0 and done_list[index - 1] != 1:
            # Dividing by 30 roughly maps raw values into the [-1, 1] range.
            dense_reward = rewarder.calc_reward_v2(reward_list[index], img_obs_dict_list[index+1], img_obs_dict_list[index],  action_list[index])
        else:
            dense_reward = 0
        dense_reward /= 30
        dense_reward = np.clip(dense_reward, -1, 1)
        dense_reward += 0.05
        dense_reward *= args.human_reward_weight
        try:
            img_dense_reward = np.clip(float(dense_reward_lsit[index]), -2, 2)
            ending_reward = float(reward_list[index])
            if ending_reward != 0:
                print("ending_reward", ending_reward)
            total_reward = img_dense_reward / 2 + dense_reward + ending_reward * 5
        except Exception as e:
            print(e)
            print(dense_reward_lsit[i])
            print(reward_list[i])
            img_dense_reward = 0
            ending_reward = 0
            total_reward = img_dense_reward / 2 + dense_reward + ending_reward * 5
        return total_reward

    pretrain_model = False
    if args.stage == 'policy' and not skip_uncertainty:
        # load trained ensemble model
        model_root_path = 'football_llm/IRL_LOG_MODEL/d3rlpy_logs'
        model_path = None
        for files in os.listdir(model_root_path):
            if formulate_param_to_name(data_param_dict) in files and 'skip-know' in files:
                # load the newest model
                newest_model_iter_num = 0
                for model_file in os.listdir(os.path.join(model_root_path, files)):
                    if model_file.endswith('.d3') and int(model_file.split('_')[1].split('.')[0]) > newest_model_iter_num:
                        newest_model_iter_num = int(model_file.split('_')[1].split('.')[0])
                model_path = os.path.join(model_root_path, files, f'model_{newest_model_iter_num}.d3')
                print('load path', model_path)
        if model_path is not None:
            bc = d3rlpy.load_learnable(model_path)
        else:
            raise ValueError("model not found, pattern", model_root_path, formulate_param_to_name(data_param_dict))
    action_list = np.array(action_list)
    action_list[action_list==None] = 0
    action_list = action_list.astype(np.int16)
    if args.stage == 'model':
        # ensemble model
        # obs + 1 (action)
        observation_signature = Signature(
        dtype=[np.float32],
        shape=[stack_obs_len + 1],  # type: ignore
        )
        # next_obs_len + 1 (reward)
        action_signature = Signature(
            dtype=[np.float32],
            shape=[obs_sample.shape[-1] + 1],  # type: ignore
        )
        reward_signature = Signature(dtype=[np.dtype(np.float32)], shape=[[1]])
        buffer = InfiniteBuffer()
        train_dataset = ReplayBuffer(
            buffer,
            env=env,
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            action_space=ActionSpace.CONTINUOUS,
            action_size=action_signature.shape[-1],
            cache_size=2000,
        )
        test_buffer = InfiniteBuffer()
        test_dataset = ReplayBuffer(test_buffer,
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            action_space=ActionSpace.CONTINUOUS,
            action_size=action_signature.shape[-1],
            cache_size=2000,
        )
        selected_dataset = train_dataset
        train_ratio = 0.8
        epi_len_dict = {}

        for i in tqdm.tqdm(range(len(obs_list)-1)):
            total_reward = compute_dense_reward_from_data(i)
            selected_dataset.append(np.append(obs_list[i], action_list[i]),
                            np.append(next_obs_list[i], [total_reward]), total_reward)
            if end_epi_list[i] == 1 or done_list[i] == 1:
                # if selected_dataset.episodes[-1].transition_count > 0:
                selected_dataset.clip_episode(done_list[i])
                # else:
                    # print("error")
                # print("len of traj", selected_dataset.episodes[-1].transition_count)
                if len(selected_dataset.episodes) == 0:
                    continue
                if selected_dataset.episodes[-1].transition_count not in epi_len_dict:
                    epi_len_dict[selected_dataset.episodes[-1].transition_count] = 1
                else:
                    epi_len_dict[selected_dataset.episodes[-1].transition_count] += 1
                if  np.random.rand() <= train_ratio:
                    selected_dataset = train_dataset
                else:
                    selected_dataset = test_dataset
        print("summary of episode length", epi_len_dict)
        encoder_factory = VectorEncoderFactory(hidden_units=[512, 256, 256, 128], activation='relu')
        # rescale and keep zero as the default action for weighting computation in the downstream
        # zone is rescaled to [-1, 1] cuz the differences larger than 1 is not common.
        samples = train_dataset.sample_transition_batch(102400)
        obs_samples = samples.observations
        acs_samples = samples.actions
        def clip_min_max(input_sample):
            min_val = input_sample.min(axis=0)
            max_val = input_sample.max(axis=0)
            max_val[(max_val - min_val) < 1] = min_val[(max_val - min_val) < 1] + 1
            return min_val, max_val
        min_acs_val, max_acs_val = clip_min_max(acs_samples)
        min_obs_val, max_obs_val = clip_min_max(obs_samples)
        obs_scaler = MinMaxObservationScaler(minimum=min_obs_val, maximum=max_obs_val)
        action_scaler = MinMaxActionScaler(minimum=min_acs_val, maximum=max_acs_val)
        bc = EnsembleBCConfig(batch_size=args.batch_size, learning_rate=1e-5,
                    encoder_factory=encoder_factory, observation_scaler=obs_scaler,
                    action_scaler=action_scaler).create(device=device)
        # start training
        if args.debug:
            n_steps_per_epoch = 100
        else:
            n_steps_per_epoch = 1000
        bc.fit(train_dataset, n_steps=int(3e5), n_steps_per_epoch=n_steps_per_epoch, save_interval=10,
            evaluators={'test_error': EnsembleDatasetErrorEvaluator(test_dataset)},
            logger_adapter=logger_adapter, experiment_name=exp_name)
        env.close()
    elif args.stage == 'policy':
        def T_R_uncertainty(obs, action):
            if len(obs.shape) == 1:
                obs = obs[np.newaxis, ...]
                action = action[np.newaxis, ...]
            if len(action.shape) == 1:
                action = action[..., np.newaxis]
            if skip_uncertainty:
                return np.zeros((obs.shape[0])), np.zeros((obs.shape[0]))
            else:
                u = bc.predict_uncertainty(np.concatenate([obs, action], axis=-1))
                u_t = np.mean(u[..., :-1], axis=-1)
                u_r = u[..., -1]
            return u_t, u_r
        observation_signature = Signature(
        dtype=[np.float32],
        shape=[stack_obs_len],  # type: ignore
        )
        action_signature = Signature(
            dtype=[np.int32],
            shape=[1],  # type: ignore
        )
        reward_signature = Signature(dtype=[np.dtype(np.float32)], shape=[[1]])
        buffer = InfiniteBuffer()
        dataset = ReplayBuffer(
            buffer,
            env=env,
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            cache_size=2000,
        )

        good_buffer = InfiniteBuffer()
        good_dataset = ReplayBuffer(
            good_buffer,
            env=env,
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            cache_size=2000,
        )
        u_t_list = []
        u_r_list = []
        # compute the uncertainty list of all data by batch
        bath_size = 40960
        for i in tqdm.tqdm(range(0, len(obs_list)-1, bath_size), desc='compute uncertainty'):
            batch_obs = obs_list[i:i+bath_size]
            batch_acs = action_list[i:i+bath_size]
            u_t, u_r = T_R_uncertainty(batch_obs, batch_acs)
            u_t_list.append(u_t)
            u_r_list.append(u_r)
        u_t_list = np.concatenate(u_t_list, axis=0)
        u_r_list = np.concatenate(u_r_list, axis=0)
        u_r_scale = np.std(u_r_list)
        u_t_scale = np.std(u_t_list)
        if u_r_scale == 0:
            u_r_scale = 1
        if u_t_scale == 0:
            u_t_scale = 1
        def scale_u_reward(u_t, u_r):
            return np.clip(u_r / u_r_scale, 0, 2), np.clip(u_t / u_t_scale, 0, 2)

        # compute the rewards
        for i in tqdm.tqdm(range(len(obs_list)-1), desc='add dataset'):
            total_reward = compute_dense_reward_from_data(i)
            u_t, u_r = u_t_list[i], u_r_list[i]
            u_t, u_r = scale_u_reward(u_t, u_r)
            reward = total_reward - args.coef_r * u_r - args.coef_t * u_t
            dataset.append(obs_list[i], action_list[i], reward)
            if end_epi_list[i] == 1 or done_list[i] == 1:
                dataset.clip_episode(done_list[i])

        ret_list = []
        original_epi_num = len(dataset.episodes)
        for e in  tqdm.tqdm(dataset.episodes, desc='filter dataset'):
            if np.random.random() > args.keep_rate:
                continue
            if np.sum(e.rewards) > 0:
                good_dataset.append_episode(e)
                ret_list.append(np.sum(e.rewards))
            else:
                if np.random.rand() < args.neg_keep_rate:
                    good_dataset.append_episode(e)
                    ret_list.append(np.sum(e.rewards))
        dataset = good_dataset
        samples = dataset.sample_transition_batch(102400)
        obs_samples = samples.observations
        rew_samples = samples.rewards
        ret_list = np.array(ret_list)
        print("reward mean", rew_samples.mean(), "std", rew_samples.std(), "max", rew_samples.max(), "min", rew_samples.min())
        print("ret mean", ret_list.mean(), "std", ret_list.std(), "max", ret_list.max(), "min", ret_list.min())
        print("episodes", len(dataset.episodes), "original episodes", original_epi_num)
        min_obs_val = obs_samples.min(axis=0)
        max_obs_val = obs_samples.max(axis=0)
        max_obs_val[(max_obs_val - min_obs_val) < 1] = min_obs_val[(max_obs_val - min_obs_val) < 1] + 1
        obs_scaler = MinMaxObservationScaler(minimum=min_obs_val, maximum=max_obs_val)
        hidden_units = [512, 512, 256, 128, 128]
        encoder_factory = VectorEncoderFactory(hidden_units=hidden_units, activation='relu', use_layer_norm=True, dropout_rate=0.1)
        # start training
        if args.debug:
            n_steps_per_epoch = 10
            env_name_list = [f'11_vs_1_level_{i}_left' for i in [0]]
        else:
            n_steps_per_epoch = 5000
            env_name_list = [f'11_vs_11_level_{i}' for i in [0, 1, 2]] # + [f'11_vs_1_level_{i}_left' for i in [0, 1, 2]]
        if args.alpha > 0:
            alpha_learning_rate = -1
        else:
            alpha_learning_rate = 1e-3
        args.video_dir = None
        args.write_goal_dumps = False
        args.write_full_episode_dumps = False
        args.render = False
        args.eval_algo = args.alg_type
        args.num_players = 1
        eval_env_list = env_list_creator(env_name_list, args)
        env_evaluator = EnvironmentEvaluator(eval_env_list, obs_type='imginary_obs',
                                             update_stack_obs=update_stack_obs,
                                             stack_obs_len=stack_obs_len, n_trials=40, acs_replace_strategy=args.strategy)
        if args.offline_algo == 'sac':
            sac = DiscreteCQLSACConfig(n_critics=4, batch_size=256,
                                    actor_encoder_factory=encoder_factory,
                                    critic_encoder_factory=encoder_factory,
                                    ent_target_coef=args.ent_target_coef,
                                        alpha_learning_rate=alpha_learning_rate,
                                        conservative_weight=args.alpha,
                                    target_value=args.target_value,
                                    observation_scaler=obs_scaler,
                                    actor_learning_rate=1e-4 * args.lr_rescale,
                                    critic_learning_rate=3e-4 * args.lr_rescale).create(device=device)
            sac.fit(dataset, n_steps=int(4e5), n_steps_per_epoch=n_steps_per_epoch,
                    save_interval=1, evaluators={'environment': env_evaluator}, logger_adapter=logger_adapter, experiment_name=exp_name)

        elif args.offline_algo == 'cql':
            cql = DiscreteCQLConfig(alpha=check_param_dict['alpha'], n_critics=2,
                                    batch_size=256, learning_rate=3e-4, alpha_learning_rate=alpha_learning_rate,
                                    encoder_factory=encoder_factory,
                                    target_value=args.target_value, observation_scaler=obs_scaler).create(device=device)

            cql.fit(dataset, n_steps=int(2e5), n_steps_per_epoch=n_steps_per_epoch,
                    save_interval=1, evaluators={'environment': env_evaluator, },
                    logger_adapter=logger_adapter, experiment_name=exp_name)

        env.close()
        for env in eval_env_list:
            env.close()
    # Tuning notes:
    # The conservative value reflects one-step performance change from behavior policy to target policy; it should relate to the one-step reward gap and stay below the maximum one-step reward.
    # Estimate target value with rew_samples.mean(); for a 10% improvement over behavior policy, use rew_samples.mean() * 0.1.
    # Legacy implementation note.
