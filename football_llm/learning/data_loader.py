
import os
import os.path as osp
import json
from pathlib import Path
from typing import Any, Optional, overload
import tqdm
import numpy as np
import gym
import argparse
from funcs import *
from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset.replay_buffer import Signature
from d3rlpy.datasets import ReplayBuffer

from llm.utils.obs2text import imaginary_data_observation, imaginary_data_to_vector, imaginary_data_observation_v2
from llm.utils.imaginary import load_inner_npz_by_index
from learning.utils import update_stack_img_obs
from learning.uncertainty_predictor import UncertaintyPredictor

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("PLFB_DATASET_PATH", PROJECT_ROOT / "data")).resolve()
OFFLINE_DATA_ROOT = Path(os.environ.get("PLFB_OFFLINE_DATASET_PATH", DATA_ROOT)).resolve()
TTT_DATA_ROOT = Path(os.environ.get("PLFB_TTT_DATA_PATH", DATA_ROOT / "tic_tac_toe_data")).resolve()
SAMPLED_DATA_ROOT = Path(os.environ.get("PLFB_SAMPLED_DATA_PATH", OFFLINE_DATA_ROOT / "sample_data")).resolve()
LEGACY_SAMPLED_DATA_ROOT = (PROJECT_ROOT / "llm" / "sampled_data").resolve()


def default_real_sample_npz() -> str:
    override = os.environ.get("PLFB_REAL_SAMPLE_NPZ")
    if override:
        return override

    candidates = (
        SAMPLED_DATA_ROOT / "sample_50-v6.npz",
        SAMPLED_DATA_ROOT / "sample_50_20240126-135504.npz",
        OFFLINE_DATA_ROOT / "sample_data" / "sample_50-v6.npz",
        LEGACY_SAMPLED_DATA_ROOT / "sample_50-v6.npz",
        LEGACY_SAMPLED_DATA_ROOT / "sample_50_20240126-135504.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])

class ImgDataset(object):
    def __init__(self, data_path: str, img_data_saved_path: str, env: gym.Env, 
                 uncertainty_predictor:UncertaintyPredictor, configs:argparse.Namespace) -> None:
        self._data_path = data_path
        self.img_data_saved_path = img_data_saved_path
        self.env = env
        self.uncertainty_predictor = uncertainty_predictor
        self.keep_rate = configs.keep_rate
        self.neg_keep_rate = configs.neg_keep_rate
        self.configs = configs
        # obs_list, action_list, reward_list, end_epi_list, done_list = self.load_raw_data()
        # self.construction(obs_list, action_list, reward_list, end_epi_list, done_list)

    def load_raw_data(self, name_filter=''):
        return None, None, None, None, None
    
    def construction(self, obs_list, action_list, reward_list, end_epi_list, done_list):
        observation_signature = Signature(
        dtype=[np.float32],
        shape=[obs_list[0].shape],  # type: ignore
        )
        action_signature = Signature(
            dtype=[np.int32],
            shape=[1],  # type: ignore
        )
        reward_signature = Signature(dtype=[np.dtype(np.float32)], shape=[[1]])
        buffer = InfiniteBuffer()
        dataset = ReplayBuffer(
            buffer,
            env=self.env,
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            cache_size=2000,
        )

        good_buffer = InfiniteBuffer()
        good_dataset = ReplayBuffer(
            good_buffer,
            env=self.env,
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
            u_t, u_r = self.uncertainty_predictor.T_R_uncertainty(batch_obs, batch_acs)
            u_t_list.append(u_t)
            u_r_list.append(u_r)
        u_t_list = np.concatenate(u_t_list, axis=0)
        u_r_list = np.concatenate(u_r_list, axis=0)
        u_r_scale = np.std(u_r_list)
        u_t_scale = np.std(u_t_list)
        u_r_min = np.min(u_r_list)
        u_t_min = np.min(u_t_list)
        if u_r_scale == 0:
            u_r_scale = 1
        if u_t_scale == 0:
            u_t_scale = 1
        def scale_u_reward(u_t, u_r):
            return np.clip((u_r - u_r_min) / u_r_scale, 0, 2), np.clip((u_t - u_t_min) / u_t_scale, 0, 2)

        # compute the rewards 
        for i in tqdm.tqdm(range(len(obs_list)-1), desc='add dataset'):
            # total_reward = compute_dense_reward_from_data(i)
            total_reward = reward_list[i]
            u_t, u_r = u_t_list[i], u_r_list[i]
            u_t, u_r = scale_u_reward(u_t, u_r)
            reward = self.uncertainty_predictor.unc_rew_v2(total_reward, u_t, u_r)
            dataset.append(obs_list[i], action_list[i], reward)
            if end_epi_list[i] == 1 or done_list[i] == 1:
                dataset.clip_episode(done_list[i])

        ret_list = []
        original_epi_num = len(dataset.episodes)
        for e in tqdm.tqdm(dataset.episodes, desc='filter dataset'):
            if np.random.rand() >= self.keep_rate:
                continue
            if np.sum(e.rewards) > 0:
                good_dataset.append_episode(e)
                ret_list.append(np.sum(e.rewards))
            else:
                if np.random.rand() < self.neg_keep_rate:
                    good_dataset.append_episode(e)
                    ret_list.append(np.sum(e.rewards))
        dataset = good_dataset
        self.dataset = dataset
        self.data_info = {
            'return_list': ret_list,
            'original_epi_num': original_epi_num
        }

class TicTacToeImgDataset(ImgDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
    
    def load_raw_data(self, pure_offline=False, name_filter=''):
        if pure_offline:
            data_path = str(TTT_DATA_ROOT)
            name_filter = 'all_data_minimax_tic_tac_toe_trajectories-noise-0.2.json'
        else:
            data_path = str(TTT_DATA_ROOT / "imaginary_data-v2")

        obs_list = []
        action_list = []
        reward_list = []
        llm_reward_list = []

        end_epi_list = []
        done_list = []
        ret_list = []
        next_obs_list = []
        original_epi_num = 0
        end_rew_bias = 1.5
        for file in os.listdir(data_path):
            # pure offline data
            if name_filter in file:
                img_saved_path = os.path.join(data_path, file)
                print("get dataset", img_saved_path)
                if pure_offline:
                    res = json.load(open(img_saved_path, 'r'))
                else:
                    res = np.load(img_saved_path, allow_pickle=True)['data'].flatten()[0]
                original_epi_num += len(res['trajectories'])
                import random
                random.shuffle(res['trajectories'])
                
                for traj in res['trajectories']:
                    cur_rew_list = []
                    if np.random.rand() >= self.keep_rate:
                        continue
                    bad_traj = False
                    try:
                        # TODO: Fix cascading issues caused by dataset alignment mismatch.
                        for idx, tuple in enumerate(traj):
                            if pure_offline:
                                if idx == len(traj) - 1:
                                    break
                                if idx % 2 == 0:
                                    obs_list.append(np.array(tuple['state'][:9]))
                                    action_list.append(np.array(traj[idx+1]['action']))
                                    reward_list.append(np.array(traj[idx+1]['reward']))
                                    next_obs_list.append(np.array(traj[idx+1]['state']))
                                    end_epi_list.append(0)
                                    done_list.append(0)
                            else:
                                if len(tuple['obs']) != 9 or len(tuple['next_obs']) != 9:
                                    bad_traj = True
                                    break
                                obs_list.append(np.array(tuple['obs'][:9]))
                                action_list.append(np.array(tuple['action']))
                                llm_reward_list.append(np.clip(float(np.array(tuple['dense_reward'])), -2, 2))
                                reward_list.append((llm_reward_list[-1]) / 4 + np.array(tuple['reward']))
                                cur_rew_list.append((llm_reward_list[-1]) / 4 + np.array(tuple['reward']))
                                next_obs_list.append(np.array(traj[idx]['next_obs'][:9]))
                                end_epi_list.append(0)
                                done_list.append(0)
                    except Exception as e:
                        print(traj)
                    if not bad_traj:
                        end_epi_list[-1] = 1
                        done_list[-1] = 1
                        reward_list[-1] = (traj[-1]['reward']) * 8 + end_rew_bias
                        ret_list.append(np.sum(cur_rew_list))
                    else:
                        obs_list = obs_list[:-idx]
                        action_list = action_list[:-idx]
                        reward_list = reward_list[:-idx]
                        next_obs_list = next_obs_list[:-idx]
                        end_epi_list = end_epi_list[:-idx]
                        done_list = done_list[:-idx]
                        llm_reward_list = llm_reward_list[:-idx]

            # else:
            #     raise ValueError("data not found")     
        obs_list = np.clip(np.array(obs_list), 0, 2)
        next_obs_list = np.clip(np.array(next_obs_list), 0, 2)
        action_list = np.clip(np.array(action_list), 0, 8)
        reward_list = np.array(reward_list) / 2
        end_epi_list = np.array(end_epi_list)
        self.data_info = {
            'return_list': ret_list,
            'original_epi_num': original_epi_num
        }
        print("original_epi_num", original_epi_num)
        return obs_list, action_list, next_obs_list, reward_list, end_epi_list, done_list 

class FootballImgDataset(ImgDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.human_reward_weight = 1.0
        self.img_reward_weight = 0.0
    
    def load_raw_data(self, fake_rollout_num, skip_gen_data,  obs_stack_num, name_filter=''):
        img_merge_saved_path = self.img_data_saved_path
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
            update_stack_obs = lambda obs, stack_obs: update_stack_img_obs(obs, stack_obs, obs_stack_num)
            obs_list = []
            img_obs_dict_list = []
            action_list = []
            reward_list = []
            dense_reward_lsit = []
            next_obs_list = []
            done_list = []
            end_epi_list = []
            sample_npz = default_real_sample_npz()
            real_data_res = np.load(sample_npz, allow_pickle=True)
            res = None
            for sample_file in sorted(os.listdir(self._data_path)):
                if not sample_file.endswith('.npz'):
                    continue
                res = npz_extractor(osp.join(self._data_path, sample_file), key_check_list=['current_obs', 'im_next_obs'])
                if res is not None:
                    break
            if res is None:
                raise FileNotFoundError(f"no valid imaginary npz files found in {self._data_path}")
            obs_sample = imaginary_data_to_vector(res['current_obs'][0], TODO_missing=True)
            stack_obs_len = obs_stack_num * obs_sample.shape[-1]
            
            def init_stack_obs(start_step, end_step, datanpz):
                stack_obs = np.zeros(stack_obs_len)
                # for i in range(max(start_point-args.trace_real_num - args.obs_stack_num, 0), start_point-args.trace_real_num ):
                for i in range(start_step, end_step):
                    vec_obs = imaginary_data_observation_v2(datanpz['obs_before_modified_by_acs'][i, 0], i, ret_type='vector')
                    stack_obs = update_stack_obs(vec_obs, stack_obs)
                return stack_obs

            for file in tqdm.tqdm(os.listdir(self._data_path), desc='loading data'):            
                if file.endswith('.npz'):
                    res = npz_extractor(osp.join(self._data_path, file), key_check_list=['current_obs', 'im_next_obs'])
                    if res is None:
                        continue
                    if fake_rollout_num < 0:
                        fake_rollout_num = int(file.split('-')[1])
                    print(file, len(res['start_point']))
                    start_points = res['start_point']
                    for idx, sp in tqdm.tqdm(enumerate(start_points), desc='find real data'):
                        data_index =sp[0]
                        start_point = sp[1]
                        if skip_gen_data:
                            continue
                        
                        found_real_data = load_inner_npz_by_index(real_data_res, data_index)
                        stack_obs = init_stack_obs(max(start_point - obs_stack_num, 0), start_point, found_real_data)
                        print("generate imaginary data from", start_point, "to", start_point+fake_rollout_num)
                        current_obs = res['current_obs'][idx]
                        im_next_obs = res['im_next_obs'][idx]
                        im_actions = res['im_next_actions'][idx]
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
    
            # raise ValueError("data not found")
        action_list[action_list == None] = 0
        action_list = action_list.astype(np.int32)
        action_list[action_list > 18] = 0 # 18 is the max action number
        
        from llm.utils.rewarder import Rewarder
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
            dense_reward *= 1.0
            ending_reward =  np.clip(float(reward_list[index]), -1, 1)
            try:
                img_dense_reward = np.clip(float(dense_reward_lsit[index]), -2, 2)
                if ending_reward != 0:
                    print("ending_reward", ending_reward)
                total_reward = img_dense_reward * self.img_reward_weight + dense_reward + ending_reward * 5
            except Exception as e:
                print(e)
                print(dense_reward_lsit[i])
                print(reward_list[i])
                img_dense_reward = 0
                ending_reward = 0
                total_reward = img_dense_reward / 2 + dense_reward + ending_reward * 5
            return total_reward
        
        # compute the rewards 
        new_reward_list = []
        for i in tqdm.tqdm(range(len(obs_list)-1), desc='add dataset'):
            total_reward = compute_dense_reward_from_data(i)
            new_reward_list.append(total_reward)
        new_reward_list = np.array(new_reward_list)
        new_reward_list = np.clip(new_reward_list, -5, 5) / 2.5
        return obs_list, action_list, next_obs_list, new_reward_list, end_epi_list, done_list
        
def get_img_dataset(data_path: str, img_merge_saved_path: str, env: gym.Env, uncertainty_predictor:UncertaintyPredictor, configs):
    if env.env_name == 'tictactoe-v0':
        return TicTacToeImgDataset(data_path, img_merge_saved_path, env, uncertainty_predictor, configs)
    elif env.env_name == 'football-11_vs_11':
        return FootballImgDataset(data_path, img_merge_saved_path, env, uncertainty_predictor, configs)
    else:
        raise ValueError(f"env {env.env_name} not supported")
