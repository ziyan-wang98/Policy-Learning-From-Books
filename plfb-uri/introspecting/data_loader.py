
import os
import json
from pathlib import Path
from typing import Any, Optional, overload
import tqdm
import numpy as np
import gym
import argparse
from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset.replay_buffer import Signature
from d3rlpy.datasets import ReplayBuffer

from introspecting.uncertainty_predictor import UncertaintyPredictor

PROJECT_ROOT = Path(os.environ.get("PLFB_URI_ROOT", Path(__file__).resolve().parents[1])).resolve()
FOOTBALL_LLM_ROOT = Path(os.environ.get("PLFB_ROOT", PROJECT_ROOT.parent / "football_llm")).resolve()
DATA_ROOT = Path(os.environ.get("PLFB_DATASET_PATH", FOOTBALL_LLM_ROOT / "data")).resolve()
TTT_DATA_ROOT = Path(os.environ.get("PLFB_TTT_DATA_PATH", DATA_ROOT / "tic_tac_toe_data")).resolve()

class ImgDataset(object):
    def __init__(self, data_path: str, env: gym.Env, 
                 uncertainty_predictor:UncertaintyPredictor, configs:argparse.Namespace) -> None:
        self._data_path = data_path
        self.env = env
        self.uncertainty_predictor = uncertainty_predictor
        self.keep_rate = configs.keep_rate
        self.neg_keep_rate = configs.neg_keep_rate
        self.configs = configs
        # obs_list, action_list, reward_list, end_epi_list, done_list = self.load_raw_data()
        # self.construction(obs_list, action_list, reward_list, end_epi_list, done_list)

    def load_raw_data(self):
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
            reward = total_reward - self.uncertainty_predictor.unc_rew(u_t, u_r)
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
    
    def load_raw_data(self):
        img_merge_saved_path = self.data_path
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
            raise ValueError("data not found")
        
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
        
        # compute the rewards 
        new_reward_list = []
        for i in tqdm.tqdm(range(len(obs_list)-1), desc='add dataset'):
            total_reward = compute_dense_reward_from_data(i)
            new_reward_list.append(total_reward)

        return obs_list, action_list, new_reward_list, end_epi_list, done_list

def get_img_dataset(data_path: str, env: gym.Env, uncertainty_predictor:UncertaintyPredictor, configs):
    if env.env_name == 'tictactoe-v0':
        return TicTacToeImgDataset(data_path, env, uncertainty_predictor, configs)
    elif env.env_name == 'football-11_vs_11_kaggle':
        return FootballImgDataset(data_path, env, uncertainty_predictor, configs)
    else:
        raise ValueError(f"env {env.env_name} not supported")
