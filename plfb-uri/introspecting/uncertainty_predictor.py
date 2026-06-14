import os
import d3rlpy
import numpy as np
import tqdm
from funcs import formulate_param_to_name

from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset.replay_buffer import Signature
from d3rlpy.datasets import ReplayBuffer
from d3rlpy.constants import ActionSpace
from d3rlpy.preprocessing import MinMaxActionScaler, MinMaxObservationScaler
from d3rlpy.algos import DiscreteBC, DiscreteBCConfig, EnsembleBC, EnsembleBCConfig
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.metrics import EnsembleDatasetErrorEvaluator


class UncertaintyPredictor(object):
    def __init__(self, env, model_root_path, model_param_dict, skip_uncertainty, coef_r, coef_t, device) -> None:
        self.model_root_path = model_root_path
        self.model_param_dict = model_param_dict
        self.skip_uncertainty  = skip_uncertainty
        self.learned_model = None
        self.coef_r = coef_r
        self.coef_t = coef_t
        self.env = env
        self.device = device
        pass

    def load_model(self) -> None:
        model_path = None
        bc = None
        model_root_path = self.model_root_path
        model_param_dict = self.model_param_dict
        model_log_path = os.path.join(model_root_path, 'd3rlpy_logs')
        longest_newest_model_iter_num = 0
        for files in os.listdir(model_log_path):
            if files.startswith(formulate_param_to_name(model_param_dict)):
                # load the newest model
                newest_model_iter_num = 0
                for model_file in os.listdir(os.path.join(model_log_path, files)):
                    if model_file.endswith('.d3') and int(model_file.split('_')[1].split('.')[0]) > newest_model_iter_num:
                        newest_model_iter_num = int(model_file.split('_')[1].split('.')[0])
                if newest_model_iter_num > longest_newest_model_iter_num:
                    longest_newest_model_iter_num = newest_model_iter_num
                    model_path = os.path.join(model_log_path, files, f'model_{longest_newest_model_iter_num}.d3')
                print('load path', model_path)
        
        if model_path is not None:
            bc = d3rlpy.load_learnable(model_path)
            self.learned_model = bc
            return True
        else:
            return False


    def train_model(self, obs_list, action_list, next_obs_list, reward_list, end_epi_list, done_list, 
                    batch_size=4096, n_steps_per_epoch=1000, ):
        # ensemble model
        # obs + 1 (action)
        # NOTE: assume dim(action)==1, dim(reward)==1
        exp_name = formulate_param_to_name(self.model_param_dict)
        logger_adapter = d3rlpy.logging.CombineAdapterFactory([
            d3rlpy.logging.FileAdapterFactory(root_dir=os.path.join(self.model_root_path, "d3rlpy_logs")),
            d3rlpy.logging.TensorboardAdapterFactory(root_dir=os.path.join(self.model_root_path, "tensorboard_logs"))])

        observation_signature = Signature(
        dtype=[np.float32],
        shape=[obs_list[0].shape[0] + 1],  # type: ignore
        )
        # next_obs_len + 1 (reward)
        action_signature = Signature(
            dtype=[np.float32],
            shape=[obs_list[0].shape[0] + 1],  # type: ignore
        )
        reward_signature = Signature(dtype=[np.dtype(np.float32)], shape=[[1]])
        buffer = InfiniteBuffer()
        train_dataset = ReplayBuffer(
            buffer,
            env=self.env,
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
            total_reward = reward_list[i]
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
        samples = train_dataset.sample_transition_batch(train_dataset.transition_count)
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
        bc = EnsembleBCConfig(batch_size=batch_size, learning_rate=1e-5, 
                    encoder_factory=encoder_factory, observation_scaler=obs_scaler, 
                    action_scaler=action_scaler).create(device=self.device)

        bc.fit(train_dataset, n_steps=int(50_000), n_steps_per_epoch=n_steps_per_epoch, save_interval=10, 
            evaluators={'test_error': EnsembleDatasetErrorEvaluator(test_dataset)}, 
            logger_adapter=logger_adapter, experiment_name=exp_name)
        self.learned_model = bc
        
    def T_R_uncertainty(self, obs, action):
        if len(obs.shape) == 1:
            obs = obs[np.newaxis, ...]
            action = action[np.newaxis, ...]
        if len(action.shape) == 1:
            action = action[..., np.newaxis]
        if self.skip_uncertainty:
            return np.zeros((obs.shape[0])), np.zeros((obs.shape[0]))
        else:
            u = self.learned_model.predict_uncertainty(np.concatenate([obs, action], axis=-1))
            u_t = np.mean(u[..., :-1], axis=-1)
            u_r = u[..., -1]
        return u_t, u_r

    def unc_rew(self, u_t, u_r):
        return self.coef_t * u_t + self.coef_r * u_r