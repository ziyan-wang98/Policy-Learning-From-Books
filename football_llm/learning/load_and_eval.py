

import sys
sys.path.append("football_llm/TiZero/submission/tizero_agent")
sys.path.append('football_llm')

import numpy as np
import d3rlpy
import os
from learning.utils import npz_extractor, update_stack_img_obs, env_list_creator
from d3rlpy.metrics import EnvironmentEvaluator
import json
import argparse

import llm.config.baseline_agent_parser as baseline_agent_parser
from llm.utils.obs2text import imaginary_data_observation, imaginary_data_to_vector

from gfootball.env.wrappers import Simple115StateWrapper_ball_owned_player

env_name_list = [f'11_vs_1_level_{i}_mid' for i in [0]]  # [f'11_vs_11_level_{i}' for i in [2]]\
    #   \
    #   + [f'11_vs_1_level_{i}_right' for i in [0, 1, 2]] \
    #   + [f'11_vs_1_level_{i}_mid' for i in [0, 1, 2]] 
# env_name_list =  [f'11_vs_1_level_{i}_left' for i in [0]]

def get_eval_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--eval_times", type=int, default=60, help="number of evaluation times")
    parser.add_argument("--epsilon", type=float, default=0.0, help="eps")
    parser.add_argument("--exp_name", type=str, default='comment=normob-lr-ent-0.4&pi_alg_type=CIQL&strategy=all_replaced&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&lr_rescale=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240312162327', )
    return parser.parse_args()
    # epsilon


if __name__ == '__main__':
    args_parser = baseline_agent_parser.parse_args(return_parser=True)
    args = get_eval_args(args_parser)

    model_root_path = 'football_llm/IRL_LOG/d3rlpy_logs'
    model_path = ''
    # 1: load the model
    # exp_name_pattern = 'comment=alpha-test-inzone-acs-v2&pi_alg_type=CIQL&strategy=parts_keeped&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240309005541'
    # exp_name_pattern = 'comment=alpha-test-inzone-acs&pi_alg_type=CIQL&strategy=all_replaced&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240309002959'
    # exp_name_pattern = 'comment=alpha-test-inzone-acs&pi_alg_type=CIQL&strategy=all_replaced&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240309003000'
    exp_name_pattern = args.exp_name #'comment=normob-lr-ent-0.9&pi_alg_type=CIQL&strategy=all_replaced&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&lr_rescale=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240309205952'
    exp_times = args.eval_times
    eps = args.epsilon
    

    for files in os.listdir(model_root_path):
        if files.startswith(exp_name_pattern):
            # load the newest model
            best_perf = -np.inf
            best_perf_model_path = None
            for model_file in os.listdir(os.path.join(model_root_path, files)):
                if model_file.endswith('.d3'):
                    cur_perf = float(model_file.split('model_rew_')[1].split('&')[0])
                    step = int(model_file.split('step_')[1].split('.')[0])
                    if step < 40000:
                        continue
                    # if step == 140000:
                    #     best_perf_model_path = model_file
                    #     break
                    if cur_perf > best_perf:
                        best_perf = cur_perf
                        best_perf_model_path = model_file
            
            model_path = os.path.join(model_root_path, files, best_perf_model_path)
            print('load path', model_path)
            parameters = json.load(open(os.path.join(model_root_path, files, 'params.json')))
            args = argparse.Namespace(**parameters)    

    if model_path is not None:
        pi = d3rlpy.load_learnable(model_path)
    else:
        raise ValueError("model not found, pattern", model_root_path, exp_name_pattern)
    
    # 2. initialize functions and parametrs

    args.eval_algo = 'CIQL'
    if eps == 1.0:
        args.eval_algo = 'random'
        
        
    # args.video_dir = None
    # args.write_goal_dumps = False
    # args.write_full_episode_dumps = False
    # args.render = False
    # args.num_players = 1
    
    eval_env_list = env_list_creator(env_name_list, args)
    check_param_dict = {
        'obs_stack_num':  int(model_path.split('obs_stack_num=')[1].split('&')[0].split('_')[0]),
        'strategy': model_path.split('strategy=')[1].split('&')[0],
    }

    update_stack_obs = lambda obs, stack_obs: update_stack_img_obs(obs, stack_obs, check_param_dict['obs_stack_num'])
    wrapper_func = Simple115StateWrapper_ball_owned_player
    wrapper = wrapper_func(eval_env_list[0])
    raw_obs = eval_env_list[0].reset()
    wrap_obs = wrapper.observation(raw_obs)
    obs_sample = imaginary_data_observation(wrap_obs[0], raw_obs[0], 0, ret_type='vector', TODO_missing=True)
    stack_obs_len = check_param_dict['obs_stack_num'] * obs_sample.shape[-1]

    # 3. initialize evaluator and evalute
    env_evaluator = EnvironmentEvaluator(eval_env_list, obs_type='imginary_obs', update_stack_obs=update_stack_obs, 
                                         stack_obs_len=stack_obs_len, n_trials=exp_times, 
                                         acs_replace_strategy='no', epsilon=eps)
    res_dict = env_evaluator(pi, None)
    print(res_dict)
    for env in eval_env_list:
        env.close()