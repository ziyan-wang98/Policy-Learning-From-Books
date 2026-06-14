
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()
TIZERO_AGENT_PATH = Path(
    os.environ.get("PLFB_TIZERO_AGENT_PATH", PROJECT_ROOT / "setup" / "TiZero" / "submission" / "tizero_agent")
)
for path in (TIZERO_AGENT_PATH, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from llm.algo.rule_base_2.gfootball import agent as rule_based_agent_2

import numpy as np
import d3rlpy
from learning.utils import npz_extractor, update_stack_img_obs, env_list_creator, ensure_gfootball_working_dir
from d3rlpy.metrics import EnvironmentEvaluator
import json
import argparse
from typing import Optional

import llm.config.baseline_agent_parser as baseline_agent_parser
from llm.utils.obs2text import imaginary_data_observation, imaginary_data_to_vector

from gfootball.env.wrappers import Simple115StateWrapper_ball_owned_player

IRL_LOG_ROOT = Path(os.environ.get("PLFB_IRL_LOG_ROOT", PROJECT_ROOT / "IRL_LOG")).resolve()
EVAL_OUTPUT_DIR = Path(os.environ.get("PLFB_EVAL_OUTPUT_DIR", PROJECT_ROOT / "learning" / "eval_result")).resolve()

DEFAULT_ENV_NAME_LIST = [f'11_vs_11_level_{i}' for i in [0, 1, 2]]


def path_param(
    path: str,
    key: str,
    default: Optional[str] = None,
    strip_timestamp: bool = False,
) -> Optional[str]:
    marker = f"{key}="
    if marker not in path:
        return default
    value = path.split(marker, 1)[1].split("&", 1)[0]
    if strip_timestamp:
        value = value.split("_", 1)[0]
    return value

def get_eval_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--eval_times", type=int, default=10, help="number of evaluation times")
    parser.add_argument("--exp_name", type=str, default='comment=normob-lr-ent-0.4&pi_alg_type=CIQL&strategy=all_replaced&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&lr_rescale=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240312162327', )
    parser.add_argument("--eval_algo", type=str, default='CIQL', help="algorithm name")
    parser.add_argument(
        "--model_root_path",
        type=str,
        default=os.environ.get("PLFB_EVAL_MODEL_ROOT", str(IRL_LOG_ROOT / "football" / "d3rlpy_logs")),
        help="Directory containing d3rlpy experiment folders.",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default=str(EVAL_OUTPUT_DIR),
        help="Directory for evaluation JSON outputs.",
    )
    parser.add_argument(
        "--eval_env_names",
        type=str,
        default=",".join(DEFAULT_ENV_NAME_LIST),
        help="Comma-separated football env names to evaluate.",
    )
    parser.add_argument(
        "--min_model_step",
        type=int,
        default=40000,
        help="Ignore CIQL checkpoints earlier than this step.",
    )
    parser.add_argument(
        "--obs_stack_num",
        type=int,
        default=4,
        help="Fallback observation stack count when the checkpoint path does not encode obs_stack_num.",
    )
    parser.add_argument(
        "--acs_replace_strategy",
        type=str,
        default="all_replaced",
        help="Fallback action-replacement strategy when the checkpoint path does not encode strategy.",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args_parser = baseline_agent_parser.parse_args(return_parser=True)
    args = get_eval_args(args_parser)
    exp_times = args.eval_times
    ensure_gfootball_working_dir()
    env_name_list = [name.strip() for name in args.eval_env_names.split(",") if name.strip()]
    if not env_name_list:
        raise ValueError("--eval_env_names did not contain any environment names")
    if args.eval_algo == 'CIQL':
        model_root_path = args.model_root_path
        model_path = None
        best_model_perf = -np.inf
        # 1: load the model
        # exp_name_pattern = 'comment=alpha-test-inzone-acs-v2&pi_alg_type=CIQL&strategy=parts_keeped&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240309005541'
        # exp_name_pattern = 'comment=alpha-test-inzone-acs&pi_alg_type=CIQL&strategy=all_replaced&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240309002959'
        # exp_name_pattern = 'comment=alpha-test-inzone-acs&pi_alg_type=CIQL&strategy=all_replaced&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240309003000'
        exp_name_pattern = args.exp_name #'comment=normob-lr-ent-0.9&pi_alg_type=CIQL&strategy=all_replaced&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&lr_rescale=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240309205952'
        if not os.path.isdir(model_root_path):
            raise ValueError(f"model root not found: {model_root_path}")
        for files in os.listdir(model_root_path):
            if files.startswith(exp_name_pattern):
                # load the newest model
                best_perf = -np.inf
                best_perf_model_path = None
                experiment_dir = os.path.join(model_root_path, files)
                for model_file in os.listdir(experiment_dir):
                    if model_file.endswith('.d3'):
                        try:
                            cur_perf = float(model_file.split('model_rew_')[1].split('&')[0])
                            step = int(model_file.split('step_')[1].split('.')[0])
                        except (IndexError, ValueError):
                            continue
                        if step < args.min_model_step:
                            continue
                        # if step == 140000:
                        #     best_perf_model_path = model_file
                        #     break
                        if cur_perf > best_perf:
                            best_perf = cur_perf
                            best_perf_model_path = model_file

                if best_perf_model_path is None:
                    continue
                if best_perf > best_model_perf:
                    best_model_perf = best_perf
                    model_path = os.path.join(experiment_dir, best_perf_model_path)
                # parameters = json.load(open(os.path.join(model_root_path, files, 'params.json')))
                # args = argparse.Namespace(**parameters)

        if model_path is not None:
            print('load path', model_path)
            pi = d3rlpy.load_learnable(model_path)
        else:
            raise ValueError(
                f"model not found: root={model_root_path}, pattern={exp_name_pattern}, "
                f"min_model_step={args.min_model_step}"
            )
    elif args.eval_algo == 'rule_based':
        pi = rule_based_agent_2
    elif args.eval_algo == 'llm_agent':
        pass
    elif args.eval_algo == 'llm_rag':
        pass
    elif args.eval_algo == 'RT2':
        pass
    else:
        raise ValueError("algorithm not supported")

    # 2. initialize functions and parametrs
    args.write_full_episode_dumps = True
    eval_env_list = env_list_creator(env_name_list, args)


    if args.eval_algo == 'CIQL':
        check_param_dict = {
            'obs_stack_num':  int(path_param(model_path, 'obs_stack_num', str(args.obs_stack_num), strip_timestamp=True)),
            'strategy': path_param(model_path, 'strategy', args.acs_replace_strategy),
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
                                            stack_obs_len=stack_obs_len, n_trials=exp_times, acs_replace_strategy=check_param_dict['strategy'])
        res_dict = env_evaluator(pi, None)
        print(res_dict)
        for env in eval_env_list:
            env.close()

    elif args.eval_algo == 'rule_based':
        env_evaluator = EnvironmentEvaluator(eval_env_list, n_trials=exp_times,rule_based = True)
        res_dict = env_evaluator(pi, None)
        print(res_dict)
        for env in eval_env_list:
            env.close()
    elif args.eval_algo == 'llm_agent':
        env_evaluator = EnvironmentEvaluator(eval_env_list, n_trials=exp_times,llm = True, llm_version='llm_agent')
        res_dict = env_evaluator(None, None)
        print(res_dict)
        for env in eval_env_list:
            env.close()
    elif args.eval_algo == 'llm_rag':
        env_evaluator = EnvironmentEvaluator(eval_env_list, n_trials=exp_times,llm = True, llm_version='llm_rag')
        res_dict = env_evaluator(None, None)
        print(res_dict)
        for env in eval_env_list:
            env.close()
    elif args.eval_algo == 'RT2':
        env_evaluator = EnvironmentEvaluator(eval_env_list, n_trials=exp_times,llm=True, llm_version='RT2')
        res_dict = env_evaluator(None, None)
        print(res_dict)
        for env in eval_env_list:
            env.close()

    # save the res_dict
    if res_dict is not None:
        save_path = args.eval_output_dir
        os.makedirs(save_path, exist_ok=True)
        import time
        # time in the format of YYYYMMDDHHMMSS
        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        file_name=f'{args.eval_algo}_eval_res_{args.exp_name}_{exp_times}_{time_str}.json'
        res_dict_path = os.path.join(save_path, file_name)
        json.dump(res_dict, open(res_dict_path, 'w'), indent=4)
        print(f"res_dict saved to {res_dict_path}")

