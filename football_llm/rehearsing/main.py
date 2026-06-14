import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()
TIZERO_AGENT_PATH = Path(
    os.environ.get("PLFB_TIZERO_AGENT_PATH", PROJECT_ROOT / "setup" / "TiZero" / "submission" / "tizero_agent")
)
for path in (TIZERO_AGENT_PATH, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import argparse
import os.path as osp
import pprint
import time
import random
import tqdm
import numpy as np
import json
from termcolor import colored
import uuid
from collections import deque

from funcs import load_jsonl_list, npz_extractor, json_str_clean
import simulator_info
from CONFIG import KnowledgeType, OFFLINE_DATASET_PATH
from retrieval.state_filed_retrieval import KnowledgeRetrieval
from rehearsing.utils import *
from rehearsing.llm_mdp import FootBallRolloutMDP
from rehearsing.prompt import code_instantiation_prompt
from llm.utils.rewarder import Rewarder
from llm.utils.openai_compat import openai_chat_query
from llm.utils.buildin_ai import obs_to_robot_acs, update_stack_obs
from llm.config.gen_main_parser import raw_policy_prompt_step1_all_code
from llm.utils.obs2text import imaginary_data_observation_v2, imaginary_data_observation, imaginary_data_to_vector, img_obs_to_text
from llm.utils.llama_index_compat import HuggingFaceEmbedding, OpenAIEmbedding
from llm.openai_server import OpenAIServer


MAX_ATTAMPTS = 2


def arg_parser():
    parser = argparse.ArgumentParser(description='rehearsing')
    default_data_path = Path(os.environ.get("PLFB_DATASET_PATH", PROJECT_ROOT / "data")) / "v4-gpt-3.5-turbo-1106-level-strict"
    parser.add_argument('--data_path', type=str, default=str(default_data_path))
    parser.add_argument('--gen_traj_index', type=int, default=12)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--gen_length', type=int, default=20)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--embed_model', type=str, default='openai')
    # add bool argument
    parser.add_argument('--just_gen_code', action='store_true')
    parser.add_argument('--skip_knowledge', action='store_true')
    parser.add_argument('--just_eval_error', action='store_true')
    parser.add_argument('--allow_repeat', action='store_true')
    # version_control
    parser.add_argument('--img_data_version', type=str, default='v3')
    return parser.parse_args()


def knowlege_to_corpus(knowledge):
    corpus = {}
    for i, know_item in enumerate(knowledge):
        new_uuid = str(uuid.uuid4())
        corpus[new_uuid] = ''
        for k, v in know_item.items():
            corpus[new_uuid] += f"{k}\n" + '\n'.join(v)
    return corpus


def load_corpus(knowledge_path):
    dirname = osp.dirname(knowledge_path)
    corpus_path = osp.join(dirname, 'corpus.jsonl')
    if osp.exists(corpus_path):
        with open(corpus_path, 'r') as file:
            return json.load(file)

    else:
        knowledge = load_jsonl_list(knowledge_path)
        corpus = knowlege_to_corpus(knowledge)
        with open(corpus_path, 'w') as file:

            json.dump(corpus, file, indent=4)
        return corpus

openai_server_plus = OpenAIServer(model="gpt-4o-2024-05-13", top_p=0.9, temp=0.1, max_token=1500)

def code_instantiation(knowledge_prompt):
    code_res = None
    code_json_load = None
    for attempt in range(MAX_ATTAMPTS):
        try:
            code_response = openai_server_plus.chat(knowledge_prompt)
            try:
                code_response_clean = json_str_clean(code_response)
                code_res = json.loads(code_response_clean)
            except json.decoder.JSONDecodeError as e:
                code = code_response_clean.split('"code": "')[1].replace("\\n", "\n")
                analyze = code_response_clean.split('"code": "')[0].replace("\\n", "\n")
                code_res = {"code": code, "analyze": analyze}
            for k, v in code_res.items():
                code_res[k] = v.split('\n')
            break
        except Exception as e:
            print(colored(f"Get Code Error: {e}, retrying...", "red"))
            print("code_json_load: ", code_response)
            continue
    return code_res

model_list = [os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini'), 'qwen2-72b-instruct', 'qwen2-7b-instruct', 'Meta-Llama-3-8B-Instruct', 'deepseek-chat']

# python rehearsing/main.py --gen_model deepseek-coder --gen_traj_index 2 --allow_repeat
# python rehearsing/main.py --gen_model deepseek-coder --gen_traj_index 2 --just_eval_error --img_data_version v2
if __name__ == '__main__':
    # knowledge embedding
    args = arg_parser()
    if args.skip_knowledge:
        args.img_data_version += '-skip'
    # model_map_id = model_list.index(args.gen_model)
    # if model_map_id * 2 <= args.gen_traj_index % (2*len(model_list)) <= model_map_id * 2 + 1:
    #     if ':rep' in args.gen_model:
    #         args.gen_model = args.gen_model.split(':rep')[0]
    #     else:
    #         args.gen_model = args.gen_model
    #     print("Start to generate code for model: ", args.gen_model, "for this index: ", args.gen_traj_index, args.gen_traj_index % (2*len(model_list)))
    # else:
    #     print("Skip this model", args.gen_model, "for this index: ", args.gen_traj_index, args.gen_traj_index % (2*len(model_list)))
    #     exit()
    policy_corpus = load_corpus(os.path.join(args.data_path, f"{KnowledgeType.Policy}/multi/best/agg-best.jsonl"))
    dynamics_corpus = load_corpus(os.path.join(args.data_path, f"{KnowledgeType.Dynamics}/multi/best/agg-best.jsonl"))
    rewards_corpus = load_corpus(os.path.join(args.data_path, f"{KnowledgeType.Reward}/multi/best/agg-best.jsonl"))
    state_field_saved_path = os.path.join(args.data_path, 'field_v1')
    os.makedirs(state_field_saved_path, exist_ok=True)
    if args.embed_model == 'openai':
        embed_model =  OpenAIEmbedding()
    elif args.embed_model == 'baai':
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en", device='cuda:2') # 0.06593406593406594
    def ret_creator(corpus, knowledge_type):
        return KnowledgeRetrieval(corpus, state_field_saved_path, model_name='gpt-4o-2024-05-13',
                                  knowledge_type=knowledge_type, top_k=args.top_k, device='cuda:2', embed_model=embed_model)
    M_retrieval = ret_creator(dynamics_corpus, KnowledgeType.Dynamics)
    Pi_retrieval = ret_creator(policy_corpus, KnowledgeType.Policy)
    R_retrieval = ret_creator(rewards_corpus, KnowledgeType.Reward)
    # load initial states
    offline_data = sampled_traj_data = get_sampled_data(OFFLINE_DATASET_PATH, load_num=50)
    print("number of valid trajectories: ", len(sampled_traj_data))

    # load start point
    start_point_path = os.path.join(OFFLINE_DATASET_PATH, 'start_points', f'score-normal-only-{args.gen_traj_index}-interval-{args.interval}.json')
    if os.path.exists(start_point_path):
        with open(start_point_path, 'r') as file:
            start_point_list = json.load(file)
    else:
        start_point_list = start_point_picker_v2(sampled_traj_data, int(args.gen_traj_index), interval=args.interval)
        if start_point_list is None:
            print("No start points found, please check the interval and gen_traj_index. It mainly because of the trajectory with all zero rewards.")
            exit()
        # save start points
        os.makedirs(os.path.dirname(start_point_path), exist_ok=True)
        with open(start_point_path, 'w') as file:
            json.dump(start_point_list, file, indent=4)
    print("Number of start points: ", len(start_point_list))

    # load reward
    try:
        rewarder = Rewarder()
    except Exception as e:
        print("Error: ", e)
        print("rewarder init failed, retrying...")
        rewarder = Rewarder()

    bc_models, robot_act_stack_hist, robot_stack_obs_num = load_bc_models()
    # random.shuffle(start_point_list)

    # code instantiation
    code_instantiated_path = os.path.join(OFFLINE_DATASET_PATH, 'instantiated_code', f'traj-{args.gen_traj_index}.jsonl')
    os.makedirs(os.path.dirname(code_instantiated_path), exist_ok=True)
    if os.path.exists(code_instantiated_path):
        with open(code_instantiated_path, 'r') as file:
            code_instantiated_dict = json.load(file)
    else:
        code_instantiated_dict = {}
    for i in tqdm.tqdm(range(len(start_point_list)), desc=f'Instantiating Code-{args.gen_traj_index}'):
        start_point = start_point_list[i]
        if start_point[1] < 0:
            continue
        if str(start_point) in code_instantiated_dict:
            print("skip the start point: ", start_point)
            # print("exist code: ")
            # pprint.pprint(code_instantiated_dict[str(start_point)])
            continue
        offline_d = load_inner_npz_by_index(offline_data, start_point[0])
        current_obs = offline_d['obs_before_modified_by_acs'][start_point[1]][0]
        current_obs = imaginary_data_observation_v2(current_obs, start_point[1], ret_type='dict')
        obs_str = img_obs_to_text(current_obs)
        pi_code = Pi_retrieval.retrieve_knowledge(obs_str)
        m_code = M_retrieval.retrieve_knowledge(obs_str)
        r_code = R_retrieval.retrieve_knowledge(obs_str)
        pi_prompt = code_instantiation_prompt(KnowledgeType.Policy, '\n\n\n === another code knowledge: \n\n'.join(pi_code), obs_str)
        m_prompt = code_instantiation_prompt(KnowledgeType.Dynamics, '\n\n\n === another code knowledge: \n\n'.join(m_code), obs_str)
        r_prompt = code_instantiation_prompt(KnowledgeType.Reward, '\n\n\n === another code knowledge: \n\n'.join(r_code), obs_str)
        pi_prompt_code = code_instantiation(pi_prompt)
        m_prompt_code = code_instantiation(m_prompt)
        r_prompt_code = code_instantiation(r_prompt)
        code_instantiated_dict[str(start_point)] = {KnowledgeType.Policy: pi_prompt_code,
                                                    KnowledgeType.Dynamics: m_prompt_code,
                                                    KnowledgeType.Reward: r_prompt_code}
        with open(code_instantiated_path, 'w') as file:
            json.dump(code_instantiated_dict, file, indent=4)
    if args.just_gen_code:
        exit()


    time_str = time.strftime("%Y%m%d-%H%M%S")

    gen_length = args.gen_length

    # hyperparameters
    act_stack_hist_len = int(robot_act_stack_hist)
    state_stack_num = int(robot_stack_obs_num)
    acs_shape = 4 # ([zone_diff, [own_the_ball_state_diff], [ball_direction_diff]])
    bc_player_num = 22
    one_debug = True
    # init dataset
    img_dataset_all = {}



    # img data
    ima_saved_path = os.path.join(OFFLINE_DATASET_PATH, f'imaginary_data-{args.img_data_version}', args.gen_model)
    os.makedirs(ima_saved_path, exist_ok=True)
    avg_rollout_loss = []
    rollout_model = FootBallRolloutMDP(model=args.gen_model, seed=args.seed, skip_knowledge=args.skip_knowledge, simulator_info=simulator_info)
    for i in tqdm.tqdm(range(len(start_point_list)), desc=f'Generating Imaginary Data-{args.gen_traj_index}'):
        start_point = start_point_list[i]
        if start_point[1] < 0:
            continue
        print("Current start point: ", start_point, "Current index: ", i , "Total: ", len(start_point_list))
        offline_d = load_inner_npz_by_index(offline_data, start_point[0])
        current_obs = offline_d['obs_before_modified_by_acs'][start_point[1]][0]
        current_obs = imaginary_data_observation_v2(current_obs, start_point[1], ret_type='dict')
        # constructy ground-truth data
        gt_next_gen_length_obs = offline_d['obs_before_modified_by_acs'][start_point[1] :start_point[1] + 1 + gen_length]
        gt_next_gen_length_actions = offline_d['action'][start_point[1] :start_point[1] + 1 + gen_length]
        gt_next_gen_length_rewards = offline_d['reward'][start_point[1] :start_point[1] + 1 + gen_length]
        gt_next_gen_length_dense_rewards = offline_d['dense_rewards'][start_point[1] :start_point[1] + 1 + gen_length]
        gt_next_gen_length_done = offline_d['done'][start_point[1] :start_point[1] + 1 + gen_length]
        codes = code_instantiated_dict[str(start_point)]
        policy_code, reward_code, transition_code = codes[KnowledgeType.Policy], codes[KnowledgeType.Reward], codes[KnowledgeType.Dynamics]

        im_next_obs_list = []
        im_action_list = []
        im_reward_list = []
        im_dense_reward_list = []
        im_llm_reward_list = []
        im_done_list = []
        im_code_list = []
        im_thought_list = []
        previous_score_diff = current_obs['score'][0] - current_obs['score'][1]
        prev_img_obs = current_obs

        cur_obs_vec = imaginary_data_to_vector(current_obs)
        # 3 is the last 3 timestep obs, 4 is the action length , 10 is the last 10 time steps action

        stack_obs_len = cur_obs_vec.shape[0] * state_stack_num + acs_shape * act_stack_hist_len
        sp1 = start_point[1]
        bc_player_stack_obs = np.zeros([bc_player_num, stack_obs_len])

        bc_player_prev_acs_list = np.zeros([bc_player_num, acs_shape])
        recap_hist_start_point = np.maximum(start_point[1] - act_stack_hist_len, 0)
        past_10_obs = offline_d['obs_before_modified_by_acs'][recap_hist_start_point: start_point[1]+1]

        acs_pred_error_list = [[] for _ in range(bc_player_num)]
        recent_dict_obs = deque(maxlen=state_stack_num + 2)
        # stage1: reconstructed the stacked obs for each robot player.
        for k in range(len(past_10_obs)-1):
            current_obs_for_stack = past_10_obs[k][0]
            current_obs_for_stack = imaginary_data_observation_v2(current_obs_for_stack, start_point[1] - 10 +  k, ret_type='dict')
            recent_dict_obs.append(current_obs_for_stack)
            for j in range(bc_player_num):
                acs = bc_player_prev_acs_list[j]
                stack_obs = bc_player_stack_obs[j]
                obs_vector = imaginary_data_to_vector(current_obs_for_stack, TODO_missing = True)
                stack_obs = update_stack_obs(obs_vector, acs, stack_obs, state_stack_num=state_stack_num)
                next_obs = imaginary_data_observation_v2(past_10_obs[k+1][0], start_point[1] -10 + k + 1, ret_type='dict')
                acs = obs_to_robot_acs(next_obs, current_obs_for_stack, j)
                bc_player_prev_acs_list[j] = acs
                bc_player_stack_obs[j] = stack_obs
                if one_debug:
                    acs_pred_error_list[j].append((np.abs(bc_models[j].predict(np.array([stack_obs])) - acs)))

        if one_debug:
            for j in range(bc_player_num):
                player_error = np.mean(acs_pred_error_list[j], axis=0)
                print("Player: ", j, "Error: ", player_error)
            error_all = np.mean(acs_pred_error_list)
            print("ALL Error: ", error_all)
            one_debug = False

        # stage2: generate the next H steps
        img_dataset_start_point = []
        img_dataset_current_obs = []

        img_dataset_gt_next_obs = []
        img_dataset_gt_next_actions = []
        img_dataset_gt_next_rewards = []
        img_dataset_gt_next_dense_rewards = []
        img_dataset_gt_next_done = []

        img_dataset_im_next_obs = []
        img_dataset_im_next_actions = []
        img_datset_im_reward = []
        img_datset_im_dense_reward = []
        img_datset_im_llm_reward = []
        img_dataset_im_done = []


        img_dataset_policy_code = []
        img_dataset_reward_code = []
        img_dataset_transition_code = []
        img_dataset_codes = []
        img_dataset_gen_times = []
        log_loss_list = []
        im_next_obs_list.append(current_obs)
        traj_name = f'no_{args.gen_traj_index}_imaginary_dataset_{start_point[1]}-{gen_length}-{args.seed}'
        file_name = os.path.join(ima_saved_path, traj_name + '.npz')
        exist_data = False
        for root, dirs, files in os.walk(os.path.dirname(ima_saved_path)):
            for file in files:
                if traj_name in file:
                    load_data = np.load(os.path.join(root, file), allow_pickle=True)
                    if 'im_next_obs' in load_data:
                        print("existing traj", os.path.join(root, file))
                    else:
                        continue

                    if args.gen_model in root:
                        if not args.allow_repeat:
                            exist_data = True
                        # compute the log loss
                        for j in range(gen_length):
                            log_vector_next_obs = imaginary_data_to_vector(load_data['im_next_obs'][0][j])
                            log_vector_gt_next_obs = imaginary_data_observation_v2(load_data['gt_next_obs'][0][j][0], start_point[1] + j , ret_type='vector')
                            log_loss = np.abs(log_vector_next_obs - log_vector_gt_next_obs)
                            log_loss_list.append(log_loss)
                        avg_rollout_loss.append(log_loss_list)
                        print(f"\n -- {gen_length} steps log loss:")
                        print("dim error", np.mean(avg_rollout_loss, axis=(0,1)))
                        print("rollout step error", np.mean(avg_rollout_loss, axis=(0, 2)))

        if args.just_eval_error or exist_data:
            print("========= Skip this start point =======")
            continue

        for j in tqdm.tqdm(range(gen_length), desc=f'Generating Single Rollout-sp-{start_point}'):
            # stack obs
            recent_dict_obs.append(current_obs)
            other_player_action = []
            for k in range(22):
                acs = bc_player_prev_acs_list[k]
                bc = bc_models[k]
                obs_vector = imaginary_data_to_vector(current_obs)
                stack_obs = bc_player_stack_obs[k]
                stack_obs = update_stack_obs(obs_vector, acs, stack_obs, state_stack_num=state_stack_num)
                acs_pred = bc.sample_action(np.array([stack_obs]))
                other_player_action.append(acs_pred)
                bc_player_stack_obs[j] = stack_obs

            im_next_obs, im_action, im_llm_reward, thought_dict = rollout_model.rollout_one_step(recent_dict_obs, policy_code, reward_code, transition_code, other_player_action)
            log_vector_next_obs = imaginary_data_to_vector(im_next_obs)
            log_vector_gt_next_obs = imaginary_data_observation_v2(gt_next_gen_length_obs[j][0], start_point[1] + j , ret_type='vector')
            log_loss = np.abs(log_vector_next_obs - log_vector_gt_next_obs)
            log_loss_list.append(log_loss)
            im_next_obs_list.append(im_next_obs)
            im_action_list.append(im_action)
            im_code_list.append(codes)
            im_thought_list.append(thought_dict)


            # Reward:
            score_diff = im_next_obs['score'][0] - im_next_obs['score'][1]
            reward = score_diff - previous_score_diff
            previous_score_diff = score_diff

            im_reward_list.append(reward)

            # Dense Reward:
            dense_rewards = rewarder.calc_reward_v2(reward, im_next_obs, prev_img_obs, im_action)
            prev_img_obs = im_next_obs
            im_dense_reward_list.append(dense_rewards)
            im_llm_reward_list.append(im_llm_reward)

            # Done:
            if reward !=0:
                done = True
            else:
                done = False

            im_done_list.append(done)

            # update

            for j in range(22):
                acs = obs_to_robot_acs(im_next_obs, current_obs, j)
                bc_player_prev_acs_list[j] = acs
            current_obs = im_next_obs
        print(f"\n -- {gen_length} steps log loss:")
        avg_rollout_loss.append(log_loss_list)
        print("dim error", np.mean(avg_rollout_loss, axis=(0, 1)))
        print("rollout step error", np.mean(avg_rollout_loss, axis=(0, 2)))

        # TODO: simplify these redundant code
        img_dataset_start_point.append(start_point)
        img_dataset_current_obs.append(current_obs)
        img_dataset_gt_next_obs.append(gt_next_gen_length_obs)
        img_dataset_gt_next_actions.append(gt_next_gen_length_actions)
        img_dataset_gt_next_rewards.append(gt_next_gen_length_rewards)
        img_dataset_gt_next_dense_rewards.append(gt_next_gen_length_dense_rewards)
        img_dataset_gt_next_done.append(gt_next_gen_length_done)

        img_dataset_im_next_obs.append(im_next_obs_list)
        img_dataset_im_next_actions.append(im_action_list)
        img_datset_im_reward.append(im_reward_list)
        img_datset_im_dense_reward.append(im_dense_reward_list)
        img_datset_im_llm_reward.append(im_llm_reward_list)
        img_dataset_im_done.append(im_done_list)

        img_dataset_policy_code.append(policy_code)
        img_dataset_reward_code.append(reward_code)
        img_dataset_transition_code.append(transition_code)
        img_dataset_codes.append(im_code_list)
        img_dataset_gen_times.append(1)

        current_wapper_obs = load_inner_npz_by_index(offline_data, start_point[0])['wrapper_obs'][start_point[1]][0]


        img_dataset_all['start_point'] = img_dataset_start_point
        img_dataset_all['current_obs'] = img_dataset_current_obs
        img_dataset_all['gt_next_obs'] = img_dataset_gt_next_obs
        img_dataset_all['gt_next_actions'] = img_dataset_gt_next_actions
        img_dataset_all['gt_reward'] = img_dataset_gt_next_rewards
        img_dataset_all['gt_dense_reward'] = img_dataset_gt_next_dense_rewards
        img_dataset_all['gt_done'] = img_dataset_gt_next_done


        img_dataset_all['im_next_obs'] = img_dataset_im_next_obs
        img_dataset_all['im_next_actions'] = img_dataset_im_next_actions
        img_dataset_all['im_reward'] = img_datset_im_reward
        img_dataset_all['im_dense_reward'] = img_datset_im_dense_reward
        img_dataset_all['im_llm_reward'] = img_datset_im_llm_reward
        img_dataset_all['im_done'] = img_dataset_im_done

        img_dataset_all['im_done'] = img_dataset_im_done
        img_dataset_all['policy_code'] = img_dataset_policy_code
        img_dataset_all['reward_code'] = img_dataset_reward_code
        img_dataset_all['transition_code'] = img_dataset_transition_code
        img_dataset_all['codes'] = img_dataset_codes

        img_dataset_all['gen_times'] = img_dataset_gen_times

        print("Number of imaginary dataset: ", len(img_dataset_all['im_next_obs']))

        # save this to a npz file
        try:
            np.savez_compressed(file_name, **img_dataset_all)
        except Exception as e:
            print("Error: ", e)
            for key in img_dataset_all:
                print("Key: ", key, "Shape: ", np.array(img_dataset_all[key]).shape)
            print("Saving to npz file failed, saving to pkl file")
            import pdb; pdb.set_trace()
            import pickle
            with open(file_name, 'wb') as f:
                pickle.dump(img_dataset_all, f)

