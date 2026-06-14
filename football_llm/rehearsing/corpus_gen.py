import sys
sys.path.append("football_llm/TiZero/submission/tizero_agent")
sys.path.append('football_llm')

import argparse
import os 
import os.path as osp
import pprint
import time
import random
import tqdm
import numpy as np
import json
import copy
from termcolor import colored
import uuid
from collections import deque

from funcs import load_jsonl_list, npz_extractor, json_str_clean

from CONFIG import KnowledgeType, OFFLINE_DATASET_PATH
from retrieval.state_filed_retrieval import KnowledgeRetrieval
from rehearsing.utils import *
from rehearsing.llm_mdp import RolloutMDP
from rehearsing.prompt import code_instantiation_prompt
from llm.utils.rewarder import Rewarder
from llm.utils.openai_compat import openai_chat_query
from llm.utils.buildin_ai import obs_to_robot_acs, update_stack_obs
from llm.config.gen_main_parser import raw_policy_prompt_step1_all_code
from llm.utils.obs2text import imaginary_data_observation_v2, imaginary_data_observation, imaginary_data_to_vector, img_obs_to_text
from llm.utils.llama_index_compat import HuggingFaceEmbedding, OpenAIEmbedding
from llm.openai_server import OpenAIServer
from envs.tictactoe import TicTacToeEnv
from rehearsing.instantiation import CodeInstantiation

MAX_ATTAMPTS = 2


def arg_parser():
    parser = argparse.ArgumentParser(description='rehearsing')
    parser.add_argument('--data_path', type=str, default=os.path.join(os.environ.get('PLFB_TTT_DATA_PATH', os.path.join(os.environ.get('PLFB_ARTIFACT_ROOT', 'plfb_artifacts'), 'tic_tac_toe_data')), 'book_knowledge'))
    parser.add_argument('--gen_traj_index', type=int, default=12)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--gen_length', type=int, default=20)
    parser.add_argument('--top_k', type=int, default=5) # topk = 5 for ttt, 20 for football.
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--embed_model', type=str, default='openai')
    parser.add_argument('--env_name', type=str, default='tictactoe')
    parser.add_argument('--eval_strategy', type=str, default='state_field')
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

openai_server_plus = OpenAIServer(model="gpt-4o", top_p=0.9, temp=0.1, max_token=1500)

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


if __name__ == '__main__':
    # knowledge embedding
    args = arg_parser()
    if args.skip_knowledge:
        args.img_data_version += '-skip'

    policy_corpus = load_corpus(os.path.join(args.data_path, f"{KnowledgeType.Policy}/multi/best/agg-best.jsonl"))
    dynamics_corpus = load_corpus(os.path.join(args.data_path, f"{KnowledgeType.Dynamics}/multi/best/agg-best.jsonl"))
    rewards_corpus = load_corpus(os.path.join(args.data_path, f"{KnowledgeType.Reward}/multi/best/agg-best.jsonl"))
    state_field_saved_path = os.path.join(args.data_path, 'field_v1')
    os.makedirs(state_field_saved_path, exist_ok=True)
    if args.embed_model == 'openai':
        embed_model =  OpenAIEmbedding()
    elif args.embed_model == 'baai':
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en", device='cuda:2') # 0.06593406593406594
    if args.env_name == 'tictactoe':
        env = TicTacToeEnv()
        import ttt_simulator_info as simulator_info
    
    def code_int_creator(corpus, knowledge_type):
        code_instantiated_path = os.path.join(OFFLINE_DATASET_PATH, 'instantiated_code', knowledge_type)
        retv = KnowledgeRetrieval(corpus, state_field_saved_path, model_name='gpt-4o', 
                                  knowledge_type=knowledge_type, top_k=args.top_k, device='cuda:2', 
                                  embed_model=embed_model, env_name=args.env_name, eval_strategy=args.eval_strategy)
        code_init = CodeInstantiation(retv, code_instantiated_path, env.vector_state_to_text, knowledge_type=knowledge_type,
                          task_info=simulator_info.TASK_DESC, element_type_prompt_dict=simulator_info.element_type_prompt_dict,
                          element_type_example_dict=simulator_info.element_type_example_dict, element_type_format_dict=simulator_info.element_type_format_dict,
                          obs_desc=simulator_info.OBSERVATION_SPACE_DESC, acs_desc=simulator_info.ACTION_SPACE_DESC)
        return code_init
    M_code_init = code_int_creator(dynamics_corpus, KnowledgeType.Dynamics)
    Pi_code_init = code_int_creator(policy_corpus, KnowledgeType.Policy)
    R_code_init = code_int_creator(rewards_corpus, KnowledgeType.Reward)