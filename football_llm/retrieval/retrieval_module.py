
import sys
sys.path.append("football_llm/TiZero/submission/tizero_agent")
sys.path.append('football_llm')
import os
import random
import numpy as np
from llm.utils.llama_index_compat import (
    BaseReader,
    HuggingFaceEmbedding,
    JSONQueryEngine,
    KeywordTableIndex,
    OpenAI,
    PromptTemplate,
    Replicate,
    SimpleDirectoryReader,
    VectorStoreQueryMode,
    default_openai_embedding,
    index_from_documents,
    make_document,
    make_service_context,
    set_global_tokenizer,
)

from book_scripts.utils import load_jsonl_list
from llm.generate_main import sample_data, start_point_picker_v2, load_inner_npz_by_index
from llm.utils.obs2text import imaginary_data_observation_v2

def get_docs(file_path):

    ### load the context from the tutorial
    # |-filter_path
    #   |- policy
    #       |- policy.jsonl
    #   |- reward
    #       |- reward.jsonl
    #   |- transition
    #       |- transition.jsonl

    assert os.path.exists(file_path)

    policy_path = os.path.join(file_path, "policy")
    reward_path = os.path.join(file_path, "reward")
    trainsition_path = os.path.join(file_path, "transition")

    # load the data
    class JsonFileReader(BaseReader):
        def load_data(self, file, extra_info=None):
            json_list = load_jsonl_list(file)
            doc_list = []
            for res in json_list:
                doc_list.append(make_document(text=str(res), extra_info=extra_info or {}))
            # load_data returns a list of Document objects
            return doc_list
    policy_doc = SimpleDirectoryReader(policy_path, file_extractor={'.jsonl': JsonFileReader()}).load_data()
    reward_doc = SimpleDirectoryReader(reward_path, file_extractor={'.jsonl': JsonFileReader()}).load_data()
    transition_doc = SimpleDirectoryReader(trainsition_path, file_extractor={'.jsonl': JsonFileReader()}).load_data()
    return policy_doc, reward_doc, transition_doc

def get_index(encorder_type, filter_path, index_type='vector'):
    if encorder_type == "defult":
        embed_model =  default_openai_embedding()
    elif encorder_type == "custom":
        # embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")
    else:
        raise NotImplementedError

    service_context_policy = make_service_context(embed_model=embed_model)
    service_context_reward = make_service_context(embed_model=embed_model)
    service_context_transition = make_service_context(embed_model=embed_model)

    policy_doc, reward_doc, transition_doc = get_docs(filter_path)
    if index_type == 'vector':
        policy_index = index_from_documents(policy_doc, service_context=service_context_policy)
        reward_index = index_from_documents(reward_doc, service_context=service_context_reward)
        transition_index = index_from_documents(
            transition_doc, service_context=service_context_transition
        )
    else:
        # Summary Index

        raise NotImplementedError
    return policy_index, reward_index, transition_index


def get_retrievers(encorder_type, filter_path, similarity_top_k=10, query_mode=VectorStoreQueryMode.DEFAULT):
    policy_index, reward_index, transition_index = get_index(encorder_type, filter_path)
    policy_retriever = policy_index.as_retriever(similarity_top_k=similarity_top_k, query_mode=query_mode)
    reward_retriever = reward_index.as_retriever(similarity_top_k=similarity_top_k, query_mode=query_mode)
    transition_retriever = transition_index.as_retriever(similarity_top_k=similarity_top_k, query_mode=query_mode)
    return policy_retriever, reward_retriever, transition_retriever


def gen_code_test(obs, policy_retriever, reward_retriever, transition_retriever, gpt_model=None):
    gpt_model = gpt_model or os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini')

    obs = json.dumps(obs)

    policy_nodes = policy_retriever.retrieve(obs)
    policy_context_str = "\n\n".join([n.node.get_content() for n in policy_nodes])
    policy_context_str = find_first_code_block(policy_context_str)


    reward_nodes = reward_retriever.retrieve(obs)
    reward_context_str = "\n\n".join([n.node.get_content() for n in reward_nodes])
    reward_context_str = find_first_code_block(reward_context_str)
    reward_path = os.environ.get('PLFB_REWARD_FINETUNE_JSONL', os.path.join(os.environ.get('PLFB_ARTIFACT_ROOT', 'plfb_artifacts'), 'book_derived', 'finetune', 'reward', 'reward.jsonl'))
    reward_length = get_jsonl_length(reward_path)
    reward_random_index = random.randint(0, reward_length - 1)
    reward_context_str = json.dumps(read_nth_jsonl(reward_path, reward_random_index))

    # transition_nodes = transition_retriever.retrieve(obs)
    # transition_context_str = "\n\n".join([n.node.get_content() for n in transition_nodes])
    # transition_context_str = find_first_code_block(transition_context_str)
    transition_json_path = os.environ.get('PLFB_TRANSITION_FINETUNE_JSONL', os.path.join(os.environ.get('PLFB_ARTIFACT_ROOT', 'plfb_artifacts'), 'book_derived', 'finetune', 'transition', 'transition.jsonl'))
    transition_length = get_jsonl_length(transition_json_path)
    transition_random_index = random.randint(0, transition_length - 1)
    transition_context_str = json.dumps(read_nth_jsonl(transition_json_path, transition_random_index))

    current_obs_prompt = ACTION_P_STEP_1_all_code.format(transition_string = transition_context_str, reward_string = reward_context_str, policy_string = policy_context_str, text_obs=obs)
    print("start to get code")
    for attempt in range(MAX_ATTAMPTS):
        try:
            code_response = openai_chat_query("", current_obs_prompt, model_name=gpt_model, req_json=True)
            code_json_load = json.loads(code_response)
            code = json.loads(code_json_load["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
            policy_code = code["policy_code"]
            reward_code = code["reward_code"]
            transition_code= code["transition_code"]
            code_thought = ""
            break
        except Exception as e:
            print(colored(f"Get Code Error: {e}, retrying...", "red"))
            print("code_json_load: ", code_json_load)
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue
    print("end to get code")
    return policy_code,reward_code, transition_code, code_thought

import sys



if __name__ == "__main__":
    print("start to test")

    encorder_type = "defult"
    llm_type = "gpt-4"
    filter_path = os.environ.get("PLFB_FILTER_PATH", os.path.join(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts"), "book_derived", "retrieval"))
    policy_retriever, reward_retriever, transition_retriever = get_retrievers(encorder_type, filter_path, similarity_top_k=100)
    print("constructed to retrievers")
    file_name = sample_data(os.environ.get("PLFB_OFFLINE_DATASET_PATH", os.path.join(os.environ.get("PLFB_DATASET_PATH", os.path.join(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts"), "football")), "offline_dataset-v4")))
    offline_data = np.load(file_name, allow_pickle=True)

    number = 5
    # start_point_list = start_point_picker(offline_data, interval=10)
    assert number < len(offline_data) , "number should be smaller than the number of files in the offline dataset"
    start_point_list = start_point_picker_v2(offline_data, number, interval=100)
    randomized_start_points = random.sample(start_point_list, len(start_point_list))

    import json
    import pprint
    for i in range(len(start_point_list)):
        start_point = start_point_list[i]
        if start_point[0] != 5:
            break
        print("Current start point: ", start_point, "Current index: ", i , "Total: ", len(start_point_list))
        offline_d = load_inner_npz_by_index(offline_data, start_point[0])
        current_obs = offline_d['obs_before_modified_by_acs'][start_point[1]][0]
        current_obs = imaginary_data_observation_v2(current_obs, start_point[1], ret_type='dict')
        obs_str = json.dumps(current_obs)
        policy_res = policy_retriever.retrieve(obs_str)
        print(pprint.pprint(current_obs))
        print("==== best ==== ")
        [pprint.pprint(eval(res.get_content())) for res in policy_res[:10]]
        print("==== worest ==== ")
        [pprint.pprint(eval(res.get_content())) for res in policy_res[-10:]]
