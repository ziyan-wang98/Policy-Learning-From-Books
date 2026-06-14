
import sys
sys.path.append("football_llm/TiZero/submission/tizero_agent")
sys.path.append('football_llm')
from CONFIG import KnowledgeType, OFFLINE_DATASET_PATH
import json
import re
import tqdm
from retrieval.prompt import *
from book_scripts.utils import query
import os
import numpy as np

from llm.utils.llama_index_compat import (
    HuggingFaceEmbedding,
    StorageContext,
    TextNode,
    default_openai_embedding,
    index_from_nodes,
    load_index_from_storage,
    make_service_context,
    set_index_service_context,
)

class KnowledgeRetrieval:
    def __init__(self, corpus, state_field_saved_path, summarize_len=4, model_name=os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
                 embed_model='baai', eval_strategy='state_field', knowledge_type=KnowledgeType.Policy,
                 top_k=15, device='cuda', env_name='football'):
        self.summarize_len = summarize_len
        self.state_field_saved_path = state_field_saved_path
        self.corpus = corpus
        self.model_name = model_name
        self.embed_model = embed_model
        self.eval_strategy = eval_strategy
        self.knowledge_type = knowledge_type
        self.top_k = top_k
        self.device = device
        self.env_name = env_name
        if self.eval_strategy != 'embedding':
            self._construct_state_field()
        else:
            self.corpus_space_dict = {}
        self._construct_index()

    def _construct_index(self):
        if self.embed_model == 'openai':
            embed_model =  default_openai_embedding()
        elif self.embed_model == 'baai':
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en", device=self.device) # 0.06593406593406594
        else:
            embed_model = self.embed_model

        service_context = make_service_context(embed_model=embed_model)

        corpus_index_dir = os.path.join(self.state_field_saved_path, self.knowledge_type, f'{self.model_name}-emb-{embed_model.model_name}-stra-{self.eval_strategy}-len-{self.summarize_len}-index')
        if os.path.exists(corpus_index_dir):
            storage_context = StorageContext.from_defaults(persist_dir=corpus_index_dir)
            index = load_index_from_storage(storage_context)
            set_index_service_context(index, service_context)
        else:
            if self.eval_strategy == 'embedding':
                nodes = [TextNode(id_=id_, text=str(text)) for id_, text in self.corpus.items()]
            elif self.eval_strategy == 'summary':
                nodes = [TextNode(id_=id_, text=str(text['preferred scope description'])) for id_, text in self.corpus_space_dict.items()]
            else:
                nodes = [TextNode(id_=id_, text="The best-fitted observation scope: \n " + str(text)) for id_, text in self.corpus_space_dict.items()]
            print("model", embed_model)
            print("num of nodes", len(nodes))
            index = index_from_nodes(nodes, service_context=service_context, show_progress=True)
            index.storage_context.persist(persist_dir=corpus_index_dir)
        if self.env_name == 'football':
            self.prompt_template = retrieve_prompt_template(self.knowledge_type)
        elif self.env_name == 'tictactoe':
            self.prompt_template = retrieve_prompt_template(self.knowledge_type, env_name=self.env_name)
        else:
            raise NotImplementedError(f"Unknown environment: {self.env_name}")
        # print("prompt_template", self.prompt_template)
        retriever = index.as_retriever(similarity_top_k=self.top_k)
        self.retriever = retriever

    def _construct_state_field(self):
        corpus_space_dict = {}
        corpus_values = list(self.corpus.values())
        corpus_keys = list(self.corpus.keys())
        def extract_numbers(s):
            pattern = r'\d+'
            numbers = re.findall(pattern, s)
            return numbers
        corpus_space_data_path = os.path.join(self.state_field_saved_path, self.knowledge_type, f'{self.model_name}-len-{self.summarize_len}-corpus-space.json')
        if os.path.exists(corpus_space_data_path):
            corpus_space_dict = json.load(open(corpus_space_data_path, 'r'))
        if len(corpus_space_dict) != len(self.corpus):
            counter = 0
            for cor_base_idx in tqdm.tqdm(range(int(len(self.corpus.values())/self.summarize_len)+1)):
                print(counter, len(corpus_space_dict))
                cors = corpus_values[cor_base_idx*self.summarize_len:(cor_base_idx+1)*self.summarize_len]
                sub_keys = corpus_keys[cor_base_idx*self.summarize_len:(cor_base_idx+1)*self.summarize_len]
                find_new = False
                for k in sub_keys:
                    if corpus_space_dict.get(k) is None:
                        find_new = True
                if not find_new:
                    counter += 1
                    continue

                if self.env_name == 'football':
                    res_prompt = code_state_action_space_gen_prompt(cors)
                elif self.env_name == 'tictactoe':
                    res_prompt = code_state_action_space_gen_prompt(cors, env_name=self.env_name)
                else:
                    raise NotImplementedError(f"Unknown environment: {self.env_name}")

                query_res = query(res_prompt, '', self.model_name, req_json=True, print_global_prompt=True)
                retry_times = 0
                while query_res is None or len(query_res.values()) != len(sub_keys):
                    query_res = query(res_prompt, '', self.model_name, req_json=True, print_global_prompt=True)
                    retry_times+=1
                    if retry_times > 2:
                        break
                if retry_times <= 2:
                    for k, v in query_res.items():
                        try:
                            number = int(extract_numbers(k)[0])
                            corpus_space_dict[sub_keys[number-1]] = v
                        except Exception as e:
                            print("error", e, k, v)
                counter += 1
            os.makedirs(os.path.dirname(corpus_space_data_path), exist_ok=True)
            json.dump(corpus_space_dict, open(corpus_space_data_path, 'w'), indent=4)
        self.corpus_space_dict = corpus_space_dict

    def retrieve_knowledge(self, observation, ret_type='text', randomized=True):
        retrieved_nodes = self.retriever.retrieve(self.prompt_template + observation)

        if self.eval_strategy == 'random':
            retrieved_ids = np.random.choice(list(self.corpus_space_dict.keys()), self.top_k, replace=False)
        else:
            retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        if randomized:
            # randomized the order
            np.random.shuffle(retrieved_ids)
        if ret_type == 'text':
            retrieved_texts = [self.corpus[id_] for id_ in retrieved_ids]
            return retrieved_texts
        elif ret_type == 'space':
            retrieved_space = [self.corpus_space_dict[id_] for id_ in retrieved_ids]
            return retrieved_space
        elif ret_type == 'ids':
            return retrieved_ids
        else:
            raise NotImplementedError(f"Unknown return type: {ret_type}")

