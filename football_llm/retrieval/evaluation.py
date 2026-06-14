
import sys
sys.path.append("football_llm/TiZero/submission/tizero_agent")
sys.path.append('football_llm')
import uuid
from CONFIG import *
import json
import tqdm
import pprint
from book_scripts.utils import *
import re
from retrieval.prompt import *
from llm.utils.llama_index_compat import HuggingFaceEmbedding, OpenAIEmbedding, TextNode, VectorStoreIndex
# from tqdm.notebook import tqdm
import pandas as pd
import argparse
from retrieval.state_filed_retrieval import KnowledgeRetrieval



np.set_printoptions(precision=2)

def arg_parser():
    parser = argparse.ArgumentParser(description='retrieval evaluation')
    parser.add_argument('--eval_strategy', type=str, default='state_field', help='evaluation strategy')
    parser.add_argument('--embed_model', type=str, default='baai', help='embedding model')
    parser.add_argument('--top_k', type=int, default=15, help='top k')
    parser.add_argument('--knowledge_rep', type=str, default='code')
    parser.add_argument('--model_name', type=str, default=os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini'), help='llm model')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    CODE_FOLDER_NAME = f'{args.knowledge_rep}_label-v2'
    # load json
    # construct qa
    # Generate a UUID4.
    code_query_data_path = os.path.join(OFFLINE_DATASET_PATH, CODE_FOLDER_NAME, 'queries-corpus.json')
    if not os.path.exists(code_query_data_path):
        dataset_path = os.path.join(OFFLINE_DATASET_PATH, CODE_FOLDER_NAME, 'code_data_dict.json')
        dataset = json.load(open(dataset_path, 'r'))
        corpus = {}
        queries = {}
        relevant_docs = {}
        embeddings = {}
        threshold = 0.97
        repeat = []
        for traj in tqdm.tqdm(dataset):
            for i in tqdm.tqdm(range(len(traj['texted_obs']))):
                if 'language' in CODE_FOLDER_NAME:
                    code = str(traj['label'][i]['knowledge'])
                else:
                    code = str(traj['label'][i]['code'])
                embedding = get_embedding(code)
                embedding_list = list(embeddings.values())
                embedding_keys = list(embeddings.keys())
                new_uuid = True
                if len(embedding_list) > 0:
                    sim_coef = get_cosine_similarity(embedding_list, embedding)
                    match_idx = np.argsort(sim_coef)[-1]
                    if sim_coef[match_idx] > threshold:
                        code_uuid = embedding_keys[match_idx]
                        code = corpus[code_uuid]
                        new_uuid = False
                        print("max sim docs", np.sort(sim_coef)[-20:], "match id", code_uuid[:-20])
                if new_uuid:
                    code_uuid = uuid.uuid4()
                    corpus[str(code_uuid)] = code
                    embeddings[str(code_uuid)] = embedding
                    pprint.pprint(code, width=400)
                repeat.append(new_uuid)
                query_id = uuid.uuid4()
                queries[str(query_id)] = traj['texted_obs'][i]
                relevant_docs[str(query_id)] = [str(code_uuid)]
                if i % 100 == 0:
                    print('new code rate:', sum(repeat) / len(repeat), "num of queries:", len(queries), "corpus size:", len(corpus))
            # break
        res_dict = {
            'corpus': corpus,
            'queries': queries,
            'relevant_docs': relevant_docs,
            'embeddings': embeddings
        }
        json.dump(res_dict, open(code_query_data_path, 'w'), indent=4)
    else:
        res_dict = json.load(open(code_query_data_path, 'r'))
        corpus = res_dict['corpus']
        queries = res_dict['queries']
        relevant_docs = res_dict['relevant_docs']
        embeddings = res_dict['embeddings']
    # delete short corpus
    corpus_to_delete = []
    for k, v in corpus.items():
        if len(v) <= 100:
            print("delete corpus ", k, v)
            corpus_to_delete.append(k)
    for k in corpus_to_delete:
        del corpus[k]
    query_to_delete = []
    for k, v in relevant_docs.items():   
        if v[0] in corpus_to_delete:
            print("delete query ", k, v)
            query_to_delete.append(k)
    for k in query_to_delete:
        del relevant_docs[k]
        del queries[k]  
                 
            
    # step2: observation field summarization.
    # summarize_len = 4
    # corpus_space_dict = {}
    # corpus_values = list(corpus.values())
    # corpus_keys = list(corpus.keys())
    # def extract_numbers(s):
    #     pattern = r'\d+'
    #     numbers = re.findall(pattern, s)
    #     return numbers
    # 
    # if os.path.exists(corpus_space_data_path):
    #     corpus_space_dict = json.load(open(corpus_space_data_path, 'r'))
    # if len(corpus_space_dict) != len(corpus):
    #     counter = 0
    #     for cor_base_idx in tqdm.tqdm(range(int(len(corpus.values())/summarize_len)+1)):
    #         print(counter, len(corpus_space_dict))
    #         cors = corpus_values[cor_base_idx*summarize_len:(cor_base_idx+1)*summarize_len]
    #         sub_keys = corpus_keys[cor_base_idx*summarize_len:(cor_base_idx+1)*summarize_len]   
    #         find_new = False
    #         for k in sub_keys:
    #             if corpus_space_dict.get(k) is None:
    #                 find_new = True
    #         if not find_new:
    #             counter += 1
    #             continue
    #         res_prompt = code_state_action_space_gen_prompt(cors)
    #         query_res = query(res_prompt, '', model_name, req_json=True, print_global_prompt=True)
    #         retry_times = 0
    #         while query_res is None or len(query_res.values()) != len(sub_keys):
    #             query_res = query(res_prompt, '', model_name, req_json=True, print_global_prompt=True)
    #             retry_times+=1 
    #             if retry_times > 2:
    #                 break
    #         if retry_times <= 2:
    #             for k, v in query_res.items():
    #                 try:
    #                     number = int(extract_numbers(k)[0])
    #                     corpus_space_dict[sub_keys[number-1]] = v
    #                 except Exception as e:
    #                     print("error", e, k, v)
    #         counter += 1
    #     json.dump(corpus_space_dict, open(corpus_space_data_path, 'w'), indent=4)
        
    
    state_field_saved_path = os.path.join(OFFLINE_DATASET_PATH, CODE_FOLDER_NAME, 'field_v2')
    os.makedirs(state_field_saved_path, exist_ok=True)
    retrieval = KnowledgeRetrieval(corpus, state_field_saved_path, summarize_len=4, model_name=args.model_name, 
                 embed_model=args.embed_model, eval_strategy=args.eval_strategy, knowledge_type=KnowledgeType.Policy,
                 top_k=args.top_k, device='cuda:2')
    # evaluate embedding method.
    # top_k = args.top_k
    # eval_strategy = args.eval_strategy
    # if args.embed_model == 'openai':
    #     embed_model =  OpenAIEmbedding()
    # else:
    #     embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en") # 0.06593406593406594
    # # 
    # if eval_strategy == 'embedding':
    #     nodes = [TextNode(id_=id_, text=str(text)) for id_, text in corpus.items()]
    # elif eval_strategy == 'summary':
    #     nodes = [TextNode(id_=id_, text=str(text['preferred scope description'])) for id_, text in corpus_space_dict.items()]
    # else:
    #     nodes = [TextNode(id_=id_, text=str(text)) for id_, text in corpus_space_dict.items()]
    # # 
    # print("model", embed_model)
    # print("num of nodes", len(nodes))
    # # prompt_template = "The optimal policy function of a football manager to make decisions in the state"
    # prompt_template = "The policy function of a football manager to be used to make decisions in the following obsevation:"
    # print("prompt_template", prompt_template)
    # index = index_from_nodes(nodes, embed_model=embed_model, show_progress=True)
    
    # retriever = index.as_retriever(similarity_top_k=top_k)
    eval_results = []
    counter = 0
    for query_id, query in tqdm.tqdm(queries.items()):
        counter += 1
        # if counter % 1000 == 0:
        #     print("hit rate", np.mean([r['is_hit'] for r in eval_results]))
        if np.random.rand() > 0.1:
            continue
        # if counter % 10 != 0:
        #     continue
        
        retrieved_ids = retrieval.retrieve_knowledge(query, ret_type='ids')
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc
        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)

    df_ada = pd.DataFrame(eval_results)
    print(df_ada["is_hit"].mean())