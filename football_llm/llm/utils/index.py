import argparse
import os

from transformers import AutoTokenizer

from llm.utils.llama_index_compat import (
    BaseReader,
    HuggingFaceEmbedding,
    JSONQueryEngine,
    KeywordTableIndex,
    OpenAI,
    PromptTemplate,
    Replicate,
    SimpleDirectoryReader,
    default_openai_embedding,
    index_from_documents,
    make_document,
    make_service_context,
    set_global_tokenizer,
)
# from book_scripts.utils import load_jsonl_list


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
    class MyFileReader(BaseReader):
        def load_data(self, file, extra_info=None):
            with open(file, "r") as f:
                text = f.read()
            # load_data returns a list of Document objects
            return [make_document(text=text + "Foobar", extra_info=extra_info or {})]

    policy_doc = SimpleDirectoryReader(policy_path).load_data()
    reward_doc = SimpleDirectoryReader(reward_path).load_data()
    transition_doc = SimpleDirectoryReader(trainsition_path).load_data()

    return policy_doc, reward_doc, transition_doc

def index_llm(llm_type):
    if llm_type == "llama":
        llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
        llm = Replicate(
            model=llama2_7b_chat,
            temperature=0.01,
            additional_kwargs={"top_p": 1, "max_new_tokens": 300},
        )
        # set tokenizer to match LLM
        set_global_tokenizer(
            AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
            )
    elif llm_type == "gpt-3":
        llm = OpenAI(temperature=0.0, model=os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    elif llm_type == "gpt-4":
        llm = OpenAI(temperature=0.0, model=os.environ.get("PLFB_OPENAI_CODE_MODEL", os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")))
    return llm

def get_index(encorder_type, llm_type, filter_path):

    if encorder_type == "defult":
        embed_model =  default_openai_embedding()
    elif encorder_type == "custom":
        # embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")
    else:
        raise NotImplementedError

    if llm_type == "llama":
            llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
            llm = Replicate(
                model=llama2_7b_chat,
                temperature=0.01,
                additional_kwargs={"top_p": 1, "max_new_tokens": 300},
            )
            # set tokenizer to match LLM
            set_global_tokenizer(
                AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
            )
    elif llm_type == "gpt-3":
        llm = OpenAI(temperature=0.0, model=os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    elif llm_type == "gpt-4":
        llm = OpenAI(temperature=0.0, model=os.environ.get("PLFB_OPENAI_CODE_MODEL", os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")))

    service_context_policy = make_service_context(embed_model=default_openai_embedding())
    service_context_reward = make_service_context(embed_model=default_openai_embedding())
    service_context_transition = make_service_context(embed_model=default_openai_embedding())

    policy_doc, reward_doc, transition_doc = get_docs(filter_path)

    policy_index = index_from_documents(policy_doc, service_context=service_context_policy)
    reward_index = index_from_documents(
        reward_doc, service_context=service_context_reward
    )
    transition_index = index_from_documents(
        transition_doc, service_context=service_context_transition
    )
    return policy_index, reward_index, transition_index


def get_index_json(encorder_type, llm_type, filter_path):

    if encorder_type == "defult":
        embed_model =  default_openai_embedding(embed_batch_size=42)
    elif encorder_type == "custom":
        # embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")
    else:
        raise NotImplementedError

    if llm_type == "llama":
            llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
            llm = Replicate(
                model=llama2_7b_chat,
                temperature=0.01,
                additional_kwargs={"top_p": 1, "max_new_tokens": 300},
            )
            # set tokenizer to match LLM
            set_global_tokenizer(
                AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
            )
    elif llm_type == "gpt-3":
        llm = OpenAI(temperature=0.0, model=os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    elif llm_type == "gpt-4":
        llm = OpenAI(temperature=0.0, model=os.environ.get("PLFB_OPENAI_CODE_MODEL", os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")))


    service_context = make_service_context(llm=llm, embed_model=embed_model)

    policy_doc, reward_doc, transition_doc = get_docs(filter_path)

    policy_json_schema = {
        "type": "object",
        "properties": {
            "policy": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string"
                    },
                    "state": {
                        "type": "string"
                    },
                    "intent": {
                        "type": "string"
                    },
                    "context": {
                        "type": "string"
                    }
                }
            }
        }
    }

    policy_index = JSONQueryEngine(
        policy_doc, json_schema=policy_json_schema, service_context=service_context
    )
    reward_index = JSONQueryEngine.from_documents(
        reward_doc, json_schema=json_schema, service_context=service_context
    )
    transition_index = JSONQueryEngine.from_documents(
        transition_doc, service_context=service_context
    )
    return policy_index, reward_index, transition_index

def get_engine(encorder_type, llm_type, filter_path):

    if encorder_type == "defult":
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    elif encorder_type == "llama":
        pass # TODO check if there are any other encoders
    else:
        raise NotImplementedError

    if llm_type == "llama":
            llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
            llm = Replicate(
                model=llama2_7b_chat,
                temperature=0.01,
                additional_kwargs={"top_p": 1, "max_new_tokens": 300},
            )
            # set tokenizer to match LLM
            set_global_tokenizer(
                AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
            )
    elif llm_type == "gpt-3":
        llm = OpenAI(temperature=0.0, model=os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    elif llm_type == "gpt-4":
        llm = OpenAI(temperature=0.0, model=os.environ.get("PLFB_OPENAI_CODE_MODEL", os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")))


    service_context = make_service_context(llm=llm, embed_model=embed_model)

    policy_doc, reward_doc, transition_doc = get_docs(filter_path)

    policy_index = index_from_documents(
        policy_doc, service_context=service_context
    )
    reward_index = index_from_documents(
        reward_doc, service_context=service_context
    )
    transition_index = index_from_documents(
        transition_doc, service_context=service_context
    )

    policy_query_engine = policy_index.as_query_engine()
    reward_query_engine = reward_index.as_query_engine()
    transition_query_engine = transition_index.as_query_engine()

    return policy_query_engine, reward_query_engine, transition_query_engine



def find_first_code_block(input_str):
    """
    Find the first substring in the input string that starts with '}\n{"code"' and ends with '}\n{"code"'.
    """
    start_pattern = "}\n{\"code\""
    end_pattern = "}\n{\"code\""

    # Find the start index of the first occurrence of start_pattern
    start_index = input_str.find(start_pattern)
    if start_index == -1:
        return None  # Start pattern not found

    # Adjust the start_index to include the start_pattern in the result
    start_index += len(start_pattern)

    # Find the end index of the first occurrence of end_pattern, starting from start_index
    end_index = input_str.find(end_pattern, start_index)
    if end_index == -1:
        return ""  # End pattern not found
    res = input_str[start_index:end_index]
    res = "{\"code\"" + res + "}"
    # Extract the substring from the adjusted start_index to the end_index
    return res

