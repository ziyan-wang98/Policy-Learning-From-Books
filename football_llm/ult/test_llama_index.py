import os

from llm.utils.llama_index_compat import (
    HuggingFaceEmbedding,
    Replicate,
    SimpleDirectoryReader,
    index_from_documents,
    make_service_context,
    set_global_tokenizer,
)


def index_gpt(text):
    # check it is text or path
    if os.path.isfile(text):
        documents = SimpleDirectoryReader(text).load_data()
    else:
        documents = [{"text": text}]
        
    documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
    index = index_from_documents(documents)
    print(type(index))
    print(index)
    return index
    

def index_llama(text):

    llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    if Replicate is None:
        raise ImportError("Replicate LlamaIndex integration is unavailable.")
    llm = Replicate(
        model=llama2_7b_chat,
        temperature=0.01,
        additional_kwargs={"top_p": 1, "max_new_tokens": 300},
    )
    # set tokenizer to match LLM
    from transformers import AutoTokenizer

    if set_global_tokenizer is not None:
        set_global_tokenizer(
            AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
        )

    if HuggingFaceEmbedding is None:
        raise ImportError("HuggingFace LlamaIndex embedding integration is unavailable.")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    service_context = make_service_context(llm=llm, embed_model=embed_model)
    
    # check it is text or path
    if os.path.isfile(text):
        documents = SimpleDirectoryReader(text).load_data()
    else:
        documents = [{"text": text}]

    documents = SimpleDirectoryReader(text).load_data()
    
    index = index_from_documents(documents, service_context=service_context)
    
    