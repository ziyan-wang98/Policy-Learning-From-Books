import os
import openai
import json
from pathlib import Path

from llm.utils.llama_index_compat import (
    CallbackManager,
    DatasetGenerator,
    Document,
    GradientBaseModelLLM,
    GradientFinetuneEngine,
    OpenAI,
    PromptTemplate,
    PyMuPDFReader,
    QueryResponseDataset,
    Replicate,
    SimpleNodeParser,
    VectorStoreIndex,
    make_service_context,
    require_optional_llama_index_symbol,
)


# Gradient credentials are read from GRADIENT_ACCESS_TOKEN and GRADIENT_WORKSPACE_ID.


def data_construction_pip():
    reader_cls = require_optional_llama_index_symbol(
        PyMuPDFReader, "PyMuPDFReader", "PDF finetuning data construction"
    )
    loader = reader_cls()
    docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

    doc_text = "\n\n".join([d.get_content() for d in docs0])
    metadata = {"paper_title": "Llama 2: Open Foundation and Fine-Tuned Chat Models"}
    docs = [Document(text=doc_text, metadata=metadata)]

    print(docs[0].get_content())

    callback_manager_cls = require_optional_llama_index_symbol(CallbackManager, "CallbackManager", "finetuning data construction")
    callback_manager = callback_manager_cls([])

    eval_context = make_service_context(
        llm=require_optional_llama_index_symbol(OpenAI, "OpenAI", "RAG data generation")(model=os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0),
        callback_manager=callback_manager,
    )
    node_parser_cls = require_optional_llama_index_symbol(SimpleNodeParser, "SimpleNodeParser", "RAG data generation")
    node_parser = node_parser_cls()
    nodes = node_parser.get_nodes_from_documents(docs)
    vector_index = VectorStoreIndex(nodes)

    # generate dataset
    DatasetGenerator_cls = require_optional_llama_index_symbol(DatasetGenerator, "DatasetGenerator", "RAG data generation")
    QueryResponseDataset_cls = require_optional_llama_index_symbol(QueryResponseDataset, "QueryResponseDataset", "RAG data generation")

    def generate_data():
        dataset_generator = DatasetGenerator_cls(
            nodes[:39],
            service_context=eval_context,
            show_progress=True,
            num_questions_per_chunk=5,  # Gradient-based LLM access does not support parallel calls here.
        )
        eval_dataset = dataset_generator.generate_dataset_from_nodes(num=60)
        eval_dataset.save_json("./data/data_rag/qa_pairs.json")
        eval_dataset = QueryResponseDataset_cls.from_json("./data/data_rag/qa_pairs.json")

    # generate RAG data
    def generate_rag_data(eval_dataset):
        qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        vector_retriever = vector_index.as_retriever(similarity_top_k=1)

        from tqdm import tqdm

        def augment_data_with_retrieval(dataset, retriever, separate_context=False):
            data_list = dataset.qr_pairs
            new_data_list = []
            for query_str, response in tqdm(data_list):
                retrieved_nodes = retriever.retrieve(query_str)
                retrieved_txts = [n.get_content() for n in retrieved_nodes]
                if separate_context:
                    for retrieved_txt in retrieved_txts:
                        fmt_query_str = qa_prompt_tmpl.format(
                            query_str=query_str, context_str=retrieved_txt
                        )
                        new_data_list.append((fmt_query_str, response))
                else:
                    context_str = "\n\n".join(retrieved_txts)
                    fmt_query_str = qa_prompt_tmpl.format(
                        query_str=query_str, context_str=context_str
                    )
                    new_data_list.append((fmt_query_str, response))
            return new_data_list

        new_qr_pairs = augment_data_with_retrieval(
            eval_dataset, vector_retriever, separate_context=False
        )
        new_eval_dataset = QueryResponseDataset_cls.from_qr_pairs(new_qr_pairs)
        new_eval_dataset.save_json("./data/data_rag/qa_pairs_ra.json")

    # generate_data()
    eval_dataset = QueryResponseDataset_cls.from_json("./data/data_rag/qa_pairs.json")
    # generate_rag_data(eval_dataset)
    new_eval_dataset = QueryResponseDataset_cls.from_json(
        "./data/data_rag/qa_pairs_ra.json"
    )

    from copy import deepcopy
    import random

    def split_train_val(dataset, train_split=0.7):
        lines = dataset.qr_pairs

        # shuffle the lines to make sure that the "train questions" cover most fo the context
        shuffled_lines = deepcopy(lines)
        random.shuffle(shuffled_lines)

        split_idx = int(train_split * len(shuffled_lines))
        train_lines = shuffled_lines[:split_idx]
        val_lines = shuffled_lines[split_idx:]
        # train_lines, val_lines = split_train_val(new_eval_dataset, train_split=0.7)

        train_dataset = QueryResponseDataset_cls.from_qr_pairs(train_lines)
        val_dataset = QueryResponseDataset_cls.from_qr_pairs(val_lines)
        train_dataset.save_json("./data/data_rag/qa_pairs_train.json")
        val_dataset.save_json("./data/data_rag/qa_pairs_val.json")

    # split_train_val(new_eval_dataset)
    train_dataset = QueryResponseDataset_cls.from_json(
        "./data/data_rag/qa_pairs_train.json"
    )
    val_dataset = QueryResponseDataset_cls.from_json("./data/data_rag/qa_pairs_val.json")

    def save_gradientai_llama_data(dataset, out_path):
        out_fp = open(out_path, "w")
        system_prompt = "You are an instruction-following model answering questions about the LLAMA 2 paper."
        train_qr_pairs = dataset.qr_pairs
        for line in train_qr_pairs:
            query, response = line
            out_dict = {
                "inputs": f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{query} [/INST] {response} </s>"
            }
            out_fp.write(json.dumps(out_dict) + "\n")

    save_gradientai_llama_data(train_dataset, "./data/data_rag/qa_pairs_gradient.jsonl")


def finetune_pipe():
    llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"

    require_optional_llama_index_symbol(GradientBaseModelLLM, "GradientBaseModelLLM", "gradient finetuning")

    # You can also use a model adapter you've trained with GradientModelAdapterLLM
    # llm = GradientBaseModelLLM(
    #     base_model_slug="llama2-7b-chat",
    #     max_tokens=400,
    # )

    base_model_slug = "llama2-7b-chat"
    # NOTE: can only specify one of base_model_slug or model_adapter_id
    finetune_engine_cls = require_optional_llama_index_symbol(GradientFinetuneEngine, "GradientFinetuneEngine", "gradient finetuning")
    finetune_engine = finetune_engine_cls(
        base_model_slug=base_model_slug,
        # model_adapter_id=os.environ.get('GRADIENT_MODEL_ADAPTER_ID'),
        name="test",
        data_path="./data/data_rag/qa_pairs_gradient.jsonl",
        verbose=True,
        max_steps=200,
        batch_size=4,
    )
    epochs = 1
    for i in range(epochs):
        print(f"** EPOCH {i} **")
        finetune_engine.finetune()
    print(finetune_engine.model_adapter_id)
    return finetune_engine.get_finetuned_model(max_tokens=300)

if __name__ == "__main__":
    # data_construction_pip()
    finetune_pipe()
