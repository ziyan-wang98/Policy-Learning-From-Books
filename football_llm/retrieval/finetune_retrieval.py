import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append("football_llm/TiZero/submission/tizero_agent")
sys.path.append("football_llm")

from llm.utils.llama_index_compat import (
    EmbeddingAdapterFinetuneEngine,
    EmbeddingQAFinetuneDataset,
    OpenAI,
    OpenAIEmbedding,
    SentenceSplitter,
    SimpleDirectoryReader,
    TextNode,
    TwoLayerNN,
    VectorStoreIndex,
    generate_qa_embedding_pairs,
    index_from_nodes,
    resolve_embed_model,
)
from tqdm import tqdm
import torch

try:
    from CONFIG import DATASET_PATH, MODEL_PATH
except Exception:
    DATASET_PATH = os.environ.get("PLFB_RETRIEVAL_DATASET_ROOT", "plfb_artifacts/book_derived/retrieval")
    MODEL_PATH = os.environ.get("PLFB_MODEL_OUTPUT_ROOT", "runs/models")


def _env_list(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [item.strip() for item in value.split(os.pathsep) if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a LlamaIndex embedding adapter on user-provided corpus files.")
    parser.add_argument("--train-files", nargs="+", default=_env_list("PLFB_RETRIEVAL_TRAIN_FILES"), help="Training corpus files readable by SimpleDirectoryReader.")
    parser.add_argument("--val-files", nargs="+", default=_env_list("PLFB_RETRIEVAL_VAL_FILES"), help="Validation corpus files readable by SimpleDirectoryReader.")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("PLFB_RETRIEVAL_OUTPUT_DIR", Path(DATASET_PATH) / "embedding")))
    parser.add_argument("--model-output-path", type=Path, default=Path(os.environ.get("PLFB_RETRIEVAL_MODEL_OUTPUT", Path(MODEL_PATH) / "embedding_model" / "adapter")))
    parser.add_argument("--qa-model", default=os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-embed-model", default=os.environ.get("PLFB_RETRIEVAL_BASE_EMBED", "local:BAAI/bge-small-en"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("PLFB_RETRIEVAL_EPOCHS", "10")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PLFB_RETRIEVAL_BATCH_SIZE", "64")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("PLFB_RETRIEVAL_LR", "3e-4")))
    parser.add_argument("--finetune", action="store_true", help="Run adapter training after generating QA pairs.")
    parser.add_argument("--evaluate-openai", action="store_true", help="Evaluate the validation set with the configured OpenAI embedding model.")
    return parser.parse_args()


def require_optional_llama_index_finetuning() -> None:
    missing = [
        name
        for name, value in {
            "SentenceSplitter": SentenceSplitter,
            "EmbeddingAdapterFinetuneEngine": EmbeddingAdapterFinetuneEngine,
            "generate_qa_embedding_pairs": generate_qa_embedding_pairs,
            "EmbeddingQAFinetuneDataset": EmbeddingQAFinetuneDataset,
            "resolve_embed_model": resolve_embed_model,
            "TwoLayerNN": TwoLayerNN,
        }.items()
        if value is None
    ]
    if missing:
        raise ImportError(
            "This optional retrieval finetuning command needs LlamaIndex finetuning "
            f"components that are unavailable in the installed package: {', '.join(missing)}"
        )


def load_corpus(files, verbose=False):
    if not files:
        raise ValueError("at least one corpus file is required")
    if verbose:
        print(f"Loading files {files}")
    reader = SimpleDirectoryReader(input_files=[str(Path(path)) for path in files])
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)
    if verbose:
        print(f"Parsed {len(nodes)} nodes")
    return nodes


def evaluate(dataset, embed_model, top_k=5):
    nodes = [TextNode(id_=node_id, text=text) for node_id, text in dataset.corpus.items()]
    index = index_from_nodes(nodes, embed_model=embed_model, show_progress=True)
    retriever = index.as_retriever(similarity_top_k=top_k)
    eval_results = []
    for query_id, query in tqdm(dataset.queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = dataset.relevant_docs[query_id][0]
        eval_results.append({"is_hit": expected_id in retrieved_ids, "retrieved": retrieved_ids, "expected": expected_id, "query": query_id})
    return eval_results


def main() -> int:
    require_optional_llama_index_finetuning()
    args = parse_args()
    if not args.train_files or not args.val_files:
        raise SystemExit("Provide --train-files and --val-files, or set PLFB_RETRIEVAL_TRAIN_FILES / PLFB_RETRIEVAL_VAL_FILES.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_nodes = load_corpus(args.train_files, verbose=True)
    val_nodes = load_corpus(args.val_files, verbose=True)
    llm = OpenAI(model=args.qa_model)
    train_dataset = generate_qa_embedding_pairs(llm=llm, nodes=train_nodes)
    val_dataset = generate_qa_embedding_pairs(llm=llm, nodes=val_nodes)

    train_path = args.output_dir / "train_dataset.json"
    val_path = args.output_dir / "val_dataset.json"
    train_dataset.save_json(str(train_path))
    val_dataset.save_json(str(val_path))

    train_dataset = EmbeddingQAFinetuneDataset.from_json(str(train_path))
    val_dataset = EmbeddingQAFinetuneDataset.from_json(str(val_path))
    base_embed_model = resolve_embed_model(args.base_embed_model)
    adapter_model = TwoLayerNN(384, 1024, 384, bias=True, add_residual=True)
    finetune_engine = EmbeddingAdapterFinetuneEngine(
        train_dataset,
        base_embed_model,
        model_output_path=str(args.model_output_path),
        adapter_model=adapter_model,
        epochs=args.epochs,
        verbose=True,
        batch_size=args.batch_size,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": args.lr},
    )
    if args.finetune:
        finetune_engine.finetune()
        finetune_engine.get_finetuned_model(adapter_cls=TwoLayerNN)
    if args.evaluate_openai:
        results = evaluate(val_dataset, OpenAIEmbedding())
        (args.output_dir / "openai_val_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote retrieval datasets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
