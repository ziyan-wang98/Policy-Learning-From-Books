"""Compatibility helpers for legacy and modern LlamaIndex releases."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Iterable


def _import_attr(candidates: Iterable[tuple[str, str]], *, required: bool = True) -> Any:
    last_exc: BaseException | None = None
    for module_name, attr_name in candidates:
        try:
            module = import_module(module_name)
            return getattr(module, attr_name)
        except Exception as exc:  # pragma: no cover - depends on installed LlamaIndex version.
            last_exc = exc
    if required:
        names = ", ".join(f"{module}.{attr}" for module, attr in candidates)
        raise ImportError(f"Could not import any LlamaIndex symbol from: {names}") from last_exc
    return None


Document = _import_attr((("llama_index.core", "Document"), ("llama_index", "Document")))
SimpleDirectoryReader = _import_attr((("llama_index.core", "SimpleDirectoryReader"), ("llama_index", "SimpleDirectoryReader")))
VectorStoreIndex = _import_attr((("llama_index.core", "VectorStoreIndex"), ("llama_index", "VectorStoreIndex")))
KeywordTableIndex = _import_attr((("llama_index.core", "KeywordTableIndex"), ("llama_index", "KeywordTableIndex")), required=False)
StorageContext = _import_attr((("llama_index.core", "StorageContext"), ("llama_index", "StorageContext")))
load_index_from_storage = _import_attr((("llama_index.core", "load_index_from_storage"), ("llama_index", "load_index_from_storage")))
set_global_tokenizer = _import_attr((("llama_index.core", "set_global_tokenizer"), ("llama_index", "set_global_tokenizer")), required=False)
Settings = _import_attr((("llama_index.core", "Settings"),), required=False)
ServiceContext = _import_attr((("llama_index.core", "ServiceContext"), ("llama_index", "ServiceContext"), ("llama_index.indices.service_context", "ServiceContext")), required=False)
BaseReader = _import_attr((("llama_index.core.readers.base", "BaseReader"), ("llama_index.readers.base", "BaseReader")))
TextNode = _import_attr((("llama_index.core.schema", "TextNode"), ("llama_index.schema", "TextNode")))
PromptTemplate = _import_attr((("llama_index.core.prompts", "PromptTemplate"), ("llama_index.prompts", "PromptTemplate")), required=False)
VectorStoreQueryMode = _import_attr((("llama_index.core.vector_stores.types", "VectorStoreQueryMode"), ("llama_index.vector_stores.types", "VectorStoreQueryMode")), required=False)
OpenAI = _import_attr((("llama_index.llms.openai", "OpenAI"), ("llama_index.llms", "OpenAI")), required=False)
Replicate = _import_attr((("llama_index.llms.replicate", "Replicate"), ("llama_index.llms", "Replicate")), required=False)
OpenAIEmbedding = _import_attr((("llama_index.embeddings.openai", "OpenAIEmbedding"), ("llama_index.embeddings", "OpenAIEmbedding")), required=False)
HuggingFaceEmbedding = _import_attr((("llama_index.embeddings.huggingface", "HuggingFaceEmbedding"), ("llama_index.embeddings", "HuggingFaceEmbedding")), required=False)
QueryBundle = _import_attr((("llama_index.core.schema", "QueryBundle"), ("llama_index.schema", "QueryBundle"), ("llama_index", "QueryBundle")), required=False)
ChatMessage = _import_attr((("llama_index.core.llms", "ChatMessage"), ("llama_index.llms", "ChatMessage"), ("llama_index.prompts", "ChatMessage")), required=False)
ChatPromptTemplate = _import_attr((("llama_index.core.prompts", "ChatPromptTemplate"), ("llama_index.prompts", "ChatPromptTemplate")), required=False)
ToolMetadata = _import_attr((("llama_index.core.tools", "ToolMetadata"), ("llama_index.tools", "ToolMetadata")), required=False)
BaseRetriever = _import_attr((("llama_index.core.retrievers", "BaseRetriever"), ("llama_index.retrievers", "BaseRetriever")), required=False)
OpenAIQuestionGenerator = _import_attr((("llama_index.core.question_gen", "OpenAIQuestionGenerator"), ("llama_index.question_gen.openai_generator", "OpenAIQuestionGenerator")), required=False)
OpenAIPydanticProgram = _import_attr((("llama_index.core.program", "OpenAIPydanticProgram"), ("llama_index.program", "OpenAIPydanticProgram")), required=False)
SimpleNodeParser = _import_attr((("llama_index.core.node_parser", "SimpleNodeParser"), ("llama_index.node_parser", "SimpleNodeParser")), required=False)
DatasetGenerator = _import_attr((("llama_index.core.evaluation", "DatasetGenerator"), ("llama_index.evaluation", "DatasetGenerator")), required=False)
QueryResponseDataset = _import_attr((("llama_index.core.evaluation", "QueryResponseDataset"), ("llama_index.evaluation", "QueryResponseDataset")), required=False)
CallbackManager = _import_attr((("llama_index.core.callbacks", "CallbackManager"), ("llama_index.callbacks", "CallbackManager")), required=False)
PyMuPDFReader = _import_attr((("llama_index.readers.file", "PyMuPDFReader"), ("llama_index.readers.file.docs", "PyMuPDFReader")), required=False)
GradientBaseModelLLM = _import_attr((("llama_index.llms.gradient", "GradientBaseModelLLM"), ("llama_index.llms", "GradientBaseModelLLM")), required=False)
GradientFinetuneEngine = _import_attr((("llama_index.finetuning.gradient.base", "GradientFinetuneEngine"),), required=False)
JSONQueryEngine = _import_attr((("llama_index.core.indices.struct_store", "JSONQueryEngine"), ("llama_index.indices.struct_store", "JSONQueryEngine")), required=False)
SentenceSplitter = _import_attr((("llama_index.core.node_parser", "SentenceSplitter"), ("llama_index.node_parser", "SentenceSplitter")), required=False)
EmbeddingAdapterFinetuneEngine = _import_attr((("llama_index.finetuning", "EmbeddingAdapterFinetuneEngine"),), required=False)
generate_qa_embedding_pairs = _import_attr((("llama_index.finetuning", "generate_qa_embedding_pairs"),), required=False)
EmbeddingQAFinetuneDataset = _import_attr((("llama_index.core.evaluation", "EmbeddingQAFinetuneDataset"), ("llama_index.evaluation", "EmbeddingQAFinetuneDataset")), required=False)
resolve_embed_model = _import_attr((("llama_index.core.embeddings", "resolve_embed_model"), ("llama_index.embeddings", "resolve_embed_model")), required=False)
TwoLayerNN = _import_attr((("llama_index.embeddings.adapter_utils", "TwoLayerNN"),), required=False)


if VectorStoreQueryMode is None:
    class VectorStoreQueryMode:  # type: ignore[no-redef]
        DEFAULT = "default"


if JSONQueryEngine is None:
    class JSONQueryEngine:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("JSONQueryEngine is unavailable in the installed LlamaIndex package.")

        @classmethod
        def from_documents(cls, *args: Any, **kwargs: Any) -> "JSONQueryEngine":
            raise ImportError("JSONQueryEngine is unavailable in the installed LlamaIndex package.")


def make_document(*, text: str, extra_info: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    metadata = extra_info or {}
    try:
        return Document(text=text, metadata=metadata, **kwargs)
    except TypeError:
        return Document(text=text, extra_info=metadata, **kwargs)


def require_optional_llama_index_symbol(symbol: Any, name: str, purpose: str) -> Any:
    if symbol is None:
        raise ImportError(
            f"LlamaIndex optional symbol {name} is unavailable; install the relevant "
            f"llama-index integration package before running {purpose}."
        )
    return symbol


def default_openai_embedding(**kwargs: Any) -> Any:
    embedding_cls = require_optional_llama_index_symbol(OpenAIEmbedding, "OpenAIEmbedding", "OpenAI embedding retrieval")
    kwargs.setdefault("model", os.environ.get("PLFB_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    try:
        return embedding_cls(**kwargs)
    except TypeError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("model", None)
        return embedding_cls(**fallback_kwargs)


def _apply_settings(kwargs: dict[str, Any]) -> None:
    if Settings is None:
        return
    mapping = {
        "llm": "llm",
        "embed_model": "embed_model",
        "callback_manager": "callback_manager",
        "node_parser": "node_parser",
        "num_output": "num_output",
        "context_window": "context_window",
        "chunk_size": "chunk_size",
        "chunk_overlap": "chunk_overlap",
    }
    for source_key, target_key in mapping.items():
        value = kwargs.get(source_key)
        if value is None:
            continue
        try:
            setattr(Settings, target_key, value)
        except Exception:
            continue


def make_service_context(**kwargs: Any) -> Any | None:
    if ServiceContext is not None:
        try:
            return ServiceContext.from_defaults(**kwargs)
        except Exception:
            if Settings is None:
                raise
    _apply_settings(kwargs)
    return None


def _with_optional_service_context(service_context: Any | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    call_kwargs = dict(kwargs)
    if service_context is not None:
        call_kwargs["service_context"] = service_context
    return call_kwargs


def index_from_documents(documents: Any, *, service_context: Any | None = None, **kwargs: Any) -> Any:
    call_kwargs = _with_optional_service_context(service_context, kwargs)
    try:
        return VectorStoreIndex.from_documents(documents, **call_kwargs)
    except TypeError:
        if "service_context" not in call_kwargs:
            raise
        call_kwargs.pop("service_context")
        return VectorStoreIndex.from_documents(documents, **call_kwargs)


def index_from_nodes(nodes: Any, *, service_context: Any | None = None, **kwargs: Any) -> Any:
    call_kwargs = _with_optional_service_context(service_context, kwargs)
    try:
        return VectorStoreIndex(nodes, **call_kwargs)
    except TypeError:
        if "service_context" not in call_kwargs:
            raise
        call_kwargs.pop("service_context")
        return VectorStoreIndex(nodes, **call_kwargs)


def set_index_service_context(index: Any, service_context: Any | None) -> Any:
    if service_context is not None:
        try:
            index._service_context = service_context
        except Exception:
            pass
    return index
