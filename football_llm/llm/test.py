import os

from llm.utils.llama_index_compat import OpenAI, make_service_context, require_optional_llama_index_symbol


def main() -> None:
    openai_llm = require_optional_llama_index_symbol(OpenAI, "OpenAI", "LlamaIndex OpenAI smoke test")
    llm = openai_llm(temperature=0.0, model=os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    make_service_context(llm=llm)
    response = llm.complete("Paul Graham is ")
    print(response)


if __name__ == "__main__":
    main()
