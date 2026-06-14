# football_llm legacy implementation

This directory contains the original football and Tic-Tac-Toe implementation used by the URI experiments. Prefer the repository-level `README.md` for setup, data, SLURM examples, and citation details.

Main components:

- `book_scripts/`: filters football tutorial text and derives policy, dynamics, and reward snippets.
- `llm/`: collects offline states and rehearses imagined transitions with LLM/RAG backends.
- `learning/`: trains and evaluates CIQL policies from imagined/offline data.
- `rehearsing/` and `retrieval/`: helper modules for trajectory generation and code retrieval.

Paths are controlled with environment variables such as `PLFB_ROOT`, `PLFB_DATASET_PATH`, `PLFB_OFFLINE_DATASET_PATH`, and `PLFB_MODEL_PATH`. API keys must be provided through environment variables, not checked into this repository.
