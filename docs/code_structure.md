# Code Structure

## `football_llm/`

- `book_scripts/`: filters and transforms external tutorial text into derived knowledge artifacts.
- `llm/` and `llm_v2/`: LLM-backed generation, baseline agents, and data-generation utilities.
- `retrieval/`: retrieval helpers for policy, reward, transition, and state-field data.
- `rehearsing/`: trajectory rehearsal utilities, including auxiliary Tic-Tac-Toe code.
- `learning/`: offline RL, CIQL training, checkpoint loading, and evaluation.
- `setup/football/`: Google Research Football source and Python package entry points.
- `d3rlpy/`: vendored d3rlpy fork used by the historical CIQL implementation.

## `plfb-uri/`

Cleaner URI entry points and configs for understanding and introspection experiments.

## `examples/`

Generic data and SLURM examples. These are templates and should be adapted to local paths and scheduler settings.


## Legacy And Auxiliary Modules

The public docs distinguish paper-critical modules from retained legacy code:

- Paper-critical football modules are mapped stage-by-stage in `docs/paper_module_operations.md` and `docs/pipeline_modules.md`. Those files are the authority for what must run in the public workflow.
- `football_llm/llm_v2/`, `football_llm/llm_baseline.py`, older `gen_finetune_data.py`-style utilities, older book-script variants, and `tictactoe/` paths are retained for compatibility or auxiliary experiments unless explicitly listed in a stage table. They are not required for final football evaluation or the retained CIQL retraining path.
- If a retained legacy module is useful for a new experiment, treat it as lower-level research code and validate it separately instead of assuming it is part of the paper-critical reproduction path.

For file-level operation commands and paper-stage mapping, see `docs/pipeline_modules.md`.
