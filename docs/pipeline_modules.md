# Pipeline Modules

This page maps the paper pipeline to the public commands and artifacts. Start with `docs/reproduction.md` for the shortest path.

| Stage | Code | Inputs | Outputs | Smoke |
| --- | --- | --- | --- | --- |
| 0 environment | `environment-universal.yml`, `scripts/check_environment.py`, `scripts/smoke_pipeline.py` | conda/mamba environment | import and artifact report | `python scripts/check_environment.py --quick`; `bash scripts/smoke_stage.sh 0` |
| 1 understanding | `plfb-uri/main_understanding.py`, `football_llm/book_scripts/*` | JSONL tutorial text, provider credentials | `book_derived/*` understanding artifacts | `bash scripts/book_understanding.sh --dry-run` |
| 2 retrieval/context | `football_llm/retrieval/*`, `scripts/normalize_retrieval_context.py` | Stage 1 outputs or retained retrieval files | `book_derived/retrieval` | `bash scripts/prepare_retrieval_context.sh --dry-run` |
| 3 imagined trajectories | `football_llm/llm/generate_main.py`, `football_llm/llm/imaginary_data_generation.py` | retrieval/context files, sampled states, optional BC checkpoints, provider credentials | `football/imaginary_dataset_0204/no_*.npz` and merged cache | `bash scripts/generate_imagined_trajectories.sh --dry-run` |
| 4 uncertainty | `football_llm/learning/uncertainty_predictor.py`, `football_llm/learning/data_loader.py` | imagined dataset cache | uncertainty checkpoint and uncertainty-shaped rewards | `bash scripts/introspect_uncertainty.sh --smoke --dry-run` |
| 5 CIQL | `football_llm/learning/imaginaryRL_v2.py`, vendored `football_llm/d3rlpy` | merged cache and uncertainty checkpoint | CIQL policy checkpoints | `bash scripts/train_ciql.sh --dry-run` |
| 6 evaluation | `football_llm/learning/load_and_eval_v2.py` | final policy checkpoint | eval JSON | `bash scripts/eval_ciql.sh --dry-run` |

## Release Artifacts Used by Default

| Role | Path |
| --- | --- |
| Final CIQL policy | `artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3` |
| Final CIQL params | `artifacts/football/final_uri_best/params.json` |
| Paper-era imagined merged cache | `football/imaginary_dataset_0204/merged_data/v3datatrace_real_num=0&extra_real_traj_num=0&obs_stack_num=4&rollout_num=0.npz` |
| Raw imagined shards | `football/imaginary_dataset_0204/no_*.npz` |
| Public first-stage checkpoint | `artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3` |
| Release reports and manifests | `reports/`, `manifests/` |

## What Is Not Distributed

Raw book text, provider credentials, historical BC policy checkpoints, unselected intermediate checkpoints, old source-style staging datasets, and auxiliary experiment material are not part of the public release.

Some lower-level research modules still support source-style dataset roots for new experiments. Public wrappers default to the curated Hugging Face layout and should be used before calling lower-level modules directly.
