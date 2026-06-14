# Paper Module Operations

This runbook expands the stage map in `docs/pipeline_modules.md`. Use public wrappers first; call lower-level modules only when developing new experiments.

## Minimal Public Workflow

```bash
python scripts/check_environment.py --quick --strict-football
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
bash scripts/download_artifacts.sh
python scripts/check_environment.py --check-artifacts --artifact-root "$PLFB_ARTIFACT_ROOT"
bash scripts/smoke_stage.sh all
bash scripts/eval_ciql.sh --dry-run
bash scripts/eval_ciql.sh
```

## Full CIQL Replay Workflow

```bash
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
export PLFB_WORK_ROOT=$PWD/runs
bash scripts/train_ciql.sh --dry-run
bash scripts/train_ciql.sh
```

The wrapper uses the public 2024-02 merged cache by default and reuses the public first-stage checkpoint when present. Override `PLFB_MERGED_DATA_CACHE_FILE` or `PLFB_UNCERTAINTY_MODEL_PATH` only for controlled experiments.

## LLM Regeneration Workflow

```bash
export PLFB_BOOK_JSONL=/path/to/book_subset.jsonl
export OPENAI_API_KEY=...
export PLFB_OPENAI_CHAT_MODEL=gpt-4o-mini
export PLFB_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
export PLFB_USE_OPENAI_COMPAT_CLIENT=1
bash scripts/book_understanding.sh
bash scripts/prepare_retrieval_context.sh
bash scripts/generate_imagined_trajectories.sh
```

Raw book text and historical BC policy checkpoints are not distributed. Use this workflow for new generation experiments with your selected source material and model provider.

## Cleanup Guidance

Keep final release artifacts, manifest files, selected evaluation JSON, and the best retained checkpoint for a run. Use `scripts/summarize_ciql_run.py` before deleting local run directories. Do not commit `.d3`, `.npz`, logs, provider credentials, runtime outputs, or machine-specific absolute paths to Git.

## Paper-critical File Coverage

This table is intentionally explicit because the public smoke suite verifies that every paper-critical module is documented in this runbook.

| Stage | File | Operation |
| --- | --- | --- |
| `stage-0-environment-and-contract` | `environment-universal.yml` | Source or wrapper used by this stage. |
| `stage-0-environment-and-contract` | `environment.yml` | Source or wrapper used by this stage. |
| `stage-0-environment-and-contract` | `environment-gfootball.yml` | Source or wrapper used by this stage. |
| `stage-0-environment-and-contract` | `requirements.txt` | Source or wrapper used by this stage. |
| `stage-0-environment-and-contract` | `docs/data_release.md` | Source or wrapper used by this stage. |
| `stage-0-environment-and-contract` | `docs/dataset_contract.md` | Source or wrapper used by this stage. |
| `stage-0-environment-and-contract` | `scripts/smoke_pipeline.py` | Source or wrapper used by this stage. |
| `stage-0-environment-and-contract` | `scripts/plfb_common.sh` | Source or wrapper used by this stage. |
| `stage-0-environment-and-contract` | `scripts/download_artifacts.sh` | Source or wrapper used by this stage. |
| `stage-0-environment-and-contract` | `scripts/smoke_stage.sh` | Source or wrapper used by this stage. |
| `stage-1-understanding` | `plfb-uri/main_understanding.py` | Source or wrapper used by this stage. |
| `stage-1-understanding` | `plfb-uri/understanding/book_reader.py` | Source or wrapper used by this stage. |
| `stage-1-understanding` | `plfb-uri/understanding/prompt_templete.py` | Source or wrapper used by this stage. |
| `stage-1-understanding` | `plfb-uri/configs/conf.yaml` | Source or wrapper used by this stage. |
| `stage-1-understanding` | `football_llm/book_scripts/filter_books_v2.py` | Source or wrapper used by this stage. |
| `stage-1-understanding` | `football_llm/book_scripts/filter_pi_t_obj.py` | Source or wrapper used by this stage. |
| `stage-1-understanding` | `football_llm/book_scripts/prompt_templete.py` | Source or wrapper used by this stage. |
| `stage-1-understanding` | `scripts/book_understanding.sh` | Source or wrapper used by this stage. |
| `stage-1-understanding` | `book_derived/v4-gpt-3.5-turbo-1106-level-strict/` | Public artifact directory consumed or produced by this stage. |
| `stage-1-understanding` | `book_derived/uri_text_results/understanding/` | Public artifact directory consumed or produced by this stage. |
| `stage-2-retrieval-and-code-instantiation` | `football_llm/llm/utils/index.py` | Source or wrapper used by this stage. |
| `stage-2-retrieval-and-code-instantiation` | `football_llm/llm/utils/llama_index_compat.py` | Source or wrapper used by this stage. |
| `stage-2-retrieval-and-code-instantiation` | `football_llm/retrieval/retrieval_module.py` | Source or wrapper used by this stage. |
| `stage-2-retrieval-and-code-instantiation` | `football_llm/retrieval/state_filed_retrieval.py` | Source or wrapper used by this stage. |
| `stage-2-retrieval-and-code-instantiation` | `football_llm/retrieval/finetune_retrieval.py` | Source or wrapper used by this stage. |
| `stage-2-retrieval-and-code-instantiation` | `football_llm/llm/config/gen_main_parser.py` | Source or wrapper used by this stage. |
| `stage-2-retrieval-and-code-instantiation` | `scripts/normalize_retrieval_context.py` | Source or wrapper used by this stage. |
| `stage-2-retrieval-and-code-instantiation` | `scripts/prepare_retrieval_context.sh` | Source or wrapper used by this stage. |
| `stage-2-retrieval-and-code-instantiation` | `book_derived/retrieval/` | Public artifact directory consumed or produced by this stage. |
| `stage-2-retrieval-and-code-instantiation` | `book_derived/uri_text_results/rehearsing/` | Public artifact directory consumed or produced by this stage. |
| `stage-3-rehearsing-imagined-trajectories` | `football_llm/llm/generate_main.py` | Source or wrapper used by this stage. |
| `stage-3-rehearsing-imagined-trajectories` | `football_llm/llm/imaginary_data_generation.py` | Source or wrapper used by this stage. |
| `stage-3-rehearsing-imagined-trajectories` | `football_llm/llm/utils/obs2text.py` | Source or wrapper used by this stage. |
| `stage-3-rehearsing-imagined-trajectories` | `football_llm/llm/utils/imaginary.py` | Source or wrapper used by this stage. |
| `stage-3-rehearsing-imagined-trajectories` | `football_llm/rehearsing/rehearse.py` | Source or wrapper used by this stage. |
| `stage-3-rehearsing-imagined-trajectories` | `scripts/generate_imagined_trajectories.sh` | Source or wrapper used by this stage. |
| `stage-3-rehearsing-imagined-trajectories` | `football/generated_llm_results/` | Public artifact directory consumed or produced by this stage. |
| `stage-3-rehearsing-imagined-trajectories` | `football/imaginary_dataset_0204/` | Public artifact directory consumed or produced by this stage. |
| `stage-3-rehearsing-imagined-trajectories` | `generated_llm_results/` | Lower-level source-style data directory for regenerated experiments. |
| `stage-3-rehearsing-imagined-trajectories` | `imaginary_dataset_0204/` | Lower-level source-style data directory for regenerated experiments. |
| `stage-4-introspecting-uncertainty-rewards` | `plfb-uri/main_introspecting.py` | Source or wrapper used by this stage. |
| `stage-4-introspecting-uncertainty-rewards` | `plfb-uri/introspecting/data_loader.py` | Source or wrapper used by this stage. |
| `stage-4-introspecting-uncertainty-rewards` | `plfb-uri/introspecting/uncertainty_predictor.py` | Source or wrapper used by this stage. |
| `stage-4-introspecting-uncertainty-rewards` | `football_llm/learning/data_loader.py` | Source or wrapper used by this stage. |
| `stage-4-introspecting-uncertainty-rewards` | `football_llm/learning/uncertainty_predictor.py` | Source or wrapper used by this stage. |
| `stage-4-introspecting-uncertainty-rewards` | `scripts/introspect_uncertainty.sh` | Source or wrapper used by this stage. |
| `stage-4-introspecting-uncertainty-rewards` | `football/imaginary_dataset_0204/` | Public artifact directory consumed or produced by this stage. |
| `stage-4-introspecting-uncertainty-rewards` | `imaginary_dataset_0204/` | Lower-level source-style data directory for regenerated experiments. |
| `stage-4-introspecting-uncertainty-rewards` | `artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3` | Release artifact or report checked by smoke tests. |
| `stage-5-ciql-training` | `football_llm/learning/imaginaryRL_v2.py` | Source or wrapper used by this stage. |
| `stage-5-ciql-training` | `football_llm/learning/data_loader.py` | Source or wrapper used by this stage. |
| `stage-5-ciql-training` | `football_llm/learning/uncertainty_predictor.py` | Source or wrapper used by this stage. |
| `stage-5-ciql-training` | `football_llm/d3rlpy/d3rlpy/algos/qlearning/cql.py` | Source or wrapper used by this stage. |
| `stage-5-ciql-training` | `football_llm/d3rlpy/d3rlpy/algos/qlearning/torch/cql_impl.py` | Source or wrapper used by this stage. |
| `stage-5-ciql-training` | `examples/slurm/train_ciql.sbatch` | Source or wrapper used by this stage. |
| `stage-5-ciql-training` | `scripts/train_ciql.sh` | Source or wrapper used by this stage. |
| `stage-5-ciql-training` | `football/imaginary_dataset_0204/` | Public artifact directory consumed or produced by this stage. |
| `stage-5-ciql-training` | `imaginary_dataset_0204/` | Lower-level source-style data directory for regenerated experiments. |
| `stage-6-final-model-selection-and-evaluation` | `football_llm/learning/load_and_eval_v2.py` | Source or wrapper used by this stage. |
| `stage-6-final-model-selection-and-evaluation` | `football_llm/learning/utils.py` | Source or wrapper used by this stage. |
| `stage-6-final-model-selection-and-evaluation` | `docs/final_ciql_model.md` | Source or wrapper used by this stage. |
| `stage-6-final-model-selection-and-evaluation` | `docs/final_ciql_traceability.json` | Source or wrapper used by this stage. |
| `stage-6-final-model-selection-and-evaluation` | `examples/slurm/eval_ciql.sbatch` | Source or wrapper used by this stage. |
| `stage-6-final-model-selection-and-evaluation` | `scripts/eval_ciql.sh` | Source or wrapper used by this stage. |
| `stage-6-final-model-selection-and-evaluation` | `scripts/summarize_ciql_run.py` | Source or wrapper used by this stage. |
| `stage-6-final-model-selection-and-evaluation` | `reports/final_ciql_release_report.json` | Release artifact or report checked by smoke tests. |
| `stage-6-final-model-selection-and-evaluation` | `reports/final_uri_best_eval_log_summary.json` | Release artifact or report checked by smoke tests. |
| `stage-6-final-model-selection-and-evaluation` | `artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3` | Release artifact or report checked by smoke tests. |
