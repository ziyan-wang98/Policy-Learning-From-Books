# Dataset Contract

The public dataset is `ziyan98/plfb` on Hugging Face. Treat it as the authoritative released artifact layout. It contains derived book artifacts, generated football imagined trajectories, the paper-aligned merged cache, the selected final CIQL model, a public first-stage replay checkpoint, reports, and manifests. It intentionally does not include raw tutorial-book corpora, provider credentials, runtime logs, old generated datasets, or non-selected checkpoints.

## Stable Paths

```text
book_derived/retrieval/                     Retrieval-ready policy, reward, and transition JSONL.
book_derived/uri_text_results/              Retained URI text outputs for inspection and replay context.
book_derived/v4-gpt-3.5-turbo-1106-level-strict/  Aggregated retained book-understanding outputs.
football/generated_llm_results/             Small retained generation logs/results.
football/imaginary_dataset_0204/no_*.npz    Raw 2024-02 imagined trajectory shards used to rebuild the final cache.
football/imaginary_dataset_0204/merged_data/v3datatrace_real_num=0&extra_real_traj_num=0&obs_stack_num=4&rollout_num=0.npz
                                             Paper-aligned final-parameter merged replay-buffer cache.
artifacts/football/final_uri_best/          Selected final CIQL checkpoint, params, training curves, and eval logs.
artifacts/football/strict_repro_first_stage_ba0e02e/  Public first-stage replay checkpoint.
reports/                                    Release reports and validation summaries.
manifests/                                  File inventories and release manifest.
```

## Artifact-To-Stage Contract

| Stage | Required public artifact group | Contract role | Validation |
| --- | --- | --- | --- |
| 0 | `manifests/`, `reports/`, `artifacts/football/final_uri_best/` | Release manifest, final model, retained reports | `python scripts/smoke_pipeline.py --mode layout --artifact-root <artifact root>` |
| 1 | `book_derived/v4-gpt-3.5-turbo-1106-level-strict/`, `book_derived/uri_text_results/understanding/` | Retained book-understanding outputs | `bash scripts/smoke_stage.sh 1` |
| 2 | `book_derived/retrieval/`, `book_derived/uri_text_results/rehearsing/` | Retrieval-ready policy/reward/transition snippets and replay context | `bash scripts/smoke_stage.sh 2` |
| 3 | `football/generated_llm_results/`, `football/imaginary_dataset_0204/no_*.npz` | Retained imagined trajectory shards and generation evidence | `bash scripts/smoke_stage.sh 3` |
| 4 | `football/imaginary_dataset_0204/merged_data/`, `artifacts/football/strict_repro_first_stage_ba0e02e/` | Merged replay-buffer cache and strict first-stage replay checkpoint | `bash scripts/smoke_stage.sh 4` |
| 5 | `football/imaginary_dataset_0204/`, `artifacts/football/final_uri_best/` | CIQL retraining inputs and selected release checkpoint/logs | `bash scripts/smoke_stage.sh 5` |
| 6 | `artifacts/football/final_uri_best/eval-environment/`, `artifacts/football/final_uri_best/eval-top_3/`, `reports/final_ciql_release_report.json`, `reports/final_uri_best_eval_log_summary.json` | Historical paper-aligned eval logs and summary reports | `bash scripts/smoke_stage.sh 6` |

## Release-critical Artifacts

| Artifact | Role |
| --- | --- |
| `artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3` | final released CIQL policy |
| `football/imaginary_dataset_0204/merged_data/v3datatrace_real_num=0&extra_real_traj_num=0&obs_stack_num=4&rollout_num=0.npz` | 2024-02 merged imagined dataset cache |
| `artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3` | public first-stage uncertainty checkpoint |

## Exclusions

The public dataset excludes raw book text, raw RedPajama/The Pile shards, old `offline_dataset-v4` and `rule_based_2_level_*` staging datasets, Tic-Tac-Toe auxiliary release material, provider credentials, private endpoints, historical BC policy checkpoints used only for optional trajectory regeneration and intermediate checkpoints that were not selected for release.

## Loading

```bash
hf download ziyan98/plfb --repo-type dataset --local-dir ./plfb_artifacts
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
export PLFB_DATASET_PATH=$PLFB_ARTIFACT_ROOT/football
export PLFB_IMAGINARY_DATASET_PATH=$PLFB_DATASET_PATH/imaginary_dataset_0204
export PLFB_MODEL_ROOT=$PLFB_ARTIFACT_ROOT/artifacts/football
python scripts/smoke_pipeline.py --mode layout --artifact-root "$PLFB_ARTIFACT_ROOT"
python scripts/smoke_pipeline.py --mode data-contract --artifact-root "$PLFB_ARTIFACT_ROOT"
```
