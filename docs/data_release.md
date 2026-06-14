# Data Release

The Hugging Face dataset `ziyan98/plfb` is the authoritative public artifact layout for the football release. It is organized by pipeline stage and is validated by `scripts/download_artifacts.sh`, `scripts/check_environment.py`, and `scripts/smoke_pipeline.py`.

## Included

- `book_derived/retrieval`: retained policy, reward, and transition retrieval/context JSONL files.
- `book_derived/uri_text_results` and `book_derived/v4-gpt-3.5-turbo-1106-level-strict`: derived book-understanding artifacts.
- `football/generated_llm_results`: retained generated LLM result records used for inspection.
- `football/imaginary_dataset_0204/no_*.npz`: 2024-02 imagined trajectory shards.
- `football/imaginary_dataset_0204/merged_data/v3datatrace_real_num=0&extra_real_traj_num=0&obs_stack_num=4&rollout_num=0.npz`: public final-parameter merged cache.
- `artifacts/football/final_uri_best`: released final CIQL policy and params.
- `artifacts/football/strict_repro_first_stage_ba0e02e`: public first-stage uncertainty checkpoint.
- `reports/` and `manifests/`: release validation reports, inventories, and stage-to-file maps.

## Excluded

- Raw book text and raw RedPajama/The Pile extracts.
- Provider API keys, private endpoints, local cache paths, and personal logs.
- Historical BC policy checkpoints used only for optional trajectory regeneration.
- Private training-only first-stage checkpoints that are not part of the public replay contract.
- Old source-style staging datasets, duplicate caches, auxiliary experiment material, and unselected intermediate checkpoints.

## Download

```bash
export PLFB_HF_REPO=${PLFB_HF_REPO:-ziyan98/plfb}
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
hf download "$PLFB_HF_REPO" --repo-type dataset --local-dir "$PLFB_ARTIFACT_ROOT"
```

Then validate the release layout and artifact contract:

```bash
python scripts/check_environment.py --check-artifacts --artifact-root "$PLFB_ARTIFACT_ROOT"
bash scripts/smoke_stage.sh all
```

## Critical Artifacts

| Role | Path |
| --- | --- |
| Final CIQL policy | `artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3` |
| Final CIQL params | `artifacts/football/final_uri_best/params.json` |
| 2024-02 merged cache | `football/imaginary_dataset_0204/merged_data/v3datatrace_real_num=0&extra_real_traj_num=0&obs_stack_num=4&rollout_num=0.npz` |
| Public first-stage checkpoint | `artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3` |

## Validation Roots

| Root | Use when | Example |
| --- | --- | --- |
| `--artifact-root` or `PLFB_ARTIFACT_ROOT` | Validating a Hugging Face dataset mirror with the public layout | `python scripts/smoke_pipeline.py --mode layout --artifact-root ./plfb_artifacts` |
| `--dataset-root` | Validating a lower-level source-style football dataset directory during new experiments | `python scripts/smoke_pipeline.py --mode data-contract --dataset-root ./my_source_style_football_data` |
| `--release-root` | Validating a split release directory with reports/model artifacts outside the dataset mirror | `python scripts/smoke_pipeline.py --mode eval-report --release-root ./release` |

`manifests/stage_file_map_20260613.json` and `manifests/inventory.jsonl` map each uploaded file to its pipeline stage and role. Use those files when mirroring or reviewing the dataset.
