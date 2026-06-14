# Reproduction Guide

This guide uses only public paths and portable commands. It does not assume any specific workstation, computing environment, or filesystem layout.

## 1. Create the Environment

Use the universal environment for the full football runtime:

```bash
git clone https://github.com/ziyan-wang98/Policy-Learning-From-Books.git
cd Policy-Learning-From-Books
micromamba create -y -n plfb-universal -f environment-universal.yml
micromamba activate plfb-universal
python -m pip install --no-deps -e football_llm/d3rlpy
python -m pip install -e football_llm/setup/football
export PYTHONPATH="$PWD/football_llm:$PWD/football_llm/d3rlpy:$PWD/football_llm/setup/football:$PWD/plfb-uri:$PYTHONPATH"
python scripts/check_environment.py --quick --strict-football
```

`mamba` is equivalent. Conda classic can work, but the football stack uses conda-forge native packages and may need more solver memory and time. Use `environment.yml` only for lightweight book/data work when GFootball is not needed. Use `environment-hf-upload.yml` only for release maintenance.

## 2. Download and Verify Artifacts

```bash
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
bash scripts/download_artifacts.sh
python scripts/check_environment.py --check-artifacts --artifact-root "$PLFB_ARTIFACT_ROOT"
bash scripts/smoke_stage.sh all
```

The artifact check validates these release-critical files:

| File | Role |
| --- | --- |
| `artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3` | final released CIQL policy |
| `artifacts/football/final_uri_best/params.json` | final released CIQL parameters |
| `football/imaginary_dataset_0204/merged_data/v3datatrace_real_num=0&extra_real_traj_num=0&obs_stack_num=4&rollout_num=0.npz` | 2024-02 merged imagined dataset cache |
| `artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3` | public first-stage uncertainty checkpoint |

## 3. Evaluate the Released Final Model

```bash
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
export PLFB_WORK_ROOT=$PWD/runs
bash scripts/eval_ciql.sh --dry-run
bash scripts/eval_ciql.sh
```

The released model is `final_uri_best/model_rew_0.5&step_48000.d3`. The evaluation wrapper loads this checkpoint from `PLFB_ARTIFACT_ROOT` and writes logs under `PLFB_WORK_ROOT/eval_result`.

## 4. Re-run CIQL Without GPT/API Calls

The wrapper defaults to the public 2024-02 merged imagined dataset cache and, when available, the public first-stage uncertainty checkpoint:

```bash
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
export PLFB_WORK_ROOT=$PWD/runs
bash scripts/train_ciql.sh --dry-run
bash scripts/train_ciql.sh
```

Important defaults:

| Variable | Default |
| --- | --- |
| `PLFB_MERGED_DATA_CACHE_FILE` | `$PLFB_ARTIFACT_ROOT/football/imaginary_dataset_0204/merged_data/v3datatrace_real_num=0&extra_real_traj_num=0&obs_stack_num=4&rollout_num=0.npz` |
| `PLFB_UNCERTAINTY_MODEL_PATH` | `$PLFB_ARTIFACT_ROOT/artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3` when the file exists |
| `PLFB_TRAIN_STEPS` | `200000` |
| `PLFB_STEPS_PER_EPOCH` | `3000` |
| `PLFB_OBS_STACK_NUM` | `4` |
| `PLFB_SEED` | `0` |

Use `bash scripts/introspect_uncertainty.sh --smoke` if you want to validate the first-stage uncertainty path. The published `strict_repro_first_stage_ba0e02e` checkpoint is the public replay checkpoint used by the wrappers.

## 5. Regenerate LLM-backed Data

LLM-backed stages are optional for final model evaluation and no-API CIQL replay. They require your own text input, provider credentials, and, for BC-guided football trajectory generation, BC policy checkpoints:

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

The OpenAI-compatible client honors `OPENAI_BASE_URL`, so newer OpenAI models or compatible providers can be used for new data-generation experiments. Such regeneration is a new experiment and is not expected to byte-match the retained 2024-02 imagined dataset.

## 6. Scheduler Use

Templates are under `examples/slurm/`. Copy a template, edit scheduler resources and activation commands for your environment, then submit it:

```bash
cp examples/slurm/train_ciql.sbatch run_train_ciql.sbatch
# edit run_train_ciql.sbatch
sbatch run_train_ciql.sbatch
```
