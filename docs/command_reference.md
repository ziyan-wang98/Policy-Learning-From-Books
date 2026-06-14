# Command Reference

## Setup

```bash
micromamba create -y -n plfb-universal -f environment-universal.yml
micromamba activate plfb-universal
python -m pip install --no-deps -e football_llm/d3rlpy
python -m pip install -e football_llm/setup/football
export PYTHONPATH="$PWD/football_llm:$PWD/football_llm/d3rlpy:$PWD/football_llm/setup/football:$PWD/plfb-uri:$PYTHONPATH"
python scripts/check_environment.py --quick --strict-football
```

## Download and Validate Artifacts

```bash
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
bash scripts/download_artifacts.sh
python scripts/check_environment.py --check-artifacts --artifact-root "$PLFB_ARTIFACT_ROOT"
bash scripts/smoke_stage.sh all
```

Override the Hugging Face dataset or local directory:

```bash
bash scripts/download_artifacts.sh --repo ziyan98/plfb --local-dir ./plfb_artifacts
bash scripts/download_artifacts.sh --skip-download --local-dir ./plfb_artifacts
```

## Stage Smoke Checks

```bash
bash scripts/smoke_stage.sh 0
bash scripts/smoke_stage.sh 1
bash scripts/smoke_stage.sh 2
bash scripts/smoke_stage.sh 3
bash scripts/smoke_stage.sh 4
bash scripts/smoke_stage.sh 5
bash scripts/smoke_stage.sh 6
bash scripts/smoke_stage.sh all
```

Set `PLFB_STRICT_IMPORTS=1` to make runtime imports fail hard. Set `PLFB_SMOKE_EVAL=1` for a one-trial final checkpoint evaluation. Set `PLFB_SMOKE_TRAIN=1` only when you intentionally want short GPU training smoke runs.

## Evaluate Final CIQL

```bash
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
export PLFB_WORK_ROOT=$PWD/runs
bash scripts/eval_ciql.sh --dry-run
bash scripts/eval_ciql.sh
```

Useful overrides:

```bash
PLFB_EVAL_TIMES=10 bash scripts/eval_ciql.sh
PLFB_EVAL_ENVS=11_vs_11_level_0 bash scripts/eval_ciql.sh
PLFB_EVAL_OUTPUT=$PWD/eval_result bash scripts/eval_ciql.sh
```

## Re-run CIQL From Public Artifacts

```bash
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
export PLFB_WORK_ROOT=$PWD/runs
bash scripts/train_ciql.sh --dry-run
bash scripts/train_ciql.sh
```

The wrapper defaults to the public 2024-02 merged cache and strict first-stage checkpoint. Override them only for controlled experiments:

```bash
PLFB_MERGED_DATA_CACHE_FILE=/path/to/cache.npz bash scripts/train_ciql.sh --dry-run
PLFB_UNCERTAINTY_MODEL_PATH=/path/to/model.d3 bash scripts/train_ciql.sh --dry-run
PLFB_TRAIN_STEPS=20 PLFB_STEPS_PER_EPOCH=10 PLFB_EVAL_TRIALS=1 bash scripts/train_ciql.sh --smoke
```

## Validate or Train First-stage Uncertainty

```bash
bash scripts/introspect_uncertainty.sh --smoke --dry-run
bash scripts/introspect_uncertainty.sh --smoke
```

The public first-stage replay checkpoint is under `artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3`.

## LLM-backed Stages

```bash
export PLFB_BOOK_JSONL=/path/to/book_subset.jsonl
export OPENAI_API_KEY=...
export PLFB_OPENAI_CHAT_MODEL=gpt-4o-mini
export PLFB_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
export PLFB_USE_OPENAI_COMPAT_CLIENT=1
bash scripts/book_understanding.sh --dry-run
bash scripts/book_understanding.sh
bash scripts/prepare_retrieval_context.sh
bash scripts/generate_imagined_trajectories.sh --dry-run
bash scripts/generate_imagined_trajectories.sh
```

Optional OpenAI-compatible backend:

```bash
export OPENAI_BASE_URL=https://your-compatible-endpoint.example/v1
export OPENAI_API_KEY=$YOUR_PROVIDER_KEY
```

## Summarize CIQL Runs

```bash
python scripts/summarize_ciql_run.py \
  --run-dir "$PLFB_WORK_ROOT/IRL_LOG/football/d3rlpy_logs/<run_name>" \
  --target-step 48000 \
  --output-json "$PLFB_WORK_ROOT/ciql_run_summary.json" \
  --pretty
```

The summarizer is read-only and reports recommended checkpoints to keep before manual curation.
