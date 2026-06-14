#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGEEOF'
Usage: bash scripts/introspect_uncertainty.sh [--smoke] [--dry-run] [extra imaginaryRL_v2.py args]

Validates the Stage 4 introspection/uncertainty path from retained artifacts.
The implementation is shared with CIQL training: imaginaryRL_v2.py loads real and
imagined data, builds/reuses merged caches, trains uncertainty rewards, then
hands the resulting rewards to CIQL. Use --smoke for a short run that stops after
minimal uncertainty and CIQL updates.
USAGEEOF
}

smoke=0
dry_run=0
extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke=1; shift ;;
    --dry-run|--print) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

source "$(dirname "$0")/plfb_common.sh"
plfb_cd_repo

if [[ "$dry_run" != 1 ]]; then
  mkdir -p "$PLFB_MODEL_LOG_ROOT" "$PLFB_MERGED_DATA_CACHE_ROOT" "$PLFB_IRL_LOG_ROOT"
fi

if [[ "$smoke" == 1 ]]; then
  : "${PLFB_TRAIN_STEPS:=20}"
  : "${PLFB_STEPS_PER_EPOCH:=10}"
  : "${PLFB_EVAL_TRIALS:=1}"
  : "${PLFB_EVAL_ENVS:=11_vs_11_level_0}"
  : "${PLFB_UNCERTAINTY_STEPS:=20}"
  : "${PLFB_UNCERTAINTY_STEPS_PER_EPOCH:=10}"
  : "${PLFB_TRAIN_COMMENT:=smoke-stage4-uncertainty}"
else
  : "${PLFB_TRAIN_STEPS:=200000}"
  : "${PLFB_STEPS_PER_EPOCH:=3000}"
  : "${PLFB_EVAL_TRIALS:=10}"
  : "${PLFB_EVAL_ENVS:=11_vs_11_level_0,11_vs_11_level_1,11_vs_11_level_2}"
  : "${PLFB_TRAIN_COMMENT:=alpha-test2}"
fi

if [[ -z "${PLFB_UNCERTAINTY_MODEL_PATH:-}" && -f "${PLFB_STRICT_FIRST_STAGE_MODEL_PATH:-}" ]]; then
  PLFB_UNCERTAINTY_MODEL_PATH="$PLFB_STRICT_FIRST_STAGE_MODEL_PATH"
fi

if [[ "$dry_run" != 1 ]]; then
  if [[ -n "${PLFB_MERGED_DATA_CACHE_FILE:-}" ]]; then
    plfb_require_file "$PLFB_MERGED_DATA_CACHE_FILE"
  else
    plfb_require_dir "$PLFB_IMAGINARY_DATASET_PATH"
  fi
  if [[ -n "${PLFB_UNCERTAINTY_MODEL_PATH:-}" ]]; then
    plfb_require_file "$PLFB_UNCERTAINTY_MODEL_PATH"
  fi
fi

cmd=(
  "$PLFB_PYTHON" football_llm/learning/imaginaryRL_v2.py
  --env_name football
  --alg_type CIQL
  --comment "$PLFB_TRAIN_COMMENT"
  --alpha "${PLFB_ALPHA:-60.0}"
  --target_value "${PLFB_TARGET_VALUE:-0.02}"
  --ent_target_coef "${PLFB_ENT_TARGET_COEF:-0.8}"
  --coef_t "${PLFB_COEF_T:-0.5}"
  --coef_r "${PLFB_COEF_R:-0.5}"
  --strategy "${PLFB_STRATEGY:-all_replaced}"
  --fake_rollout_num "${PLFB_FAKE_ROLLOUT_NUM:--1}"
  --data_version "${PLFB_DATA_VERSION:-v1}"
  --device "${PLFB_DEVICE:-cuda:0}"
  --eval_trials "$PLFB_EVAL_TRIALS"
  --eval_env_names "$PLFB_EVAL_ENVS"
  --n_steps "$PLFB_TRAIN_STEPS"
  --n_steps_per_epoch "$PLFB_STEPS_PER_EPOCH"
  --obs_stack_num "${PLFB_OBS_STACK_NUM:-4}"
)

if [[ -n "${PLFB_UNCERTAINTY_STEPS:-}" ]]; then
  cmd+=(--uncertainty_n_steps "$PLFB_UNCERTAINTY_STEPS")
fi
if [[ -n "${PLFB_UNCERTAINTY_STEPS_PER_EPOCH:-}" ]]; then
  cmd+=(--uncertainty_n_steps_per_epoch "$PLFB_UNCERTAINTY_STEPS_PER_EPOCH")
fi
if [[ -n "${PLFB_UNCERTAINTY_BATCH_SIZE:-}" ]]; then
  cmd+=(--uncertainty_batch_size "$PLFB_UNCERTAINTY_BATCH_SIZE")
fi
if [[ -n "${PLFB_UNCERTAINTY_MODEL_PATH:-}" ]]; then
  cmd+=(--uncertainty_model_path "$PLFB_UNCERTAINTY_MODEL_PATH")
fi
if [[ -n "${PLFB_MERGED_DATA_CACHE_FILE:-}" ]]; then
  cmd+=(--merged_data_cache_file "$PLFB_MERGED_DATA_CACHE_FILE")
fi
if [[ "${PLFB_SKIP_GEN_DATA:-0}" == 1 ]]; then
  cmd+=(--skip_gen_data)
fi
cmd+=("${extra_args[@]}")

if [[ "$dry_run" == 1 ]]; then
  plfb_print_env
  plfb_print_command "${cmd[@]}"
else
  plfb_run "${cmd[@]}"
fi
