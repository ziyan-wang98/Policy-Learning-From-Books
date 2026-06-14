#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGEEOF'
Usage: bash scripts/train_ciql.sh [--smoke] [--dry-run] [extra imaginaryRL_v2.py args]

Runs the final CIQL training configuration from retained public artifacts.
Use --smoke for a short GPU run that exercises data loading, uncertainty, and CIQL.
Set PLFB_MERGED_DATA_CACHE_FILE=/path/to/cache.npz to force an exact merged cache.
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
  mkdir -p "$PLFB_IRL_LOG_ROOT" "$PLFB_MODEL_LOG_ROOT" "$PLFB_MERGED_DATA_CACHE_ROOT"
fi

if [[ "$smoke" == 1 ]]; then
  : "${PLFB_TRAIN_STEPS:=20}"
  : "${PLFB_STEPS_PER_EPOCH:=10}"
  : "${PLFB_EVAL_TRIALS:=1}"
  : "${PLFB_EVAL_ENVS:=11_vs_11_level_0}"
  : "${PLFB_UNCERTAINTY_STEPS:=20}"
  : "${PLFB_UNCERTAINTY_STEPS_PER_EPOCH:=10}"
  : "${PLFB_TRAIN_COMMENT:=smoke-public-layout}"
else
  : "${PLFB_TRAIN_STEPS:=200000}"
  : "${PLFB_STEPS_PER_EPOCH:=3000}"
  : "${PLFB_EVAL_TRIALS:=10}"
  : "${PLFB_EVAL_ENVS:=11_vs_11_level_0,11_vs_11_level_1,11_vs_11_level_2}"
  : "${PLFB_TRAIN_COMMENT:=alpha-test2}"
fi

: "${PLFB_ALG_TYPE:=CIQL}"
: "${PLFB_ALPHA:=60.0}"
: "${PLFB_TARGET_VALUE:=0.02}"
: "${PLFB_ENT_TARGET_COEF:=0.8}"
: "${PLFB_COEF_T:=0.5}"
: "${PLFB_COEF_R:=0.5}"
: "${PLFB_STRATEGY:=all_replaced}"
: "${PLFB_FAKE_ROLLOUT_NUM:=-1}"
: "${PLFB_OBS_STACK_NUM:=4}"
: "${PLFB_TRACE_REAL_NUM:=0}"
: "${PLFB_EXTRA_REAL_TRAJ_NUM:=0}"
: "${PLFB_ROLLOUT_NUM:=0}"
: "${PLFB_SKIP_GEN_DATA:=false}"
: "${PLFB_OBSERVATION_SCALER:=none}"
: "${PLFB_SEED:=0}"
: "${PLFB_UNCERTAINTY_MODEL_PATH:=}"
: "${PLFB_MERGED_DATA_CACHE_FILE:=}"
if [[ -z "$PLFB_UNCERTAINTY_MODEL_PATH" && -f "${PLFB_STRICT_FIRST_STAGE_MODEL_PATH:-}" ]]; then
  PLFB_UNCERTAINTY_MODEL_PATH="$PLFB_STRICT_FIRST_STAGE_MODEL_PATH"
fi

if [[ "$dry_run" != 1 ]]; then
  if [[ -n "$PLFB_MERGED_DATA_CACHE_FILE" ]]; then
    plfb_require_file "$PLFB_MERGED_DATA_CACHE_FILE"
  else
    plfb_require_dir "$PLFB_IMAGINARY_DATASET_PATH"
  fi
  if [[ -n "$PLFB_UNCERTAINTY_MODEL_PATH" ]]; then
    plfb_require_file "$PLFB_UNCERTAINTY_MODEL_PATH"
  fi
fi

PLFB_TRAIN_COMMENT_MAX_CHARS=24
if (( ${#PLFB_TRAIN_COMMENT} > PLFB_TRAIN_COMMENT_MAX_CHARS )); then
  plfb_die "PLFB_TRAIN_COMMENT is ${#PLFB_TRAIN_COMMENT} characters; keep it <= ${PLFB_TRAIN_COMMENT_MAX_CHARS} to stay below d3rlpy filesystem name limits. Use a short label such as alpha-test2 or ciqlresume."
fi

cmd=(
  env
  "PLFB_TRACE_REAL_NUM=$PLFB_TRACE_REAL_NUM"
  "PLFB_EXTRA_REAL_TRAJ_NUM=$PLFB_EXTRA_REAL_TRAJ_NUM"
  "PLFB_ROLLOUT_NUM=$PLFB_ROLLOUT_NUM"
  "PLFB_SKIP_GEN_DATA=$PLFB_SKIP_GEN_DATA"
  "PLFB_MERGED_DATA_CACHE_FILE=$PLFB_MERGED_DATA_CACHE_FILE"
  "$PLFB_PYTHON" football_llm/learning/imaginaryRL_v2.py
  --env_name football
  --alg_type "$PLFB_ALG_TYPE"
  --comment "$PLFB_TRAIN_COMMENT"
  --alpha "$PLFB_ALPHA"
  --target_value "$PLFB_TARGET_VALUE"
  --ent_target_coef "$PLFB_ENT_TARGET_COEF"
  --coef_t "$PLFB_COEF_T"
  --coef_r "$PLFB_COEF_R"
  --strategy "$PLFB_STRATEGY"
  --fake_rollout_num "$PLFB_FAKE_ROLLOUT_NUM"
  --data_version "${PLFB_DATA_VERSION:-v1}"
  --device "${PLFB_DEVICE:-cuda:0}"
  --seed "$PLFB_SEED"
  --eval_trials "$PLFB_EVAL_TRIALS"
  --eval_env_names "$PLFB_EVAL_ENVS"
  --n_steps "$PLFB_TRAIN_STEPS"
  --n_steps_per_epoch "$PLFB_STEPS_PER_EPOCH"
  --obs_stack_num "$PLFB_OBS_STACK_NUM"
  --observation_scaler "$PLFB_OBSERVATION_SCALER"
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
case "${PLFB_SKIP_GEN_DATA,,}" in
  1|true|yes|on) cmd+=(--skip_gen_data) ;;
  0|false|no|off) ;;
  *) plfb_die "PLFB_SKIP_GEN_DATA must be true or false" ;;
esac
cmd+=("${extra_args[@]}")

plfb_print_final_ciql_defaults() {
  cat <<ENVEOF
Final CIQL traceability defaults:
  comment=$PLFB_TRAIN_COMMENT
  algorithm=$PLFB_ALG_TYPE
  alpha=$PLFB_ALPHA
  target_value=$PLFB_TARGET_VALUE
  ent_target_coef=$PLFB_ENT_TARGET_COEF
  coef_t=$PLFB_COEF_T
  coef_r=$PLFB_COEF_R
  strategy=$PLFB_STRATEGY
  fake_rollout_num=$PLFB_FAKE_ROLLOUT_NUM
  trace_real_num=$PLFB_TRACE_REAL_NUM
  extra_real_traj_num=$PLFB_EXTRA_REAL_TRAJ_NUM
  rollout_num=$PLFB_ROLLOUT_NUM
  skip_gen_data=$PLFB_SKIP_GEN_DATA
  obs_stack_num=$PLFB_OBS_STACK_NUM
  observation_scaler=$PLFB_OBSERVATION_SCALER
  seed=$PLFB_SEED
  uncertainty_model_path=${PLFB_UNCERTAINTY_MODEL_PATH:-}
  merged_data_cache_file=${PLFB_MERGED_DATA_CACHE_FILE:-}
  n_steps=$PLFB_TRAIN_STEPS
  n_steps_per_epoch=$PLFB_STEPS_PER_EPOCH
ENVEOF
}

if [[ "$dry_run" == 1 ]]; then
  plfb_print_env
  plfb_print_final_ciql_defaults
  plfb_print_command "${cmd[@]}"
else
  plfb_run "${cmd[@]}"
fi
