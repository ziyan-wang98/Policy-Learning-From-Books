#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGEEOF'
Usage: bash scripts/eval_ciql.sh [--dry-run]

Evaluates the released final CIQL checkpoint. Configure with:
  PLFB_ARTIFACT_ROOT, PLFB_MODEL_ROOT, PLFB_EVAL_OUTPUT, PLFB_EVAL_TIMES,
  PLFB_EVAL_ENVS, PLFB_MIN_MODEL_STEP, PLFB_DEVICE-related scheduler setup.
USAGEEOF
}

dry_run=0
extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--print) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

source "$(dirname "$0")/plfb_common.sh"
plfb_cd_repo
if [[ "$dry_run" != 1 ]]; then
  mkdir -p "$PLFB_EVAL_OUTPUT"
  plfb_require_file "$PLFB_MODEL_ROOT/final_uri_best/model_rew_0.5&step_48000.d3"
fi

cmd=(
  "$PLFB_PYTHON" football_llm/learning/load_and_eval_v2.py
  --eval_algo "${PLFB_EVAL_ALGO:-CIQL}"
  --eval_times "${PLFB_EVAL_TIMES:-40}"
  --exp_name "${PLFB_EVAL_EXP_NAME:-final_uri_best}"
  --model_root_path "$PLFB_MODEL_ROOT"
  --eval_output_dir "$PLFB_EVAL_OUTPUT"
  --min_model_step "${PLFB_MIN_MODEL_STEP:-40000}"
  --eval_env_names "${PLFB_EVAL_ENVS:-11_vs_11_level_0,11_vs_11_level_1,11_vs_11_level_2}"
  --obs_stack_num "${PLFB_OBS_STACK_NUM:-4}"
  --acs_replace_strategy "${PLFB_STRATEGY:-all_replaced}"
)
cmd+=("${extra_args[@]}")

if [[ "$dry_run" == 1 ]]; then
  plfb_print_env
  plfb_print_command "${cmd[@]}"
else
  plfb_run "${cmd[@]}"
fi
