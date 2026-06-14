#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGEEOF'
Usage: bash scripts/smoke_stage.sh STAGE

STAGE can be 0, 1, 2, 3, 4, 5, 6, or all.
Set PLFB_STRICT_IMPORTS=1 to make runtime imports fail hard.
By default, each stage runs static/data checks plus the public wrapper dry-run.
Set PLFB_SMOKE_TRAIN=1 to run short GPU training smokes in stage 4/5/all.
Set PLFB_SMOKE_EVAL=1 to run a one-trial final checkpoint evaluation in stage 6/all.
Set PLFB_EVAL_TIMES to override the one-trial smoke default.
USAGEEOF
}

stage="${1:-all}"
case "$stage" in
  0|1|2|3|4|5|6|all) ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

source "$(dirname "$0")/plfb_common.sh"
plfb_cd_repo

strict_args=()
if [[ "${PLFB_STRICT_IMPORTS:-0}" == 1 ]]; then
  strict_args+=(--strict-imports)
fi

smoke() {
  local mode="$1"
  shift || true
  plfb_run "$PLFB_PYTHON" scripts/smoke_pipeline.py --mode "$mode" --repo-root "$PLFB_REPO_ROOT" --artifact-root "$PLFB_ARTIFACT_ROOT" "${strict_args[@]}" "$@"
}

run_eval_smoke() {
  plfb_run env PLFB_EVAL_TIMES="${PLFB_EVAL_TIMES:-1}" bash scripts/eval_ciql.sh
}

run_wrapper_dry_run() {
  plfb_run bash "$@" --dry-run
}

run_retrieval_normalization_dry_run() {
  local stage1_root="${PLFB_STAGE1_ROOT:-$PLFB_ARTIFACT_ROOT/book_derived/v4-gpt-3.5-turbo-1106-level-strict}"
  local output_root="${PLFB_STAGE2_NORMALIZED_ROOT:-$PLFB_WORK_ROOT/book_derived/retrieval}"
  local limit="${PLFB_STAGE2_NORMALIZE_SMOKE_LIMIT:-2}"
  plfb_run "$PLFB_PYTHON" scripts/normalize_retrieval_context.py \
    --stage1-root "$stage1_root" --output-root "$output_root" --limit "$limit" --dry-run
}

run_stage() {
  case "$1" in
    0)
      env_args=(--quick --repo-root "$PLFB_REPO_ROOT")
      if [[ "${PLFB_STRICT_IMPORTS:-0}" == 1 ]]; then
        env_args+=(--strict-football)
      fi
      plfb_run "$PLFB_PYTHON" scripts/check_environment.py "${env_args[@]}"
      smoke source
      smoke api
      smoke imports
      ;;
    1)
      smoke source
      smoke module-map
      smoke layout
      run_wrapper_dry_run scripts/book_understanding.sh
      ;;
    2)
      smoke module-map
      smoke layout
      smoke data-contract
      run_retrieval_normalization_dry_run
      run_wrapper_dry_run scripts/prepare_retrieval_context.sh
      ;;
    3)
      smoke module-map
      smoke layout
      smoke data-contract
      run_wrapper_dry_run scripts/generate_imagined_trajectories.sh
      ;;
    4)
      smoke module-map
      smoke layout
      smoke data-contract
      plfb_run bash scripts/introspect_uncertainty.sh --smoke --dry-run
      if [[ "${PLFB_SMOKE_TRAIN:-0}" == 1 ]]; then
        plfb_run bash scripts/introspect_uncertainty.sh --smoke
      fi
      ;;
    5)
      smoke module-map
      smoke layout
      smoke data-contract
      plfb_run bash scripts/train_ciql.sh --smoke --dry-run
      if [[ "${PLFB_SMOKE_TRAIN:-0}" == 1 ]]; then
        plfb_run bash scripts/train_ciql.sh --smoke
      fi
      ;;
    6)
      smoke eval-report
      smoke layout
      smoke data-contract
      run_wrapper_dry_run scripts/eval_ciql.sh
      if [[ "${PLFB_SMOKE_EVAL:-0}" == 1 ]]; then
        run_eval_smoke
      fi
      ;;
  esac
}

if [[ "$stage" == all ]]; then
  for single_stage in 0 1 2 3 4 5 6; do
    run_stage "$single_stage"
  done
else
  run_stage "$stage"
fi
