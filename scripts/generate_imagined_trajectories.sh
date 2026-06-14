#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGEEOF'
Usage: bash scripts/generate_imagined_trajectories.sh [--dry-run] [extra generate_main.py args]

Runs Stage 3 LLM-backed imagined trajectory generation. The public artifact
workflow does not need this step because generated trajectories are retained.
Required for a real regeneration run: OPENAI_API_KEY and the relevant data dirs.
Set PLFB_BC_MODEL_ROOT when using BC-guided generation modes.
PLFB_SAMPLED_DATA_PATH defaults to PLFB_WORK_ROOT/sampled_data.
Set PLFB_OPENAI_CHAT_MODEL to choose the chat model and OPENAI_BASE_URL for compatible providers.
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
: "${PLFB_USE_OPENAI_COMPAT_CLIENT:=1}"
export PLFB_USE_OPENAI_COMPAT_CLIENT

cmd=("$PLFB_PYTHON" football_llm/llm/generate_main.py)
cmd+=("${extra_args[@]}")

if [[ "$dry_run" == 1 ]]; then
  plfb_print_env
  plfb_print_command "${cmd[@]}"
  exit 0
fi

[[ -n "${OPENAI_API_KEY:-}" ]] || plfb_die "set OPENAI_API_KEY for LLM-backed imagined trajectory generation"
mkdir -p "$PLFB_IMAGINARY_DATASET_PATH" "$PLFB_SAMPLED_DATA_PATH"
plfb_require_dir "$PLFB_OFFLINE_DATASET_PATH"
plfb_require_dir "$PLFB_FILTER_PATH"
if [[ -z "${PLFB_BC_MODEL_ROOT:-}" || ! -d "${PLFB_BC_MODEL_ROOT:-}" ]]; then
  printf 'warning: PLFB_BC_MODEL_ROOT is not set to an existing directory; BC-guided modes may fail.\n' >&2
fi
plfb_run "${cmd[@]}"
