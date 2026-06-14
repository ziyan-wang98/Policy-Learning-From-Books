#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGEEOF'
Usage: bash scripts/book_understanding.sh [--dry-run]

Runs Stage 1 book understanding with user-supplied tutorial JSONL.
Required for a real run: PLFB_BOOK_JSONL and OPENAI_API_KEY.
Set PLFB_OPENAI_CHAT_MODEL and PLFB_OPENAI_AGG_MODEL to choose current models.
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
: "${PLFB_UNDERSTANDING_DATA_DIR:=$PLFB_WORK_ROOT/understanding_data}"
: "${PLFB_UNDERSTANDING_RES_ROOT:=football_data}"

if [[ "$dry_run" != 1 ]]; then
  mkdir -p "$PLFB_UNDERSTANDING_DATA_DIR"
  [[ -n "${PLFB_BOOK_JSONL:-}" ]] || plfb_die "set PLFB_BOOK_JSONL to a tutorial JSONL file"
  [[ -n "${OPENAI_API_KEY:-}" ]] || plfb_die "set OPENAI_API_KEY for LLM-backed understanding"
  plfb_require_file "$PLFB_BOOK_JSONL"
  cp "$PLFB_BOOK_JSONL" "$PLFB_UNDERSTANDING_DATA_DIR/book_subset.jsonl"
fi

cmd=(
  "$PLFB_PYTHON" plfb-uri/main_understanding.py
  sim_info=fb
  path=fb
  path.root="$PLFB_REPO_ROOT/plfb-uri"
  path.data_path="$PLFB_UNDERSTANDING_DATA_DIR"
  path.book=book_subset.jsonl
  path.res_root="$PLFB_UNDERSTANDING_RES_ROOT"
)
cmd+=("${extra_args[@]}")

if [[ "$dry_run" == 1 ]]; then
  plfb_print_env
  plfb_print_command "${cmd[@]}"
else
  plfb_run "${cmd[@]}"
fi
