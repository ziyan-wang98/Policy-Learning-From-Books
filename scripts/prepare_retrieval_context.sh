#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGEEOF'
Usage: bash scripts/prepare_retrieval_context.sh [--dry-run]

Validates the Stage 2 retrieval/context artifact layout used by generation.
The public workflow uses retained retrieval files under book_derived/retrieval.
USAGEEOF
}

dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--print) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/plfb_common.sh"
plfb_cd_repo

if [[ "$dry_run" == 1 ]]; then
  plfb_print_env
  printf 'retrieval context path: %s\n' "$PLFB_FILTER_PATH"
  exit 0
fi

plfb_require_dir "$PLFB_FILTER_PATH"
plfb_run "$PLFB_PYTHON" scripts/smoke_pipeline.py --mode module-map --repo-root "$PLFB_REPO_ROOT" --artifact-root "$PLFB_ARTIFACT_ROOT"
plfb_run "$PLFB_PYTHON" scripts/smoke_pipeline.py --mode layout --repo-root "$PLFB_REPO_ROOT" --artifact-root "$PLFB_ARTIFACT_ROOT"
plfb_run "$PLFB_PYTHON" scripts/smoke_pipeline.py --mode data-contract --repo-root "$PLFB_REPO_ROOT" --artifact-root "$PLFB_ARTIFACT_ROOT"
