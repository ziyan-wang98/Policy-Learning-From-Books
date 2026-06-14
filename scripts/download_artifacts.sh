#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGEEOF'
Usage: bash scripts/download_artifacts.sh [--skip-download] [--repo DATASET_ID] [--local-dir PATH]

Downloads the public PLfB artifact layout from Hugging Face and verifies the
final checkpoint checksum plus layout/report smoke checks.
USAGEEOF
}

skip_download=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) skip_download=1; shift ;;
    --repo) PLFB_HF_REPO="$2"; shift 2 ;;
    --local-dir) PLFB_ARTIFACT_ROOT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/plfb_common.sh"
plfb_cd_repo
mkdir -p "$PLFB_ARTIFACT_ROOT"

if [[ "$skip_download" != 1 ]]; then
  command -v hf >/dev/null 2>&1 || plfb_die "hf CLI not found; install huggingface_hub first"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  plfb_run hf download "$PLFB_HF_REPO" --repo-type dataset --local-dir "$PLFB_ARTIFACT_ROOT"
fi

final_ckpt="$PLFB_MODEL_ROOT/final_uri_best/model_rew_0.5&step_48000.d3"
plfb_require_file "$final_ckpt"
"$PLFB_PYTHON" - "$final_ckpt" <<'PYEOF'
from pathlib import Path
import hashlib
import sys
expected = "625a387b8701295838ff10beb631dd5052d1bb8eafb9b01af77947164119cd67"
path = Path(sys.argv[1])
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"checkpoint sha256 mismatch for {path}: {actual}")
print(f"checkpoint sha256 ok: {actual}")
PYEOF

plfb_run "$PLFB_PYTHON" scripts/smoke_pipeline.py --mode module-map --repo-root "$PLFB_REPO_ROOT" --artifact-root "$PLFB_ARTIFACT_ROOT"
plfb_run "$PLFB_PYTHON" scripts/smoke_pipeline.py --mode layout --repo-root "$PLFB_REPO_ROOT" --artifact-root "$PLFB_ARTIFACT_ROOT"
plfb_run "$PLFB_PYTHON" scripts/smoke_pipeline.py --mode data-contract --repo-root "$PLFB_REPO_ROOT" --artifact-root "$PLFB_ARTIFACT_ROOT"
plfb_run "$PLFB_PYTHON" scripts/smoke_pipeline.py --mode eval-report --repo-root "$PLFB_REPO_ROOT" --artifact-root "$PLFB_ARTIFACT_ROOT"
