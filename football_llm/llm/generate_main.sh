#!/usr/bin/env bash
set -euo pipefail

# This script is used to collect the dataset for the LLM model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set OPENAI_API_KEY before running. Use OPENAI_BASE_URL for OpenAI-compatible providers." >&2
    exit 1
fi

if [[ -n "${PLFB_CONDA_SH:-}" ]]; then
    source "$PLFB_CONDA_SH"
fi
if [[ -n "${PLFB_CONDA_PREFIX:-}" ]]; then
    conda activate "$PLFB_CONDA_PREFIX"
elif [[ -n "${PLFB_CONDA_ENV:-}" ]]; then
    conda activate "$PLFB_CONDA_ENV"
fi

cd "$SCRIPT_DIR"

for i in {0..49}
do  
    echo "Collecting data for $i / 50"
    python generate_main.py --number $((i % 50)) &

    # Wait for all background processes to complete after processing every 50 iterations
    if [ $((i % 50)) -eq 49 ]; then 
        wait
    fi
done
