#!/usr/bin/env bash
# Template only: edit resource lines and activation commands for your scheduler.
#SBATCH --job-name=plfb-book-understanding
#SBATCH --partition=<cpu-or-gpu-partition>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${PLFB_REPO_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${PLFB_WORK_ROOT:=$PLFB_REPO_ROOT/runs}"

cd "$PLFB_REPO_ROOT"

# Replace this with your environment activation command.
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate plfb

bash scripts/book_understanding.sh
