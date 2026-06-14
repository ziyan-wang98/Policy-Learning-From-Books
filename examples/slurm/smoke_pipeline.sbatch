#!/usr/bin/env bash
# Template only: edit resource lines and activation commands for your scheduler.
#SBATCH --job-name=plfb-smoke
#SBATCH --partition=<gpu-or-cpu-partition>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${PLFB_REPO_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${PLFB_ARTIFACT_ROOT:=$PLFB_REPO_ROOT/plfb_artifacts}"

cd "$PLFB_REPO_ROOT"

# Replace this with your environment activation command.
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate plfb-gfootball

bash scripts/smoke_stage.sh all
