#!/usr/bin/env bash
# Template only: edit resource lines and activation commands for your scheduler.
#SBATCH --job-name=plfb-generate-trajectories
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${PLFB_REPO_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${PLFB_ARTIFACT_ROOT:=$PLFB_REPO_ROOT/plfb_artifacts}"
: "${PLFB_WORK_ROOT:=$PLFB_REPO_ROOT/runs}"

cd "$PLFB_REPO_ROOT"

# Replace this with your environment activation command.
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate plfb-gfootball

bash scripts/generate_imagined_trajectories.sh
