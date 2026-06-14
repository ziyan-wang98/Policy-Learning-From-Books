#!/usr/bin/env bash
set -euo pipefail

# This script is used to collect the dataset for the LLM model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PLFB_CONDA_SH:-}" ]]; then
    source "$PLFB_CONDA_SH"
fi
if [[ -n "${PLFB_CONDA_PREFIX:-}" ]]; then
    conda activate "$PLFB_CONDA_PREFIX"
elif [[ -n "${PLFB_CONDA_ENV:-}" ]]; then
    conda activate "$PLFB_CONDA_ENV"
fi

cd "$SCRIPT_DIR"

# algos=("tizero_agent" "rule_based_2")
algos=("rule_based_2")
envs=("11_vs_11_level_1" "11_vs_11_level_2" "11_vs_11_level_3")
# More than three parallel workers can saturate CPU and starve other processes.
batch_run=3
let num_games_1=100/batch_run

for ((i=1; i<=$num_games_1; i++))
do
    for algo in "${algos[@]}"
    do
        for env in "${envs[@]}"
        do
            echo "Collecting data for $algo number $i / $num_games_1 in $env"
            for ((j=1; j<=$batch_run; j++))
            do
                python main.py --logall --algo $algo --offline_dataset_collection --environment $env &
                sleep 10
            done
            wait
        done
    done
done
wait
