#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"


start_idx="$1"
end_idx="$2"
scope="$3" # <17

for ((i=start_idx; i<=end_idx; i++))
do  
    echo "Collecting data for $i"
    python "$PROJECT_ROOT/rehearsing/main.py" --gen_traj_index "$i" "${@:4}" --gen_length 10 &
    sleep 60
    # Wait for all background processes to complete after processing every $scope iterations
    if [ $((($i + 1) % $scope)) -eq 0 ]; then 
        wait
    fi
done
wait
