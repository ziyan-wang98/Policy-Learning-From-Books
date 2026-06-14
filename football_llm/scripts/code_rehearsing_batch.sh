#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"


for i in {45..50}
do  
    echo "Collecting data for $i"
    python "$PROJECT_ROOT/rehearsing/main.py" --gen_traj_index "$i" --just_gen_code &
    sleep 60
    # Wait for all background processes to complete after processing every 10 iterations
    if [ $((($i + 1) % 15)) -eq 0 ]; then 
        wait
    fi
done
wait
