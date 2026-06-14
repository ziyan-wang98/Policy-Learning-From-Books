#!/bin/bash


CMD="python load_and_eval_v2.py --eval_times 1 --exp_name RT2_test --eval_algo RT2"

PARALLEL_RUNS=5

ROUNDS=2


for round in $(seq 1 $ROUNDS)
do
    echo "Starting round $round"

    for i in $(seq 1 $PARALLEL_RUNS)
    do
        $CMD &
    done
    
    wait
    
    echo "Finished round $round"
done

echo "All rounds completed"