#!/bin/bash


num_pl=21
for ((i=0; i<=num_pl; i++))
do
    echo "Pretraining for player $i / $num_pl"
    cuda_num=$((i%3))
    python pretrain_builtin.py --comment "final-nz-10x-fix-feat" --player $i --batch_size 4096 --policy_type "stochastic_weighted" --regen_data --device cuda:$cuda_num &
    sleep 60
done

wait