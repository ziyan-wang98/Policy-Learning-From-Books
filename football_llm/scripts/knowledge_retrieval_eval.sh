#!/usr/bin/env bash
set -euo pipefail


# knowledge_retrieval_eval.sh: 5: Syntax error: "(" unexpected 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

eval_strategy_list=("state_field" "summary") 
knowledge_rep_list=("code" "language")
embed_model_list=("openai" "baai")

for eval_strategy in "${eval_strategy_list[@]}"
do
    for embed_model in "${embed_model_list[@]}"
    do
        for knowledge_rep in "${knowledge_rep_list[@]}"
        do
            echo "Evaluating $eval_strategy with $knowledge_rep and $embed_model"
            python "$PROJECT_ROOT/retrieval/evaluation.py" --eval_strategy "$eval_strategy" --knowledge_rep "$knowledge_rep" --embed_model "$embed_model" --top_k 10
        done
    done
done
wait
