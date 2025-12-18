#!/usr/bin/env bash
set -e

source .venv/bin/activate

task=$1
model=$2
config=$3
exp=$4

configs=($config)
exps=($exp)

for (( i=0; i<${#configs[@]}; i++ ));
do
    config=${configs[$i]}
    exp=${exps[$i]}
    echo "Evaluating: $model"
    python tasks/evaluate.py --task $task --model $model --config configs/$config.yaml --experiment $exp
done