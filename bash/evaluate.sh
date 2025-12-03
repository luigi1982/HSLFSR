#!/usr/bin/env bash
set -e

source .venv/bin/activate

task=$1
config=$2
exp=$3

configs=($config)
exps=($exp)

for (( i=0; i<${#configs[@]}; i++ ));
do
    config=${configs[$i]}
    exp=${exps[$i]}
    echo "Evaluating: $config"
    python tasks/evaluate.py --task $task --config configs/$config.yaml --experiment $exp
done