#!/usr/bin/env bash
set -e

source .venv/bin/activate

config=$1
exp=$2

configs=($config)
exps=($exp)

for (( i=0; i<${#configs[@]}; i++ ));
do
    config=${configs[$i]}
    exp=${exps[$i]}
    echo "Evaluating: $config"
    python tasks/evaluate.py --task lfsr --config configs/$config.yaml --experiment $exp
done