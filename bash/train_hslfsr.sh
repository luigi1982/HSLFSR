#!/usr/bin/env bash
set -e

source .venv/bin/activate

configs=("epit")
models=("short")
task="lfsr"

for (( i=0; i<${#configs[@]}; i++ ));
do
    config=${configs[$i]}
    model=${models[$i]}
    echo "Training: $config $model"
    python tasks/train.py --task $task --config configs/$config.yaml --modification ${config}_$model
done