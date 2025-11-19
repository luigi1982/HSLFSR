#!/usr/bin/env bash
set -e

source .venv/bin/activate

config=("epit")
models=("swin")
task="lfsr"

for model in "${models[@]}"
do
    echo "Training: $config $model"
    python tasks/train.py --task $task --config configs/$config.yaml --modification ${config}_$model
done