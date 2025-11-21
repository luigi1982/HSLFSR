#!/usr/bin/env bash
set -e

source .venv/bin/activate

config=("f3dun")
models=("f4dun")
task="lfsr"

for model in "${models[@]}"
do
    echo "Training: $config $model"
    python tasks/train.py --task $task --config configs/$config.yaml --modification $model
done