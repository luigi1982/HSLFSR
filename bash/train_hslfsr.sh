#!/usr/bin/env bash
set -e

source .venv/bin/activate

config=("det")
models=("swin_angular_cascaded" "swin_spatial" "swin_angular_spatial")
task="lfsr"

for model in "${models[@]}"
do
    echo "Training: $config $model"
    python tasks/train.py --task $task --config configs/$config.yaml --modification ${config}_$model
done