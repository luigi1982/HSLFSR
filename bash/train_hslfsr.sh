#!/usr/bin/env bash
set -e

source .venv/bin/activate

config=("det")
models=("swin_angular_cascaded" "swin_angular" "ablation_angular")
task="lfsr"

for model in "${models[@]}"
do
    echo "Training: $config $model"
    python tasks/train.py --task $task --config configs/$config.yaml --modification ${config}_$model
done