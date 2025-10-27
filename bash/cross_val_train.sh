#!/usr/bin/env bash
set -e

source .venv/bin/activate

configs=("distg" "adam" "epit" "f3dun" "ssaformer")

for config in "${configs[@]}"
do
    echo "Training: $config"
    python tasks/cross_val_training.py --task lfsr --config configs/$config.yaml
done