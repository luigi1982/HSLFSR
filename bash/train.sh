#!/usr/bin/env bash
set -e

source .venv/bin/activate

configs=("epit")

for config in "${configs[@]}"
do
    echo "Training: $config"
    python train.py --config configs/$config.yaml
done