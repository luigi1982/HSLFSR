#!/usr/bin/env bash
set -e

source .venv/bin/activate

configs=("drcan" "swinir" "hat")

for config in "${configs[@]}"
do
    echo "Training: $config"
    python sisr/train.py --config configs/$config.yaml
done