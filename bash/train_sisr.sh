#!/usr/bin/env bash
set -e

source .venv/bin/activate

configs=("drcan" "swinir" "hat")

for config in "${configs[@]}"
do
    echo "Training: $config"
    python tasks/train.py --task sisr --config configs/$config.yaml
done