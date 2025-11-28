#!/usr/bin/env bash
set -e

source .venv/bin/activate

configs=("epit" "det" "distg" "f3dun")

for config in "${configs[@]}"
do
    echo "Training: $config"
    python tasks/train.py --task lfsr --config configs/$config.yaml
done