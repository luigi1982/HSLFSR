#!/usr/bin/env bash
set -e

source .venv/bin/activate

configs=("lft") # "f3dun" "ssaformer" "mst")

for config in "${configs[@]}"
do
    echo "Training: $config"
    python tasks/train.py --task lfsr --config configs/$config.yaml --from_checkpoint real_data-run1-1103-1714
done