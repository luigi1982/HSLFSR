#!/usr/bin/env bash
set -e

source .venv/bin/activate
python tasks/train.py --task diff --config configs/distg_unet.yaml