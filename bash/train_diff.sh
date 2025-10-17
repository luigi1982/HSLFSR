#!/usr/bin/env bash
set -e

source .venv/bin/activate
python diffusion/train.py --config configs/distg_unet.yaml