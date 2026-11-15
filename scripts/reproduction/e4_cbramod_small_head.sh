#!/usr/bin/env bash
# E4: CBraMod with a small head (mean-pool + linear, 201 params) instead of the 20.6M MLP head (E1). 3 seeds.
set -euo pipefail
cd "$(dirname "$0")/../.."
for seed in 0 1 2; do
  python train.py --model cbramod --seed "$seed" --name e4_cbramod_small_head --set head=small head_lr=5.0e-4
done
