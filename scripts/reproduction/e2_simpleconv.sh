#!/usr/bin/env bash
# E2: EEG-SimpleConv from scratch, same data / loop / metrics as E1, lr chosen on val. 5 seeds.
set -euo pipefail
cd "$(dirname "$0")/../.."
lr=$(python scripts/reproduction/best_lr.py)
for seed in 0 1 2 3 4; do
  python train.py --model simpleconv --seed "$seed" --name e2_simpleconv --set lr="$lr"
done
