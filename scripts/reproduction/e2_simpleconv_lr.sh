#!/usr/bin/env bash
# E2 (step 1): EEG-SimpleConv lr search on val, 1 seed. The best lr is read by best_lr.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
for lr in 1.0e-4 3.0e-4 1.0e-3; do
  python train.py --model simpleconv --seed 0 --name "e2_lr_$lr" --set lr="$lr"
done
