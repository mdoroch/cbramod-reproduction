#!/usr/bin/env bash
# E8: Euclidean Alignment per subject (He & Wu, 2019) for both models; otherwise as E7 (CBraMod) and E2 (SimpleConv). 3 seeds.
# EA uses the unlabeled trials of each subject, test subjects included.
set -euo pipefail
cd "$(dirname "$0")/../.."
lr=$(python scripts/reproduction/best_lr.py)
for seed in 0 1 2; do
  python train.py --model cbramod --seed "$seed" --name e8_cbramod_ea --set ea=true
  python train.py --model simpleconv --seed "$seed" --name e8_simpleconv_ea --set ea=true lr="$lr"
done
