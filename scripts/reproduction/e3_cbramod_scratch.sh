#!/usr/bin/env bash
# E3: CBraMod from random init, same recipe as E1. Does pretraining pay off? 3 seeds.
set -euo pipefail
cd "$(dirname "$0")/../.."
for seed in 0 1 2; do
  python train.py --model cbramod --seed "$seed" --name e3_cbramod_scratch --set pretrained=false head_lr=5.0e-4
done
# Same with one lr 1e-4 for backbone and head (paper Table 6), to compare with E7 at equal recipe
for seed in 0 1 2; do
  python train.py --model cbramod --seed "$seed" --name e3_cbramod_scratch_single_lr --set pretrained=false
done
