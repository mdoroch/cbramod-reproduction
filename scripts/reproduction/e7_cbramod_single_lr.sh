#!/usr/bin/env bash
# E7: E1 with one lr (1e-4) for backbone and head, as written in the paper (Table 6), instead of the repo's head lr 5e-4. 5 seeds. One lr is the config default.
set -euo pipefail
cd "$(dirname "$0")/../.."
for seed in 0 1 2 3 4; do
  python train.py --model cbramod --seed "$seed" --name e7_cbramod_single_lr
done
