#!/usr/bin/env bash
# E6: CBraMod with the pretrained backbone frozen, only the head is trained (paper Table 18). 3 seeds.
set -euo pipefail
cd "$(dirname "$0")/../.."
for seed in 0 1 2; do
  python train.py --model cbramod --seed "$seed" --name e6_cbramod_frozen --set frozen=true head_lr=5.0e-4
done
