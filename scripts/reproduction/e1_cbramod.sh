#!/usr/bin/env bash
# E1: CBraMod, pretrained weights, full fine-tuning, official repo lr (head 5e-4, multi_lr). 5 seeds.
# Paper Table 3 is reproduced with the paper's single lr: E7.
set -euo pipefail
cd "$(dirname "$0")/../.."
for seed in 0 1 2 3 4; do
  python train.py --model cbramod --seed "$seed" --name e1_cbramod --set head_lr=5.0e-4
done
