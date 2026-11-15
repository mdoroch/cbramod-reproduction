#!/usr/bin/env bash
# E5: learning curve over the number of train subjects (seeded random subsets of 1-15). Val / test unchanged.
# CBraMod with the repo lr (head 5e-4) and with one lr 1e-4 (paper).
# n = 15 is covered by E1 (pretrained), E3 (scratch) and E2 (SimpleConv). 3 seeds.
set -euo pipefail
cd "$(dirname "$0")/../.."
lr=$(python scripts/reproduction/best_lr.py)
for n in 3 7; do
  for seed in 0 1 2; do
    python train.py --model cbramod --seed "$seed" --name "e5_cbramod_n$n" --set n_train_subjects="$n" head_lr=5.0e-4
    python train.py --model cbramod --seed "$seed" --name "e5_cbramod_scratch_n$n" --set n_train_subjects="$n" pretrained=false head_lr=5.0e-4
    python train.py --model simpleconv --seed "$seed" --name "e5_simpleconv_n$n" --set n_train_subjects="$n" lr="$lr"
    # CBraMod with one lr 1e-4 (paper, config default), pretrained and scratch
    python train.py --model cbramod --seed "$seed" --name "e5_cbramod_single_lr_n$n" --set n_train_subjects="$n"
    python train.py --model cbramod --seed "$seed" --name "e5_cbramod_scratch_single_lr_n$n" --set n_train_subjects="$n" pretrained=false
  done
done
