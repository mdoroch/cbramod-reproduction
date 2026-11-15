#!/usr/bin/env bash
# All reproduction experiments in order, then the summary table (results/summary.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
bash scripts/reproduction/e1_cbramod.sh
bash scripts/reproduction/e2_simpleconv_lr.sh
bash scripts/reproduction/e2_simpleconv.sh
bash scripts/reproduction/e3_cbramod_scratch.sh
bash scripts/reproduction/e4_cbramod_small_head.sh
bash scripts/reproduction/e5_low_data.sh
bash scripts/reproduction/e6_cbramod_frozen.sh
bash scripts/reproduction/e7_cbramod_single_lr.sh
bash scripts/reproduction/e8_ea.sh
python scripts/aggregate.py
