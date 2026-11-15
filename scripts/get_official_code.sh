#!/usr/bin/env bash
# Official CBraMod and EEGSimpleConv code at fixed commits -> third_party/ (used by check_parity.py only)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
clone() {  # url dir commit
  [ -d "third_party/$2" ] || git clone -q "$1" "third_party/$2"
  git -C "third_party/$2" checkout -q "$3"
}
clone https://github.com/wjq-learning/CBraMod.git CBraMod b9e961003214326972c567eff390e75b0287e32a
clone https://github.com/elouayas/EEGSimpleConv.git EEGSimpleConv a732d9a3758ede6f31ddd5de1fece2e790ef4d0f
