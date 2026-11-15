#!/usr/bin/env bash
# Pretrained CBraMod weights (https://huggingface.co/weighting666/CBraMod) -> weights/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
curl -L -o weights/pretrained_weights.pth \
  https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth
