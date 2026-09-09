#!/usr/bin/env bash
set -euo pipefail

cd /raid/nbe_tmp/markus_frey/asp
export CUDA_VISIBLE_DEVICES=1

echo '=== Merging all modes under data/scenes ==='
.venv/bin/python bench/merge_bank.py --root data/scenes

MODELS=(
  vit_base_patch14_dinov2.lvd142m
  vit_so400m_patch14_siglip_384.webli
  vit_base_patch16_clip_224.openai
  resnet50.a1_in1k
)

BANK=data/scenes/c4_valley

echo '=== Scoring c4_valley on GPU 1 ==='
for model in "${MODELS[@]}"; do
  echo "--- c4_valley  x  $model"
  .venv/bin/python bench/metrics.py  --bank "$BANK" --model "$model" --appearance both
  .venv/bin/python bench/exchange.py --bank "$BANK" --model "$model"
done
