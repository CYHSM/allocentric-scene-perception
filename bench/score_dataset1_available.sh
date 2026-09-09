#!/usr/bin/env bash
set -euo pipefail

cd /raid/nbe_tmp/markus_frey/asp
export CUDA_VISIBLE_DEVICES=1

MODELS=(
  vit_base_patch14_dinov2.lvd142m
  vit_so400m_patch14_siglip_384.webli
  vit_base_patch16_clip_224.openai
  resnet50.a1_in1k
)

BANKS=(
  data/scenes/c0_shape_colour
  data/scenes/c1_shape
  data/scenes/c2_colour
  data/scenes/c3_peaks_bare
)

echo "=== Scoring completed modes on GPU 1 ==="
for bank in "${BANKS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "--- $(basename "$bank")  x  $model"
    .venv/bin/python bench/metrics.py  --bank "$bank" --model "$model" --appearance both
    .venv/bin/python bench/exchange.py --bank "$bank" --model "$model"
  done
done

echo "=== Identity Gate Check ==="
.venv/bin/python - "data/scenes" << "PY"
import glob, json, os, sys
bad = 0
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*", "metrics.json"))):
    for r in json.load(open(p)):
        if r.get("appearance") != "same":
            continue
        cell = r["by_delta"].get("0") or r["by_delta"].get(0)
        if not cell:
            continue
        v = cell["recall@1"]
        ok = v > 0.999
        bad += not ok
        print(f"  {r[bank]:20s} {r[model]:38s} {v:.3f}  [{OK if ok else FAILED}]")
if bad:
    print(f"\nWARNING: {bad} cell(s) failed identity gate.")
else:
    print("\nALL CELLS PASSED IDENTITY GATE (Recall@1 = 1.000).")
PY
