#!/usr/bin/env bash
#
# The whole read-out, in one command, in the only order it is valid in.
#
#     bash bench/score_all.sh [data/scenes]
#
# `merge_bank.py` runs first and `set -e` makes it a gate, not a step: if the
# five modes do not hold the same scenes at the same places, nothing downstream
# runs. A half-rendered bank scores perfectly happily and says nothing about it.
set -euo pipefail

ROOT="${1:-data/scenes}"
MODELS=(
  vit_base_patch14_dinov2.lvd142m
  vit_so400m_patch14_siglip_384.webli
  vit_base_patch16_clip_224.openai
  resnet50.a1_in1k
)

echo "=== 1/4  merge + check (the gate) ==========================="
python bench/merge_bank.py --root "$ROOT"

echo
echo "=== 2/4  Figure 1: look at the dataset ======================"
python bench/figure_dataset.py --root "$ROOT" --out figures/fig1_dataset.png

echo
echo "=== 3/4  score ============================================="
for bank in "$ROOT"/*/; do
  for model in "${MODELS[@]}"; do
    echo "--- $(basename "$bank")  x  $model"
    # `--appearance both`: the "same" arm is the identity gate (Recall@1 at
    # delta 0 must be exactly 1.000), and it costs one extra embedding pass.
    python bench/metrics.py  --bank "$bank" --model "$model" --appearance both
    python bench/exchange.py --bank "$bank" --model "$model"
  done
done

echo
echo "=== 4/4  Figures 2-3 and Table 1 ==========================="
python bench/figure_results.py --root "$ROOT"

echo
echo "identity gate -- every cell below must read 1.000:"
python - "$ROOT" <<'PY'
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
        print(f"  {r['bank']:20s} {r['model']:38s} {v:.3f}  "
              f"[{'OK' if ok else 'FAILED'}]")
if bad:
    raise SystemExit(f"\n{bad} cell(s) failed the identity gate -- "
                     f"the query was not matched to its own file; do not "
                     f"report these numbers.")
print("\nidentity gate passed.")
PY
