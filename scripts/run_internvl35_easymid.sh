#!/usr/bin/env bash
# Evaluate InternVL3.5 series on easy_m and mid_m benchmarks on dgx2.
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/raid/nbe_tmp/markus_frey/cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

MODELS=(
  "OpenGVLab/InternVL3_5-8B-HF"
  "OpenGVLab/InternVL3_5-14B-HF"
  "OpenGVLab/InternVL3_5-38B-HF"
)

for BENCH in "data/benchmarks/vlm_benchmark_4afc_easy_m.json" "data/benchmarks/vlm_benchmark_4afc_mid_m.json"; do
    BENCH_TAG=$(basename "$BENCH" .json | sed "s/^vlm_benchmark_//")
    for MODEL in "${MODELS[@]}"; do
        SLUG=$(echo "$MODEL" | tr "/[:upper:]" "_[:lower:]" | tr -cd "a-z0-9_.-")
        OUT="results/${SLUG}_${BENCH_TAG}_cot_anyview_n100.json"
        LOG="logs/${SLUG}_${BENCH_TAG}.log"
        
        if [ -s "$OUT" ] && .venv/bin/python -c "import json,sys; d=json.load(open('$OUT')); sys.exit(0 if len(d.get('results',[]))>=100 else 1)" 2>/dev/null; then
            echo "[skip] $MODEL already complete on $BENCH_TAG"
            continue
        fi
        
        echo "=== $(date) Starting $MODEL on $BENCH_TAG ==="
        .venv/bin/python bench/evaluate_vlm.py \
            --benchmark "$BENCH" \
            --model "$MODEL" \
            --prompt_style cot_anyview \
            --max_tokens 8000 \
            --max_trials 100 \
            --out "$OUT" > "$LOG" 2>&1
        echo "=== $(date) Finished $MODEL on $BENCH_TAG ==="
    done
done
echo "ALL DONE at $(date)"
