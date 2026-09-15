#!/usr/bin/env bash
set -uo pipefail

cd /raid/nbe_tmp/markus_frey/asp

export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"
export CUDA_VISIBLE_DEVICES="7"

mkdir -p results/prompts_c0 logs/prompts_c0

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
BENCH="data/vlm_benchmark_4afc_c0.json"

PROMPTS=(
  "direct"
  "cot_anyview"
  "neutral_anyview"
  "mental_rotation"
  "anchor"
  "birdseye"
  "elimination"
  "hybrid"
  "c0_mental_rotation"
  "c0_cyclic_topology"
  "c0_anchor_triangulation"
  "c0_ego_to_allo"
  "c0_falsification"
  "c0_birdseye_grid"
)

echo "=== Starting Prompt Sweep on c0_shape_colour ==="
echo "Model: $MODEL"
echo "Benchmark: $BENCH (100 trials)"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Total Prompts: ${#PROMPTS[@]}"
echo "================================================"

for PROMPT in "${PROMPTS[@]}"; do
    OUT="results/prompts_c0/qwen_7b_c0_${PROMPT}.json"
    LOG="logs/prompts_c0/qwen_7b_c0_${PROMPT}.log"
    
    if [ -s "$OUT" ] && .venv/bin/python -c "import json,sys; d=json.load(open('$OUT')); sys.exit(0 if len(d.get('results',[]))>=100 else 1)" 2>/dev/null; then
        echo "[skip] $PROMPT already complete ($OUT)"
        continue
    fi
    
    echo "=== [$(date +'%T')] Running prompt style: $PROMPT ==="
    .venv/bin/python bench/evaluate_vlm.py \
        --benchmark "$BENCH" \
        --model "$MODEL" \
        --prompt_style "$PROMPT" \
        --max_trials 100 \
        --out "$OUT" > "$LOG" 2>&1
        
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        ACC=$(.venv/bin/python -c "import json; d=json.load(open('$OUT')); print(f\"{d.get('summary',{}).get('overall_accuracy',0)*100:.1f}%\")" 2>/dev/null || echo "N/A")
        echo "=== [$(date +'%T')] Finished $PROMPT -> Overall Acc: $ACC ==="
    else
        echo "=== [$(date +'%T')] ERROR on $PROMPT (exit code $STATUS) ==="
        tail -n 20 "$LOG"
    fi
done

echo "=== [$(date +'%T')] ALL PROMPTS COMPLETED ==="
