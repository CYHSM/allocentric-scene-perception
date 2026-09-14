#!/usr/bin/env bash
# GPT-5.6 Luna across the four landmark-count banks.
# A per-run budget ceiling, because a provider-side price change is exactly what
# the probe just caught: measured $0.28 per 100 trials, ceiling set at $1.00 so
# a 3x surprise stops the run instead of the card.
set -uo pipefail
cd "$(dirname "$0")/.."
for n in n01 n02 n04 n06; do
  out="results/or_openai_gpt-5.6-luna_cot_anyview_setsize_${n}_n100.json"
  if [ -s "$out" ]; then echo "[skip] $n already present"; continue; fi
  echo "=== $(date -u +%H:%M:%S)  setsize $n ==="
  python3 bench/evaluate_vlm.py \
    --benchmark "data/vlm_benchmark_setsize_${n}.json" \
    --model openai/gpt-5.6-luna --prompt_style cot_anyview --max_tokens 8000 \
    --max_trials 100 --budget_usd 1.00 \
    --api_base "https://openrouter.ai/api/v1" --api_key "$OPENROUTER_API_KEY" \
    --out "$out" 2>&1 | grep -aiE "^API:|Overall Accuracy|budget|Traceback|Error"
done
echo "LUNA SETSIZE DONE $(date -u +%H:%M:%S)"
