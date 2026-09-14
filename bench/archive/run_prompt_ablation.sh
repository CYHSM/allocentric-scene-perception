#!/usr/bin/env bash
# run_prompt_ablation.sh: Direct empirical comparison of prompt strategies on GPU 3
set -euo pipefail

export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"
mkdir -p results logs

GPU=3
BENCHMARK="data/vlm_benchmark_calibration.json"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

PROMPTS=("cot" "mental_rotation" "anchor" "birdseye" "elimination" "elevation")

echo "=========================================================="
echo "STARTING PROMPT STRATEGY ABLATION ON GPU $GPU"
echo "Model: $MODEL | Benchmark: $BENCHMARK (50 trials)"
echo "=========================================================="

for PROMPT in "${PROMPTS[@]}"; do
    echo "----------------------------------------------------------"
    echo "Running Prompt Style: $PROMPT"
    echo "----------------------------------------------------------"
    OUT="results/calib_7b_${PROMPT}.json"
    LOG="logs/calib_7b_${PROMPT}.log"
    
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python bench/evaluate_vlm.py \
        --benchmark "$BENCHMARK" \
        --model "$MODEL" \
        --prompt_style "$PROMPT" \
        --max_tokens 512 \
        --no_resume \
        --out "$OUT" 2>&1 | tee "$LOG"
done

echo "=========================================================="
echo "ALL PROMPTS EVALUATED! GENERATING COMPARISON SUMMARY..."
echo "=========================================================="

.venv/bin/python -c "
import json
import glob
from collections import defaultdict

prompts = ['cot', 'mental_rotation', 'anchor', 'birdseye', 'elimination', 'elevation']
deltas = [0, 45, 90, 135, 180]
modes = ['c0_shape_colour', 'c1_shape', 'c2_colour', 'c3_peaks_bare', 'c4_valley']

print(f'\\n{\"\":18s} | {\"Overall\":8s} | {\"d000\":6s} | {\"d045\":6s} | {\"d090\":6s} | {\"d135\":6s} | {\"d180\":6s}')
print('-' * 72)

for p in prompts:
    path = f'results/calib_7b_{p}.json'
    try:
        with open(path) as f:
            data = json.load(f)
        trials = data['results']
        overall = sum(t['is_correct'] for t in trials) / len(trials) * 100
        
        by_d = defaultdict(list)
        for t in trials:
            by_d[t['delta']].append(t['is_correct'])
        d_accs = [sum(by_d[d])/len(by_d[d])*100 if by_d[d] else 0.0 for d in deltas]
        
        d_str = ' | '.join(f'{acc:5.1f}%' for acc in d_accs)
        print(f'{p:18s} | {overall:7.1f}% | {d_str}')
    except Exception as e:
        print(f'{p:18s} | Error: {e}')
"
