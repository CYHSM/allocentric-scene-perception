#!/usr/bin/env bash
# launch_32b_queue.sh: Wait for 3B run to complete, merge results, and launch Qwen2.5-VL-32B-Instruct
set -euo pipefail

export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"
mkdir -p results logs

echo "=========================================================="
echo "Phase 1: Waiting for Qwen2.5-VL-3B evaluation to finish..."
echo "=========================================================="

while pgrep -f "evaluate_vlm.*Qwen2.5-VL-3B" > /dev/null; do
    echo "[$(date +'%H:%M:%S')] Still waiting for 3B workers..."
    sleep 10
done

echo "[$(date +'%H:%M:%S')] All 3B workers completed!"
echo "Merging 3B results..."

.venv/bin/python bench/merge_vlm_results.py \
    results/qwen2_5_vl_3b_4afc_c0_shape_colour.json \
    results/qwen2_5_vl_3b_4afc_c1_shape.json \
    results/qwen2_5_vl_3b_4afc_c2_colour.json \
    results/qwen2_5_vl_3b_4afc_c3_peaks_bare.json \
    results/qwen2_5_vl_3b_4afc_c4_valley.json \
    --out results/qwen2_5_vl_3b_4afc_full.json

echo "Generating Figure 4 with 3B and 7B results..."
.venv/bin/python bench/figure_vlm.py \
    --results results/qwen2_5_vl_3b_4afc_full.json results/qwen2_5_vl_7b_4afc_full.json \
    --out paper/fig4_vlm.png \
    --table paper/table_vlm.tex

echo "=========================================================="
echo "Phase 2: Launching Qwen/Qwen2.5-VL-32B-Instruct across GPUs"
echo "=========================================================="

MODEL="Qwen/Qwen2.5-VL-32B-Instruct"
PREFIX="qwen2_5_vl_32b_4afc"
PROMPT_STYLE="cot"
MAX_TOKENS=512
BENCHMARK="data/vlm_benchmark_4afc.json"

# Use all available high-memory GPUs (0, 1, 4, 5, 6, 7) with NVLink
export CUDA_VISIBLE_DEVICES="0,1,4,5,6,7"

echo "Evaluating $MODEL across GPUs $CUDA_VISIBLE_DEVICES..."
.venv/bin/python bench/evaluate_vlm.py \
    --benchmark "$BENCHMARK" \
    --model "$MODEL" \
    --prompt_style "$PROMPT_STYLE" \
    --max_tokens "$MAX_TOKENS" \
    --out "results/${PREFIX}_full.json"

echo "Generating updated Figure 4 with 3B, 7B, and 32B results..."
.venv/bin/python bench/figure_vlm.py \
    --results results/qwen2_5_vl_3b_4afc_full.json results/qwen2_5_vl_7b_4afc_full.json "results/${PREFIX}_full.json" \
    --out paper/fig4_vlm.png \
    --table paper/table_vlm.tex

echo "All evaluations complete!"
