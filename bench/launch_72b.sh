#!/usr/bin/env bash
# launch_72b.sh: Evaluate flagship Qwen2.5-VL-72B-Instruct across A100 GPUs using NVLink
set -euo pipefail

export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"
mkdir -p results logs paper

MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
BENCHMARK="data/vlm_benchmark_4afc.json"
OUT="results/qwen2_5_vl_72b_4afc_full.json"
LOG="logs/qwen2_5_vl_72b_4afc.log"

echo "=========================================================="
echo "QUEUING QWEN2.5-VL-72B-INSTRUCT EVALUATION"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

# Wait for 32B run and overnight battery to finish if active
echo "Waiting for currently active VLM evaluations to finish..."
while pgrep -f "evaluate_vlm.*32B" > /dev/null; do
    echo "[$(date +'%H:%M:%S')] Waiting for 32B evaluation..."
    sleep 30
done

echo "Active 32B process complete. Checking GPU memory..."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Use all high-bandwidth NVLink GPUs (0, 1, 4, 5, 6, 7)
export CUDA_VISIBLE_DEVICES="0,1,4,5,6,7"

echo "Launching $MODEL on GPUs $CUDA_VISIBLE_DEVICES..."
echo "Using empirically optimized 'mental_rotation' prompt style..."

.venv/bin/python bench/evaluate_vlm.py \
    --benchmark "$BENCHMARK" \
    --model "$MODEL" \
    --prompt_style mental_rotation \
    --max_tokens 512 \
    --out "$OUT" 2>&1 | tee "$LOG"

echo "72B evaluation completed successfully at $(date)!"

# Update publication Figure 4 with full scaling hierarchy: 3B, 7B, 32B, 72B
.venv/bin/python bench/figure_vlm.py \
    --results results/qwen2_5_vl_3b_4afc_full.json results/qwen2_5_vl_7b_4afc_full.json results/qwen2_5_vl_32b_4afc_full.json "$OUT" \
    --out paper/fig4_vlm.png \
    --table paper/table_vlm.tex

echo "Updated Figure 4 and Table 1 with 72B results!"
