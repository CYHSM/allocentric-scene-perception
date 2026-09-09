#!/usr/bin/env bash
# run_overnight_battery.sh: Comprehensive overnight evaluation of prompt variations,
# 2AFC task format, and model scaling on dgx2.
set -euo pipefail

export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"
mkdir -p results logs paper

LOG="logs/overnight_battery.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "STARTING OVERNIGHT VLM EXPERIMENTAL BATTERY"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

# -----------------------------------------------------------------------------
# Phase 1: Wait for running 32B model to complete
# -----------------------------------------------------------------------------
echo "[Phase 1] Waiting for Qwen2.5-VL-32B run to complete..."
while pgrep -f "evaluate_vlm.*32B" > /dev/null; do
    sleep 20
done
echo "[Phase 1] Qwen2.5-VL-32B completed at $(date)!"

if [ -f "results/qwen2_5_vl_32b_4afc_full.json" ]; then
    echo "Updating Figure 4 with 3B, 7B, and 32B..."
    .venv/bin/python bench/figure_vlm.py \
        --results results/qwen2_5_vl_3b_4afc_full.json results/qwen2_5_vl_7b_4afc_full.json results/qwen2_5_vl_32b_4afc_full.json \
        --out paper/fig4_vlm.png \
        --table paper/table_vlm.tex
fi

# -----------------------------------------------------------------------------
# Phase 2: Prompt Strategy 1 — Mental Rotation Prompting on 7B (500 Trials)
# -----------------------------------------------------------------------------
echo "=========================================================="
echo "[Phase 2] Evaluating Mental Rotation Prompting on Qwen2.5-VL-7B..."
echo "=========================================================="

MODES=("c0_shape_colour" "c1_shape" "c2_colour" "c3_peaks_bare" "c4_valley")
GPUS=(0 1 4 6 7)
PIDS=()

for i in "${!MODES[@]}"; do
    MODE="${MODES[$i]}"
    GPU="${GPUS[$i]}"
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python bench/evaluate_vlm.py \
        --benchmark data/vlm_benchmark_4afc.json \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --prompt_style mental_rotation \
        --max_tokens 512 \
        --modes "$MODE" \
        --out "results/qwen2_5_vl_7b_mental_rot_${MODE}.json" > "logs/7b_mental_rot_${MODE}.log" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done

.venv/bin/python bench/merge_vlm_results.py \
    results/qwen2_5_vl_7b_mental_rot_c0_shape_colour.json \
    results/qwen2_5_vl_7b_mental_rot_c1_shape.json \
    results/qwen2_5_vl_7b_mental_rot_c2_colour.json \
    results/qwen2_5_vl_7b_mental_rot_c3_peaks_bare.json \
    results/qwen2_5_vl_7b_mental_rot_c4_valley.json \
    --out results/qwen2_5_vl_7b_mental_rot_full.json

echo "[Phase 2] Mental Rotation evaluation complete!"

# -----------------------------------------------------------------------------
# Phase 3: Prompt Strategy 2 — Landmark Anchor Prompting on 7B (500 Trials)
# -----------------------------------------------------------------------------
echo "=========================================================="
echo "[Phase 3] Evaluating Landmark Anchor Prompting on Qwen2.5-VL-7B..."
echo "=========================================================="

PIDS=()
for i in "${!MODES[@]}"; do
    MODE="${MODES[$i]}"
    GPU="${GPUS[$i]}"
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python bench/evaluate_vlm.py \
        --benchmark data/vlm_benchmark_4afc.json \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --prompt_style anchor \
        --max_tokens 512 \
        --modes "$MODE" \
        --out "results/qwen2_5_vl_7b_anchor_${MODE}.json" > "logs/7b_anchor_${MODE}.log" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done

.venv/bin/python bench/merge_vlm_results.py \
    results/qwen2_5_vl_7b_anchor_c0_shape_colour.json \
    results/qwen2_5_vl_7b_anchor_c1_shape.json \
    results/qwen2_5_vl_7b_anchor_c2_colour.json \
    results/qwen2_5_vl_7b_anchor_c3_peaks_bare.json \
    results/qwen2_5_vl_7b_anchor_c4_valley.json \
    --out results/qwen2_5_vl_7b_anchor_full.json

echo "[Phase 3] Landmark Anchor evaluation complete!"

# -----------------------------------------------------------------------------
# Phase 4: Task Format Ablation — Pairwise 2AFC on 7B (500 Trials, Chance = 50%)
# -----------------------------------------------------------------------------
echo "=========================================================="
echo "[Phase 4] Evaluating Pairwise 2AFC on Qwen2.5-VL-7B..."
echo "=========================================================="

PIDS=()
for i in "${!MODES[@]}"; do
    MODE="${MODES[$i]}"
    GPU="${GPUS[$i]}"
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python bench/evaluate_vlm.py \
        --benchmark data/vlm_benchmark_2afc.json \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --prompt_style cot \
        --max_tokens 512 \
        --modes "$MODE" \
        --out "results/qwen2_5_vl_7b_2afc_${MODE}.json" > "logs/7b_2afc_${MODE}.log" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done

.venv/bin/python bench/merge_vlm_results.py \
    results/qwen2_5_vl_7b_2afc_c0_shape_colour.json \
    results/qwen2_5_vl_7b_2afc_c1_shape.json \
    results/qwen2_5_vl_7b_2afc_c2_colour.json \
    results/qwen2_5_vl_7b_2afc_c3_peaks_bare.json \
    results/qwen2_5_vl_7b_2afc_c4_valley.json \
    --out results/qwen2_5_vl_7b_2afc_full.json

echo "[Phase 4] 2AFC evaluation complete!"

# -----------------------------------------------------------------------------
# Phase 5: Multi-Prompt & Format Figure Synthesis
# -----------------------------------------------------------------------------
echo "=========================================================="
echo "[Phase 5] Generating Multi-Prompt & Format Comparative Plots..."
echo "=========================================================="

.venv/bin/python bench/figure_vlm.py \
    --results \
        results/qwen2_5_vl_3b_4afc_full.json \
        results/qwen2_5_vl_7b_4afc_full.json \
        results/qwen2_5_vl_7b_mental_rot_full.json \
        results/qwen2_5_vl_7b_anchor_full.json \
        results/qwen2_5_vl_32b_4afc_full.json \
    --out paper/fig4_vlm.png \
    --table paper/table_vlm.tex

echo "=========================================================="
echo "ALL PROMPT & FORMAT EXPERIMENTS COMPLETED AT $(date)"
echo "=========================================================="

# -----------------------------------------------------------------------------
# Phase 6: Flagship Scaling Evaluation (Qwen2.5-VL-72B-Instruct)
# -----------------------------------------------------------------------------
echo "=========================================================="
echo "[Phase 6] Launching Flagship Qwen2.5-VL-72B-Instruct..."
echo "=========================================================="

bash bench/launch_72b.sh

echo "=========================================================="
echo "COMPLETE BATTERY AND 72B RUN COMPLETED AT $(date)"
echo "=========================================================="

