#!/usr/bin/env bash
# ==============================================================================
# COSYNE 2027: Working Memory Delay Latent Perturbation Sweeps
#
# Simulates working memory delay (e.g. 2s) via targeted latent noise injection
# into the study scene visual representation (image 0), leaving probe options clean.
# Pinned to idle GPUs (e.g. GPU 7 or 4) on dgx2.
# ==============================================================================
set -euo pipefail

# Ensure working directory is repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

GPU_DEVICE="${1:-7}"
MODEL="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
BENCHMARK="${3:-data/vlm_benchmark_4afc_easy_m.json}"
OUT_DIR="${4:-results/delayed_sweeps}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"

mkdir -p "${OUT_DIR}"
mkdir -p logs

BENCH_TAG=$(basename "${BENCHMARK}" .json | sed "s/^vlm_benchmark_//")
SLUG=$(echo "${MODEL}" | tr "/[:upper:]" "_[:lower:]" | tr -cd "a-z0-9_.-")
LOG_FILE="logs/delayed_${SLUG}_${BENCH_TAG}.log"

echo "=============================================================================="
echo "STARTING WORKING MEMORY DELAY SWEEPS"
echo "Model:      ${MODEL}"
echo "Benchmark:  ${BENCHMARK} (${BENCH_TAG})"
echo "GPU Device: ${GPU_DEVICE}"
echo "Out Dir:    ${OUT_DIR}"
echo "Log:        ${LOG_FILE}"
echo "Date:       $(date)"
echo "=============================================================================="

# Sweep 1: Temporal Delay Duration (t in seconds, D=0.05)
# Tests t in {0.0, 0.5, 1.0, 2.0, 5.0} seconds
echo "--- Running Temporal Delay Duration Sweep (t = 0.0, 0.5, 1.0, 2.0, 5.0s) ---"
.venv/bin/python bench/delayed_vlm.py \
    --model "${MODEL}" \
    --benchmark "${BENCHMARK}" \
    --delay_seconds "0.0,0.5,1.0,2.0,5.0" \
    --diffusion_rate 0.05 \
    --noise_type "gaussian" \
    --noise_target "study" \
    --locus "vision_embed" \
    --prompt_style "cot_anyview" \
    --max_trials 100 \
    --seed 42 \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "--- Running Noise Scale Psychometric Sweep (sigma = 0.0, 0.05, 0.1, 0.25, 0.5, 1.0) ---"
.venv/bin/python bench/delayed_vlm.py \
    --model "${MODEL}" \
    --benchmark "${BENCHMARK}" \
    --noise_scales "0.0,0.05,0.1,0.25,0.5,1.0" \
    --noise_type "gaussian" \
    --noise_target "study" \
    --locus "vision_embed" \
    --prompt_style "cot_anyview" \
    --max_trials 100 \
    --seed 42 \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "--- Running Probe Control Sweep (sigma = 0.0, 0.1, 0.25, 0.5 on options) ---"
.venv/bin/python bench/delayed_vlm.py \
    --model "${MODEL}" \
    --benchmark "${BENCHMARK}" \
    --noise_scales "0.0,0.1,0.25,0.5" \
    --noise_type "gaussian" \
    --noise_target "options" \
    --locus "vision_embed" \
    --prompt_style "cot_anyview" \
    --max_trials 100 \
    --seed 42 \
    --out_dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "=============================================================================="
echo "DELAY SWEEPS COMPLETE: $(date)"
echo "Analyzing results..."
.venv/bin/python bench/analyze_delay.py \
    --dir "${OUT_DIR}" \
    --pattern "delayed_${SLUG}_${BENCH_TAG}_*.json" \
    --out_prefix "paper/cosyne/figures/working_memory_decay_${SLUG}_${BENCH_TAG}"

echo "All figures and LaTeX tables generated successfully!"
