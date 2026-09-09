#!/usr/bin/env bash
# launch_vlm_parallel.sh: Run 5 modes of VLM benchmark across 5 A100 GPUs in parallel
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-VL-3B-Instruct}"
PREFIX="${2:-qwen2_5_vl_3b_4afc}"
PROMPT_STYLE="${3:-cot}"
MAX_TOKENS="${4:-512}"
BENCHMARK="${5:-data/vlm_benchmark_4afc.json}"

export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"
mkdir -p results logs

echo "=========================================================="
echo "Launching Parallel VLM Evaluation across 5 GPUs"
echo "Model:        $MODEL"
echo "Prefix:       $PREFIX"
echo "Prompt Style: $PROMPT_STYLE"
echo "Max Tokens:   $MAX_TOKENS"
echo "Benchmark:    $BENCHMARK"
echo "=========================================================="

MODES=("c0_shape_colour" "c1_shape" "c2_colour" "c3_peaks_bare" "c4_valley")
GPUS=(0 1 4 6 7)

PIDS=()
for i in "${!MODES[@]}"; do
    MODE="${MODES[$i]}"
    GPU="${GPUS[$i]}"
    OUT="results/${PREFIX}_${MODE}.json"
    LOG="logs/${PREFIX}_${MODE}.log"

    echo "[GPU $GPU] Starting $MODE -> $OUT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python bench/evaluate_vlm.py \
        --benchmark "$BENCHMARK" \
        --model "$MODEL" \
        --prompt_style "$PROMPT_STYLE" \
        --max_tokens "$MAX_TOKENS" \
        --modes "$MODE" \
        --out "$OUT" > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "Launched 5 parallel workers: ${PIDS[*]}"
echo "Waiting for all workers to complete..."

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

if [ "$FAIL" -eq 0 ]; then
    echo "All 5 mode evaluations completed successfully!"
    echo "Merging results..."
    .venv/bin/python bench/merge_vlm_results.py \
        "results/${PREFIX}_c0_shape_colour.json" \
        "results/${PREFIX}_c1_shape.json" \
        "results/${PREFIX}_c2_colour.json" \
        "results/${PREFIX}_c3_peaks_bare.json" \
        "results/${PREFIX}_c4_valley.json" \
        --out "results/${PREFIX}_full.json"
    echo "Saved merged results to results/${PREFIX}_full.json"
else
    echo "One or more workers failed. Check logs in logs/${PREFIX}_*.log"
    exit 1
fi
