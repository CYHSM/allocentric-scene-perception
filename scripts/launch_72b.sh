#!/usr/bin/env bash
# Evaluate Qwen2.5-VL-72B-Instruct on the 4AFC bank.
#
# Three things went wrong the last time this ran and are fixed here.
#
# 1. It ran with --prompt_style mental_rotation while 3B, 7B and 32B all ran
#    cot, so 72B could not go on the scaling axis: the point would have differed
#    in prompt as well as size. On the full 500 trials mental_rotation is not
#    significantly better than cot anyway (29.4% vs 26.2%, McNemar p = 0.14).
#    The scaling series is cot.
#
# 2. Two copies were launched at once by two copies of the overnight battery.
#    evaluate_vlm.py writes its output with a plain open(path,"w") *and* resumes
#    from that same file, so the two processes read each other's partial results
#    and interleaved their writes: after 34 minutes the output held 20 trials.
#    The flock below makes a second copy exit instead of racing.
#
# 3. It pinned CUDA_VISIBLE_DEVICES to a fixed list including GPUs another user
#    was on. dgx2 has no scheduler and 20+ users, so the devices are chosen at
#    launch from whatever is actually idle.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
BENCHMARK="data/vlm_benchmark_4afc_hard.json"
PROMPT="${PROMPT_STYLE:-cot}"
OUT="results/qwen2_5_vl_72b_4afc_${PROMPT}.json"
LOG="logs/qwen2_5_vl_72b_4afc_${PROMPT}.log"
LOCK="logs/.vlm_gpu.lock"   # shared: one GPU eval at a time on this box

export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"
mkdir -p results logs paper

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another GPU evaluation already holds $LOCK -- exiting rather than racing it"
    exit 0
fi

# Fewest-megabytes-first among GPUs under 2 GB; 72B in bf16 is ~145 GB, so six
# 80 GB cards is comfortable and four is the floor.
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | awk -F', ' '$2 < 2000 {print $2","$1}' | sort -n | cut -d, -f2)
if [ "${#FREE[@]}" -lt 4 ]; then
    echo "only ${#FREE[@]} idle GPUs; 72B needs at least 4. Aborting."
    nvidia-smi --query-gpu=index,memory.used --format=csv
    exit 1
fi
DEVICES=$(IFS=,; echo "${FREE[*]:0:6}")
export CUDA_VISIBLE_DEVICES="$DEVICES"

echo "=========================================================="
echo "Qwen2.5-VL-72B-Instruct  |  prompt=$PROMPT  |  GPUs=$DEVICES"
echo "Host: $(hostname)  |  $(date)"
echo "=========================================================="

.venv/bin/python bench/evaluate_vlm.py \
    --benchmark "$BENCHMARK" \
    --model "$MODEL" \
    --prompt_style "$PROMPT" \
    --max_tokens 512 \
    --out "$OUT" 2>&1 | tee "$LOG"

echo "72B ($PROMPT) finished at $(date)"

# The scaling axis, regenerated. analyze_vii is the one that reports VII and the
# positional-prior controls; figure_vlm.py is the older raw-accuracy figure.
.venv/bin/python bench/archive/analyze_vii.py --json figures/vii_rows.json | tail -30
