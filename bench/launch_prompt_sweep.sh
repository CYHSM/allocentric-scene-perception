#!/usr/bin/env bash
# One model, one benchmark, many prompts: does the conclusion depend on how the
# question was asked?
#
#   nohup bash bench/launch_prompt_sweep.sh > logs/prompt_sweep.log 2>&1 &
#
# Qwen2.5-VL-32B by default, because it is the model whose result is most
# surprising -- its rotated accuracy is below its own answer prior -- so the
# sweep doubles as the robustness check on that finding.
#
# **Read only the rotated trials from this sweep.** Six of the nine prompt
# styles contain the clause that asserts the scene *is* shown from a different
# viewpoint. That is false on every delta=0 trial and was worth 5 points of
# accuracy to Gemini when it was fixed; it is true on every rotated trial. So
# the rotated subset is the part of this sweep where all styles ask the same
# honest question, and it is also the paper's dependent variable.
#
# The earlier prompt ablation (results/calib_7b_*.json) cannot be used: it ran
# on a superseded benchmark with the unfixed prompt and no anyview control.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
BENCHMARK="${BENCHMARK:-data/vlm_benchmark_4afc_hard.json}"
BENCH_TAG=$(basename "$BENCHMARK" .json | sed 's/^vlm_benchmark_//')
TRIALS=100
MAX_TOKENS=8000
LOCK="logs/.vlm_gpu.lock"
export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"
mkdir -p results logs

# cot_anyview is the locked prompt and is already run; it is listed so the
# comparison table has its reference row without a special case.
PROMPTS=(cot_anyview cot mental_rotation anchor birdseye elimination)

SLUG=$(echo "$MODEL" | tr '/[:upper:]' '_[:lower:]' | tr -cd 'a-z0-9_.-')

exec 9>"$LOCK"
echo "waiting for the shared GPU lock (the foil ladder holds it first)..."
flock 9
while pgrep -f "evaluate_vlm.py" >/dev/null; do
    echo "[$(date +%H:%M:%S)] another evaluate_vlm.py is running; waiting..."
    sleep 120
done
echo "lock acquired at $(date)"

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | awk -F', ' '$2 < 2000 {print $2","$1}' | sort -n | cut -d, -f2)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${FREE[*]:0:6}")
echo "GPUs $CUDA_VISIBLE_DEVICES"

for P in "${PROMPTS[@]}"; do
    OUT="results/${SLUG}_${BENCH_TAG}_${P}_n${TRIALS}.json"
    if [ -s "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
sys.exit(0 if len(d.get('results',[]))>=$TRIALS and not d.get('summary',{}).get('in_progress') else 1)" 2>/dev/null; then
        echo "[skip] $P already complete at $OUT"
        continue
    fi
    echo "=========================================================="
    echo "$MODEL  |  prompt=$P  |  $(date)"
    .venv/bin/python bench/evaluate_vlm.py --benchmark "$BENCHMARK" --model "$MODEL" \
        --prompt_style "$P" --max_tokens "$MAX_TOKENS" --max_trials "$TRIALS" \
        --out "$OUT" 2>&1 | tail -20
done
echo "PROMPT SWEEP DONE at $(date)"
