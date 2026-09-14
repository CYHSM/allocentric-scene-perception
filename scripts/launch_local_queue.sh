#!/usr/bin/env bash
# Run the open VLMs on one foil bank, one model at a time.
#
#   nohup bash scripts/launch_local_queue.sh > logs/queue_hard.log 2>&1 &
#   BENCHMARK=data/vlm_benchmark_4afc_mid_m.json \
#     nohup bash scripts/launch_local_queue.sh > logs/queue_mid.log 2>&1 &
#   MODELS="OpenGVLab/InternVL3_5-1B-HF OpenGVLab/InternVL3_5-2B-HF" GPUS=3,4,5,6,7 \
#     nohup bash scripts/launch_local_queue.sh > logs/queue_internvl35.log 2>&1 &
#
# `MODELS` overrides the default list; `GPUS` pins the run to specific devices
# instead of taking whatever is idle. Pin them when the box is shared: the idle
# scan takes any GPU under 2 GB, so a neighbour's job that has just finished
# loading looks free and the run lands on top of it.
#
# The bank is the only thing that varies. Its name goes in the output filename,
# because the same model at the same prompt against two banks otherwise lands on
# one path and silently overwrites itself.
#
# Configuration is fixed here and matches the OpenRouter runs exactly, because
# the point of this queue is a scaling axis that can sit in the same table as
# Gemini, Luna and Qwen3-VL-235B:
#
#   benchmark    $BENCHMARK (default: the hard band)
#   prompt       cot_anyview   -- `cot` asserts the answer is rotated, which is
#                                 false for every delta=0 trial
#   max_tokens   8000          -- a cap, not a target; open instruct models stop
#                                 at EOS long before it, so it costs nothing and
#                                 keeps the setting identical across runs
#   trials       100, stratified seed 0 -- the SAME 100 the API models answered
#
# Models are attempted in order and a failure does not stop the queue: the
# cached ones must land even if a later download fails. Each gets a 3-trial
# smoke test first -- a run where nothing parses produces a file of nulls that
# scores at chance and looks like a finding.
set -uo pipefail
cd "$(dirname "$0")/.."

BENCHMARK="${BENCHMARK:-data/vlm_benchmark_4afc_hard.json}"
BENCH_TAG=$(basename "$BENCHMARK" .json | sed 's/^vlm_benchmark_//')
PROMPT="cot_anyview"
MAX_TOKENS=8000
TRIALS=100
LOCK="logs/.vlm_gpu.lock"
export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"
mkdir -p results logs

if [ -n "${MODELS:-}" ]; then
  read -r -a MODELS <<< "$MODELS"
else
MODELS=(
  # --- already in the HF cache: no download, these must land first ---
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-32B-Instruct"
  "OpenGVLab/InternVL3-38B-hf"
  "Qwen/Qwen2.5-VL-72B-Instruct"
)
fi
# The default list needs no downloading. /raid was 99% full (483 GB) when this was
# written, and these five plus the four OpenRouter models are already a
# complete table: a four-point Qwen2.5-VL scaling axis, InternVL3-38B, and
# four API models on the identical 100 trials.

run_one() {
    local MODEL="$1"
    local SLUG; SLUG=$(echo "$MODEL" | tr '/[:upper:]' '_[:lower:]' | tr -cd 'a-z0-9_.-')
    local OUT="results/${SLUG}_${BENCH_TAG}_${PROMPT}_n${TRIALS}.json"
    local SMOKE="results/smoke_${SLUG}_${BENCH_TAG}.json"
    local LOG="logs/${SLUG}_${BENCH_TAG}.log"

    if [ -s "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
sys.exit(0 if len(d.get('results',[]))>=$TRIALS and not d.get('summary',{}).get('in_progress') else 1)" 2>/dev/null; then
        echo "[skip] $MODEL already complete at $OUT"
        return 0
    fi

    echo "=========================================================="
    echo "$MODEL  |  $(date)"
    echo "=========================================================="
    if [ -n "${GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="$GPUS"
    else
        mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                            | awk -F', ' '$2 < 2000 {print $2","$1}' | sort -n | cut -d, -f2)
        if [ "${#FREE[@]}" -lt 1 ]; then
            echo "no idle GPUs; skipping $MODEL"; nvidia-smi --query-gpu=index,memory.used --format=csv
            return 1
        fi
        export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${FREE[*]:0:6}")
    fi
    echo "GPUs $CUDA_VISIBLE_DEVICES"

    rm -f "$SMOKE"
    if ! .venv/bin/python bench/evaluate_vlm.py --benchmark "$BENCHMARK" --model "$MODEL" \
         --prompt_style "$PROMPT" --max_tokens "$MAX_TOKENS" --max_trials 3 \
         --out "$SMOKE" > "${LOG%.log}.smoke.log" 2>&1; then
        echo "SMOKE FAILED for $MODEL -- see ${LOG%.log}.smoke.log"; tail -5 "${LOG%.log}.smoke.log"
        return 1
    fi
    if ! .venv/bin/python - "$SMOKE" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["results"]
ok = [x for x in r if x.get("model_choice") is not None]
print(f"smoke: {len(r)} trials, {len(ok)} parsed")
sys.exit(0 if ok else 1)
PY
    then
        echo "SMOKE PARSED NOTHING for $MODEL -- skipping"; return 1
    fi

    .venv/bin/python bench/evaluate_vlm.py --benchmark "$BENCHMARK" --model "$MODEL" \
        --prompt_style "$PROMPT" --max_tokens "$MAX_TOKENS" --max_trials "$TRIALS" \
        --out "$OUT" 2>&1 | tee "$LOG"
    echo "$MODEL finished at $(date)"
}

exec 9>"$LOCK"
echo "waiting for the shared GPU lock..."
flock 9
while pgrep -f "evaluate_vlm.py" >/dev/null; do
    echo "[$(date +%H:%M:%S)] another evaluate_vlm.py is running; waiting..."
    sleep 120
done
echo "lock acquired at $(date)"

for M in "${MODELS[@]}"; do
    run_one "$M" || echo "[continue] $M did not complete"
done
echo "QUEUE DONE ($BENCH_TAG) at $(date)"
