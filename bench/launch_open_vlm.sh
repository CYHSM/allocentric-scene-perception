#!/usr/bin/env bash
# Evaluate any open vision-language model on the 4AFC bank.
#
#   bash bench/launch_open_vlm.sh OpenGVLab/InternVL3-38B-hf
#   PROMPT_STYLE=mental_rotation bash bench/launch_open_vlm.sh <model>
#
# Waits for the shared GPU lock rather than racing whatever is running, so this
# can be fired while the 72B job is still going and it will start when that
# finishes. dgx2 has no scheduler, so the lock is the scheduler.
#
# A three-trial smoke test runs first. Every model outside the Qwen family takes
# a different processor signature, and finding that out after two hours of
# generation -- or worse, after a run that silently returned no parseable choice
# on every trial -- is the failure this guards against.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: launch_open_vlm.sh <hf-model-id>}"
PROMPT="${PROMPT_STYLE:-cot}"
SLUG=$(echo "$MODEL" | tr '/[:upper:]' '_[:lower:]' | tr -cd 'a-z0-9_.-')
BENCHMARK="data/vlm_benchmark_4afc.json"
OUT="results/${SLUG}_4afc_${PROMPT}.json"
SMOKE="results/smoke_${SLUG}.json"
LOG="logs/${SLUG}_4afc_${PROMPT}.log"
LOCK="logs/.vlm_gpu.lock"

export HF_HOME="/raid/nbe_tmp/markus_frey/cache/huggingface"
mkdir -p results logs

exec 9>"$LOCK"
echo "waiting for the GPU lock ($LOCK)..."
flock 9

# The lock alone is not enough during the changeover: the currently running 72B
# job was started by the previous version of launch_72b.sh, which held a
# differently-named lock file. Wait for any evaluate_vlm.py to exit as well, so
# this cannot start alongside a job that predates the shared lock.
while pgrep -f "evaluate_vlm.py" | grep -qv "^$$\\|^${BASHPID:-0}$"; do
    echo "[$(date +%H:%M:%S)] another evaluate_vlm.py is running; waiting..."
    sleep 120
done
echo "lock acquired and GPUs clear at $(date)"

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | awk -F', ' '$2 < 2000 {print $2","$1}' | sort -n | cut -d, -f2)
if [ "${#FREE[@]}" -lt 2 ]; then
    echo "only ${#FREE[@]} idle GPUs. Aborting."; nvidia-smi --query-gpu=index,memory.used --format=csv; exit 1
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${FREE[*]:0:6}")

echo "=========================================================="
echo "$MODEL  |  prompt=$PROMPT  |  GPUs=$CUDA_VISIBLE_DEVICES"
echo "Host: $(hostname)  |  $(date)"
echo "=========================================================="

echo "--- smoke test, 3 trials ---"
rm -f "$SMOKE"
.venv/bin/python bench/evaluate_vlm.py \
    --benchmark "$BENCHMARK" --model "$MODEL" --prompt_style "$PROMPT" \
    --max_tokens 512 --max_trials 3 --out "$SMOKE" 2>&1 | tee "${LOG%.log}.smoke.log"

# A run in which nothing parses is worse than a crash: it produces a full result
# file of nulls that scores at chance and looks like a finding.
.venv/bin/python - "$SMOKE" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["results"]
answered = [x for x in r if x.get("model_choice") is not None]
errs = [x for x in r if x.get("error")]
print(f"smoke: {len(r)} trials, {len(answered)} parsed, {len(errs)} errors")
if errs:
    print("first error:", errs[0]["error"]); sys.exit(1)
if not answered:
    print("no trial returned a parseable choice -- the processor or the prompt "
          "is wrong for this model family; not starting the full run")
    sys.exit(1)
PY

echo "--- full run, 500 trials ---"
.venv/bin/python bench/evaluate_vlm.py \
    --benchmark "$BENCHMARK" --model "$MODEL" --prompt_style "$PROMPT" \
    --max_tokens 512 --out "$OUT" 2>&1 | tee "$LOG"

echo "$MODEL ($PROMPT) finished at $(date)"
.venv/bin/python bench/archive/analyze_vii.py --json figures/vii_rows.json | tail -30
