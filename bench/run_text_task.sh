#!/usr/bin/env bash
# The 2AFC task with the pictures replaced by coordinates.
#
#   bash bench/run_text_task.sh id   Qwen/Qwen2.5-VL-7B-Instruct        # local
#   bash bench/run_text_task.sh noid Qwen/Qwen2.5-VL-7B-Instruct
#   OPENROUTER_API_KEY=sk-or-... bash bench/run_text_task.sh id openai/gpt-5 100 0.50
#
# Both variants run the same 100 trials; `id` names the landmarks and `noid`
# does not, so the identity effect is within-trial. Text trials carry no images,
# so a paid run here costs a fraction of the image arm and any text-only model
# can take it.
#
# delta = 0 is an attention check in this channel, not an appearance gate: with
# exact coordinates the correct candidate is a byte-for-byte copy of the study
# block. Read it as "did the model read the list", and take the viewpoint
# result from delta >= 45.
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT="${1:-id}"
MODEL="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
N="${3:-100}"
BUDGET="${4:-0}"
PROMPT="${PROMPT_STYLE:-neutral}"
BENCH="data/text_benchmark_2afc_${VARIANT}.json"

case "$VARIANT" in id|noid) ;; *) echo "variant must be id or noid" >&2; exit 1;; esac
[ -f "$BENCH" ] || python3 bench/build_text_benchmark.py

SLUG=$(echo "$MODEL" | tr '/:[:upper:]' '__[:lower:]' | tr -cd 'a-z0-9_.-')
OUT="results/text_${VARIANT}_${SLUG}_${PROMPT}.json"
mkdir -p results

ARGS=(--benchmark "$BENCH" --model "$MODEL" --prompt_style "$PROMPT"
      --max_trials "$N" --max_tokens 512 --out "$OUT")
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    ARGS+=(--api_base "https://openrouter.ai/api/v1" --api_key "$OPENROUTER_API_KEY")
    if [ "$BUDGET" = "0" ]; then
        echo "Refusing a paid endpoint with no budget. Pass one as argument 4." >&2
        exit 1
    fi
    ARGS+=(--budget_usd "$BUDGET")
fi

echo "variant  $VARIANT   model $MODEL   trials $N   prompt $PROMPT"
echo "out      $OUT"
python3 bench/evaluate_vlm.py "${ARGS[@]}"

python3 - "$OUT" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1])); r = d["results"]
ans = [x for x in r if x.get("model_choice") is not None]
print(f"\nparsed {len(ans)}/{len(r)}")
by = collections.defaultdict(list)
for x in r:
    by[x["delta"]].append(x["is_correct"])
for k in sorted(by):
    v = by[k]
    tag = "  (attention check)" if k == 0 else ""
    print(f"  delta {k:3d}: {sum(v)}/{len(v)} = {100*sum(v)/len(v):5.1f}%{tag}")
ch = collections.Counter(x.get("model_choice") for x in r)
print("choice distribution:", dict(ch))
PY
