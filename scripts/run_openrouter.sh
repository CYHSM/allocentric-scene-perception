#!/usr/bin/env bash
# Frontier models on the four-mountains bank, via OpenRouter.
#
#   export OPENROUTER_API_KEY=sk-or-...
#   bash scripts/run_openrouter.sh                       # the free smoke test
#   bash scripts/run_openrouter.sh <model> <n> <budget>  # a paid run
#
# The default does a free model at n=20 and spends nothing: it exists to prove
# the request format, the image encoding and the answer parsing work end to end
# before any money is involved. Only widen it once that has passed.
#
# Every paid invocation needs a budget in dollars and stops when the provider's
# reported spend reaches it, keeping the trials already paid for. A 4AFC trial
# ships five 640x440 PNGs, so cost is dominated by image tokens and scales with
# n, not with how long the replies are.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-meta-llama/llama-3.2-11b-vision-instruct:free}"
N="${2:-20}"
BUDGET="${3:-0}"
PROMPT="${PROMPT_STYLE:-neutral}"
BENCH="${BENCHMARK:-data/vlm_benchmark_4afc_hard.json}"
# Reasoning models spend the whole budget on hidden thinking and return an empty
# content field; raise this for them (dots-3 needs ~2000) or the parse sees "".
MAX_TOKENS="${MAX_TOKENS:-512}"
# Parallel in-flight requests. The run is provider latency end to end, so this
# is close to a linear speedup until OpenRouter rate limits.
WORKERS="${WORKERS:-8}"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "OPENROUTER_API_KEY is not set. Get a key at https://openrouter.ai/keys" >&2
    exit 1
fi

SLUG=$(echo "$MODEL" | tr '/:[:upper:]' '__[:lower:]' | tr -cd 'a-z0-9_.-')
# The benchmark name is part of the identity of a run. Without it, the same
# model+prompt+n against the random-foil bank and against the hard-foil bank
# both land on one filename and silently overwrite each other.
BENCH_TAG=$(basename "$BENCH" .json | sed 's/^vlm_benchmark_//; s/^task$/human50/')
OUT="results/or_${SLUG}_${PROMPT}_${BENCH_TAG}_n${N}.json"
mkdir -p results

echo "model    $MODEL"
echo "trials   $N (stratified over the 25 mode x delta cells)"
echo "prompt   $PROMPT"
echo "bench    $BENCH"
echo "budget   \$$BUDGET"
echo "out      $OUT"
echo

# bash 3.2 (macOS) errors on ${arr[@]} for an empty array under `set -u`;
# the ${arr[@]+...} guard makes the empty case expand to nothing instead.
BUDGET_ARG=()
if [ "$BUDGET" != "0" ]; then BUDGET_ARG=(--budget_usd "$BUDGET"); fi

python3 bench/evaluate_vlm.py \
    --benchmark "$BENCH" \
    --model "$MODEL" \
    --api_base "https://openrouter.ai/api/v1" \
    --api_key "$OPENROUTER_API_KEY" \
    --prompt_style "$PROMPT" \
    --max_trials "$N" \
    --max_tokens "$MAX_TOKENS" \
    --workers "$WORKERS" \
    ${BUDGET_ARG[@]+"${BUDGET_ARG[@]}"} \
    --out "$OUT"

echo
python3 - "$OUT" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1])); r = d["results"]
ans = [x for x in r if x.get("model_choice") is not None]
err = [x for x in r if x.get("error")]
print(f"parsed {len(ans)}/{len(r)}   errors {len(err)}")
if err:
    print("first error:", err[0]["error"][:200])
if not ans:
    print("NOTHING PARSED -- do not spend money on this model until the reply "
          "format is understood. One raw reply:")
    print((r[0].get('reply') or '')[:400]); sys.exit(1)
u = d["summary"].get("api_usage", {})
if u:
    per = u.get("usd_per_call") or 0
    print(f"spent ${u['spent_usd']:.4f} over {u['calls']} calls "
          f"(${per:.5f}/call)  ->  500 trials would be about ${per * 500:.2f}")
PY
