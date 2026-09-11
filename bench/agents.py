"""
One record per agent, whoever or whatever the agent is.

Frozen encoders are scored by retrieval, VLMs by forced choice, people by the
same forced choice in a browser. The figures need them on one axis, and they
need to keep absorbing new ones -- 72B, InternVL, the frontier models -- without
anyone editing a plotting script. So discovery and metadata live here, and the
figure scripts consume `collect()`.

Each record carries what a figure needs to place it: `family` and `params_b` for
colour and ordering, `arm` and `m` for the chance level, `accuracy` and `vii` by
delta, `ci` for the error bars, and the positional-prior controls. Anything that
cannot be parsed from a filename is read from the result file's own summary
rather than guessed.

Two rules the discovery follows, both of which have already bitten:

* **A run is only a scaling point if it differs from the others in scale
  alone.** The prompt ablations are the same 7B model on the same trials; they
  are tagged `role="ablation"` so a scaling plot does not show them as four
  model sizes when there are two.
* **An incomplete or contaminated run is labelled, not silently averaged.** The
  32B table was published from a 360/500 snapshot missing a whole mode, and the
  500-trial 2AFC ran against a superseded benchmark file. Both are visible here
  as `complete` and `benchmark`.
"""

import collections
import functools
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vii as V

MODES = ["c0_shape_colour", "c1_shape", "c2_colour", "c3_peaks_bare", "c4_valley"]
DELTAS = [0, 45, 90, 135, 180]

ENCODER_NAMES = {
    "vit_base_patch14_dinov2.lvd142m": ("DINOv2-B/14", "dinov2", 0.086),
    "vit_so400m_patch14_siglip_384.webli": ("SigLIP-so400m", "siglip", 0.400),
    "vit_base_patch16_clip_224.openai": ("CLIP-B/16", "clip", 0.086),
    "resnet50.a1_in1k": ("ResNet-50", "resnet", 0.026),
}

# Prompt styles that count as the primary arm. Everything else is an ablation:
# same model, same trials, different instructions.
PRIMARY_STYLES = {"cot", "neutral", "direct", None}

FAMILY_PATTERNS = [
    (r"human", "human"), (r"qwen", "qwen"), (r"internvl", "internvl"),
    (r"gpt|o[34]-|openai", "openai"), (r"gemini", "gemini"),
    (r"claude", "anthropic"), (r"llama", "llama"), (r"pixtral|mistral", "mistral"),
    (r"molmo", "molmo"), (r"gemma", "gemma"),
]


def _family(name):
    low = name.lower()
    for pat, fam in FAMILY_PATTERNS:
        if re.search(pat, low):
            return fam
    return "other"


def _params_b(name):
    """Parameter count in billions, from the model id. None when unstated."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name)
    return float(m.group(1)) if m else None


def _pretty(model_id, prompt_style=None):
    """A label a reader recognises, without the vendor path or the -hf suffix."""
    if model_id.startswith("human:"):
        return "Human " + model_id.split(":", 1)[1]
    name = model_id.split("/")[-1]
    name = re.sub(r"-(hf|Instruct|instruct|it)$", "", name).replace("-Instruct", "")
    name = name.replace("Qwen2.5-VL", "Qwen2.5-VL").replace("InternVL3", "InternVL3")
    if prompt_style and prompt_style not in ("cot", None):
        name += f" ({prompt_style})"
    return name


def wilson(k, n, z=1.96):
    """Wilson score interval. The normal approximation is useless at the two
    places this data lives -- 0/1600 and 49/50 -- and both are load-bearing."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _vii_ci(acc, n_by_delta, m, n_boot=2000, seed=0):
    """
    Bootstrap VII over trials, per delta.

    VII is a ratio, and at ceiling or floor its sampling distribution is
    strongly asymmetric -- a human at 10/10 and 9/10 has a VII whose spread is
    nothing like symmetric. A normal error bar there would be a lie, so the
    interval is percentile bootstrap.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n0 = n_by_delta.get(0)
    if not n0:
        return {}
    k0 = rng.binomial(n0, acc[0], n_boot) / n0
    out = {}
    for d, p in acc.items():
        if d == 0:
            continue
        nd = n_by_delta.get(d) or n0
        kd = rng.binomial(nd, p, n_boot) / nd
        vals = [V.vii(a, b, m, n_trials=min(nd, n0)) for a, b in zip(kd, k0)]
        vals = [v for v in vals if v is not None]
        out[d] = (float(np.percentile(vals, 2.5)),
                  float(np.percentile(vals, 97.5))) if vals else None
    return out


# --------------------------------------------------------------------------- #
# Frozen encoders
# --------------------------------------------------------------------------- #

def encoder_records(root="data/scenes_100", appearance="changed", per_mode=False):
    """
    One record per encoder, averaged over the five cue modes (or one per mode
    with `per_mode`). The `same` appearance records are the identity gate, a
    pipeline check, and never a result.
    """
    per = collections.defaultdict(dict)
    for mode in MODES:
        path = os.path.join(root, mode, "metrics.json")
        if not os.path.exists(path):
            continue
        for rec in json.load(open(path)):
            if rec["appearance"] != appearance:
                continue
            per[rec["model"]][mode] = rec

    out = []
    for model_id, by_mode in per.items():
        label, family, pb = ENCODER_NAMES.get(
            model_id, (model_id, _family(model_id), _params_b(model_id)))
        targets = [(m, {m: r}) for m, r in by_mode.items()] if per_mode \
            else [(None, by_mode)]
        for mode_tag, group in targets:
            m = next(iter(group.values()))["n_scenes"]
            acc, ns, rsa = {}, {}, {}
            for d in DELTAS:
                cells = [r["by_delta"][str(d)] for r in group.values()
                         if str(d) in r["by_delta"]]
                if not cells:
                    continue
                n = sum(c["n"] for c in cells)
                acc[d] = sum(c["recall@1"] * c["n"] for c in cells) / n
                ns[d] = int(n)
                rsa[d] = sum(c["rsa"] for c in cells) / len(cells)
            out.append({
                "label": label + (f" · {mode_tag}" if mode_tag else ""),
                "model_id": model_id, "family": family, "params_b": pb,
                "arm": "retrieval", "m": m, "mode": mode_tag,
                "role": "primary", "kind": "encoder", "complete": True,
                "prompt_style": None, "benchmark": root,
                "accuracy": acc, "n": ns, "rsa": rsa,
                "ci": {d: wilson(round(acc[d] * ns[d]), ns[d]) for d in acc},
                "vii": V.vii_curve(acc, m, n_trials=min(ns.values())),
                "vii_ci": _vii_ci(acc, ns, m),
                "control": None,
            })
    return out


# --------------------------------------------------------------------------- #
# Forced choice: VLMs and people, from the same schema
# --------------------------------------------------------------------------- #

SKIP = re.compile(r"smoke|_test|_c[0-4]_|corrupt|calib|^_")


def afc_records(dirs=("results", "human_task"), min_trials=40, per_mode=False):
    """
    Every forced-choice result file that is not a smoke test or a per-mode shard.

    Per-mode shards are skipped because their `_full` concatenation is also
    present; counting both would double every model.
    """
    out = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            base = os.path.basename(path)
            if SKIP.search(base):
                continue
            try:
                blob = json.load(open(path))
            except ValueError:
                continue
            res = blob.get("results")
            if not res or len(res) < min_trials:
                continue
            out.extend(_afc_record(path, blob, per_mode))
    return out


@functools.lru_cache(maxsize=32)
def _benchmark_answers(name):
    """{trial_id: correct_choice} for a benchmark file, if it is still on disk."""
    for cand in (name, os.path.join("data", name), os.path.join("human_task", name)):
        if os.path.exists(cand):
            try:
                return {t["id"]: t["correct_choice"]
                        for t in json.load(open(cand))["trials"]}
            except (ValueError, KeyError):
                return None
    return None


def _stale(res, benchmark):
    """
    True when the run's own recorded answers disagree with the benchmark file of
    that name as it stands now.

    This is not paranoia. The 500-trial 2AFC run was executed against a
    superseded copy of `vlm_benchmark_2afc.json`: it disagrees with the current
    file on 257 of 500 trials, and nothing in the result file says so. Without
    this check it collates as a clean run and lands in a figure.
    """
    answers = _benchmark_answers(benchmark)
    if not answers:
        return None                       # cannot tell; do not claim either way
    bad = sum(1 for r in res
              if r["trial_id"] in answers
              and answers[r["trial_id"]] != r["correct_choice"])
    return bad > 0


def _infer_m(res):
    """
    How many alternatives a run had, when nothing recorded it.

    A run still in progress has no `n_options` in its summary yet, and the old
    fallback was a bare `4`. That is not a harmless default: it puts a 2AFC run
    on a 25% chance level, which inflates every d' and sends VII off the top --
    the half-finished Gemini 2AFC run collated at VII(90) = 5.19. Read it off
    the answers instead, which are present from the first trial.
    """
    seen = {r.get("correct_choice") for r in res}
    seen |= {r.get("model_choice") for r in res}
    seen.discard(None)
    return max(seen) if seen else 4


def _afc_record(path, blob, per_mode):
    res = blob["results"]
    summary = blob.get("summary", {})
    model_id = summary.get("model") or os.path.splitext(os.path.basename(path))[0]
    style = summary.get("prompt_style")
    m = summary.get("n_options") or res[0].get("n_options") or _infer_m(res)
    is_human = str(model_id).startswith("human")

    groups = ([(mo, [r for r in res if r["mode"] == mo]) for mo in MODES]
              if per_mode else [(None, res)])
    recs = []
    for mode_tag, rows in groups:
        if not rows:
            continue
        by = collections.defaultdict(list)
        for r in rows:
            by[r["delta"]].append(bool(r["is_correct"]))
        acc = {d: sum(v) / len(v) for d, v in by.items()}
        ns = {d: len(v) for d, v in by.items()}
        if 0 not in acc:
            continue

        cells = collections.Counter((r["mode"], r["delta"]) for r in rows)
        bench_name = os.path.basename(summary.get("benchmark", "?"))
        stale = _stale(res, bench_name)
        complete = (bool(summary) and not summary.get("in_progress")
                    and (per_mode or len({r["mode"] for r in rows}) == len(MODES))
                    and len(set(cells.values())) == 1
                    and stale is not True)

        recs.append({
            "label": _pretty(model_id, style) + (f" · {mode_tag}" if mode_tag else ""),
            "model_id": model_id, "family": _family(model_id),
            "params_b": None if is_human else _params_b(model_id),
            "arm": f"{m}AFC", "m": m, "mode": mode_tag,
            "role": "primary" if style in PRIMARY_STYLES else "ablation",
            "kind": "human" if is_human else "vlm",
            "complete": complete, "prompt_style": style,
            "stale_benchmark": stale,
            "benchmark": bench_name,
            "path": path,
            "accuracy": acc, "n": ns, "rsa": {},
            "ci": {d: wilson(sum(by[d]), ns[d]) for d in acc},
            "vii": V.vii_curve(acc, m, n_trials=min(ns.values())),
            "vii_ci": _vii_ci(acc, ns, m),
            "control": V.constant_answer_rate(
                [r.get("model_choice") for r in rows],
                [r["correct_choice"] for r in rows]),
            "overall": sum(r["is_correct"] for r in rows) / len(rows),
        })
    return recs


def pool_humans(records):
    """
    Collapse participants into one `Human (n=k)` record.

    Reported as the mean over participants with a bootstrap interval over
    participants, not over trials: with several people the between-person spread
    is the uncertainty that matters, and pooling trials would understate it.
    A single participant is left as a single participant and labelled so.
    """
    import numpy as np
    humans = [r for r in records if r["kind"] == "human"]
    if len(humans) < 2:
        return records
    others = [r for r in records if r["kind"] != "human"]
    m = humans[0]["m"]
    acc, ns, ci = {}, {}, {}
    rng = np.random.default_rng(0)
    for d in DELTAS:
        vals = [h["accuracy"][d] for h in humans if d in h["accuracy"]]
        if not vals:
            continue
        acc[d] = float(np.mean(vals))
        ns[d] = sum(h["n"].get(d, 0) for h in humans)
        boot = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(2000)]
        ci[d] = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    pooled = {
        "label": f"Human (n={len(humans)})", "model_id": "human:pooled",
        "family": "human", "params_b": None, "arm": humans[0]["arm"], "m": m,
        "mode": None, "role": "primary", "kind": "human", "complete": True,
        "prompt_style": humans[0]["prompt_style"],
        "benchmark": humans[0]["benchmark"],
        "accuracy": acc, "n": ns, "rsa": {}, "ci": ci,
        "vii": V.vii_curve(acc, m, n_trials=min(ns.values())),
        "vii_ci": _vii_ci(acc, ns, m),
        "control": {"best_fixed": 1.0 / m, "distribution": 1.0 / m},
        "n_participants": len(humans),
    }
    return others + [pooled]


def collect(root="data/scenes_100", dirs=("results", "human_task"),
            per_mode=False, primary_only=True, complete_only=False,
            pool=True):
    recs = encoder_records(root, per_mode=per_mode) + afc_records(dirs, per_mode=per_mode)
    if pool:
        recs = pool_humans(recs)
    if primary_only:
        recs = [r for r in recs if r["role"] == "primary"]
    if complete_only:
        recs = [r for r in recs if r["complete"]]
    order = {"human": 0, "encoder": 2, "vlm": 1}
    recs.sort(key=lambda r: (order.get(r["kind"], 3), r["family"],
                             r["params_b"] if r["params_b"] is not None else 1e9,
                             r["label"]))
    return recs


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per_mode", action="store_true")
    ap.add_argument("--all", action="store_true", help="include ablation runs")
    ap.add_argument("--json", help="write the records here")
    args = ap.parse_args()

    recs = collect(per_mode=args.per_mode, primary_only=not args.all)
    head = " ".join(f"VII{d:>3d}" for d in DELTAS[1:])
    print(f"{'agent':30s} {'arm':9s} {'n':>5s} {'acc0':>6s} {head}  flags")
    for r in recs:
        vii = " ".join("   n/a" if r["vii"].get(d) is None else f"{r['vii'][d]:6.3f}"
                       for d in DELTAS[1:])
        flags = []
        if r.get("stale_benchmark"):
            flags.append("STALE-BENCHMARK")
        elif not r["complete"]:
            flags.append("INCOMPLETE")
        if r["control"] and r.get("overall", 1) <= max(
                r["control"]["best_fixed"], r["control"]["distribution"]):
            flags.append("at-its-own-prior")
        print(f"{r['label']:30s} {r['arm']:9s} {min(r['n'].values()):5d} "
              f"{r['accuracy'][0]*100:5.1f}% {vii}  {','.join(flags)}")

    if args.json:
        json.dump(recs, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
