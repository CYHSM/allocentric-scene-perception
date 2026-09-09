"""
Every model on one axis.

Reads the two arms -- frozen-encoder retrieval from `data/scenes_100/*/metrics.json`
and forced-choice runs from `results/*.json` -- converts each cell to d', and
reports VII(delta) = d'(delta)/d'(0). See `bench/vii.py` for why the raw numbers
cannot be compared directly.

Every forced-choice row carries its own positional-prior controls beside it.
Qwen2.5-VL-7B answers "4" on 59% of trials, so "26.2% correct against 25% chance"
is not by itself evidence of anything; the honest comparison is against what that
model's own answer distribution scores while ignoring the images.

    python bench/analyze_vii.py                  # table
    python bench/analyze_vii.py --json out.json  # machine-readable, for figures
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vii as V

MODES = ["c0_shape_colour", "c1_shape", "c2_colour", "c3_peaks_bare", "c4_valley"]
DELTAS = [0, 45, 90, 135, 180]

# timm ids are unreadable in a table and unstable as figure labels.
ENCODER_NAMES = {
    "vit_base_patch14_dinov2.lvd142m": "DINOv2-B/14",
    "vit_so400m_patch14_siglip_384.webli": "SigLIP-so400m",
    "vit_base_patch16_clip_224.openai": "CLIP-B/16",
    "resnet50.a1_in1k": "ResNet-50",
}


def encoder_rows(root="data/scenes_100", appearance="changed"):
    """
    One row per (model, mode). `appearance="changed"` is the real condition: the
    delta=0 cell is the same place from the same bearing under a different
    appearance sample, which is the model's appearance gate and the denominator
    of VII. The `same` records are the identity gate (all exactly 1.000) and are
    a pipeline check, not a result.
    """
    out = []
    for mode in MODES:
        path = os.path.join(root, mode, "metrics.json")
        if not os.path.exists(path):
            print(f"[WARN] {path} missing; skipping {mode}", file=sys.stderr)
            continue
        for rec in json.load(open(path)):
            if rec["appearance"] != appearance:
                continue
            by_delta = {int(k): v["recall@1"] for k, v in rec["by_delta"].items()}
            n = min(v["n"] for v in rec["by_delta"].values())
            m = rec["n_scenes"]          # retrieval over N scenes is N-AFC
            out.append({
                "arm": "retrieval",
                "model": ENCODER_NAMES.get(rec["model"], rec["model"]),
                "mode": mode,
                "m": m,
                "n_trials": n,
                "accuracy": by_delta,
                "vii": V.vii_curve(by_delta, m, n_trials=n),
            })
    return out


def afc_rows(paths, min_cell=20):
    """
    One row per forced-choice result file, plus a per-mode breakdown.

    Files still carrying `summary.in_progress` are reported but flagged: the
    published `table_vlm.tex` was generated from a 360/500 snapshot in which
    `c4_valley` had no trials at all, so the delta columns averaged over three
    modes in one row and five in the others.
    """
    out = []
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"[WARN] {path} missing; skipping {name}", file=sys.stderr)
            continue
        blob = json.load(open(path))
        res = blob.get("results", [])
        if not res:
            continue
        summary = blob.get("summary", {})
        m = res[0].get("n_options") or summary.get("n_options") or 4

        by = collections.defaultdict(list)
        for r in res:
            by[r["delta"]].append(bool(r["is_correct"]))
        acc = {d: sum(v) / len(v) for d, v in by.items()}
        n = min(len(v) for v in by.values())

        modes_seen = sorted({r["mode"] for r in res})
        cells = collections.Counter((r["mode"], r["delta"]) for r in res)
        ragged = len(set(cells.values())) > 1

        ctl = V.constant_answer_rate(
            [r.get("model_choice") for r in res],
            [r["correct_choice"] for r in res])

        out.append({
            "arm": f"{m}AFC",
            "model": name,
            "path": path,
            "m": m,
            "n_total": len(res),
            "n_trials": n,
            "accuracy": acc,
            "vii": V.vii_curve(acc, m, n_trials=n) if 0 in acc else {},
            "overall": sum(r["is_correct"] for r in res) / len(res),
            "control": ctl,
            "modes": modes_seen,
            "incomplete": bool(summary.get("in_progress"))
                          or len(modes_seen) < len(MODES) or ragged,
        })
    return out


def _fmt(v):
    return "   n/a" if v is None else f"{v:6.3f}"


def print_table(enc, afc):
    print("=" * 84)
    print("FROZEN ENCODERS   cross-view Recall@1 over a 100-scene gallery (m = 100)")
    print("=" * 84)
    hdr = " ".join(f"VII{d:>3d}" for d in DELTAS[1:])
    print(f"{'model':16s} {'mode':16s} {'R@1(0)':>7s} {hdr}")
    for r in enc:
        print(f"{r['model']:16s} {r['mode']:16s} {r['accuracy'][0]*100:6.1f}% "
              + " ".join(_fmt(r["vii"].get(d)) for d in DELTAS[1:]))

    print()
    print("=" * 84)
    print("FORCED CHOICE     with each model's own positional-prior controls")
    print("=" * 84)
    print(f"{'model':18s} {'n':>4s} {'acc(0)':>7s} {hdr}  {'overall':>7s} "
          f"{'bestfix':>7s} {'prior':>6s}")
    for r in afc:
        flag = "  << INCOMPLETE" if r["incomplete"] else ""
        a0 = r["accuracy"].get(0)
        print(f"{r['model']:18s} {r['n_total']:4d} "
              f"{(a0*100 if a0 is not None else float('nan')):6.1f}% "
              + " ".join(_fmt(r["vii"].get(d)) for d in DELTAS[1:])
              + f"  {r['overall']:7.3f} {r['control']['best_fixed']:7.3f} "
                f"{r['control']['distribution']:6.3f}{flag}")
        floor = max(r["control"]["best_fixed"], r["control"]["distribution"])
        if r["overall"] <= floor:
            which = ("always answering the most common option"
                     if r["control"]["best_fixed"] >= r["control"]["distribution"]
                     else "sampling its own answer distribution")
            print(f"{'':18s}   ^ NOT ABOVE ITS OWN PRIOR: {which} scores "
                  f"{floor:.3f} against this model's {r['overall']:.3f}")


DEFAULT_AFC = {
    "Qwen2.5-VL-3B": "results/qwen2_5_vl_3b_4afc_full.json",
    "Qwen2.5-VL-7B": "results/qwen2_5_vl_7b_4afc_full.json",
    "Qwen2.5-VL-32B": "results/qwen2_5_vl_32b_4afc_full.json",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/scenes_100")
    ap.add_argument("--results", default="results",
                    help="directory scanned for extra *_4afc_full.json / *_2afc*.json")
    ap.add_argument("--json", help="write the collated rows here")
    args = ap.parse_args()

    paths = dict(DEFAULT_AFC)
    for f in sorted(os.listdir(args.results)) if os.path.isdir(args.results) else []:
        full = os.path.join(args.results, f)
        if full in paths.values() or not f.endswith(".json"):
            continue
        if "_full" in f or "2afc" in f:
            paths.setdefault(os.path.splitext(f)[0], full)

    enc, afc = encoder_rows(args.root), afc_rows(paths)
    print_table(enc, afc)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"encoders": enc, "afc": afc}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
