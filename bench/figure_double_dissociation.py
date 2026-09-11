"""
Human and frontier VLM fail on orthogonal axes.

Two panels, because the claim is an interaction and not a gap:

  A  accuracy against viewpoint change. The person pays for rotation; the model
     does not.
  B  accuracy against how confusable the foils are. The model pays for
     similarity; the person does not.

Panel B uses every mode, on a percentile axis rather than metres. Absolute
metres are not comparable across modes: c0/c1/c2 have a median pairwise layout
distance of 32.4 m and c3/c4 of 7.3 m, because c3/c4 minimise over landmark
permutations as well. Pooling raw metres put 32 of the 37 shortest-distance
trials in c3/c4 -- which are the model's easiest modes -- and produced a
U-shaped curve that was really a mode effect. Ranking each trial's nearest-foil
distance within *its own mode's* full pairwise distribution removes that: the
five modes then span the same 0-7th percentile range (medians 1.6 to 2.5), so a
bin means the same thing everywhere and no trial is discarded.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DELTAS = [0, 45, 90, 135, 180]
CTRL = {"c0_shape_colour", "c1_shape", "c2_colour"}


def percentile_ranker(npz="data/layout_distance.npz"):
    """trial distance -> its percentile within that mode's own distance distribution."""
    store = np.load(npz, allow_pickle=True)
    tables = {}
    for key in store.files:
        if key.endswith("__D"):
            mode = key[:-3]
            D = store[key]
            tables[mode] = np.sort(D[~np.eye(len(D), dtype=bool)])
    return lambda mode, d: 100.0 * np.searchsorted(tables[mode], d) / len(tables[mode])
HUMAN = "#111111"
MODEL = "#c8791a"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def curve(ax, xs, ks, ns, colour, label, marker="o"):
    ys = [k / n if n else np.nan for k, n in zip(ks, ns)]
    lo = [wilson(k, n)[0] for k, n in zip(ks, ns)]
    hi = [wilson(k, n)[1] for k, n in zip(ks, ns)]
    ax.plot(xs, ys, marker=marker, color=colour, lw=2, ms=6, label=label)
    ax.fill_between(xs, lo, hi, color=colour, alpha=0.13, lw=0)
    for x, k, n in zip(xs, ks, ns):
        if n:
            ax.annotate(f"{k}/{n}", (x, k / n), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=6.5, color=colour)


def main(human_path, model_path, out="figures/fig_double_dissociation.png"):
    bench = {t["id"]: t for t in
             json.load(open("data/vlm_benchmark_4afc_hard.json"))["trials"]}
    H = json.load(open(human_path))["results"]
    G = {x["trial_id"]: x for x in json.load(open(model_path))["results"]}
    # Pair strictly: the person has not answered every trial the model has.
    P = [(h, G[h["trial_id"]]) for h in H if h["trial_id"] in G]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

    # --- A: viewpoint ------------------------------------------------------
    # Each observer on every trial it has answered, not only the paired subset.
    # The model has done all 100 and the person 40, and restricting the model to
    # the person's 40 threw away more than half its data and made a null
    # rotation effect (12/20 8/20 12/20 8/20 12/20 over 100, Fisher p = 0.65)
    # look like a decline. Panel B stays paired, because there the comparison is
    # trial-for-trial.
    ax = axes[0]
    Gall = list(json.load(open(model_path))["results"])
    for rows, colour, name, marker in ((H, HUMAN, f"Human p01 (n={len(H)})", "o"),
                                       (Gall, MODEL, f"Gemini 3.8 Flash (n={len(Gall)})", "s")):
        ks = [sum(x["is_correct"] for x in rows if x["delta"] == d) for d in DELTAS]
        ns = [sum(1 for x in rows if x["delta"] == d) for d in DELTAS]
        curve(ax, DELTAS, ks, ns, colour, name, marker)
    ax.axhline(0.25, color="#555", ls="--", lw=1)
    ax.annotate("chance", (183, 0.25), fontsize=7, va="center", color="#555")
    ax.set_xticks(DELTAS)
    ax.set_xlabel("viewpoint change $\\Delta$ (degrees)")
    ax.set_ylabel("accuracy")
    ax.set_title("A. Rotation costs the person (p = 0.047).\nThe model shows no effect (p = 0.65).", fontsize=10)

    # --- B: foil similarity, every mode, on a per-mode percentile axis -----
    ax = axes[1]
    rank = percentile_ranker()
    def pr(h):
        return rank(h["mode"], bench[h["trial_id"]]["min_foil_distance_m"])

    vals = sorted(pr(h) for h, _ in P)
    # Quartiles of the trials actually run, so the four bins are equally filled
    # and no bin's interval is dominated by having two trials in it.
    edges = [0.0] + [vals[int(len(vals) * q)] for q in (0.25, 0.5, 0.75)] + [100.0]
    centres, labels = [], []
    ks_h, ns_h, ks_g, ns_g = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [p for p in P if lo <= pr(p[0]) < hi or (hi == 100.0 and pr(p[0]) >= lo)]
        if not sel:
            continue
        centres.append(len(centres))
        labels.append(f"{lo:.1f}-{min(hi, max(vals)):.1f}")
        ks_h.append(sum(p[0]["is_correct"] for p in sel)); ns_h.append(len(sel))
        ks_g.append(sum(p[1]["is_correct"] for p in sel)); ns_g.append(len(sel))
    curve(ax, centres, ks_h, ns_h, HUMAN, f"Human p01 (paired, n={len(P)})", "o")
    curve(ax, centres, ks_g, ns_g, MODEL, f"Gemini 3.8 Flash (paired, n={len(P)})", "s")
    ax.axhline(0.25, color="#555", ls="--", lw=1)
    ax.set_xticks(centres, labels, fontsize=8)
    ax.set_xlabel("nearest foil, as a percentile of that mode's own\n"
                  "layout-distance distribution  (all five modes)")
    ax.set_title("B. Foil similarity costs the model.\nThe person is flat across it.",
                 fontsize=10)

    for ax in axes:
        ax.set_ylim(-0.03, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower left")

    fig.suptitle("Hard-foil benchmark (4AFC, chance 25%): the two observers fail "
                 "on orthogonal axes", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    fig.savefig(out.replace(".png", ".pdf"))
    print("wrote", out, f"({len(P)} paired trials, all {len({h['mode'] for h,_ in P})} modes in panel B)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "human_task_hard/human_p01_4afc_partial40.json",
         sys.argv[2] if len(sys.argv) > 2
         else "results/or_google_gemini-3.8-flash_cot_anyview_hardfoils_n100.json")
