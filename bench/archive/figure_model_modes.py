"""
One model across the cue ladder: accuracy per (mode, delta), with intervals.

Written because a per-delta average can hide how few trials sit under it. The
50-trial human-matched slice puts **2 trials in each of the 25 cells**, so every
cell is one of {0%, 50%, 100%} and a Wilson interval on it spans most of the
axis. The figure therefore plots the interval, prints the raw k/n in each cell,
and marks the chance line: anything whose interval crosses chance is not
evidence yet, however dark the cell looks.

Usage:
    python3 bench/figure_model_modes.py results/or_google_gemini-3.8-flash_cot_n50.json
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agents

DELTAS = [0, 45, 90, 135, 180]
MODES = agents.MODES
NICE = {"c0_shape_colour": "c0  shape+colour", "c1_shape": "c1  shape",
        "c2_colour": "c2  colour", "c3_peaks_bare": "c3  peaks bare",
        "c4_valley": "c4  valley"}


def wilson(k, n, z=1.96):
    """Wilson score interval -- behaves at k=0 and k=n, which Wald does not."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cells(res):
    out = {}
    for mode in MODES:
        for d in DELTAS:
            rows = [r for r in res if r["mode"] == mode and r["delta"] == d]
            out[(mode, d)] = (sum(r["is_correct"] for r in rows), len(rows))
    return out


def main(path, overlay=None, out=None):
    blob = json.load(open(path))
    res = blob["results"]
    m = blob["summary"].get("n_options") or res[0].get("n_options") or 2
    chance = 1.0 / m
    name = blob["summary"].get("model", os.path.basename(path))
    grid = cells(res)
    per_cell_n = {n for (_, n) in grid.values() if n}

    over = None
    if overlay and os.path.exists(overlay):
        ov = json.load(open(overlay))
        over = cells(ov["results"])

    fig = plt.figure(figsize=(12.4, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.35, 1.0], wspace=0.35)

    # --- A: the grid, with raw counts written in ------------------------------
    ax = fig.add_subplot(gs[0])
    img = np.full((len(MODES), len(DELTAS)), np.nan)
    for i, mode in enumerate(MODES):
        for j, d in enumerate(DELTAS):
            k, n = grid[(mode, d)]
            if n:
                img[i, j] = k / n - chance
    im = ax.imshow(img, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for i, mode in enumerate(MODES):
        for j, d in enumerate(DELTAS):
            k, n = grid[(mode, d)]
            ax.text(j, i, f"{k}/{n}", ha="center", va="center", fontsize=7.5)
    ax.set_xticks(range(len(DELTAS)), DELTAS, fontsize=8)
    ax.set_yticks(range(len(MODES)), [NICE[m_] for m_ in MODES], fontsize=8)
    ax.set_xlabel("$\\Delta$ (degrees)", fontsize=9)
    ax.set_title("A. accuracy $-$ chance, per cell", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # --- B: per-mode curves with Wilson intervals -----------------------------
    ax = fig.add_subplot(gs[1])
    cmap = plt.get_cmap("viridis")
    for i, mode in enumerate(MODES):
        ks = [grid[(mode, d)] for d in DELTAS]
        ys = [k / n if n else np.nan for k, n in ks]
        lo = [wilson(k, n)[0] for k, n in ks]
        hi = [wilson(k, n)[1] for k, n in ks]
        col = cmap(i / (len(MODES) - 1))
        ax.plot(DELTAS, ys, "-o", ms=4, color=col, label=NICE[mode], lw=1.5)
        ax.fill_between(DELTAS, lo, hi, color=col, alpha=0.12, lw=0)
    ax.axhline(chance, color="#333", ls="--", lw=1.0)
    ax.annotate("chance", (182, chance), fontsize=7, va="center", color="#333")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(DELTAS)
    ax.set_xlabel("$\\Delta$ (degrees)", fontsize=9)
    ax.set_ylabel("accuracy", fontsize=9)
    ax.set_title("B. per mode, with 95% Wilson intervals\n"
                 "(shaded band = what these data actually constrain)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # --- C: pooled over delta, per mode; plus the corrected delta=0 -----------
    ax = fig.add_subplot(gs[2])
    x = np.arange(len(MODES))
    ys, err = [], [[], []]
    for mode in MODES:
        k = sum(grid[(mode, d)][0] for d in DELTAS)
        n = sum(grid[(mode, d)][1] for d in DELTAS)
        p = k / n if n else np.nan
        lo, hi = wilson(k, n)
        ys.append(p)
        err[0].append(p - lo)
        err[1].append(hi - p)
    ax.bar(x, ys, color="#c8791a", yerr=err, capsize=3, error_kw={"lw": 1})
    ax.axhline(chance, color="#333", ls="--", lw=1.0)
    if over:
        ok = [over[(mo, 0)] for mo in MODES]
        ax.plot(x, [k / n if n else np.nan for k, n in ok], "k*", ms=11,
                label="$\\Delta=0$ re-run,\ncorrected prompt", ls="none")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_xticks(x, [m_.split("_")[0] for m_ in MODES], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy (pooled over $\\Delta$)", fontsize=9)
    ax.set_title(f"C. per mode, n={sum(n for _, n in grid.values()) // len(MODES)} each",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{name} on the cue ladder   "
                 f"({sum(n for _, n in grid.values())} trials, "
                 f"{sorted(per_cell_n)} per cell)", fontsize=11)
    out = out or f"figures/fig_modes_{os.path.splitext(os.path.basename(path))[0]}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", out)

    print(f"\n{'mode':18s} {'k/n':>7s} {'acc':>6s}  95% Wilson")
    for mode in MODES:
        k = sum(grid[(mode, d)][0] for d in DELTAS)
        n = sum(grid[(mode, d)][1] for d in DELTAS)
        lo, hi = wilson(k, n)
        print(f"{NICE[mode]:18s} {k:3d}/{n:<3d} {k/n:6.2f}  [{lo:.2f}, {hi:.2f}]"
              f"{'   crosses chance' if lo <= chance else ''}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "results/or_google_gemini-3.8-flash_cot_n50.json"
    main(p, overlay="results/or_google_gemini-3.8-flash_cot_anyview_n10.json")
