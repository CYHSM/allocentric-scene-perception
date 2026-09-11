"""
The hard-foil benchmark split every way: by mode, and by mode x delta.

Three things on one sheet, because the per-mode story is where the stimulus
design shows up and the pooled numbers hide it:

  A  accuracy per stimulus mode, for the model on random foils, the model on
     hard foils, and the person on hard foils. The gap between the two model
     bars is what foil selection bought.
  B  the model's mode x delta grid (4 trials per cell).
  C  the person's mode x delta grid, same axes. Sparse -- the person has done
     40 of the 100 -- so the raw k/n is printed in every cell rather than
     leaving a colour to imply a precision that is not there.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = ["c0_shape_colour", "c1_shape", "c2_colour", "c3_peaks_bare", "c4_valley"]
SHORT = ["c0\nshape+colour", "c1\nshape", "c2\ncolour", "c3\npeaks bare", "c4\nvalley"]
DELTAS = [0, 45, 90, 135, 180]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def by_mode(res):
    return {m: (sum(x["is_correct"] for x in res if x["mode"] == m),
                sum(1 for x in res if x["mode"] == m)) for m in MODES}


def grid(res):
    g = np.full((len(MODES), len(DELTAS)), np.nan)
    counts = {}
    for i, m in enumerate(MODES):
        for j, d in enumerate(DELTAS):
            rows = [x for x in res if x["mode"] == m and x["delta"] == d]
            counts[(i, j)] = (sum(x["is_correct"] for x in rows), len(rows))
            if rows:
                g[i, j] = sum(x["is_correct"] for x in rows) / len(rows)
    return g, counts


def draw_grid(ax, g, counts, title, chance):
    im = ax.imshow(g - chance, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for (i, j), (k, n) in counts.items():
        ax.text(j, i, f"{k}/{n}" if n else "-", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(DELTAS)), DELTAS, fontsize=8)
    ax.set_yticks(range(len(MODES)), [m.split("_")[0] for m in MODES], fontsize=8)
    ax.set_xlabel("$\\Delta$", fontsize=9)
    ax.set_title(title, fontsize=9.5)
    return im


def panel_a_only(out, keep_deltas, subtitle):
    """
    Panel A alone, over a chosen subset of viewpoint changes.

    Delta = 0 is the appearance gate, not a viewpoint change: study and target
    sit at the same azimuth and only the weather is resampled. Pooling it into a
    per-mode accuracy mixes "can it match an appearance" with "does the match
    survive a turn", which are the two things this benchmark exists to separate.
    """
    rand = [x for x in json.load(open("results/or_google_gemini-3.8-flash_cot_anyview_randomfoils_n100.json"))["results"] if x["delta"] in keep_deltas]
    hard = [x for x in json.load(open("results/or_google_gemini-3.8-flash_cot_anyview_hardfoils_n100.json"))["results"] if x["delta"] in keep_deltas]
    hum = [x for x in json.load(open("human_task_hard/human_p01_4afc_partial40.json"))["results"] if x["delta"] in keep_deltas]
    chance = 0.25

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    series = [("Gemini, random foils", by_mode(rand), "#e0a44f"),
              ("Gemini, hard foils", by_mode(hard), "#b8620a"),
              ("Human p01, hard foils", by_mode(hum), "#111111")]
    x = np.arange(len(MODES))
    w = 0.26
    for si, (name, d, col) in enumerate(series):
        n_tot = sum(v[1] for v in d.values())
        ys, err = [], [[], []]
        for m in MODES:
            k, n = d[m]
            p = k / n if n else np.nan
            lo, hi = wilson(k, n)
            ys.append(p)
            err[0].append(p - lo if n else 0)
            err[1].append(hi - p if n else 0)
        ax.bar(x + (si - 1) * w, ys, w, label=f"{name} (n={n_tot})", color=col,
               yerr=err, capsize=2.5, error_kw={"lw": 1})
        for xi, m in zip(x, MODES):
            k, n = d[m]
            if n:
                ax.text(xi + (si - 1) * w, 0.025, f"{k}/{n}", ha="center",
                        fontsize=7, rotation=90,
                        color="#fff" if si == 2 else "#333")
    ax.axhline(chance, color="#555", ls="--", lw=1.1)
    ax.annotate("chance", (len(MODES) - 0.5, chance + 0.02), fontsize=8, color="#555")
    ax.set_xticks(x, SHORT, fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("accuracy")
    ax.set_title(subtitle, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    fig.savefig(out.replace(".png", ".pdf"))
    print("wrote", out)
    a, b, c = by_mode(rand), by_mode(hard), by_mode(hum)
    print(f"\n{'mode':18s} {'Gem random':>11s} {'Gem hard':>10s} {'Human hard':>11s}")
    for m in MODES:
        print(f"{m:18s} {a[m][0]:3d}/{a[m][1]:<7d} {b[m][0]:3d}/{b[m][1]:<6d} {c[m][0]:3d}/{c[m][1]:<7d}")


def main(out="figures/fig_modes_full.png"):
    rand = json.load(open("results/or_google_gemini-3.8-flash_cot_anyview_randomfoils_n100.json"))["results"]
    hard = json.load(open("results/or_google_gemini-3.8-flash_cot_anyview_hardfoils_n100.json"))["results"]
    hum = json.load(open("human_task_hard/human_p01_4afc_partial40.json"))["results"]
    chance = 0.25

    fig = plt.figure(figsize=(14.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1, 1], wspace=0.3)

    # --- A ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    series = [("Gemini, random foils (n=100)", by_mode(rand), "#e0b070"),
              ("Gemini, hard foils (n=100)", by_mode(hard), "#c8791a"),
              ("Human p01, hard foils (n=40)", by_mode(hum), "#111111")]
    x = np.arange(len(MODES))
    w = 0.26
    for s, (name, d, col) in enumerate(series):
        ys = [d[m][0] / d[m][1] if d[m][1] else np.nan for m in MODES]
        err = [[], []]
        for m in MODES:
            k, n = d[m]
            p = k / n if n else np.nan
            lo, hi = wilson(k, n)
            err[0].append(p - lo if n else 0)
            err[1].append(hi - p if n else 0)
        ax.bar(x + (s - 1) * w, ys, w, label=name, color=col,
               yerr=err, capsize=2, error_kw={"lw": 0.9})
        for xi, m in zip(x, MODES):
            k, n = d[m]
            if n:
                ax.text(xi + (s - 1) * w, 0.02, f"{k}/{n}", ha="center",
                        fontsize=6, rotation=90,
                        color="#fff" if s == 2 else "#333")
    ax.axhline(chance, color="#555", ls="--", lw=1)
    ax.annotate("chance", (len(MODES) - 0.45, chance + 0.02), fontsize=7, color="#555")
    ax.set_xticks(x, SHORT, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("A. Per stimulus mode", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # --- B, C ---------------------------------------------------------------
    g1, c1 = grid(hard)
    ax1 = fig.add_subplot(gs[1])
    im = draw_grid(ax1, g1, c1, "B. Gemini, hard foils\n(4 per cell)", chance)
    g2, c2 = grid(hum)
    ax2 = fig.add_subplot(gs[2])
    draw_grid(ax2, g2, c2, "C. Human p01, hard foils\n(1-3 per cell, 40 of 100 done)", chance)
    fig.colorbar(im, ax=[ax1, ax2], label="accuracy $-$ chance", shrink=0.82)

    fig.suptitle("Hard-foil benchmark, 4AFC (chance 25%) — full split by mode and viewpoint",
                 fontsize=11.5)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", out)
    print(f"\n{'mode':18s} {'Gem random':>11s} {'Gem hard':>9s} {'Human hard':>11s}")
    a, b, c = by_mode(rand), by_mode(hard), by_mode(hum)
    for m in MODES:
        print(f"{m:18s} {a[m][0]:3d}/{a[m][1]:<7d} {b[m][0]:3d}/{b[m][1]:<5d} {c[m][0]:3d}/{c[m][1]:<7d}")


if __name__ == "__main__":
    main()
