"""
Panel A of `figure_modes_full.py`, opened up: every mode against both axes.

Five columns, one per stimulus mode. Top row splits by viewpoint change, bottom
row by how close the nearest foil is. Three observers on each: the model on
random foils, the model on hard foils, the person on hard foils.

Two things make this readable at these sample sizes.

* **Coarse bins.** Per mode there are 20 model trials and 5-12 human ones, so a
  five-level delta axis would put 4 and 1 trials in a cell. Delta is collapsed to
  0 / 45-90 / 135-180 and distance to three percentile bands.
* **The two model runs share the distance axis.** Trial ids collide between the
  random and hard benchmark files, so each result is looked up in the benchmark
  it was actually run against, and the random run's foils -- median 32nd
  percentile against the hard run's 2nd -- extend the x-range instead of being a
  separate condition. Distance is ranked within each mode's own distribution,
  because absolute metres are not comparable across modes.

Every point prints k/n. Where n is small the interval is most of the axis, and
that is the honest picture rather than a smooth line through single trials.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = ["c0_shape_colour", "c1_shape", "c2_colour", "c3_peaks_bare", "c4_valley"]
TITLE = {"c0_shape_colour": "c0  shape+colour", "c1_shape": "c1  shape",
         "c2_colour": "c2  colour", "c3_peaks_bare": "c3  peaks bare",
         "c4_valley": "c4  valley"}
DBINS = [("0°", lambda d: d == 0), ("45-90°", lambda d: d in (45, 90)),
         ("135-180°", lambda d: d in (135, 180))]
PBINS = [("<2", 0, 2), ("2-10", 2, 10), (">10", 10, 101)]
CHANCE = 0.25


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ranker(npz="data/layout_distance.npz"):
    store = np.load(npz, allow_pickle=True)
    tab = {k[:-3]: np.sort(store[k][~np.eye(len(store[k]), dtype=bool)])
           for k in store.files if k.endswith("__D")}
    return lambda mode, d: 100.0 * np.searchsorted(tab[mode], d) / len(tab[mode])


def load(result_path, benchmark_path):
    """Results tagged with the foil distance from their OWN benchmark file."""
    bench = {t["id"]: t for t in json.load(open(benchmark_path))["trials"]}
    out = []
    for x in json.load(open(result_path))["results"]:
        t = bench.get(x["trial_id"])
        if t is None or x.get("model_choice") is None:
            continue
        out.append({**x, "min_foil": t["min_foil_distance_m"]})
    return out


def min_foil_for_random(benchmark_path, npz="data/layout_distance.npz"):
    """The random-foil benchmark records no distances; compute them."""
    store = np.load(npz, allow_pickle=True)
    blob = json.load(open(benchmark_path))
    for t in blob["trials"]:
        ids = list(store[f"{t['mode']}__ids"])
        D = store[f"{t['mode']}__D"]
        idx = {s: i for i, s in enumerate(ids)}
        tgt = next(o["scene_id"] for o in t["options"] if o["is_target"])
        t["min_foil_distance_m"] = round(float(min(
            D[idx[tgt], idx[o["scene_id"]]] for o in t["options"] if not o["is_target"])), 2)
    return blob


def series_points(rows, mode, selectors):
    ks, ns = [], []
    for sel in selectors:
        r = [x for x in rows if x["mode"] == mode and sel(x)]
        ks.append(sum(x["is_correct"] for x in r))
        ns.append(len(r))
    return ks, ns


def plot_row(axes, rowdata, labels, series, ylabel, xlabel):
    for ax, mode in zip(axes, MODES):
        for name, rows, col, mk in series:
            ks, ns = series_points(rows, mode, rowdata)
            xs = np.arange(len(labels))
            ys = [k / n if n else np.nan for k, n in zip(ks, ns)]
            lo = [wilson(k, n)[0] for k, n in zip(ks, ns)]
            hi = [wilson(k, n)[1] for k, n in zip(ks, ns)]
            ax.plot(xs, ys, marker=mk, color=col, lw=1.7, ms=5, label=name)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.11, lw=0)
            for xi, k, n in zip(xs, ks, ns):
                if n:
                    ax.annotate(f"{k}/{n}", (xi, k / n), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=5.6, color=col)
        ax.axhline(CHANCE, color="#555", ls="--", lw=0.9)
        ax.set_xticks(range(len(labels)), labels, fontsize=7.5)
        ax.set_ylim(-0.05, 1.12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=9)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[2].set_xlabel(xlabel, fontsize=9)


def main(out="figures/fig_mode_detail.png"):
    rank = ranker()
    randb = min_foil_for_random("data/vlm_benchmark_4afc.json")
    tmp = "/tmp/_rand_with_dist.json"
    json.dump(randb, open(tmp, "w"))

    rand = load("results/or_google_gemini-3.8-flash_cot_anyview_randomfoils_n100.json", tmp)
    hard = load("results/or_google_gemini-3.8-flash_cot_anyview_hardfoils_n100.json",
                "data/vlm_benchmark_4afc_hard.json")
    hum = load("human_task/hard/human_p01_4afc.json",
               "data/vlm_benchmark_4afc_hard.json")
    for rows in (rand, hard, hum):
        for x in rows:
            x["pct"] = rank(x["mode"], x["min_foil"])

    series = [("Gemini, random foils", rand, "#e0a44f", "^"),
              ("Gemini, hard foils", hard, "#b8620a", "s"),
              ("Human p01, hard foils", hum, "#111111", "o")]

    fig, axes = plt.subplots(2, len(MODES), figsize=(15.5, 6.6), sharey="row")
    for ax, mode in zip(axes[0], MODES):
        ax.set_title(TITLE[mode], fontsize=9.5)

    plot_row(axes[0], [f for _, f in [(l, (lambda d: lambda x: d(x["delta"]))(f))
                                      for l, f in DBINS]],
             [l for l, _ in DBINS], series,
             "accuracy", "viewpoint change $\\Delta$")

    plot_row(axes[1],
             [(lambda lo, hi: lambda x: lo <= x["pct"] < hi)(lo, hi) for _, lo, hi in PBINS],
             [l for l, _, _ in PBINS], series,
             "accuracy", "nearest foil (percentile within mode)")

    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Every mode against both axes — 4AFC, chance 25% (counts are correct/n)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    fig.savefig(out.replace(".png", ".pdf"))
    print("wrote", out)
    os.remove(tmp)


if __name__ == "__main__":
    main()
