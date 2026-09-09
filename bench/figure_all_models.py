"""
Figure 1b: every model on one axis.

    python bench/figure_all_models.py --rows figures/vii_rows.json --out figures/fig1_all_models.png

Raw accuracy cannot carry this figure. The frozen encoders are scored by
retrieval against a 100-scene gallery (chance 1%), the VLMs by 4AFC (chance
25%); plotted together, the arms would separate by task format and not by model.
So the y axis is VII -- each cell converted to d' under the m-AFC model and
divided by that model's own delta=0 appearance gate. See bench/vii.py.

Two lines are drawn that are usually left implicit and matter more than any
curve here:

  * VII = 0 is chance. Not "near the bottom of the axis" -- exactly chance,
    because d' is defined to be 0 there for every m.
  * VII = 1 is a representation the walk does not disturb. The gap between the
    curves and that line is the finding.

The delta=0 panel is separate and is *not* a VII: it is the appearance gate
itself, the denominator, and the only place where scaling does anything.
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DELTAS = [45, 90, 135, 180]

ENCODERS = ["DINOv2-B/14", "SigLIP-so400m", "CLIP-B/16", "ResNet-50"]
ENC_COLOURS = dict(zip(ENCODERS, ["#1b4965", "#2f6690", "#5a8fb8", "#9dbfd6"]))
VLM_COLOURS = ["#f2b880", "#e07a5f", "#9e2a2b"]


def _mean_curve(rows, model):
    """VII averaged over the five cue modes, and its spread."""
    per = [r["vii"] for r in rows if r["model"] == model]
    out = {}
    for d in DELTAS:
        vals = [p[str(d)] if str(d) in p else p.get(d) for p in per]
        vals = [v for v in vals if v is not None]
        out[d] = (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="figures/vii_rows.json",
                    help="output of bench/analyze_vii.py --json")
    ap.add_argument("--out", default="figures/fig1_all_models.png")
    args = ap.parse_args()

    blob = json.load(open(args.rows))
    enc, afc = blob["encoders"], blob["afc"]

    # Only complete forced-choice runs belong on a published axis. The 32B row
    # in the earlier table averaged three modes where the others averaged five,
    # because the run was still going when the figure was made.
    # The scaling series only: every point must differ in scale and nothing
    # else. The prompt variants (mental_rotation, anchor) are the same 7B model
    # on the same trials and belong in the ablation, not on this axis -- putting
    # them here would read as four model sizes when it is two.
    vlms = [r for r in afc if r["arm"] == "4AFC" and r["n_total"] >= 400
            and re.fullmatch(r"Qwen2\.5-VL-\d+B", r["model"])]

    def _params(name):
        m = re.search(r"(\d+(?:\.\d+)?)\s*B", name)
        return float(m.group(1)) if m else 0.0

    vlms.sort(key=lambda r: _params(r["model"]))       # 3B, 7B, 32B, not file size

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.55]})

    # --- (i) the appearance gate: what scaling actually buys ----------------
    ax = axes[0]
    enc_gate = {m: np.mean([r["accuracy"]["0"] if "0" in r["accuracy"]
                            else r["accuracy"][0]
                            for r in enc if r["model"] == m]) for m in ENCODERS}
    labels, vals, cols, spread = [], [], [], []
    for m in ENCODERS:
        labels.append(m)
        vals.append(enc_gate[m] * 100)
        cols.append(ENC_COLOURS[m])
        # The encoder bar is a mean over five cue modes that range from 7% to
        # 45%; the bar alone would hide that, so every mode is drawn on it.
        spread.append([(r["accuracy"]["0"] if "0" in r["accuracy"]
                        else r["accuracy"][0]) * 100
                       for r in enc if r["model"] == m])
    for i, r in enumerate(vlms):
        labels.append(r["model"].replace("Qwen2.5-VL-", "Qwen2.5-VL\n"))
        a0 = r["accuracy"]["0"] if "0" in r["accuracy"] else r["accuracy"][0]
        vals.append(a0 * 100)
        cols.append(VLM_COLOURS[i % len(VLM_COLOURS)])
        spread.append([])
    ax.bar(range(len(vals)), vals, color=cols, width=0.72)
    for x, pts in enumerate(spread):
        if pts:
            ax.scatter([x] * len(pts), pts, s=9, color="#22303c", zorder=3,
                       linewidths=0)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("accuracy at $\\Delta=0^\\circ$  (%)")
    ax.set_title("appearance gate\nsame place, same bearing", fontsize=9.5)
    # The two arms have different chance levels, so both are drawn.
    ax.axhline(1.0, color="#888", lw=0.9, ls=":")
    ax.text(-0.42, 2.2, "chance, retrieval (1%)", fontsize=6.5, color="#666",
            ha="left")
    ax.axhline(25.0, color="#c0392b", lw=0.9, ls=":")
    ax.text(-0.42, 26.2, "chance, 4AFC (25%)", fontsize=6.5, color="#c0392b",
            ha="left")
    ax.set_ylim(0, 78)
    ax.spines[["top", "right"]].set_visible(False)

    # --- (ii) VII: what it does not buy -------------------------------------
    ax = axes[1]
    ax.axhspan(-0.02, 0.0, color="#d9d9d9", alpha=0.7, zorder=0)
    ax.axhline(0.0, color="#444", lw=1.0)
    ax.axhline(1.0, color="#2a9d8f", lw=1.0, ls="--")
    ax.text(37, 1.02, "perfect invariance", va="bottom", fontsize=7.5,
            color="#2a9d8f")
    ax.text(181, 0.0, " chance", va="center", fontsize=7.5, color="#444")

    for m in ENCODERS:
        c = _mean_curve(enc, m)
        ys = [c[d][0] for d in DELTAS]
        ax.plot(DELTAS, ys, "-o", ms=4, lw=1.4, color=ENC_COLOURS[m], label=m)

    for i, r in enumerate(vlms):
        v = r["vii"]
        ys = [v.get(str(d), v.get(d)) for d in DELTAS]
        ys = [np.nan if y is None else y for y in ys]
        ax.plot(DELTAS, ys, "-s", ms=5, lw=1.8, color=VLM_COLOURS[i % 3],
                label=r["model"])

    ax.set_xticks(DELTAS)
    ax.set_xlabel("viewpoint change  $\\Delta$  (degrees)")
    ax.set_ylabel("VII  =  $d'(\\Delta)\\,/\\,d'(0)$")
    ax.set_title("viewpoint invariance, normalised by each model's own gate",
                 fontsize=9.5)
    ax.set_xlim(35, 190)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.92))
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")

    # State the numbers the figure is drawn from, so a caption can be written
    # without re-reading the JSON.
    print("\nencoder VII, averaged over the five cue modes:")
    for m in ENCODERS:
        c = _mean_curve(enc, m)
        print(f"  {m:15s} " + "  ".join(f"{d}deg {c[d][0]:.3f}" for d in DELTAS))
    print("forced choice:")
    for r in vlms:
        v = r["vii"]
        a0 = r["accuracy"]["0"] if "0" in r["accuracy"] else r["accuracy"][0]
        print(f"  {r['model']:15s} gate {a0*100:.1f}%  " + "  ".join(
            f"{d}deg {(v.get(str(d)) if v.get(str(d)) is not None else float('nan')):.3f}"
            for d in DELTAS))


if __name__ == "__main__":
    main()
