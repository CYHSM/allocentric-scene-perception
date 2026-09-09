"""
The main result: can a model find the place again from a new viewpoint?

    python bench/figure_retrieval.py --root data/scenes_100 --out figures/fig_retrieval.png

Every scene in this gallery holds the *same* landmarks, so no model can identify
a place by noticing which objects are in it. What is left is the arrangement.
Recall@1 against turn angle therefore reads as: how much of the model's ability
to recognise a place survives moving around it.

The chance line is drawn, not implied. With a 100-scene gallery chance is 1%,
and the difference between "5%" and "at chance" is the whole result -- a curve
that ends near the floor says something categorical, and a reader should not
have to divide by the gallery size to see it.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = [("c0_shape_colour", "c0  shape + colour"),
         ("c1_shape", "c1  shape only"),
         ("c2_colour", "c2  colour only"),
         ("c3_peaks_bare", "c3  landforms"),
         ("c4_valley", "c4  full valley")]

SHORT = {"vit_base_patch14_dinov2.lvd142m": "DINOv2-B/14",
         "vit_so400m_patch14_siglip_384.webli": "SigLIP-so400m",
         "vit_base_patch16_clip_224.openai": "CLIP-B/16",
         "resnet50.a1_in1k": "ResNet-50"}


def load(root):
    out = {}
    for mode, _ in MODES:
        p = os.path.join(root, mode, "metrics.json")
        if os.path.exists(p):
            try:
                out[mode] = json.load(open(p))
            except ValueError:
                print(f"[fig] {p} unreadable -- skipped")
    return out


def series(rows, model, key, appearance="changed"):
    for r in rows:
        if r.get("model") == model and r.get("appearance") == appearance:
            bd = r["by_delta"]
            xs = sorted(float(k) for k in bd)
            return xs, [bd[str(int(x)) if str(int(x)) in bd else str(x)][key]
                        for x in xs], r["n_scenes"]
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/scenes_100")
    ap.add_argument("--out", default="figures/fig_retrieval.png")
    ap.add_argument("--metric", default="recall@1")
    args = ap.parse_args()

    mets = load(args.root)
    if not mets:
        raise SystemExit(f"no metrics.json under {args.root}/*/")
    models = []
    for rows in mets.values():
        for r in rows:
            if r["model"] not in models:
                models.append(r["model"])

    have = [(m, lab) for m, lab in MODES if m in mets]
    fig, axes = plt.subplots(1, len(have), figsize=(2.75 * len(have), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    chance = None
    for ax, (mode, label) in zip(axes, have):
        for k, model in enumerate(models):
            xs, ys, n = series(mets[mode], model, args.metric)
            if xs is None:
                continue
            if args.metric == "rsa":
                # RSA at delta 0 is a matrix against itself: exactly 1.0 by
                # definition. Plotting it compresses every informative point
                # into the bottom of the axis to make room for a constant.
                xs, ys = zip(*[(x, y) for x, y in zip(xs, ys) if x > 0])
            chance = 100.0 / n
            scale = 1.0 if args.metric == "rsa" else 100.0
            ax.plot(xs, [scale * v for v in ys], "o-", ms=4, lw=1.5,
                    color=f"C{k}", label=SHORT.get(model, model))
        if args.metric == "rsa":
            # RSA is a correlation: its null is 0, and it is not a percentage,
            # so neither the log axis nor the 1/N chance line applies.
            ax.axhline(0.0, color="0.35", lw=1.1, ls=(0, (4, 3)), zorder=1)
            ax.set_xlim(32, 193)
        else:
            if chance:
                ax.axhline(chance, color="0.35", lw=1.1, ls=(0, (4, 3)), zorder=1)
                ax.text(182, chance, " chance", va="center", fontsize=7.5,
                        color="0.35")
            ax.set_yscale("log")
            ax.set_ylim(0.05, 130)
            ax.set_yticks([0.1, 1, 10, 100])
            ax.set_yticklabels(["0.1", "1", "10", "100"])
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("turn  $\\Delta$  [deg]", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    if args.metric == "rsa":
        # One shared limit, taken from the data. A hard-coded ceiling silently
        # clipped c2's 180 deg point, which is the largest value in the figure.
        lo = min(min(l.get_ydata()) for ax in axes for l in ax.get_lines()
                 if len(l.get_ydata()) > 1)
        hi = max(max(l.get_ydata()) for ax in axes for l in ax.get_lines()
                 if len(l.get_ydata()) > 1)
        pad = 0.08 * (hi - lo)
        for ax in axes:
            ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("Spearman RSA" if args.metric == "rsa"
                       else "Recall@1  [%]", fontsize=9.5)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    miss = [m for m, _ in MODES if m not in mets]
    title = ("Does the scene-by-scene geometry survive the turn?"
             if args.metric == "rsa" else
             "Finding the place again from a new viewpoint, "
             "in a gallery where every scene holds the same landmarks")
    fig.suptitle(title, fontsize=11.5, y=1.04)
    if miss:
        print(f"[fig] MISSING metrics for: {', '.join(miss)}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"[fig-retrieval] -> {args.out}")


if __name__ == "__main__":
    main()
