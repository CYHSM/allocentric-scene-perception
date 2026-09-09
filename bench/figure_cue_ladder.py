"""
Figure 3: which cue is the residual signal living on?

    python bench/figure_cue_ladder.py --root data/scenes_100 --out figures/fig3_cue_ladder.png

The five modes render the *same 100 layouts*, so cue is a within-scene factor and
every comparison across this axis is paired. c0 gives each landmark a distinct
shape and colour; c1 removes colour, c2 removes shape; c3 replaces the discrete
objects with parametric landforms; c4 puts those same landforms in a full valley
with terrain, water and vegetation.

Three read-outs are drawn rather than one, because they disagree and the
disagreement is the point:

  * **Recall@1 at delta=0** -- the appearance gate. Falls from ~45% on c0 to
    ~7-12% on c2: a model that cannot tell the landmarks apart cannot index the
    place even without moving.
  * **RSA at delta=45** -- how much of the arrangement survives the turn in the
    representation's geometry, independent of whether retrieval succeeds. This
    is the only read-out still off the floor, and it falls monotonically c0 -> c4.
  * **Recall@1 at delta=45** against the 1% chance line.

RSA falling from c0 to c4 while retrieval at delta=45 does not is not a
contradiction: retrieval at 45 degrees is at chance in *every* mode, so it has no
resolution left to rank them with. RSA is what carries the cue effect, and that
is worth saying in the caption rather than leaving a reader to reconcile two
panels.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = [("c0_shape_colour", "c0\nshape+colour"),
         ("c1_shape", "c1\nshape"),
         ("c2_colour", "c2\ncolour"),
         ("c3_peaks_bare", "c3\nlandforms"),
         ("c4_valley", "c4\nvalley")]

SHORT = {"vit_base_patch14_dinov2.lvd142m": "DINOv2-B/14",
         "vit_so400m_patch14_siglip_384.webli": "SigLIP-so400m",
         "vit_base_patch16_clip_224.openai": "CLIP-B/16",
         "resnet50.a1_in1k": "ResNet-50"}
COLOURS = {"DINOv2-B/14": "#1b4965", "SigLIP-so400m": "#2f6690",
           "CLIP-B/16": "#e07a5f", "ResNet-50": "#8d99ae"}


def load(root, appearance="changed"):
    """{model: {mode: by_delta}}, keeping the record for the real condition."""
    out = {}
    for mode, _ in MODES:
        path = os.path.join(root, mode, "metrics.json")
        if not os.path.exists(path):
            print(f"[fig] {path} missing -- {mode} left out")
            continue
        for rec in json.load(open(path)):
            if rec["appearance"] != appearance:
                continue
            name = SHORT.get(rec["model"], rec["model"])
            out.setdefault(name, {})[mode] = {
                int(k): v for k, v in rec["by_delta"].items()}
    return out


def _series(data, model, key, delta):
    ys = []
    for mode, _ in MODES:
        rec = data.get(model, {}).get(mode)
        ys.append(rec[delta][key] if rec else np.nan)
    return ys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/scenes_100")
    ap.add_argument("--out", default="figures/fig3_cue_ladder.png")
    args = ap.parse_args()

    data = load(args.root)
    if not data:
        raise SystemExit(f"no metrics under {args.root}")
    models = [m for m in COLOURS if m in data]
    x = np.arange(len(MODES))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharex=True)

    panels = [
        ("Recall@1 at $\\Delta=0^\\circ$\nthe appearance gate", "recall@1", 0, True),
        ("Recall@1 at $\\Delta=45^\\circ$\nafter one turn", "recall@1", 45, True),
        ("RSA at $\\Delta=45^\\circ$\nrepresentational geometry", "rsa", 45, False),
    ]

    for ax, (title, key, delta, is_recall) in zip(axes, panels):
        for m in models:
            ys = np.array(_series(data, m, key, delta), dtype=float)
            if is_recall:
                ys = ys * 100.0
            ax.plot(x, ys, "-o", ms=5, lw=1.6, color=COLOURS[m], label=m)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in MODES], fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        if is_recall:
            ax.set_ylabel("Recall@1  (%)")
            # Chance is 1% with a 100-scene gallery. Drawn, not implied: the
            # whole claim in the middle panel is that the curves are *on* it.
            ax.axhline(1.0, color="#c0392b", lw=0.9, ls=":")
            ax.text(x[-1], 1.0, " chance", color="#c0392b", fontsize=7,
                    va="bottom", ha="right")
        else:
            ax.set_ylabel("RSA  (Spearman $\\rho$)")
            ax.axhline(0.0, color="#444", lw=0.9)

    axes[1].set_ylim(0, 5.5)      # the panel is a floor; do not share the c0 scale
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.supxlabel("what tells the landmarks apart", fontsize=9, y=0.005)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}\n")

    for title, key, delta, _ in panels:
        head = title.split("\n")[0]
        print(head)
        for m in models:
            ys = _series(data, m, key, delta)
            print(f"  {m:15s} " + "  ".join(
                f"{lbl.split(chr(10))[0]} {v:.3f}" for (_, lbl), v in zip(MODES, ys)))


if __name__ == "__main__":
    main()
