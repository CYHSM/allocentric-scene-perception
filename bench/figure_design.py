"""
Figure 2: the experiment in one picture.

    python bench/figure_design.py --root data/scenes --out figures/fig2_design.png

Two rows, sharing a scene and an axis each:

* **top** -- the same place, walking around it. Nothing about the world changed.
* **bottom** -- the same viewpoint, the place changed by d metres.

Every image in the figure differs from the top-left one. The experiment is the
question of *which kind* of difference a model finds larger, and lambda is the
d at which the two rows cost it the same.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def path(root, mode, scene, app, az):
    p = os.path.join(root, mode, scene, f"{scene}_{app}_az{az:03d}.png")
    return p if os.path.exists(p) else None


def panel(ax, p, title, sub=None, edge=None):
    ax.set_xticks([]); ax.set_yticks([])
    if p:
        ax.imshow(mpimg.imread(p))
    else:
        ax.text(.5, .5, "not yet rendered", ha="center", va="center",
                fontsize=8, color="#b23a2e", transform=ax.transAxes)
    ax.set_title(title, fontsize=10)
    if sub:
        ax.set_xlabel(sub, fontsize=8.5)
    if edge:
        for s in ax.spines.values():
            s.set_color(edge); s.set_linewidth(2.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/scenes")
    ap.add_argument("--out", default="figures/fig2_design.png")
    ap.add_argument("--mode", default="c0_shape_colour")
    ap.add_argument("--scene", default="a000")
    ap.add_argument("--deltas", type=int, nargs="+", default=[0, 45, 90, 135, 180])
    ap.add_argument("--ladder", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()

    n = max(len(args.deltas), len(args.ladder))
    fig = plt.figure(figsize=(2.55 * n, 6.4))
    gs = fig.add_gridspec(2, n, hspace=0.42, wspace=0.08,
                          top=0.80, bottom=0.09)

    for i, dz in enumerate(args.deltas):
        ax = fig.add_subplot(gs[0, i])
        panel(ax, path(args.root, args.mode, args.scene, "A", dz % 360),
              f"$\\Delta$ = {dz}°",
              "the reference" if dz == 0 else "same place, I moved",
              edge="#2e6da4" if dz == 0 else None)

    for j, d in enumerate(args.ladder):
        ax = fig.add_subplot(gs[1, j])
        sid = f"{args.scene}p{d:g}"
        panel(ax, path(args.root, args.mode, sid, "A", args.deltas[0] % 360),
              f"$d$ = {d:g} m", "same viewpoint, place changed")

    fig.text(0.5, 0.955, "What the benchmark asks",
             ha="center", fontsize=14)
    fig.text(0.5, 0.905,
             "Two images of a place differ for exactly two reasons. "
             "Both rows depart from the blue reference.",
             ha="center", fontsize=10, color="0.3")
    fig.text(0.5, 0.855,
             "TOP: you moved.     BOTTOM: the place changed.     "
             "$\\lambda$ = the $d$ at which a model finds them equally large.",
             ha="center", fontsize=10.5)
    fig.text(0.012, 0.60, "VIEWPOINT", rotation=90, va="center",
             fontsize=10, color="#2e6da4", weight="bold")
    fig.text(0.012, 0.24, "SCENE", rotation=90, va="center",
             fontsize=10, color="#b23a2e", weight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=135, bbox_inches="tight")
    print(f"[fig-design] -> {args.out}")


if __name__ == "__main__":
    main()
