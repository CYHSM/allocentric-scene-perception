"""
Figure 1: what the dataset is.

    python bench/figure_dataset.py --root data/scenes --out figures/fig1_dataset.png

Two rows, and each makes one of the design's two claims checkable by eye:

* **top** -- one scene in all five modes. Same place, same camera, same
  landmarks; only the cue that tells them apart changes. If these are not
  obviously the same layout, the pairing is broken.
* **bottom** -- the probe ladder. The same scene with one landmark moved
  1..24 m, each panel labelled with the fraction of pixels that actually
  changed. Whether the smallest rungs are *visible* and whether they are
  *rendered* are different questions, and only the second one decides what a
  flat D_scene(1 m) would mean: if the image barely changed, the floor is the
  stimulus; if it changed and the model did not notice, the floor is the model.
  Measuring it here is what keeps that inference out of the results section.

The bottom row is also where a real bug was caught: the identity draw had been
seeded on the scene id rather than the anchor's, so every probe showed different
objects instead of the same scene displaced. Nothing else would have shown it --
each probe is a perfectly plausible scene on its own.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

MODES = [("c0_shape_colour", "c0   shape + colour"),
         ("c1_shape", "c1   shape only"),
         ("c2_colour", "c2   colour only"),
         ("c3_peaks_bare", "c3   landforms, bare"),
         ("c4_valley", "c4   full valley")]


def frame(root, mode, scene, app="A", az=0):
    p = os.path.join(root, mode, scene, f"{scene}_{app}_az{az:03d}.png")
    return p if os.path.exists(p) else None


def changed_fraction(ref_png, png, tol=2):
    """Fraction of pixels differing from the anchor by more than `tol` levels.

    `tol` is above PNG quantisation but far below anything visible, so this
    measures whether the renderer resolved the displacement at all -- not
    whether a person would spot it.
    """
    a = mpimg.imread(ref_png)[..., :3]
    b = mpimg.imread(png)[..., :3]
    if a.shape != b.shape:
        return None
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)).max(axis=2) * 255.0
    return float((d > tol).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/scenes")
    ap.add_argument("--out", default="figures/fig1_dataset.png")
    ap.add_argument("--scene", default="a000")
    ap.add_argument("--azimuth", type=int, default=0)
    ap.add_argument("--ladder", type=float, nargs="+",
                    default=[1, 2, 4, 8, 16, 24])
    args = ap.parse_args()

    anchor_png = frame(args.root, MODES[0][0], args.scene, az=args.azimuth)
    ladder_report = []

    ncol = max(len(MODES), len(args.ladder))
    fig = plt.figure(figsize=(2.7 * ncol, 5.6))
    gs = fig.add_gridspec(2, ncol, hspace=0.28, wspace=0.10)
    missing = []

    for i, (mode, label) in enumerate(MODES):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xticks([]); ax.set_yticks([])
        p = frame(args.root, mode, args.scene, az=args.azimuth)
        if p:
            ax.imshow(mpimg.imread(p))
        else:
            missing.append(f"{mode}/{args.scene}")
            ax.text(.5, .5, "missing", ha="center", va="center",
                    color="#b23a2e", fontsize=9, transform=ax.transAxes)
        ax.set_title(label, fontsize=10.5)
    for i in range(len(MODES), ncol):
        fig.add_subplot(gs[0, i]).axis("off")

    for j, d in enumerate(args.ladder):
        ax = fig.add_subplot(gs[1, j])
        ax.set_xticks([]); ax.set_yticks([])
        sid = f"{args.scene}p{d:g}"
        p = frame(args.root, MODES[0][0], sid, az=args.azimuth)
        if p:
            ax.imshow(mpimg.imread(p))
            frac = None if anchor_png is None else changed_fraction(anchor_png, p)
            ax.set_title(f"$d$ = {d:g} m", fontsize=10.5)
            if frac is not None:
                ax.set_xlabel(f"{100 * frac:.1f}% of pixels changed", fontsize=8.5)
                ladder_report.append((d, frac))
        else:
            missing.append(f"{MODES[0][0]}/{sid}")
            ax.text(.5, .5, "missing", ha="center", va="center",
                    color="#b23a2e", fontsize=9, transform=ax.transAxes)
            ax.set_title(f"$d$ = {d:g} m", fontsize=10.5)
    for j in range(len(args.ladder), ncol):
        fig.add_subplot(gs[1, j]).axis("off")

    fig.text(0.5, 0.965,
             f"Scene {args.scene} at {args.azimuth}°.  "
             "Top: the same place in all five modes.  "
             "Bottom: one landmark moved $d$ metres.",
             ha="center", fontsize=11.5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=125, bbox_inches="tight")
    print(f"[fig1] -> {args.out}")
    if ladder_report:
        print("[fig1] the ladder against the renderer's floor:")
        for d, frac in ladder_report:
            print(f"         d = {d:>4g} m   {100 * frac:5.2f}% of pixels changed")
        lo = min(f for _, f in ladder_report)
        print(f"       smallest rung moves {100 * lo:.2f}% of the frame -- "
              + ("above the floor, so a flat D_scene there is the model's doing"
                 if lo > 0.005 else
                 "AT THE FLOOR: the stimulus, not the model, may be the limit"))
    if missing:
        # A blank panel that says nothing is how a half-rendered bank gets
        # reported as a finished one.
        print(f"[fig1] MISSING {len(missing)}: " + ", ".join(missing[:8]))


if __name__ == "__main__":
    main()
