"""
Figure: the bank. Scenes down, modes across.

    python bench/figure_bank.py --root data/scenes --out figures/fig3_bank.png

Each row is one place; each column is the same place with a different set of
identity cues. Reading across a row checks the claim the whole design rests on
and that no single image can show: these are the same landmarks in the same
positions, and only what tells them apart has changed.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

MODES = [("c0_shape_colour", "c0\nshape + colour"),
         ("c1_shape", "c1\nshape only"),
         ("c2_colour", "c2\ncolour only"),
         ("c3_peaks_bare", "c3\nlandforms, bare"),
         ("c4_valley", "c4\nfull valley")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/scenes")
    ap.add_argument("--out", default="figures/fig3_bank.png")
    ap.add_argument("--scenes", nargs="+",
                    default=["a000", "a001", "a002", "a003", "a004"])
    ap.add_argument("--azimuth", type=int, default=0)
    args = ap.parse_args()

    nr, nc = len(args.scenes), len(MODES)
    fig, axes = plt.subplots(nr, nc, figsize=(2.45 * nc, 1.85 * nr))
    missing = 0
    for r, sid in enumerate(args.scenes):
        for c, (mode, label) in enumerate(MODES):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            p = os.path.join(args.root, mode, sid,
                             f"{sid}_A_az{args.azimuth:03d}.png")
            if os.path.exists(p):
                ax.imshow(mpimg.imread(p))
            else:
                missing += 1
                ax.text(.5, .5, "not yet\nrendered", ha="center", va="center",
                        fontsize=7.5, color="#b23a2e", transform=ax.transAxes)
            if r == 0:
                ax.set_title(label, fontsize=10)
            if c == 0:
                ax.set_ylabel(sid, fontsize=10)
    fig.suptitle(f"The bank at {args.azimuth}°: one set of places, rendered five ways",
                 fontsize=13, y=0.995)
    fig.text(0.5, 0.005,
             "Across a row: same landmarks, same positions, same camera. "
             "Only the identity cue and the world change.",
             ha="center", fontsize=9.5, color="0.3")
    fig.tight_layout(rect=[0, 0.018, 1, 0.975])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=135, bbox_inches="tight")
    print(f"[fig-bank] -> {args.out}" + (f"   MISSING {missing}" if missing else ""))


if __name__ == "__main__":
    main()
