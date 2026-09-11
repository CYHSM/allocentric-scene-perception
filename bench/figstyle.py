"""
One look for every figure in the paper.

Colour carries family, not identity: all Qwen models share a hue and separate by
lightness, so a scaling series reads as a series. Chance is drawn the same way
everywhere, because on a 4AFC task with an 80-trial rotated cell the distance
from chance is the entire result and a reader should never have to hunt for the
line.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_spec as SPEC

# Figures are placed two-up at ~0.48\textwidth, which is about 3.4 inches. A
# 5-inch figure is then scaled to 0.65 and a 7pt label renders at 4.5pt. So the
# paper figures are authored near their printed size and the font sizes below
# are the sizes that actually appear on the page.
TWO_UP = (3.5, 3.1)      # side by side in a two-column float
FULL = (5.6, 3.4)        # one figure across the text block

RC = {
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.8, "lines.markersize": 5,
}


# One palette for everything that is not a model. Family hues come from
# SPEC.FAMILY_COLOUR; these are the greys and the two accents that carry
# meaning on their own, so that the same grey never means two things in two
# figures.
GREY = "#b0b0b0"      # a mark that is deliberately not the point
INK = "#333333"       # rules, edges and error bars
MUTED = "#888888"     # footnotes and counts
CHANCE = "#444444"    # the chance reference, wherever it is drawn
OK = "#1a7f4f"        # the correct option, in the stimulus figure only


def use():
    plt.rcParams.update(RC)


def colour(family, params_b=None, family_sizes=None):
    """
    Family hue, lightened by rank within the family's size series.

    A scaling series plotted in five unrelated colours looks like five unrelated
    models; the point of that panel is that they are one model at five sizes.
    """
    base = SPEC.FAMILY_COLOUR.get(family, "#777777")
    if not family_sizes or params_b is None or len(family_sizes) < 2:
        return base
    order = sorted(family_sizes)
    frac = order.index(params_b) / (len(order) - 1)
    rgb = tuple(int(base[i:i + 2], 16) / 255 for i in (1, 3, 5))
    light = 0.62 - 0.62 * frac           # 0 = base colour, >0 = toward white
    return tuple(c + (1 - c) * light for c in rgb)


def chance_line(ax, level=None, label=True):
    level = SPEC.ARM["chance"] if level is None else level
    ax.axhline(level, color=CHANCE, ls=(0, (4, 3)), lw=1.0, zorder=0)
    if label:
        ax.annotate("chance", xy=(0.995, level), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7, color=CHANCE)


def save(fig, stem, outdir="figures"):
    import os
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/{stem}.{ext}")
    plt.close(fig)
    return f"{outdir}/{stem}.png"
