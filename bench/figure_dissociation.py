"""
The dissociation figure set: a gate that scales, an invariance that does not.

Everything here reads `agents.collect()` so a new run appears by being written
to `results/`. Four figures, each answering one question a reviewer will ask:

  fig_dissociation  does scale buy invariance, or only the appearance gate?
  fig_vii_curves    how does invariance fall off with the size of the turn?
  fig_cue_ladder    which visual cues carry what is left?
  fig_retinotopic   are the failures random, or systematically below chance?

Why d' and not accuracy or kappa: the arms have different chance levels (2AFC,
4AFC), so accuracy cannot be compared across them, and kappa -- while it does
correct for chance -- still collapses "can it tell scenes apart" and "does that
survive rotation" into a single scalar. The claim of this paper is a
*dissociation* between those two, and a single scalar cannot show one.
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
import vii as V

DELTAS = [0, 45, 90, 135, 180]
OUT = "figures"

FAMILY_COLOUR = {"qwen": "#3b6ea5", "internvl": "#2e8b74", "gemini": "#c8791a",
                 "openai": "#7a4fa3", "anthropic": "#b5533c", "human": "#111111",
                 "other": "#777777"}


def _rows(min_trials=40):
    """Primary forced-choice runs, ordered small model to large, human last."""
    recs = [r for r in agents.afc_records(min_trials=min_trials)
            if r["role"] == "primary"]
    out = []
    for r in recs:
        acc = dict(r["accuracy"])
        ns = dict(r["n"])
        if 0 not in acc:
            continue
        m = r["m"]
        d0 = V.dprime(acc[0], m, ns[0])
        r = dict(r)
        r["acc"] = acc
        r["ns"] = ns
        r["d0"] = d0
        r["dprime"] = {d: V.dprime(acc[d], m, ns[d]) for d in acc}
        r["viic"] = {d: (r["dprime"][d] / d0 if d0 > 0 and np.isfinite(d0) else None)
                     for d in acc}
        out.append(r)
    human = [r for r in out if r["family"] == "human"]
    model = [r for r in out if r["family"] != "human"]
    model.sort(key=lambda r: (r["params_b"] or 1e9))
    return model + human


def _label(r):
    tag = r["label"]
    if not r["complete"]:
        tag += f"\n(partial, n={sum(r['ns'].values())})"
    return tag


def _colour(r):
    return FAMILY_COLOUR.get(r["family"], FAMILY_COLOUR["other"])


def fig_dissociation(rows, path):
    """The gate on one axis, what survives the turn on the other."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = [_label(r) for r in rows]
    x = np.arange(len(rows))
    cols = [_colour(r) for r in rows]

    gate = [r["d0"] if np.isfinite(r["d0"]) else np.nan for r in rows]
    axes[0].bar(x, gate, color=cols)
    for xi, g in zip(x, gate):
        if not np.isfinite(g) or g <= 0:
            axes[0].annotate("at chance\nd'(0) = 0", (xi, 0.05), ha="center",
                             fontsize=6.5, color="#a33")
    axes[0].set_ylabel("d' at $\\Delta=0$")
    axes[0].set_title("A. The appearance gate\n(same viewpoint, resampled appearance)",
                      fontsize=10)

    # Invariance is what is left once the viewpoint actually moves, so average
    # the turns that require it and leave the 0 cell out of it.
    inv = []
    for r in rows:
        vals = [r["viic"][d] for d in (90, 135, 180)
                if d in r["viic"] and r["viic"][d] is not None]
        inv.append(np.mean(vals) if vals else np.nan)
    axes[1].bar(x, inv, color=cols)
    # A model at chance at delta=0 has no gate to normalise by, so VII does not
    # exist for it. That is a limitation of the index, not a score of zero, and
    # it has to be legible on the figure -- an absent bar reads as "failed".
    for xi, v in zip(x, inv):
        if not np.isfinite(v):
            axes[1].annotate("VII undefined\n(no gate to\nnormalise by)",
                             (xi, 0.05), ha="center", fontsize=6.5, color="#a33")
    axes[1].axhline(0, color="#333", lw=0.8)
    axes[1].set_ylabel("mean VII over $\\Delta \\geq 90^\\circ$")
    axes[1].set_title("B. What survives the turn\n(VII = d'($\\Delta$) / d'(0))",
                      fontsize=10)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Scale buys the gate, not the map", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_vii_curves(rows, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for r in rows:
        xs = [d for d in DELTAS if d in r["viic"] and r["viic"][d] is not None]
        ys = [r["viic"][d] for d in xs]
        if not xs:
            continue
        ax.plot(xs, ys, "-o", ms=4, color=_colour(r), label=_label(r).replace("\n", " "),
                lw=2.2 if r["family"] == "human" else 1.4,
                ls="-" if r["complete"] else "--")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(DELTAS)
    ax.set_xlabel("viewpoint change $\\Delta$ (degrees)")
    ax.set_ylabel("VII = d'($\\Delta$) / d'(0)")
    ax.set_title("Viewpoint invariance falls to the floor by 90$^\\circ$", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_cue_ladder(path, min_trials=40):
    """Accuracy above chance, per stimulus mode and per turn, per model."""
    recs = [r for r in agents.afc_records(min_trials=min_trials, per_mode=True)
            if r["role"] == "primary"]
    by_model = {}
    for r in recs:
        by_model.setdefault(r["label"], {})[r["mode"]] = r
    names = [n for n in by_model if "gemini" not in n.lower()]
    names.sort(key=lambda n: (list(by_model[n].values())[0]["params_b"] or 1e9))
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(2.5 * len(names), 3.4),
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        grid = np.full((len(agents.MODES), len(DELTAS)), np.nan)
        for i, mode in enumerate(agents.MODES):
            rec = by_model[name].get(mode)
            if not rec:
                continue
            acc = dict(rec["accuracy"])
            for j, d in enumerate(DELTAS):
                if d in acc:
                    grid[i, j] = acc[d] - 1.0 / rec["m"]
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
        ax.set_xticks(range(len(DELTAS)), DELTAS, fontsize=7)
        ax.set_yticks(range(len(agents.MODES)),
                      [m.replace("_", " ") for m in agents.MODES], fontsize=7)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("$\\Delta$", fontsize=8)
    for ax in axes[0][1:]:
        ax.set_yticklabels([])
    fig.colorbar(im, ax=axes[0], label="accuracy $-$ chance", shrink=0.85)
    fig.suptitle("The cue ladder: only the $\\Delta=0$ column is above chance",
                 fontsize=11)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_retinotopic(rows, path):
    """
    Accuracy against chance with a binomial interval, so "at chance" and
    "reliably below chance" are told apart. VII cannot do this: d' is clipped at
    zero, so every below-chance cell collates as VII = 0 and the trap is
    invisible in the invariance axis.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for r in rows:
        chance = 1.0 / r["m"]
        xs = [d for d in DELTAS if d in r["acc"]]
        ys = [r["acc"][d] - chance for d in xs]
        err = [1.96 * np.sqrt(max(r["acc"][d], 1e-9) * (1 - r["acc"][d]) / r["ns"][d])
               for d in xs]
        ax.errorbar(xs, ys, yerr=err, fmt="-o", ms=4, capsize=2, lw=1.4,
                    color=_colour(r), label=_label(r).replace("\n", " "),
                    ls="-" if r["complete"] else "--")
    ax.axhline(0, color="#333", lw=1.0)
    ax.set_xticks(DELTAS)
    ax.set_xlabel("viewpoint change $\\Delta$ (degrees)")
    ax.set_ylabel("accuracy $-$ chance")
    ax.set_title("Below chance is not noise: the retinotopic trap", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# The two stimulus modes whose foils are not inventory-controlled. In c0/c1/c2
# all 100 scenes are built from ONE landmark set (e.g. every c0 scene contains
# exactly a cyan pyramid, a green cylinder, a purple cone and a yellow dome), so
# the only thing that distinguishes a scene from a foil is where those landmarks
# stand -- which is the spatial question the benchmark means to ask. In c3 and
# c4 every scene has its own unique set of terrain morphologies, so "which
# option contains the same landmarks" answers the trial without any spatial
# reasoning at all.
INVENTORY_CONTROLLED = ["c0_shape_colour", "c1_shape", "c2_colour"]
INVENTORY_DIAGNOSTIC = ["c3_peaks_bare", "c4_valley"]


def fig_inventory_shortcut(path, min_trials=40):
    """
    Above-chance accuracy at large turns, split by whether landmark identity is
    diagnostic. If a model were doing the spatial task, the split would not
    matter. If it is matching landmark inventories, only the right-hand bar
    survives.
    """
    recs = [r for r in agents.afc_records(min_trials=min_trials)
            if r["role"] == "primary"]
    recs.sort(key=lambda r: (r["family"] == "human", r["params_b"] or 1e9))

    names, ctrl, diag = [], [], []
    for rec in recs:
        res = json.load(open(rec["path"]))["results"]
        chance = 1.0 / rec["m"]
        vals = []
        for modes in (INVENTORY_CONTROLLED, INVENTORY_DIAGNOSTIC):
            rows = [r for r in res if r["mode"] in modes and r["delta"] >= 90]
            vals.append(sum(r["is_correct"] for r in rows) / len(rows) - chance
                        if rows else np.nan)
        names.append(f"{rec['label']} {rec['arm']}")
        ctrl.append(vals[0])
        diag.append(vals[1])

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x - 0.2, ctrl, 0.4, label="landmark inventory CONTROLLED\n(c0, c1, c2)",
           color="#3b6ea5")
    ax.bar(x + 0.2, diag, 0.4, label="landmark inventory DIAGNOSTIC\n(c3, c4)",
           color="#c8791a")
    ax.axhline(0, color="#333", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("accuracy $-$ chance,  $\\Delta \\geq 90^\\circ$")
    ax.set_title("Where the residual performance at large turns comes from",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = _rows()
    fig_dissociation(rows, f"{OUT}/fig_dissociation.png")
    fig_vii_curves(rows, f"{OUT}/fig_vii_curves.png")
    fig_cue_ladder(f"{OUT}/fig_cue_ladder.png")
    fig_retinotopic(rows, f"{OUT}/fig_retinotopic.png")
    fig_inventory_shortcut(f"{OUT}/fig_inventory_shortcut.png")
    print(f"{len(rows)} runs plotted:")
    for r in rows:
        inv = [r["viic"][d] for d in (90, 135, 180) if r["viic"].get(d) is not None]
        print(f"  {r['label']:42s} {r['arm']}  n={sum(r['ns'].values()):4d}  "
              f"d'(0)={r['d0']:.2f}  VII>=90={np.mean(inv) if inv else float('nan'):.3f}"
              f"{'' if r['complete'] else '   [partial]'}")


if __name__ == "__main__":
    main()
