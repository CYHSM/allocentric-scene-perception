"""
Figure 1: what turning around costs a person, and what it costs a model.

    python bench/figure_gap.py --out figures/fig1_gap.png

Driven entirely by `bench/agents.py`, so a new model joins the figure as soon as
its result file lands. Colour is by family, marker by kind, ordering by
parameter count.

The axis is **d'**, not accuracy and not a ratio.

Accuracy cannot carry it: the arms have different chance levels -- retrieval over
a 100-scene gallery is 1%, 4AFC is 25%, 2AFC is 50% -- so 54% is barely off the
floor on one arm and extraordinary on another. d' under the m-AFC model removes
that, which is what lets a person, a 32B VLM and a 26M-parameter ResNet share an
axis.

An earlier version plotted VII = d'(delta)/d'(0), each agent normalised by its
own zero-turn performance. That is the right *summary* number and it is still in
the table, but it is the wrong thing to draw: d'(0) for Qwen2.5-VL-3B is 0.39, so
its VII is a small number over a small number and the interval covers the whole
panel. Plotting d' directly keeps the gate and the collapse in one line and the
intervals well behaved.

**Left: d' against turn angle.** delta=0 is the appearance gate -- same place,
same bearing, a different appearance sample -- and everything to the right of it
is the cost of walking around the place.

**Right: the same two quantities against model size.** The claim the paper makes
is that scale moves d'(0) and not d'(45), and this is the panel that can be
wrong.

A point at the ceiling its trial count can resolve (a person answering 10 of 10)
is drawn hollow with an upward arrow: d' there is a lower bound, not a
measurement. Intervals are Wilson on the accuracy, mapped through d'.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agents as A
import vii as V

DELTAS = [45, 90, 135, 180]

FAMILY_COLOUR = {
    "human":    "#1d3f2b",
    "qwen":     "#9e2a2b",
    "internvl": "#c1666b",
    "openai":   "#0f766e",
    "gemini":   "#7c3aed",
    "anthropic": "#c2410c",
    "llama":    "#b45309",
    "mistral":  "#a16207",
    "gemma":    "#7f1d1d",
    "dinov2":   "#1b4965",
    "siglip":   "#2f6690",
    "clip":     "#5a8fb8",
    "resnet":   "#9dbfd6",
    "other":    "#6b7280",
}
KIND_MARKER = {"human": "*", "vlm": "s", "encoder": "o"}
KIND_SIZE = {"human": 15, "vlm": 6, "encoder": 5}


def colour(rec):
    return FAMILY_COLOUR.get(rec["family"], FAMILY_COLOUR["other"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/scenes_100")
    ap.add_argument("--results", nargs="+", default=["results", "human_task"])
    ap.add_argument("--out", default="figures/fig1_gap.png")
    ap.add_argument("--include_incomplete", action="store_true",
                    help="also plot runs flagged incomplete or stale-benchmark")
    args = ap.parse_args()

    recs = A.collect(root=args.root, dirs=tuple(args.results))
    if not args.include_incomplete:
        dropped = [r for r in recs if not r["complete"]]
        recs = [r for r in recs if r["complete"]]
        for r in dropped:
            why = "stale benchmark" if r.get("stale_benchmark") else "incomplete"
            print(f"[fig] left out {r['label']} ({r['arm']}): {why}")
    if not recs:
        raise SystemExit("no complete records to plot")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    all_d = [0] + DELTAS

    def dprime_with_ci(r, d):
        """d', its Wilson interval mapped through d', and whether it is capped."""
        n = r["n"][d]
        p = r["accuracy"][d]
        lo, hi = r["ci"][d]
        cap = (n - 0.5) / n
        return (V.dprime(p, r["m"], n_trials=n),
                V.dprime(lo, r["m"], n_trials=n),
                V.dprime(hi, r["m"], n_trials=n),
                p >= cap)

    # ---- left: d' against how far you walked -------------------------------
    ax = axes[0]
    ax.axhline(0.0, color="#444", lw=1.0)
    ax.text(183, 0.0, " chance", va="center", fontsize=7.5, color="#444")

    rng = np.random.default_rng(0)
    for r in recs:
        human = r["kind"] == "human"
        c = colour(r)
        xs, ys, lo, hi, capped = [], [], [], [], []
        for d in all_d:
            if d not in r["accuracy"]:
                continue
            v, l, h, cp = dprime_with_ci(r, d)
            xs.append(d); ys.append(v); lo.append(v - l); hi.append(h - v)
            capped.append(cp)
        jit = 0 if human else rng.uniform(-2.0, 2.0)
        ax.errorbar(np.array(xs) + jit, ys, yerr=[lo, hi], fmt="-",
                    marker=KIND_MARKER[r["kind"]], ms=KIND_SIZE[r["kind"]],
                    lw=2.8 if human else 1.5, color=c, zorder=6 if human else 2,
                    elinewidth=0.9, capsize=2, alpha=1.0 if human else 0.9,
                    markerfacecolor=c, markeredgecolor=c,
                    label=f"{r['label']}  ({r['arm']})")
        # A point at the resolution ceiling of its own trial count is a lower
        # bound, not a measurement, and must not read as one.
        for x, y, cp in zip(xs, ys, capped):
            if cp:
                ax.plot(x + jit, y, KIND_MARKER[r["kind"]],
                        ms=KIND_SIZE[r["kind"]], markerfacecolor="white",
                        markeredgecolor=c, markeredgewidth=1.4, zorder=7)
                ax.annotate("", xy=(x + jit, y + 0.28), xytext=(x + jit, y + 0.05),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1))

    ax.set_xticks(all_d)
    ax.set_xlabel("how far you walked around the place,  $\\Delta$ (degrees)")
    ax.set_ylabel("$d'$")
    ax.set_title("$\\Delta=0$ is the appearance gate; everything right of it is "
                 "the cost of turning", fontsize=9)
    ax.set_xlim(-9, 190)
    ax.legend(fontsize=6.9, ncol=2, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.80), columnspacing=1.0,
              handletextpad=.5, labelspacing=.35)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- right: does scale buy either of them? -----------------------------
    ax = axes[1]
    # Only the VLMs form a scaling series. Joining ResNet-50 to CLIP to DINOv2
    # to a 3B VLM would draw a line through four unrelated training regimes and
    # produce a dip that is a change of family, not an effect of size. The
    # encoders are shown as what they are: separate points on the same axes.
    vlms = sorted((r for r in recs if r["kind"] == "vlm" and r["params_b"]
                   and 45 in r["accuracy"]), key=lambda r: r["params_b"])
    encs = [r for r in recs if r["kind"] == "encoder" and r["params_b"]]

    SERIES = [(0, "#9e2a2b", "-o", "$\\Delta=0^\\circ$  appearance gate"),
              (45, "#5a8fb8", "--s", "$\\Delta=45^\\circ$  one turn"),
              (90, "#8d99ae", ":^", "$\\Delta=90^\\circ$  quarter way round")]
    slopes = {}
    for d, col, style, tag in SERIES:
        if vlms and all(d in r["accuracy"] for r in vlms):
            xs = np.array([np.log10(r["params_b"]) for r in vlms])
            ys = np.array([dprime_with_ci(r, d)[0] for r in vlms])
            ax.plot(10 ** xs, ys, style, color=col, lw=1.7, ms=6,
                    label="VLMs, " + tag, zorder=3)
            if len(xs) > 1:
                slopes[d] = np.polyfit(xs, ys, 1)
        if encs and d in encs[0]["accuracy"]:
            ax.scatter([r["params_b"] for r in encs],
                       [dprime_with_ci(r, d)[0] for r in encs],
                       marker={0: "o", 45: "s", 90: "^"}[d], s=34,
                       facecolors="none", edgecolors=col, linewidths=1.3,
                       zorder=3, label="encoders, " + tag)

    for r in vlms + encs:
        ax.annotate(r["label"].replace("Qwen2.5-VL-", "").replace("InternVL3-", "IV3-"),
                    (r["params_b"], dprime_with_ci(r, 0)[0]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=6.6, color="#5b625c")

    human = next((r for r in recs if r["kind"] == "human"), None)
    if human:
        v = dprime_with_ci(human, 0)[0]
        ax.axhline(v, color=FAMILY_COLOUR["human"], lw=1.3, ls=":", zorder=1)
        ax.text(0.015, v + .06, "human, every angle", fontsize=7.5,
                color=FAMILY_COLOUR["human"], transform=ax.get_yaxis_transform())
    # The number the paper quotes. Three points fitted log-linearly and
    # extrapolated is illustrative, not a prediction, and the caption must say
    # so -- but the contrast between the slopes does not depend on the fit:
    # d' at 90 degrees and beyond is exactly zero at every scale measured.
    if human and slopes:
        lines = []
        for d, (a, b) in sorted(slopes.items()):
            h = dprime_with_ci(human, d)[0] if d in human["accuracy"] else None
            if a <= 0.02:
                lines.append(f"$\\Delta={d}^\\circ$: no slope")
            elif h:
                need = 10 ** ((h - b) / a)
                need_s = (f"{need:,.0f}B" if need < 1e4
                          else f"$10^{{{np.log10(need):.0f}}}$B")
                lines.append(f"$\\Delta={d}^\\circ$: {a:+.2f} $d'$/decade "
                             f"$\\to$ human at ~{need_s}")
        ax.text(0.985, 0.985, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=6.6, color="#3a3f3c",
                linespacing=1.5)

    ax.axhline(0, color="#444", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (billions, log)")
    ax.set_ylabel("$d'$")
    ax.set_title("both improve with scale; only one is heading anywhere",
                 fontsize=9)
    ax.set_ylim(-0.35, 3.05)
    ax.legend(fontsize=6.4, frameon=False, loc="lower left",
              bbox_to_anchor=(-0.01, -0.02), labelspacing=.3, ncol=2,
              columnspacing=.8, handletextpad=.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")

    print(f"\n{'agent':26s} {'arm':9s} " +
          " ".join(f"d'({d}){'':>1s}" for d in [0] + DELTAS) + "   VII45")
    for r in recs:
        ds = " ".join(
            f"{V.dprime(r['accuracy'][d], r['m'], n_trials=r['n'][d]):6.2f}"
            if d in r["accuracy"] else "   n/a" for d in [0] + DELTAS)
        v45 = r["vii"].get(45)
        print(f"{r['label']:26s} {r['arm']:9s} {ds}   "
              f"{'n/a' if v45 is None else f'{v45:.3f}'}")


if __name__ == "__main__":
    main()
