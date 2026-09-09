"""
Figures 2 and 3, and Table 1: what the models did.

    python bench/figure_results.py --root data/scenes --model <timm id>

Reads the JSON that `metrics.py` and `exchange.py` leave in each mode
directory. Nothing is computed here that is not already in those files, so a
figure can never disagree with the number it is drawn from.

**Fig 2** -- lambda(45 deg) for every model in every mode. Small lambda is the
allocentric end. A lambda that falls off the probe ladder is drawn *at the
boundary with an arrow*, not omitted and not clamped: for a saturated model
that is the entire result, and a missing bar reads as a missing run.

**Fig 3** -- the two curves that lambda is read off, overlaid, one panel per
mode. This is the figure that shows whether lambda was interpolated between two
real points or extrapolated off the end, which a single number cannot say.
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


def load(root, name):
    """{mode: [rows]} for one of the two result files, skipping absent modes."""
    out = {}
    for mode, _ in MODES:
        p = os.path.join(root, mode, name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    out[mode] = json.load(f)
            except ValueError:
                print(f"[figs] {p} is not readable JSON -- skipped")
    return out


def _lam_row(rows, model, delta):
    for r in rows:
        if r.get("model") == model:
            return r.get("lambda", {}).get(str(delta), {})
    return {}


def fig_lambda(exch, models, delta, ladder, out):
    """
    Points, not bars. On a log axis a bar's length measures nothing -- it runs
    from an arbitrary floor -- and here it would also cover the very band that
    marks the allocentric end of the scale.
    """
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    lo_m, hi_m = ladder[0], ladder[-1]
    width = 0.72 / max(len(models), 1)
    xs = np.arange(len(MODES))

    ax.axhspan(hi_m, hi_m * 2.4, color="#d9534f", alpha=.08, zorder=0)
    ax.axhspan(lo_m / 2.4, lo_m, color="#2e7d5b", alpha=.08, zorder=0)
    ax.text(-.45, hi_m * 1.55, "off the ladder: appearance end",
            ha="left", va="center", fontsize=7.5, color="#8c2f2b")
    ax.text(-.45, lo_m / 1.55, "off the ladder: allocentric end",
            ha="left", va="center", fontsize=7.5, color="#1d5c42")

    for k, model in enumerate(models):
        off = (k - (len(models) - 1) / 2) * width
        for i, (mode, _) in enumerate(MODES):
            v = _lam_row(exch.get(mode, []), model, delta)
            if not v:
                continue
            x = xs[i] + off
            if v.get("m") is not None:
                if v.get("lo") is not None:
                    ax.plot([x, x], [v["lo"], v["hi"]], color=f"C{k}",
                            lw=1.6, solid_capstyle="butt", zorder=3)
                ax.plot([x], [v["m"]], "o", color=f"C{k}", ms=6.5,
                        mec="white", mew=.9, zorder=4)
            else:
                # Off the ladder is not a missing run; it is the result. Draw it
                # at the end it fell off, hollow, with an arrow saying which way.
                bound = v.get("bound")
                if bound not in ("below", "above"):
                    continue
                y = hi_m * 1.25 if bound == "above" else lo_m / 1.25
                ax.plot([x], [y], marker="^" if bound == "above" else "v",
                        mfc="none", mec=f"C{k}", mew=1.6, ms=7, zorder=4)

    for k, model in enumerate(models):
        ax.plot([], [], "o", color=f"C{k}", label=SHORT.get(model, model))

    for i in range(1, len(MODES)):
        ax.axvline(i - .5, color="0.88", lw=.8, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(lo_m / 2.4, hi_m * 2.4)
    ax.set_yticks(ladder)
    ax.set_yticklabels([f"{v:g}" for v in ladder])
    ax.set_xlim(-.6, len(MODES) - .4)
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in MODES], fontsize=8.5)
    ax.set_ylabel(f"$\\lambda$({delta}\u00b0)  [m]")
    ax.set_title(f"How far the landmarks must move to cost what a {delta}\u00b0 walk "
                 "costs\n(lower = the turn is cheaper = more viewpoint-invariant)",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=len(models), frameon=False,
              loc="upper left", bbox_to_anchor=(0, -0.18))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[fig2] -> {out}")


def fig_curves(exch, model, delta, out):
    have = [(m, l) for m, l in MODES if _row(exch.get(m, []), model)]
    if not have:
        print("[fig3] no exchange.json rows for this model -- skipped")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(2.5 * len(have), 2.9),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (mode, label) in zip(axes, have):
        r = _row(exch[mode], model)
        ds = sorted((float(k), v) for k, v in r["d_scene"].items())
        x = [k for k, _ in ds]; y = [v for _, v in ds]
        ax.plot(x, y, "o-", color="C0", ms=3.5, lw=1.4, label="$D_{scene}(d)$")
        tgt = r["d_view"].get(str(delta))
        if tgt is not None:
            ax.axhline(tgt, color="C3", lw=1.3, ls="--",
                       label=f"$D_{{view}}({delta}°)$")
        lam = r["lambda"].get(str(delta), {}).get("m")
        if lam is not None:
            ax.plot([lam], [tgt], marker="*", ms=11, color="k", zorder=5)
            ax.annotate(f"$\\lambda$={lam:.1f} m", (lam, tgt), fontsize=8,
                        xytext=(3, -11), textcoords="offset points")
        else:
            b = r["lambda"].get(str(delta), {}).get("bound")
            ax.text(.5, .06, {"above": "$\\lambda$ > ladder",
                              "below": "$\\lambda$ < ladder"}.get(b, ""),
                    transform=ax.transAxes, ha="center", fontsize=8,
                    color="#8c2f2b")
        ax.set_xscale("log"); ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x], fontsize=7.5)
        ax.set_title(label.replace("\n", "  "), fontsize=9)
        ax.set_xlabel("$d_{pos}$  [m]", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("embedding distance")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(f"{SHORT.get(model, model)}: $\\lambda$ is where the scene "
                 f"curve meets the cost of the turn", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[fig3] -> {out}")


def _row(rows, model):
    for r in rows:
        if r.get("model") == model:
            return r
    return None


def _metric(mets, mode, model, delta, key):
    for r in mets.get(mode, []):
        if r.get("model") == model and r.get("appearance") == "changed":
            cell = r.get("by_delta", {}).get(str(delta)) or \
                   r.get("by_delta", {}).get(delta)
            if cell:
                return cell.get(key)
    return None


def table1(exch, mets, models, deltas, out):
    """Table 1 as LaTeX. Unmeasured lambdas print their direction, not a dash."""
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             "model & mode & " +
             " & ".join([f"$\\lambda({d}^\\circ)$" for d in deltas]) +
             r" & R@1$_{45}$ & RSA$_{45}$ \\", r"\midrule"]
    for model in models:
        first = True
        for mode, label in MODES:
            if not _row(exch.get(mode, []), model):
                continue
            cells = []
            for d in deltas:
                v = _lam_row(exch[mode], model, d)
                if v.get("m") is not None:
                    cells.append(f"{v['m']:.1f}")
                else:
                    cells.append({"above": r"$>$24", "below": r"$<$1"}
                                 .get(v.get("bound"), "--"))
            r1 = _metric(mets, mode, model, 45, "recall@1")
            rsa = _metric(mets, mode, model, 45, "rsa")
            lines.append(
                f"{SHORT.get(model, model) if first else ''} & "
                f"{label.replace(chr(10), ' ')} & " + " & ".join(cells) +
                f" & {'--' if r1 is None else f'{r1*100:.1f}'}"
                f" & {'--' if rsa is None else f'{rsa:+.2f}'} \\\\")
            first = False
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table1] -> {out}")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/scenes")
    ap.add_argument("--delta", type=int, default=45)
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m",
                    help="the model Fig 3 draws curves for")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--table", default="paper/table1.tex")
    args = ap.parse_args()

    exch, mets = load(args.root, "exchange.json"), load(args.root, "metrics.json")
    if not exch:
        raise SystemExit(f"no exchange.json under {args.root}/*/ -- run "
                         f"bench/exchange.py first")
    models, ladder = [], set()
    for rows in exch.values():
        for r in rows:
            if r["model"] not in models:
                models.append(r["model"])
            ladder |= {float(k) for k in r["d_scene"]}
    ladder = sorted(ladder)

    missing = [m for m, _ in MODES if m not in exch]
    if missing:
        print(f"[figs] MISSING exchange.json for {len(missing)} mode(s): "
              f"{', '.join(missing)}")

    os.makedirs(args.figdir, exist_ok=True)
    fig_lambda(exch, models, args.delta, ladder,
               os.path.join(args.figdir, "fig2_lambda.png"))
    fig_curves(exch, args.model, args.delta,
               os.path.join(args.figdir, "fig3_curves.png"))
    table1(exch, mets, models, [45, 90, 180], args.table)


if __name__ == "__main__":
    main()
