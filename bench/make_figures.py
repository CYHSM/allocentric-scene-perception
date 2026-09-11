#!/usr/bin/env python3
"""
The paper's figures, from paper/*.csv.

    python3 bench/make_figures.py            # all of them
    python3 bench/make_figures.py --only f4

Like the tables, these read only what `collate.py` wrote. No figure opens a
result file or recomputes an accuracy, so a figure and a table cannot disagree.

    fig0_stimuli             what the task looks like, and one trial of it
    fig1_gate_and_scaling    the gate against the walk, and both against size
    fig3_modes               the stimulus ladder, rotated trials only
    fig4_error_structure     when wrong, which foil is chosen?
    fig5_agreement           do a pair agree on which trials are solvable?
    fig6_item_difficulty     are some trials hard for every model at once?
    fig7_prompts             one model across six instruction styles
    fig8_setsize             landmark count against accuracy

Three earlier figures were retired rather than kept for completeness. The
viewpoint curve and the foil ladder are tables now -- both are grids of numbers
that a reader wants to compare exactly, and neither had a shape worth drawing.
The within-band foil-distance panel showed nothing and could not: that arm draws
every foil from the hardest decile, so its whole contrast is a few metres.
"""

import argparse
import collections
import csv
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_spec as SPEC
import figstyle as FS
import stats as S
import vii as V
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

PAPER = os.path.join(SPEC.REPO, "paper")
FIGS = os.path.join(SPEC.REPO, "figures")


def _rows(name):
    with open(os.path.join(PAPER, name)) as f:
        return list(csv.DictReader(f))


def _observers():
    obs = _rows("observers.csv")
    sizes = collections.defaultdict(list)
    for o in obs:
        if o["params_b"]:
            sizes[o["family"]].append(float(o["params_b"]))
    for o in obs:
        o["_colour"] = FS.colour(
            o["family"], float(o["params_b"]) if o["params_b"] else None,
            sizes.get(o["family"]))
        o["_partial"] = bool(o["blocking"])
        o["_star"] = " *" if o["_partial"] else ""
    return obs


def _by_rotated(obs):
    """Descending rotated accuracy, human first. The ordering every panel uses."""
    return sorted(obs, key=lambda o: (0 if o["access"] == "human" else 1,
                                      -float(o["acc_drot"] or 0)))


def _trial_map(trials):
    by = collections.defaultdict(dict)
    for t in trials:
        by[t["key"]][t["trial_id"]] = int(t["is_correct"])
    return by


# ------------------------------------------------------------------ fig 1 ---

def fig1(obs):
    """
    The two things a cognitive map has to do, and what scale does to them.

    Left: the appearance gate against what survives a walk. x is whether the
    observer can name the place at all with no viewpoint change, y is whether it
    still can once the camera moves. An observer that reads scenes by appearance
    is far right and flat on the floor; an observer that is merely bad sits near
    the origin. Those are different failures and a single accuracy number cannot
    tell them apart.

    Right: the same two quantities against parameter count, in two families.
    Qwen2.5-VL has no more public sizes, so the axis could only be replicated
    rather than extended; InternVL3.5 gives six dense points from 1B to 38B in
    an unrelated architecture. Both are drawn because a flat rotated line in one
    family is a property of that family, and the same flat line in two is a
    property of the task.

    Grey is the unrotated line throughout. It is the part that scale does move,
    and putting it in the background is the argument: what the figure is about
    is the coloured line underneath it, which does not move. The Qwen series
    carries a second rotated line on the simple trials, where the distractors
    are far apart, and the strip of API models on the far right is its positive
    control: the same widening that moves nothing in the open families moves
    both API models by 20 to 45 points.

    No confidence intervals here. Eleven series of them turned the panel into a
    thicket, and every interval in the figure is in Table 1.
    """
    lad = _rows("ladder.csv")
    have = collections.defaultdict(dict)
    for r in lad:
        if r["complete"] == "1" and r["acc_drot"]:
            have[r["bank"]][r["key"]] = r

    # Authored at the printed width. A wider canvas scaled down to
    # \textwidth shrinks every label with it.
    fig = plt.figure(figsize=(FS.FULL[0], FS.FULL[1] * 0.98))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.28, 2.0], wspace=0.30,
                             left=0.065, right=0.985, top=0.90, bottom=0.235)
    inner = outer[0, 1].subgridspec(1, 2, width_ratios=[3.5, 1.0], wspace=0.07)
    ax = fig.add_subplot(outer[0, 0])
    lx = fig.add_subplot(inner[0, 0])
    bx = fig.add_subplot(inner[0, 1], sharey=lx)

    # ---- left: the gate against the walk ------------------------------------
    ch = SPEC.ARM["chance"]
    ax.axhline(ch, color=FS.CHANCE, ls=(0, (4, 3)), lw=0.9, alpha=0.7, zorder=0)
    ax.axvline(ch, color=FS.CHANCE, ls=(0, (4, 3)), lw=0.9, alpha=0.7, zorder=0)
    ax.plot([0, 1], [0, 1], color="#dcdcdc", lw=0.9, zorder=0)

    named = [o for o in obs if o["key"] in SPEC.FIG_HIGHLIGHT]
    rest = [o for o in obs if o["key"] not in SPEC.FIG_HIGHLIGHT
            and o["acc_d0"] and o["acc_drot"]]
    for o in rest:
        ax.plot(float(o["acc_d0"]), float(o["acc_drot"]), "o", ms=4.5,
                color=FS.GREY, mec=FS.MUTED, mew=0.6, zorder=2)
    for o in named:
        if not o["acc_d0"] or not o["acc_drot"]:
            continue
        x, y = float(o["acc_d0"]), float(o["acc_drot"])
        ax.plot(x, y, "o", ms=7.5 if o["access"] == "human" else 6,
                color=o["_colour"],
                mfc="white" if o["_partial"] else o["_colour"],
                mec=o["_colour"], mew=1.6, zorder=4)
        dx, dy = _F1_LABEL.get(o["key"], (7, -3))
        ax.annotate(_SHORT.get(o["key"], o["label"]) + o["_star"], (x, y),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="right" if dx < 0 else "left", fontsize=6.8,
                    color=FS.INK)

    ax.set_xlabel("appearance gate  (Δ = 0)", fontsize=8)
    ax.set_ylabel("rotated accuracy  (Δ ≥ 45°)", fontsize=8)
    ax.set_xlim(0.15, 1.14)
    ax.set_ylim(0.03, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(labelsize=7)

    # ---- right: the same two quantities against size ------------------------
    # One Qwen line per condition, 3B to 235B. InternVL3.5 ran the same ladder
    # and lands on top of this one; it is in Table 1 rather than here, because
    # two families drawn together needed a second visual channel and the panel
    # was carrying four lines to make one point.
    series = sorted([o for o in obs
                     if o["family"] == "qwen" and o["params_b"]
                     and o.get("role") == "primary"],
                    key=lambda o: float(o["params_b"]))
    all_x = {float(o["params_b"]) for o in series}

    def _pts(getter):
        out = []
        for o in series:
            v = getter(have, o["key"])
            if v:
                out.append((float(o["params_b"]), v[0]))
        return sorted(out)

    api = [o for o in obs if o["key"] in ("luna", "gemini38")]
    api.sort(key=lambda o: o["key"] != "luna")

    # Each condition is one line that starts in the parameter panel and finishes
    # in the strip, crossing the axis break. The two API models disclose no size
    # and cannot sit on the axis, but they are the same measurement, and three
    # pairs of unconnected dots do not read as that.
    # The unrotated points are drawn without a line. Joining them invites the
    # reader to trace a trend across models that differ in more than size, and
    # the trend is not the claim: the claim is where the points sit relative to
    # the rotated ones below them.
    CONDS = [
        dict(getter=_hard_gate, colour=GATE_C, alpha=1.0, marker="o", lw=0,
             z=2, label="unrotated (Δ=0)"),
        dict(getter=_pool_wide, colour=HARD_C, alpha=SIMPLE_A, marker="s",
             lw=1.7, z=3, label="rotated, simple trials"),
        dict(getter=_hard_rot, colour=HARD_C, alpha=1.0, marker="s", lw=1.7,
             z=4, label="rotated, hard trials"),
    ]
    for c in CONDS:
        pts = _pts(c["getter"])
        kw = dict(marker=c["marker"], color=c["colour"], lw=c["lw"], ms=4.4,
                  alpha=c["alpha"], zorder=c["z"], mfc=c["colour"], mew=1.2)
        if pts:
            lx.plot([x for x, _ in pts], [p for _, p in pts], label=c["label"],
                    **kw)
        strip = [(i, c["getter"](have, o["key"]))
                 for i, o in enumerate(api)]
        strip = [(i, v[0]) for i, v in strip if v]
        if strip:
            bx.plot([i for i, _ in strip], [p for _, p in strip], **kw)
        # The crossing segment, drawn on the figure so neither axes clips it.
        if c["lw"] and pts and strip and pts[-1][0] == max(all_x):
            fig.add_artist(ConnectionPatch(
                xyA=pts[-1], coordsA=lx.transData,
                xyB=(strip[0][0], strip[0][1]), coordsB=bx.transData,
                color=c["colour"], lw=c["lw"], alpha=c["alpha"] * 0.9,
                zorder=c["z"]))

    FS.chance_line(lx, label=False)
    lx.set_xscale("log")
    # A tick per model collides at 32/72. The exact sizes are in Table 1; the
    # axis only needs a readable ruler.
    ruler = [x for x in (4, 8, 16, 32, 64, 128) if min(all_x) <= x <= max(all_x)]
    lx.set_xticks(ruler)
    lx.set_xticklabels([f"{x:.0f}" for x in ruler])
    lx.minorticks_off()
    # Padding, or the 3B marker is cut in half by the spine.
    lx.set_xlim(min(all_x) * 0.72, max(all_x) * 1.45)
    lx.set_xlabel("parameters (B, log scale)", fontsize=8)
    lx.set_ylabel("accuracy", fontsize=8)
    lx.set_ylim(0, 1.02)
    lx.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    lx.tick_params(labelsize=7)

    # ---- the strip: models with no disclosed parameter count ----------------
    # Neither panel has a clear stretch of the 25% line to write "chance" on,
    # so it goes in the legend instead of on top of a marker.
    FS.chance_line(bx, label=False)
    bx.set_xlim(-0.45, len(api) - 0.25)
    bx.set_xticks(range(len(api)))
    bx.set_xticklabels([_SHORT.get(o["key"], o["label"]) for o in api],
                       fontsize=7, rotation=30, ha="right",
                       rotation_mode="anchor")
    # No title on the strip: it collides with the panel title beside it, and
    # the caption already says these two disclose no parameter count.
    bx.tick_params(axis="y", left=False, labelleft=False)

    import matplotlib.lines as mlines
    handles, labels = lx.get_legend_handles_labels()
    order = [labels.index(c["label"]) for c in CONDS if c["label"] in labels]
    hs = [handles[i] for i in order] + [
        mlines.Line2D([], [], color=FS.CHANCE, ls=(0, (4, 3)), lw=1.0)]
    ls = [labels[i] for i in order] + ["25% chance"]
    lx.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.22, -0.23),
              fontsize=6.6, ncol=4, columnspacing=1.2, handlelength=2.2,
              handletextpad=0.5)
    return FS.save(fig, "fig1_gate_and_scaling", FIGS)


# Hand-placed labels for the five named points, which sit close together on the
# right-hand wall. No automatic placement separates them legibly.
# Short forms for the point labels. The panel is half as wide as it used to
# be and the full roster names do not fit beside the markers.
_SHORT = {
    "luna": "GPT-5.6", "gemini38": "Gemini 3.8",
    "qwen25vl_3b": "Qwen2.5-VL 3B", "qwen25vl_72b": "Qwen2.5-VL 72B",
}

_F1_LABEL = {
    "human_p01": (-9, -4), "luna": (6, 7), "gemini38": (8, 3),
    "qwen25vl_3b": (8, -3), "qwen25vl_72b": (9, -10),
}


# ------------------------------------------------------------------ fig 2 ---

def _pool_wide(have, key):
    """
    The mid and easy rungs, pooled into one "wide foils" condition.

    Three rotated lines put two of them on top of each other: at 31 m and at
    43 m of foil separation every open model sits at chance and the two curves
    are within a few points everywhere. Pooling them trades a distinction no
    model responds to for a panel a reader can parse.

    The two rungs re-ask the *same* questions with different foils, so pooling
    doubles the trial count without doubling the evidence. The point estimate
    uses the pooled counts; the interval is computed on the number of distinct
    items, which is the single-rung n. That is the conservative choice.
    """
    rs = [have.get(b, {}).get(key) for b in ("mid", "easy")]
    rs = [r for r in rs if r]
    if not rs:
        return None
    ns = [int(r["n_drot"]) for r in rs]
    ks = [round(float(r["acc_drot"]) * n) for r, n in zip(rs, ns)]
    p = sum(ks) / sum(ns)
    _, lo, hi = S.wilson(round(p * max(ns)), max(ns))
    return p, lo, hi


def _hard_rot(have, key):
    r = have.get("hard", {}).get(key)
    if not r:
        return None
    p, n = float(r["acc_drot"]), int(r["n_drot"])
    _, lo, hi = S.wilson(round(p * n), n)
    return p, lo, hi


def _hard_gate(have, key):
    r = have.get("hard", {}).get(key)
    if not r:
        return None
    p, n = float(r["acc_d0"]), int(r["n_d0"])
    _, lo, hi = S.wilson(round(p * n), n)
    return p, lo, hi


# The unrotated line is grey in every panel. It is the line that moves, and
# keeping it in the background is the argument: the reader is meant to end up
# looking at the coloured line underneath it, which does not.
GATE_C = FS.GREY

# Colour carries the trial difficulty, not the model. There is one series per
# condition running the whole width of the panel, so a family encoding would be
# a second meaning on a channel that already has one.
HARD_C = SPEC.FAMILY_COLOUR["qwen"]
SIMPLE_A = 0.42


# ------------------------------------------------------------------ fig 3 ---

def fig3(obs, cells):
    """
    The stimulus ladder as a heatmap: observers x modes, rotated trials.

    A line plot of ten observers over five categorical positions suggested a
    trend between modes that does not exist -- the modes are not ordered by any
    underlying quantity. A grid says the same numbers without implying one.

    Rows are sorted by rotated accuracy, which is what the cells show, so the
    row order and the cell values are the same quantity. Diverging around
    chance, because each cell answers "is this above 25%", not "how large".
    """
    import matplotlib.colors as mcolors
    agg = collections.defaultdict(lambda: [0, 0])
    for c in cells:
        if int(c["delta"]) == 0:
            continue
        agg[(c["key"], c["mode"])][0] += int(c["k"])
        agg[(c["key"], c["mode"])][1] += int(c["n"])

    rows = [o for o in _by_rotated(obs)
            if any(agg.get((o["key"], m), (0, 0))[1] for m in SPEC.PLOT_MODES)]
    M = np.full((len(rows), len(SPEC.PLOT_MODES)), np.nan)
    for i, o in enumerate(rows):
        for j, mode in enumerate(SPEC.PLOT_MODES):
            k, n = agg.get((o["key"], mode), (0, 0))
            if n:
                M[i, j] = k / n

    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(rows) + 2.0))
    ch = SPEC.ARM["chance"]
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=ch, vmax=1.0)
    im = ax.imshow(M, cmap="RdBu", norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center",
                    fontsize=7.5,
                    # RdBu is dark at both ends, so contrast is distance from
                    # the white centre in normalised units: a 0% cell is as
                    # dark as a 100% one.
                    color="white" if abs(norm(M[i, j]) - 0.5) > 0.32 else "#222")
    ax.set_xticks(range(len(SPEC.PLOT_MODES)))
    ax.set_xticklabels([SPEC.MODE_SHORT[m] for m in SPEC.PLOT_MODES], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([o["label"] + o["_star"] for o in rows], fontsize=8)
    ax.set_title("Rotated accuracy by stimulus mode", loc="left")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("accuracy on rotated trials", fontsize=8)
    cb.ax.axhline(ch, color="#222", lw=1.2)
    cb.ax.tick_params(labelsize=7)
    cb.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.annotate("rows sorted by rotated accuracy   ·   white = chance (25%)",
                xy=(1.0, -0.13), xycoords="axes fraction", ha="right",
                fontsize=6.8, color="#888")
    return FS.save(fig, "fig3_modes", FIGS)


# ------------------------------------------------------------------ fig 4 ---

def fig4(obs, trials, ax=None):
    """
    When the observer is wrong, which of the three foils did it pick?

    Each error offers three wrong answers, and they are not interchangeable: one
    is the nearest to the target in rotation-optimal layout distance, one the
    middle, one the farthest. A graded representation of layout confuses a place
    with its nearest neighbour more often than with the far ones, so it should
    lean on the left segment. Independent of geometry means three equal thirds.

    Every model is flat. Pooled over 581 model errors the split is 30 / 39 / 32
    against 33 / 33 / 33, and no individual model departs from a third by more
    than its interval. **This is a null result and is drawn as one** -- the
    models-pooled row exists to show that the null holds with real power, not to
    manufacture an effect. The signal in the panel is the contrast with the
    human, whose errors do lean nearest, and that rests on five errors.

    What would turn this from a null into a mechanism is the appearance
    control: rank the options by embedding distance to the study image instead
    of by layout distance, and ask whether model choices track *that*. Errors
    that are random with respect to geometry but ordered with respect to
    appearance would be the paper's claim shown directly rather than inferred.
    """
    import json as _json
    with open(os.path.join(SPEC.REPO, SPEC.ARM["benchmark"])) as f:
        T = {t["id"]: t for t in _json.load(f)["trials"]}

    agg = collections.defaultdict(lambda: [0, 0, 0])
    for t in trials:
        if int(t["is_correct"]) or not t["model_choice"]:
            continue
        opts = T[t["trial_id"]]["options"]
        foils = sorted([(i, o["layout_distance_m"])
                        for i, o in enumerate(opts) if not o["is_target"]],
                       key=lambda x: x[1])
        chosen = int(t["model_choice"]) - 1
        for rank, (i, _) in enumerate(foils):
            if i == chosen:
                agg[t["key"]][rank] += 1

    rows = []
    for o in _by_rotated(obs):
        v = agg.get(o["key"])
        if v and sum(v) >= 5:
            rows.append((o["label"] + o["_star"], v, o["access"] == "human"))
    pooled = [0, 0, 0]
    for o in obs:
        if o["access"] == "human":
            continue
        for i, c in enumerate(agg.get(o["key"], [0, 0, 0])):
            pooled[i] += c
    if sum(pooled):
        rows.append((f"all {sum(1 for o in obs if o['access'] != 'human')} "
                     f"models pooled", pooled, False))

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.0, 0.40 * len(rows) + 1.9))
    small = 0 if own else 1.0
    seg_c = ["#c2410c", "#fdba74", "#e5e7eb"]
    seg_l = ["nearest foil", "middle", "farthest foil"]
    y = range(len(rows))
    for i, (_, v, is_human) in zip(y, rows):
        n = sum(v)
        left = 0.0
        for j in range(3):
            w = v[j] / n
            ax.barh(i, w, left=left, height=0.64, color=seg_c[j],
                    edgecolor=FS.INK if is_human else "white", lw=0.8,
                    label=seg_l[j] if i == 0 else None)
            if w > (0.09 if own else 0.15):
                ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center",
                        fontsize=7 - small, color=FS.INK if j else "white")
            left += w
        ax.annotate(f"n={n}", xy=(1.01, i), va="center",
                    fontsize=6.5 - small, color=FS.MUTED)
    for x in (1 / 3, 2 / 3):
        ax.axvline(x, color=FS.INK, ls=(0, (3, 3)), lw=1.0, zorder=5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8 - small)
    # Room past 1.0 for the n= labels. The human's n is 5, and a reader must
    # see that before reading anything into the 80% bar.
    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 1 / 3, 2 / 3, 1])
    ax.set_xticklabels(["0", "⅓", "⅔", "1"])
    ax.set_xlabel("share of this observer's errors", fontsize=9 - small)
    if own:
        ax.set_title("When it is wrong, which foil does it pick?", loc="left")
    ax.grid(False)
    ax.invert_yaxis()
    ax.legend(ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.10 if not own else
                              (-0.22 if len(rows) > 6 else -0.34)),
              fontsize=7.5 - small, columnspacing=1.2, handlelength=1.4)
    if own:
        ax.annotate("dashed = thirds, what geometry-blind errors look like",
                    xy=(1.0, -0.13), xycoords="axes fraction", ha="right",
                    fontsize=6.8, color=FS.MUTED)
    return FS.save(fig, "fig4_error_structure", FIGS) if own else None


def delta_strip(scene="a014", study_az=90, mode="c0_shape_colour"):
    """
    One scene at the five viewpoint changes, as five small images.

    Not a figure in its own right. These sit in the header of the viewpoint
    table, above the column they belong to, so a reader sees what
    "$\\Delta=135^\\circ$" actually looks like while reading the number under it.
    The sky band and the outer margins carry nothing, so they are cropped away;
    at 0.8 inches wide on the page, the objects are all there is room for.
    """
    from PIL import Image
    out = os.path.join(FIGS, "delta_strip")
    os.makedirs(out, exist_ok=True)
    made = []
    for d in SPEC.DELTAS:
        az = (study_az + d) % 360
        src = os.path.join(SPEC.REPO, "data/scenes_100", mode, scene,
                           f"{scene}_B_az{az:03d}.png")
        if not os.path.exists(src):
            return None
        im = Image.open(src).convert("RGB")
        w, h = im.size
        im = im.crop((int(w * 0.06), int(h * 0.30), int(w * 0.94),
                      int(h * 0.98)))
        im = im.resize((360, round(360 * im.size[1] / im.size[0])),
                       Image.LANCZOS)
        for ext in ("png", "pdf"):
            im.save(os.path.join(out, f"delta_{d:03d}.{ext}"))
        made.append(f"delta_strip/delta_{d:03d}")
    return os.path.join(out, "delta_000.png")


def mode_strip(scene="a014", az=90):
    """
    One place in each stimulus mode, as five small images for a table header.

    The same crop everywhere, and a shallower one than the viewpoint strip: in
    c4 the peaks sit high in the frame, so cutting the top third to remove sky
    would cut the landmarks with it.
    """
    from PIL import Image
    out = os.path.join(FIGS, "mode_strip")
    os.makedirs(out, exist_ok=True)
    for mode in SPEC.MODES:
        src = os.path.join(SPEC.REPO, "data/scenes_100", mode, scene,
                           f"{scene}_A_az{az:03d}.png")
        if not os.path.exists(src):
            return None
        im = Image.open(src).convert("RGB")
        w, h = im.size
        im = im.crop((int(w * 0.06), int(h * 0.16), int(w * 0.94),
                      int(h * 0.98)))
        im = im.resize((360, round(360 * im.size[1] / im.size[0])),
                       Image.LANCZOS)
        for ext in ("png", "pdf"):
            im.save(os.path.join(out, f"{mode.split('_')[0]}.{ext}"))
    return os.path.join(out, "c0.png")


def fig4_agreement_and_errors(obs, trials):
    """
    The two appendix panels that say the failures are shared and unstructured.

    Left, who agrees with whom on which trials are solvable; right, which of the
    three distractors gets picked when an answer is wrong. They were two floats.
    One figure, because they are one argument: the models fail together, and
    when they fail they are not confusing a place with its nearest neighbour.

    Drawn at 7.2 inches and scaled down by the page rather than authored at the
    printed width. The kappa matrix is 17x17 and the error panel is 18 rows, so
    at 5.5 inches the labels would collide; the panels carry their own reduced
    font sizes for this layout.
    """
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.74], wspace=1.00,
                          left=0.175, right=0.945, top=0.975, bottom=0.285)
    fig5(obs, trials, ax=fig.add_subplot(gs[0, 0]))
    fig4(obs, trials, ax=fig.add_subplot(gs[0, 1]))
    return FS.save(fig, "fig4_agreement_and_errors", FIGS)


# ------------------------------------------------------------------ fig 5 ---

def fig5(obs, trials, ax=None):
    """
    Do two observers agree about which trials are solvable?

    Cohen's kappa on trial-level correctness, every pair. High kappa means the
    pair succeeds and fails on the same trials; kappa near zero means their
    successes are unrelated. Raw agreement would not do -- two observers both at
    20% agree on most trials by both being wrong a lot -- so this is corrected
    for the agreement independence alone predicts.

    The human is pinned to the first row -- it is the reference every other row
    is read against, and a reader should not have to hunt for it. The models
    below are ordered by average-linkage clustering on 1 - kappa, because the
    question among them is who resembles whom, and a similarity matrix only
    shows its blocks when the similar rows are adjacent. Accuracy ordering
    scatters the block across the grid.

    Sequential colour, white at zero, not diverging: everything meaningful is
    positive, the largest negative is -0.05 which on 80 trials is noise, and a
    diverging map spends half its range dramatising that noise.
    """
    by = _trial_map(trials)
    present = [o for o in obs if by.get(o["key"])]
    n = len(present)
    if n < 3:
        return None

    K = np.full((n, n), np.nan)
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            if i == j:
                continue
            ids = [x for x in by[a["key"]] if x in by[b["key"]]]
            if len(ids) < 20:
                continue
            K[i, j] = S.cohen_kappa([by[a["key"]][x] for x in ids],
                                    [by[b["key"]][x] for x in ids])

    # Seriate the models among themselves, then put the human back on top.
    idx_h = [i for i, o in enumerate(present) if o["access"] == "human"]
    idx_m = [i for i, o in enumerate(present) if o["access"] != "human"]
    order = idx_h + [idx_m[i] for i in _seriate(K[np.ix_(idx_m, idx_m)])]
    present = [present[i] for i in order]
    K = K[np.ix_(order, order)]

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
    fig = ax.figure
    hi_v = float(np.nanmax(K))
    # Square cells at full width; filling the cell when this panel shares a
      # figure, so its title lines up with the one beside it.
    im = ax.imshow(K, cmap="Oranges", vmin=0.0, vmax=hi_v,
                   aspect="equal" if own else "auto")
    # The per-cell numbers only fit at full width. Half width, they would be
    # 3pt and the colour says the same thing.
    if own:
        for i in range(n):
            for j in range(n):
                if np.isnan(K[i, j]):
                    continue
                ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if K[i, j] > 0.62 * hi_v else FS.INK)
    labels = [o["label"] + o["_star"] for o in present]
    small = 0 if own else 1.0
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right",
                                                fontsize=7 - small)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7 - small)
    if own:
        ax.set_title("Do they agree on which trials are solvable?  (Cohen's κ)",
                     loc="left")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04 if own else 0.030,
                      pad=0.02 if own else 0.012)
    cb.set_label("κ    higher = more agreement" if own else "κ",
                 fontsize=8 - small, labelpad=2)
    cb.ax.tick_params(labelsize=7 - small)
    n_neg = int(np.nansum(K < 0))
    neg = (f"{n_neg // 2} pairs are slightly negative "
           f"(min {float(np.nanmin(K)):.2f}), floored to 0 in colour only"
           if n_neg else "")
    # Paired with another panel, the long-form note doubles the width of the
    # canvas and everything on the page shrinks to fit it. The caption says it.
    ax.annotate(("high κ = this pair succeeds and fails on the same trials   ·   "
                 "white = their successes are unrelated\n"
                 "human first, then models ordered by similarity to each other"
                 + ("   ·   " + neg if neg else "")) if own else neg,
                xy=(0.0, -0.30 if own else -0.33), xycoords="axes fraction",
                fontsize=6.8 - small, color=FS.MUTED, va="top")
    return FS.save(fig, "fig5_agreement", FIGS) if own else None


def _seriate(K):
    """Leaf order from average-linkage clustering on 1 - kappa."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    D = 1.0 - np.nan_to_num(K, nan=0.0)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0                       # enforce exact symmetry
    D[D < 0] = 0.0
    return list(leaves_list(linkage(squareform(D, checks=False), "average")))


# ------------------------------------------------------------------ fig 6 ---

def fig6(obs, trials, n_iter=5000, seed=0):
    """
    Are some trials hard for every model at once?

    Sort the rotated trials by how many of the models solved each, and plot that
    curve. Then do the same after shuffling each model's own correct/incorrect
    vector across trials -- which keeps every model's accuracy exactly and
    destroys only the alignment between models -- and shade where those shuffled
    curves fall.

    If the models succeeded on unrelated trials the real curve would sit inside
    the band. It runs above the band on the left, where a few trials are solved
    by most models, and below it on the right, where a block defeats all of
    them. Difficulty is a property of the trial, shared across models from four
    organisations.

    The human is drawn on the same axis once enough of their session exists to
    support it: their accuracy within each model-difficulty bin says whether the
    two difficulty axes are the same one. On a partial session it is an
    annotation instead, because a curve over seven bins built from twenty-odd
    trials invites a reading it cannot support.
    """
    rng = random.Random(seed)
    by = _trial_map(trials)
    models = [o["key"] for o in obs if o["access"] != "human" and by.get(o["key"])]
    rot = sorted({t["trial_id"] for t in trials if int(t["delta"]) > 0})
    ids = [t for t in rot if all(t in by[m] for m in models)]
    if len(ids) < 20 or len(models) < 3:
        return None

    observed = sorted((sum(by[m][t] for m in models) for t in ids), reverse=True)
    var_obs = statistics.pvariance(observed)

    curves, var_null = [], []
    for _ in range(n_iter):
        cols = []
        for m in models:
            v = [by[m][t] for t in ids]
            rng.shuffle(v)
            cols.append(v)
        tot = [sum(c[i] for c in cols) for i in range(len(ids))]
        var_null.append(statistics.pvariance(tot))
        curves.append(sorted(tot, reverse=True))
    p = (sum(1 for v in var_null if v >= var_obs) + 1) / (n_iter + 1)

    arr = np.array(curves)
    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    x = np.arange(1, len(ids) + 1)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.fill_between(x, lo, hi, color="#bbbbbb", alpha=0.55, step="mid",
                    label="if each model succeeded on unrelated trials")
    ax.step(x, observed, where="mid", color="#8b5cf6", lw=2.2, label="observed")

    n_none = sum(1 for v in observed if v == 0)
    exp_none = float(np.mean([sum(1 for v in c if v == 0) for c in curves]))
    ax.annotate(f"{n_none} trials that no model solves\n(expected {exp_none:.0f})",
                xy=(len(ids) - n_none / 2, 0.05), xytext=(len(ids) - 30, 2.1),
                ha="left", va="bottom", fontsize=7.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.9,
                                shrinkA=0, shrinkB=2))

    ax.set_xlabel(f"the {len(ids)} rotated trials, sorted from most-solved to least")
    ax.set_ylabel(f"how many of the {len(models)} models solved it")
    ax.set_ylim(-0.3, len(models) + 0.3)
    ax.set_yticks(range(0, len(models) + 1, 2))
    ax.set_xlim(0.5, len(ids) + 0.5)
    ax.set_title("Some trials are hard for every model at once", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.annotate(f"spread of trial difficulty is "
                f"{var_obs / statistics.mean(var_null):.1f}× what independence "
                f"predicts  (permutation p = {p:.4f})",
                xy=(0.0, -0.17), xycoords="axes fraction", fontsize=7,
                color="#555", va="top")

    hk = [o for o in obs if o["access"] == "human"]
    if hk:
        hkey = hk[0]["key"]
        none_ids = [t for t in ids if sum(by[m][t] for m in models) == 0]
        seen = [t for t in none_ids if t in by[hkey]]
        if seen:
            ax.annotate(f"of those, the human answered {len(seen)} and got "
                        f"{sum(by[hkey][t] for t in seen)} right",
                        xy=(0.0, -0.245), xycoords="axes fraction", fontsize=7,
                        color="#555", va="top")
    return FS.save(fig, "fig6_item_difficulty", FIGS)


# ------------------------------------------------------------------ fig 7 ---

# The instruction names, cut to one or two words. At half the width the full
# blurbs eat the panel, and the caption carries what each one is short for.
_PROMPT_SHORT = {
    "cot_anyview": "CoT",
    "cot": "CoT + View",
    "mental_rotation": "rotate",
    "anchor": "anchor",
    "birdseye": "plan view",
    "elimination": "eliminate",
}

# The three models with all four landmark banks, worst rotated accuracy first.
# Gemini 3.8 Flash has no landmark runs at all, so the best-performing panel is
# GPT-5.6 Luna.
# Two-line names: at a third of the panel row each, one line runs into its
# neighbour's title.
SETSIZE_MODELS = [
    ("OpenGVLab/InternVL3_5-38B-HF", "InternVL3.5\n38B", "internvl35"),
    ("Qwen/Qwen2.5-VL-72B-Instruct", "Qwen2.5-VL\n72B", "qwen"),
    ("openai/gpt-5.6-luna", "GPT-5.6\nLuna", "openai"),
]


def fig2_landmarks_and_prompts():
    """
    Two controls on one row: what the task is made of, and how it is asked.

    Left: landmark count against accuracy, one panel per model. An earlier
    version drew every model in one axes. Three models times two conditions is
    too many crossing lines: the reader has to track colour and dash pattern at
    once, and the comparison that matters (does the rotated line follow the
    unrotated one?) is within a model, not between them. The shaded band between
    the two lines is the quantity the paper is about: what identifying a place
    buys you once the camera moves. Panels run worst to best, so the row ends on
    the one model that closes the band.

    The x axis is landmark count and the median distractor distance that comes
    with it, printed together, because fewer landmarks compress layout space and
    the two cannot be separated.

    Right: one model under six instruction styles, rotated trials only. Six of
    the nine styles in the codebase assert that the test views come from a
    different viewpoint. That is true at every delta >= 45 and false at
    delta = 0, where it is worth several points, so the rotated subset is where
    every style asks the same question. Plotted as an interval per style against
    the chance line rather than as bars, because the finding is that the
    intervals all contain roughly the same value.
    """
    import json as _json
    import matplotlib.lines as mlines

    rows = [r for r in _rows("setsize.csv") if r["complete"] == "1"]
    banks = {b["key"]: b for b in SPEC.SETSIZE_BANKS}
    order = [b["key"] for b in SPEC.SETSIZE_BANKS]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["model"]][r["bank"]] = r
    models = [(k, lab, fam) for k, lab, fam in SETSIZE_MODELS
              if len(by.get(k, {})) >= 3]

    sweep = []
    for style, _ in SPEC.SWEEP_STYLES:
        path = os.path.join(SPEC.REPO, SPEC.SWEEP_PATH.format(style=style))
        if not os.path.exists(path):
            continue
        with open(path) as f:
            trials = _json.load(f)["results"]
        if len(trials) < SPEC.ARM["n_trials"]:
            continue
        rot = [t for t in trials if int(t["delta"]) > 0]
        p, lo, hi = S.wilson(sum(bool(t["is_correct"]) for t in rot), len(rot))
        sweep.append((_PROMPT_SHORT.get(style, style), style, p, lo, hi))
    if not models or len(sweep) < 2:
        return None
    sweep.sort(key=lambda r: r[2])

    # Explicit positions rather than a gridspec: the right panel's row labels
    # are wider than the panel itself, and a gridspec column sizes the axes
    # without them, so the labels land on top of the panel to its left.
    fig = plt.figure(figsize=(FS.FULL[0], 2.55))
    TOP, BOT, GAP = 0.83, 0.30, 0.012
    # px first: its row labels are wider than the panel, so it needs the left
    # margin that the landmark row does not.
    P0, P1, L0, L1 = 0.145, 0.400, 0.500, 0.985
    px = fig.add_axes([P0, BOT, P1 - P0, TOP - BOT])
    w = (L1 - L0 - GAP * (len(models) - 1)) / len(models)
    axes = [fig.add_axes([L0 + i * (w + GAP), BOT, w, TOP - BOT])
            for i in range(len(models))]

    # ---- left: landmark count ----------------------------------------------
    xs = list(range(len(order)))
    for ax, (key, lab, fam) in zip(axes, models):
        cells, colour = by[key], SPEC.FAMILY_COLOUR[fam]
        series = {}
        for field in ("acc_d0", "acc_drot"):
            nf = "n_d0" if field == "acc_d0" else "n_drot"
            series[field] = [(i, float(cells[b][field]), int(cells[b][nf]))
                             for i, b in enumerate(order)
                             if b in cells and cells[b][field]]
        common = sorted({i for i, _, _ in series["acc_d0"]}
                        & {i for i, _, _ in series["acc_drot"]})
        if common:
            g = {i: p for i, p, _ in series["acc_d0"]}
            r = {i: p for i, p, _ in series["acc_drot"]}
            ax.fill_between(common, [r[i] for i in common],
                            [g[i] for i in common], color=colour, alpha=0.13,
                            lw=0)
        for field, ls, marker, fill in (("acc_d0", (0, (2, 2)), "o", False),
                                        ("acc_drot", "-", "s", True)):
            pts = series[field]
            if not pts:
                continue
            lo = [p - S.wilson(round(p * k), k)[1] for _, p, k in pts]
            hi = [S.wilson(round(p * k), k)[2] - p for _, p, k in pts]
            ax.errorbar([i for i, _, _ in pts], [p for _, p, _ in pts],
                        yerr=[lo, hi], fmt=marker, ls=ls, color=colour,
                        capsize=1.8, elinewidth=0.8, ms=4, lw=1.5,
                        mfc=colour if fill else "white")
        FS.chance_line(ax, label=False)
        ax.set_title(lab, fontsize=7, color=colour, pad=3,
                     linespacing=1.15)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(banks[b]["n_landmarks"]) for b in order],
                           fontsize=7)
        ax.set_xlim(-0.35, len(order) - 0.65)
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="y", left=(ax is axes[0]),
                       labelleft=(ax is axes[0]), labelsize=7)
    axes[0].set_ylabel("accuracy", fontsize=8)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    # One label for the row of panels. Three identical axis labels, or one wide
    # enough to run under its neighbours, both read as a mistake. The distractor
    # distance that comes with each landmark count is in the caption: it does
    # not fit here and a reader needs it once, not on every glance.
    span = (L0 + L1) / 2
    fig.text(span, 0.145, "landmarks per scene", fontsize=7.5, ha="center")
    proxies = [mlines.Line2D([], [], color=FS.INK, ls=(0, (2, 2)), marker="o",
                             mfc="white", ms=4, lw=1.5, label="unrotated (Δ=0)"),
               mlines.Line2D([], [], color=FS.INK, ls="-", marker="s", ms=4,
                             lw=1.5, label="rotated (Δ≥45)")]
    fig.legend(handles=proxies, loc="lower center", ncol=2, fontsize=7,
               bbox_to_anchor=(span, -0.01), frameon=False, columnspacing=1.8,
               handlelength=2.2, handletextpad=0.5)

    # ---- right: instruction styles -----------------------------------------
    # No FS.chance_line here: that draws a *horizontal* reference, and on this
    # panel accuracy is the x axis.
    px.axvline(SPEC.ARM["chance"], color=FS.CHANCE, ls=(0, (4, 3)), lw=1.0,
               zorder=0)
    for i, (blurb, style, p, lo, hi) in enumerate(sweep):
        locked = style == SPEC.ARM["prompt_style_model"]
        px.plot([lo, hi], [i, i], color=FS.INK, lw=1.1, zorder=3)
        px.plot([p], [i], "o", ms=6 if locked else 4.8,
                color=(SPEC.FAMILY_COLOUR["qwen"] if locked else FS.GREY),
                mec=FS.INK, mew=0.9, zorder=4)
    px.set_yticks(range(len(sweep)))
    px.set_yticklabels([r[0] for r in sweep], fontsize=7)
    px.set_xlabel("accuracy on rotated trials", fontsize=7.5)
    px.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    px.set_xlim(0, 0.45)
    px.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    px.tick_params(labelsize=7)
    # As a title on `px` this runs off the right edge of the canvas, and a
    # tight bounding box then widens the whole figure to hold it. Centred over
    # the panel and its row labels, it stays inside.
    fig.text((P0 + P1) / 2 - 0.10, TOP + 0.035, SPEC.SWEEP_MODEL, fontsize=8,
             ha="center", va="bottom")
    px.grid(axis="y", visible=False)
    px.margins(y=0.14)
    return FS.save(fig, "fig2_landmarks_and_prompts", FIGS)


def fig0_stimuli(mode_scene="a014", mode_az=90):
    """
    What the task actually looks like.

    Three rows. The top row is one place in all five stimulus modes at one
    viewpoint, in build order c0 to c4, which is the appearance ladder: both
    cues, then shape alone, then colour alone, then the objects go entirely,
    then the camera drops into the valley. (Other figures order the modes by
    cue rather than by index. Here the index order is the right one, because
    this figure is where the reader learns what c0 to c4 mean.)

    The two lower rows are single trials, a study image and the four candidates
    from a viewpoint 135 degrees away, with each candidate's layout distance to
    the target printed underneath. A reader who has not seen these images cannot
    judge whether 6.8 m of layout separation is a lot or a little, and that
    quantity carries the paper's main claim.

    One trial is c0 and one is c4 because the two read differently. In c0 a
    reader can check the answer by matching a coloured object, and may come away
    thinking the task is about object identity. c4 has no objects and no colour,
    so the only thing separating the target from the distractors is where the
    peaks are, which is the quantity the paper measures. Showing both says the
    question is the same in either case and only the cues differ.
    """
    import matplotlib.image as mpimg

    bpath = os.path.join(SPEC.REPO, SPEC.ARM["benchmark"])
    if not os.path.exists(bpath):
        return None
    with open(bpath) as f:
        trials = json.load(f)["trials"]

    def _trial(mode):
        for t in trials:
            if t["mode"] == mode and t["delta"] == 135:
                return t
        return None

    rows = [(m, _trial(m)) for m in ("c0_shape_colour", "c4_valley")]
    if any(t is None for _, t in rows):
        return None

    def _box(ax, colour, lw):
        # figstyle hides the top and right spines globally; an image frame needs
        # all four or the highlight reads as a stray corner mark.
        for side, sp in ax.spines.items():
            sp.set_visible(True); sp.set_linewidth(lw); sp.set_color(colour)

    def _im(rel):
        full = os.path.join(SPEC.REPO, rel)
        return mpimg.imread(full) if os.path.exists(full) else None

    top = []
    for mode in SPEC.MODES:
        rel = (f"data/scenes_100/{mode}/{mode_scene}/"
               f"{mode_scene}_A_az{mode_az:03d}.png")
        top.append((mode, _im(rel)))
    if any(im is None for _, im in top):
        return None

    # The renders are 640x440. Sizing the figure to that aspect is what stops
    # imshow from leaving a band of white above and below every row.
    h, w = top[0][1].shape[:2]
    cell_w = FS.FULL[0] / 5.0
    row_h = cell_w * (h / w)
    gap, top_pad, bot_pad = 0.56, 0.28, 0.24
    H = 3 * row_h + 2 * gap + top_pad + bot_pad
    fig = plt.figure(figsize=(FS.FULL[0], H))
    gs = fig.add_gridspec(3, 5, hspace=gap / row_h, wspace=0.04,
                          top=1 - top_pad / H, bottom=bot_pad / H,
                          left=0.005, right=0.995)

    for i, (mode, im) in enumerate(top):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        _box(ax, FS.GREY, 0.6)
        ax.set_title(SPEC.MODE_LABELS[mode], fontsize=7, pad=3)

    for row, (mode, trial) in enumerate(rows, start=1):
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(_im(trial["study_image"]))
        ax.set_xticks([]); ax.set_yticks([])
        _box(ax, SPEC.FAMILY_COLOUR["human"], 1.4)
        ax.set_title(f"study  ({trial['study_azimuth']}°)", fontsize=7, pad=3)
        ax.set_xlabel("this place", fontsize=6.5, labelpad=2)

        for i, opt in enumerate(trial["options"]):
            ax = fig.add_subplot(gs[row, i + 1])
            im = _im(opt["image_path"])
            if im is None:
                return None
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            tgt = opt["is_target"]
            _box(ax, FS.OK if tgt else FS.GREY, 1.8 if tgt else 0.6)
            ax.set_title(f"option {i + 1}" + ("  ✓" if tgt else ""),
                         fontsize=7, pad=3,
                         color=FS.OK if tgt else FS.INK)
            ax.set_xlabel("same place" if tgt
                          else f"{opt['layout_distance_m']:.1f} m away",
                          fontsize=6.5, labelpad=2,
                          color=FS.OK if tgt else FS.INK)

    # Row headers sit a fixed distance above each row rather than at hand-tuned
    # figure coordinates, so they stay put if the row height changes.
    heads = ["One place, five stimulus modes"] + [
        f"One trial in {' '.join(SPEC.MODE_LABELS[m].split())}" for m, _ in rows]
    for row, text in enumerate(heads):
        y = 1 - (top_pad * 0.42 + row * (row_h + gap)) / H
        fig.text(0.005, y, text, fontsize=8.5, ha="left", va="center",
                 weight="bold")
    return FS.save(fig, "fig0_stimuli", FIGS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="f0 ... f8")
    args = ap.parse_args()
    FS.use()
    obs, cells, trials = _observers(), _rows("cells.csv"), _rows("trials.csv")
    # fig3_modes and fig6_item_difficulty were retired: the mode heatmap says
    # what Table 2 says, and the item-difficulty curve says what one sentence of
    # Results says.
    plan = [("f0", fig0_stimuli), ("f1", lambda: fig1(obs)),
            ("f2", fig2_landmarks_and_prompts),
            ("f4", lambda: fig4_agreement_and_errors(obs, trials)),
            ("fd", delta_strip), ("fm", mode_strip)]
    for name, fn in plan:
        if args.only in (None, name):
            out = fn()
            print("wrote " + out if out else f"{name} skipped: not enough data")


if __name__ == "__main__":
    main()
