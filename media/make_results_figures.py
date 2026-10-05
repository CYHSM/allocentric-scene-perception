"""
Redraw the paper's result figures in the style of architecture_mse.tex.

    python media/make_results_figures.py      # writes media/results_*.tex
    cd media && for f in results_*.tex; do pdflatex $f; done

The numbers come from media/data/*.json, which were digitized from the
published figures (see media/data/README.md) because the original arrays were
not available. Everything is emitted as plain TikZ so each figure is one
self-contained .tex file with its data inlined.
"""

import json
import math
import os
import random

import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

PREAMBLE = r"""\documentclass[border=6pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,decorations.pathreplacing}
\definecolor{spaceC}{HTML}{3B6EA5}
\definecolor{timeC}{HTML}{D9822B}
\definecolor{hcC}{HTML}{6B5B95}
\definecolor{vcC}{HTML}{8A7F72}
\definecolor{ink}{HTML}{2B2B2B}
\definecolor{muted}{HTML}{6B6B6B}
\definecolor{gridC}{HTML}{E6E4E0}
\definecolor{nanC}{HTML}{EFEDEA}
\definecolor{s1}{HTML}{2A78D6}
\definecolor{s2}{HTML}{EB6834}
\definecolor{s3}{HTML}{1BAF7A}
\definecolor{s4}{HTML}{EDA100}
\definecolor{s5}{HTML}{E87BA4}
\begin{document}
\begin{tikzpicture}[font=\footnotesize,
  tick/.style={font=\scriptsize,text=muted},
  lab/.style={font=\footnotesize,text=ink},
  ttl/.style={font=\small\bfseries,anchor=west,text=ink}]
"""
END = "\\end{tikzpicture}\n\\end{document}\n"

# The same region colours as the architecture figures: pre-trained visual
# cortex grey, the time-averaged (spatial) pathway blue, the space-averaged
# pathway orange, hippocampus purple.
REGION = {"Retina": "vcC", "V1": "vcC", "V2": "vcC", "V4": "vcC", "IT": "vcC",
          "PHC": "spaceC", "RSC": "spaceC", "MEC": "spaceC",
          "PRC": "timeC", "LEC": "timeC",
          "CA3": "hcC", "CA1": "hcC"}

VIRIDIS = matplotlib.colormaps["viridis"]


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def f(v):
    return f"{v:.3f}"


# --------------------------------------------------------------- axes ------

def y_axis(x0, x1, ymap, ticks, fmt, label=None, label_x=None):
    out = []
    for t in ticks:
        y = ymap(t)
        out.append(rf"\draw[gridC,line width=0.4pt] ({f(x0)},{f(y)})--({f(x1)},{f(y)});")
        out.append(rf"\node[tick,anchor=east] at ({f(x0-0.08)},{f(y)}) {{{fmt(t)}}};")
    if label:
        ym = (ymap(ticks[0]) + ymap(ticks[-1])) / 2
        out.append(rf"\node[lab,rotate=90,anchor=south] at ({f(label_x)},{f(ym)}) {{{label}}};")
    return out


def boxplot(x, st, colour, ymap, w=0.46, train=None, test=None, seed=0, jitter=0.17):
    """One box: whiskers, IQR box, median, and the recovered points on top."""
    out = []
    lo, q1, med, q3, hi = (ymap(st[k]) for k in ("lo", "q1", "med", "q3", "hi"))
    c = f"{colour}!85!black"
    out.append(rf"\draw[{c},line width=0.6pt] ({f(x)},{f(q3)})--({f(x)},{f(hi)}) ({f(x-w*0.22)},{f(hi)})--({f(x+w*0.22)},{f(hi)});")
    out.append(rf"\draw[{c},line width=0.6pt] ({f(x)},{f(q1)})--({f(x)},{f(lo)}) ({f(x-w*0.22)},{f(lo)})--({f(x+w*0.22)},{f(lo)});")
    out.append(rf"\filldraw[fill={colour}!16,draw={c},line width=0.6pt,rounded corners=0.6pt] ({f(x-w/2)},{f(q1)}) rectangle ({f(x+w/2)},{f(q3)});")
    rng = random.Random(seed)
    for vals, style in ((train or [], "fill=black!50,draw=white,line width=0.25pt"),
                        (test or [], "fill=white,draw=ink,line width=0.35pt")):
        for v in vals:
            dx = (rng.random() - 0.5) * 2 * jitter
            out.append(rf"\filldraw[{style}] ({f(x+dx)},{f(ymap(v))}) circle (0.034);")
    out.append(rf"\draw[ink,line width=1.0pt] ({f(x-w/2)},{f(med)})--({f(x+w/2)},{f(med)});")
    return out


def cells(x0, y0, grid, cw, ch, vmin=None, vmax=None):
    """A ratemap as filled rectangles; row 0 at the top, None drawn as a gap."""
    vals = [v for row in grid for v in row if v is not None]
    vmin = min(vals) if vmin is None else vmin
    vmax = max(vals) if vmax is None else vmax
    n, m = len(grid), len(grid[0])
    out = [rf"\fill[nanC] ({f(x0)},{f(y0-n*ch)}) rectangle ({f(x0+m*cw)},{f(y0)});"]
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v is None:
                continue
            r, g, b, _ = VIRIDIS((v - vmin) / (vmax - vmin + 1e-12))
            col = f"{{rgb,255:red,{int(r*255)};green,{int(g*255)};blue,{int(b*255)}}}"
            out.append(rf"\fill[fill={col}] ({f(x0+j*cw)},{f(y0-(i+1)*ch)}) rectangle ({f(x0+(j+1)*cw+0.002)},{f(y0-i*ch+0.002)});")
    out.append(rf"\draw[black!35,line width=0.3pt] ({f(x0)},{f(y0-n*ch)}) rectangle ({f(x0+m*cw)},{f(y0)});")
    return out


def colourbar(x, y0, h, w=0.14, label="lowest", label_hi="highest"):
    out = []
    steps = 40
    for k in range(steps):
        r, g, b, _ = VIRIDIS(k / (steps - 1))
        col = f"{{rgb,255:red,{int(r*255)};green,{int(g*255)};blue,{int(b*255)}}}"
        out.append(rf"\fill[fill={col}] ({f(x)},{f(y0+k*h/steps)}) rectangle ({f(x+w)},{f(y0+(k+1)*h/steps+0.002)});")
    out.append(rf"\draw[black!35,line width=0.3pt] ({f(x)},{f(y0)}) rectangle ({f(x+w)},{f(y0+h)});")
    out.append(rf"\node[tick,anchor=west] at ({f(x+w+0.05)},{f(y0+h)}) {{{label_hi}}};")
    out.append(rf"\node[tick,anchor=west] at ({f(x+w+0.05)},{f(y0)}) {{{label}}};")
    out.append(rf"\fill[nanC] ({f(x)},{f(y0-0.55)}) rectangle ({f(x+w)},{f(y0-0.41)});")
    out.append(rf"\draw[black!35,line width=0.3pt] ({f(x)},{f(y0-0.55)}) rectangle ({f(x+w)},{f(y0-0.41)});")
    out.append(rf"\node[tick,anchor=west] at ({f(x+w+0.05)},{f(y0-0.48)}) {{not visited}};")
    return out


def legend_points(x, y):
    return [rf"\filldraw[fill=black!50,draw=white,line width=0.25pt] ({f(x)},{f(y)}) circle (0.05);",
            rf"\node[tick,anchor=west] at ({f(x+0.1)},{f(y)}) {{training scenes}};",
            rf"\filldraw[fill=white,draw=ink,line width=0.35pt] ({f(x+2.0)},{f(y)}) circle (0.05);",
            rf"\node[tick,anchor=west] at ({f(x+2.1)},{f(y)}) {{held-out scenes}};"]


# ------------------------------------------------------------ figure 3 -----

def fig_scene_discrimination():
    d = load("fig3_scene_discrimination.json")["layers"]
    names = list(d)
    H = 5.2
    ymap = lambda v: v * H
    dx = 1.05
    xs = {n: 0.7 + i * dx for i, n in enumerate(names)}
    x1 = xs[names[-1]] + 0.7
    out = [PREAMBLE]
    out.append(r"\node[ttl] at (-1.1,5.95) {Which two of three images show the same place? Accuracy of every layer};")
    out += y_axis(0, x1, ymap, [0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda t: f"{t:.1f}".replace("0.", ".") if t not in (0, 1) else str(int(t)),
                  label="triplet accuracy", label_x=-0.72)
    c = ymap(1 / 3)
    out.append(rf"\draw[muted,densely dashed,line width=0.5pt] (0,{f(c)})--({f(x1)},{f(c)});")
    out.append(rf"\node[tick,anchor=south east] at ({f(x1)},{f(c)}) {{chance}};")
    for i, n in enumerate(names):
        out += boxplot(xs[n], d[n], REGION[n], ymap, train=d[n]["train"], test=d[n]["test"], seed=i)
        out.append(rf"\node[lab,anchor=north] at ({f(xs[n])},-0.12) {{{n}}};")
    # region brackets
    def bracket(a, b, text, y=-0.62):
        return [rf"\draw[black!45,line width=0.5pt] ({f(xs[a]-0.3)},{f(y+0.08)})--({f(xs[a]-0.3)},{f(y)})--({f(xs[b]+0.3)},{f(y)})--({f(xs[b]+0.3)},{f(y+0.08)});",
                rf"\node[tick,anchor=north] at ({f((xs[a]+xs[b])/2)},{f(y-0.03)}) {{{text}}};"]
    out += bracket("V1", "IT", "CORnet-Z, pre-trained")
    out += bracket("PHC", "LEC", "parahippocampal, retrosplenial, entorhinal")
    out += bracket("CA3", "CA1", "hippocampus")
    out += legend_points(x1 - 4.1, 5.55)
    out.append(END)
    return "".join(s + "\n" for s in out)


# ----------------------------------------------------------- figure S4 -----

def fig_representations():
    alloc = load("figS4_allocentricity.json")["layers"]
    ro = load("figS4_readout.json")
    out = [PREAMBLE]
    # a: allocentricity
    H = 4.6
    ymap = lambda v: (v + 10) / 20 * H
    names = list(alloc)
    xs = {n: 0.6 + i * 0.95 for i, n in enumerate(names)}
    x1 = xs[names[-1]] + 0.6
    out.append(r"\node[ttl] at (-1.1,5.35) {a\quad Allocentricity of each layer};")
    out += y_axis(0, x1, ymap, [-10, -5, 0, 5, 10], lambda t: f"{t:d}".replace("-", "$-$"), label="allocentricity", label_x=-0.62)
    for i, n in enumerate(names):
        out += boxplot(xs[n], alloc[n], REGION[n], ymap, w=0.5, train=alloc[n]["train"], test=alloc[n]["test"], seed=10 + i, jitter=0.18)
        out.append(rf"\node[lab,anchor=north] at ({f(xs[n])},-0.12) {{{n}}};")
    out += legend_points(xs["PHC"] - 0.2, 4.95)

    # b: linear readout
    ox = x1 + 2.2
    W, H2 = 6.8, 4.6
    layers = ro["layers"]
    lx = {n: ox + 0.35 + i * (W - 0.7) / (len(layers) - 1) for i, n in enumerate(layers)}
    ymap2 = lambda v: (v + 2.5) / 5.0 * H2
    out.append(rf"\node[ttl] at ({f(ox-1.0)},5.35) {{b\quad Linear read-out of scene properties}};")
    out += y_axis(ox, ox + W, ymap2, [-2, -1, 0, 1, 2], lambda t: f"{t:d}".replace("-", "$-$"), label="read-out performance (z-scored)", label_x=ox - 0.55)
    for n in layers:
        out.append(rf"\node[lab,anchor=north] at ({f(lx[n])},-0.12) {{{n}}};")
    marks = {"Object screen": ("s1", "circle"), "Object world": ("s2", "square"), "Object size": ("s3", "diamond"),
             "Camera position": ("s4", "triangle"), "Camera angle": ("s5", "circle")}
    est = ro.get("estimated", {})
    ends = []
    for name, vals in ro["series"].items():
        col, shape = marks[name]
        pts = " -- ".join(f"({f(lx[n])},{f(ymap2(v))})" for n, v in zip(layers, vals))
        out.append(rf"\draw[{col},line width=1.1pt,line join=round] {pts};")
        for n, v in zip(layers, vals):
            x, y = lx[n], ymap2(v)
            fill = "white" if n in est.get(name, []) else col
            if shape == "circle":
                m = rf"\filldraw[fill={fill},draw={col},line width=0.7pt] ({f(x)},{f(y)}) circle (0.075);"
            elif shape == "square":
                m = rf"\filldraw[fill={fill},draw={col},line width=0.7pt] ({f(x-0.065)},{f(y-0.065)}) rectangle ({f(x+0.065)},{f(y+0.065)});"
            elif shape == "diamond":
                m = rf"\filldraw[fill={fill},draw={col},line width=0.7pt] ({f(x)},{f(y+0.09)})--({f(x+0.09)},{f(y)})--({f(x)},{f(y-0.09)})--({f(x-0.09)},{f(y)})--cycle;"
            else:
                m = rf"\filldraw[fill={fill},draw={col},line width=0.7pt] ({f(x)},{f(y+0.09)})--({f(x+0.08)},{f(y-0.055)})--({f(x-0.08)},{f(y-0.055)})--cycle;"
            if name == "Camera angle":
                m = m.replace(f"fill={fill}", "fill=white")
            out.append(m)
        ends.append((ymap2(vals[-1]), name, col))
    # direct labels at the CA1 end, spread so they do not collide
    ends.sort()
    placed = []
    for y, name, col in ends:
        y = max(y, placed[-1] + 0.27) if placed else y
        placed.append(y)
        out.append(rf"\draw[{col},line width=0.6pt] ({f(lx[layers[-1]]+0.12)},{f(y)})--({f(lx[layers[-1]]+0.3)},{f(y)});")
        out.append(rf"\node[tick,text=ink,anchor=west] at ({f(lx[layers[-1]]+0.32)},{f(y)}) {{{name}}};")
    out.append(END)
    return "".join(s + "\n" for s in out)


# ----------------------------------------------------------- figure S5 -----

def fig_lesion_remapping():
    les = load("figS5_lesion.json")["lesioned_percent"]
    rm = load("figS5_remapping.json")["maps"]
    out = [PREAMBLE]
    H = 4.8
    ymap = lambda v: v * H
    names = list(les)
    xs = {n: 0.6 + i * 0.8 for i, n in enumerate(names)}
    x1 = xs[names[-1]] + 0.55
    out.append(r"\node[ttl] at (-1.1,5.55) {a\quad Scene discrimination after lesioning units};")
    out += y_axis(0, x1, ymap, [0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda t: f"{t:.1f}".replace("0.", ".") if t not in (0, 1) else str(int(t)),
                  label="triplet accuracy", label_x=-0.62)
    c = ymap(1 / 3)
    out.append(rf"\draw[muted,densely dashed,line width=0.5pt] (0,{f(c)})--({f(x1)},{f(c)});")
    out.append(rf"\node[tick,anchor=north east] at ({f(x1)},{f(c)}) {{chance}};")
    for i, n in enumerate(names):
        out += boxplot(xs[n], les[n], "hcC", ymap, w=0.42, train=les[n]["train"], test=les[n]["test"], seed=30 + i, jitter=0.15)
        out.append(rf"\node[lab,anchor=north] at ({f(xs[n])},-0.12) {{{n}}};")
    out.append(rf"\node[lab,anchor=north] at ({f((xs[names[0]]+xs[names[-1]])/2)},-0.55) {{units lesioned (\%)}};")
    out += legend_points(xs[names[0]] - 0.2, 5.15)

    # b: remapping
    ox = x1 + 1.6
    cw, ch = 0.155, 0.12
    mw, mh = 15 * cw, 15 * ch
    out.append(rf"\node[ttl] at ({f(ox-0.4)},5.55) {{b\quad The same two units in three scenes}};")
    for r, unit in enumerate(("n1", "n2")):
        top = 4.95 - r * (mh + 0.75)
        out.append(rf"\node[lab,rotate=90,anchor=south] at ({f(ox-0.12)},{f(top-mh/2)}) {{unit {r+1}}};")
        for s in range(3):
            x0 = ox + s * (mw + 0.35)
            out += cells(x0, top, rm[f"{unit}_s{s+1}"], cw, ch)
            if r == 0:
                out.append(rf"\node[lab,anchor=south] at ({f(x0+mw/2)},{f(top+0.03)}) {{scene {s+1}}};")
    out += colourbar(ox + 3 * (mw + 0.35) - 0.1, 4.95 - 2 * mh - 0.75 + 0.1, 1.5)
    out.append(rf"\node[tick,anchor=north] at ({f(ox+1.5*mw+0.35)},{f(4.95-2*mh-0.8)}) {{each map: object position in the arena, $15{{\times}}15$ bins}};")
    out.append(END)
    return "".join(s + "\n" for s in out)


# ------------------------------------------------- layers: accuracy + allocentricity ---

# Processing depth of every layer, laid out the way the network is wired: one
# visual path, then the view-averaged (PH, MEC) and space-averaged (PR, LEC)
# streams side by side, merging in CA3. RSC hangs off MEC as a side branch.
STAGE = {"Retina": (0, 0.0), "V1": (1, 0.0), "V2": (2, 0.0), "V4": (3, -0.14), "IT": (3, 0.14),
         "PHC": (4, -0.14), "PRC": (4, 0.14), "MEC": (5, -0.14), "LEC": (5, 0.14),
         "RSC": (5, 0.48), "CA3": (6, 0.0), "CA1": (7, 0.0)}
EDGES = [("Retina", "V1", "vcC"), ("V1", "V2", "vcC"), ("V2", "V4", "vcC"), ("V2", "IT", "vcC"),
         ("V4", "PHC", "spaceC"), ("IT", "PRC", "timeC"), ("PHC", "MEC", "spaceC"), ("PRC", "LEC", "timeC"),
         ("MEC", "CA3", "spaceC"), ("LEC", "CA3", "timeC"), ("CA3", "CA1", "hcC")]
STAGE_LABEL = ["pixels", "V1", "V2", "V4 $\\cdot$ IT", "PH $\\cdot$ PR", "MEC $\\cdot$ LEC", "CA3", "CA1"]


def layer_panel(stats, ymap, xstage, names, show_points=True):
    out = []
    xy = {}
    for n in names:
        s, off = STAGE[n]
        xy[n] = (xstage(s) + off, ymap(stats[n]["med"]))
    for a, b, col in EDGES:
        if a in xy and b in xy:
            out.append(rf"\draw[{col}!70,line width=0.9pt] ({f(xy[a][0])},{f(xy[a][1])})--({f(xy[b][0])},{f(xy[b][1])});")
    if "RSC" in xy and "MEC" in xy:
        out.append(rf"\draw[spaceC!55,line width=0.7pt,densely dashed] ({f(xy['MEC'][0])},{f(xy['MEC'][1])})--({f(xy['RSC'][0])},{f(xy['RSC'][1])});")
    for n in names:
        x, y = xy[n]
        col = REGION[n]
        q1, q3 = ymap(stats[n]["q1"]), ymap(stats[n]["q3"])
        out.append(rf"\draw[{col},line width=2.4pt,line cap=round,opacity=0.35] ({f(x)},{f(q1)})--({f(x)},{f(q3)});")
        out.append(rf"\filldraw[fill={col},draw=white,line width=0.7pt] ({f(x)},{f(y)}) circle (0.085);")
    return out, xy


def fig_layers():
    acc = load("fig3_scene_discrimination.json")["layers"]
    alc = load("figS4_allocentricity.json")["layers"]
    W = 1.45
    xstage = lambda s: 0.7 + s * W
    x1 = xstage(7) + 0.75
    out = [PREAMBLE]

    # a: accuracy
    H = 4.2
    ymap = lambda v: 3.7 + (v - 0.2) / 0.8 * H     # 0.2..1.0
    out.append(r"\node[ttl] at (-1.2,8.75) {a\quad Scene identity emerges late: which two of three images show the same place?};")
    out += y_axis(0, x1, ymap, [0.2, 0.4, 0.6, 0.8, 1.0], lambda t: f"{t:.1f}", label="triplet accuracy", label_x=-0.72)
    c = ymap(1 / 3)
    out.append(rf"\draw[muted,densely dashed,line width=0.5pt] (0,{f(c)})--({f(x1)},{f(c)});")
    out.append(rf"\node[tick,anchor=north east] at ({f(x1)},{f(c)}) {{chance}};")
    names = list(acc)
    p, xy = layer_panel(acc, ymap, xstage, names)
    out += p
    # sparse direct labels: the end of each stream, the side branch, the summary values
    out.append(rf"\node[tick,text=ink,anchor=west] at ({f(xy['RSC'][0]+0.12)},{f(xy['RSC'][1])}) {{RSC}};")
    out.append(rf"\node[tick,text=ink,anchor=south] at ({f(xy['CA3'][0])},{f(ymap(acc['CA3']['q3'])+0.06)}) {{{acc['CA3']['med']:.2f}}};")
    out.append(rf"\node[tick,text=ink,anchor=south] at ({f(xy['CA1'][0])},{f(ymap(acc['CA1']['q3'])+0.06)}) {{{acc['CA1']['med']:.2f}}};")
    out.append(rf"\node[tick,text=ink,anchor=south] at ({f(xy['IT'][0]+0.05)},{f(ymap(acc['IT']['q3'])+0.06)}) {{{acc['IT']['med']:.2f}}};")
    # bracket for the frozen part
    yb = 8.2
    out.append(rf"\draw[black!40,line width=0.5pt] ({f(xstage(1)-0.3)},{f(yb-0.08)})--({f(xstage(1)-0.3)},{f(yb)})--({f(xstage(3)+0.3)},{f(yb)})--({f(xstage(3)+0.3)},{f(yb-0.08)});")
    out.append(rf"\node[tick,anchor=south] at ({f(xstage(2))},{f(yb)}) {{CORnet-Z, pre-trained}};")
    out.append(rf"\draw[black!40,line width=0.5pt] ({f(xstage(4)-0.3)},{f(yb-0.08)})--({f(xstage(4)-0.3)},{f(yb)})--({f(xstage(7)+0.3)},{f(yb)})--({f(xstage(7)+0.3)},{f(yb-0.08)});")
    out.append(rf"\node[tick,anchor=south] at ({f(xstage(5.5))},{f(yb)}) {{trained with the triplet loss}};")

    # b: allocentricity, same x positions
    H2 = 2.5
    ymap2 = lambda v: (v + 7.5) / 17.5 * H2         # -7.5..10
    out.append(r"\node[ttl] at (-1.2,2.75) {b\quad The same layers become view-invariant};")
    out += y_axis(0, x1, ymap2, [-5, 0, 5, 10], lambda t: f"{t:d}".replace("-", "$-$"), label="allocentricity", label_x=-0.72)
    out.append(rf"\draw[black!45,line width=0.5pt] (0,{f(ymap2(0))})--({f(x1)},{f(ymap2(0))});")
    p2, xy2 = layer_panel(alc, ymap2, xstage, list(alc))
    out += p2
    out.append(rf"\node[tick,text=ink,anchor=west] at ({f(xy2['RSC'][0]+0.12)},{f(xy2['RSC'][1])}) {{RSC}};")
    for i, lab in enumerate(STAGE_LABEL):
        out.append(rf"\node[lab,anchor=north] at ({f(xstage(i))},-0.12) {{{lab}}};")
    out.append(rf"\node[tick,anchor=north] at ({f(xstage(1))},{f(ymap2(2))}) {{not measured}};")
    # legend
    lx, ly = xstage(0) - 0.4, -0.95
    items = [("vcC", "visual cortex"), ("spaceC", "averaged over views (PH, MEC, RSC)"),
             ("timeC", "averaged over space (PR, LEC)"), ("hcC", "hippocampus")]
    for k, (col, txt) in enumerate(items):
        x = lx + (k % 2) * 5.6; y = ly - (k // 2) * 0.32
        out.append(rf"\filldraw[fill={col},draw=white,line width=0.6pt] ({f(x)},{f(y)}) circle (0.075);")
        out.append(rf"\node[tick,text=ink,anchor=west] at ({f(x+0.1)},{f(y)}) {{{txt}}};")
    x = lx + 11.2
    out.append(rf"\draw[black!45,line width=2.4pt,opacity=0.35,line cap=round] ({f(x)},{f(ly-0.12)})--({f(x)},{f(ly+0.12)});")
    out.append(rf"\filldraw[fill=black!45,draw=white,line width=0.6pt] ({f(x)},{f(ly)}) circle (0.075);")
    out.append(rf"\node[tick,text=ink,anchor=west] at ({f(x+0.12)},{f(ly)}) {{median and IQR}};")
    out.append(rf"\node[tick,text=ink,anchor=west] at ({f(x+0.12)},{f(ly-0.32)}) {{over scenes}};")
    out.append(END)
    return "".join(s + "\n" for s in out)


# --------------------------------------- accuracy by layer: three treatments ---

ACC_ORDER = ["Retina", "V1", "V2", "V4", "IT", "PHC", "PRC", "MEC", "LEC", "CA3", "CA1"]


def _acc():
    d = load("fig3_scene_discrimination.json")["layers"]
    return {k: v for k, v in d.items() if k != "RSC"}


def _acc_axes(x1, ymap):
    out = y_axis(0, x1, ymap, [0.2, 0.4, 0.6, 0.8, 1.0], lambda t: f"{t:.1f}", label="triplet accuracy", label_x=-0.72)
    c = ymap(1 / 3)
    out.append(rf"\draw[muted,densely dashed,line width=0.5pt] (0,{f(c)})--({f(x1)},{f(c)});")
    out.append(rf"\node[tick,anchor=north east] at ({f(x1)},{f(c)}) {{chance}};")
    return out


def _stream_legend(x, y, items):
    out = []
    for k, (col, txt) in enumerate(items):
        xx = x + (k % 2) * 5.0; yy = y - (k // 2) * 0.32
        out.append(rf"\filldraw[fill={col},draw=white,line width=0.6pt] ({f(xx)},{f(yy)}) circle (0.075);")
        out.append(rf"\node[tick,text=ink,anchor=west] at ({f(xx+0.1)},{f(yy)}) {{{txt}}};")
    return out


STREAMS = [("vcC", "visual cortex"), ("spaceC", "averaged over views (PH, MEC)"),
           ("timeC", "averaged over space (PR, LEC)"), ("hcC", "hippocampus")]


def fig_layers_opt1():
    """Option 1: the network's wiring, median and IQR per layer."""
    acc = _acc()
    W = 1.45; xstage = lambda s: 0.7 + s * W; x1 = xstage(7) + 0.75
    H = 4.2; ymap = lambda v: (v - 0.2) / 0.8 * H
    out = [PREAMBLE]
    out += _acc_axes(x1, ymap)
    p, xy = layer_panel(acc, ymap, xstage, list(acc))
    out += p
    for n, dy in (("IT", 0.06), ("CA3", 0.06), ("CA1", 0.06)):
        out.append(rf"\node[tick,text=ink,anchor=south] at ({f(xy[n][0])},{f(ymap(acc[n]['q3'])+dy)}) {{{acc[n]['med']:.2f}}};")
    for i, lab in enumerate(STAGE_LABEL):
        out.append(rf"\node[lab,anchor=north] at ({f(xstage(i))},-0.12) {{{lab}}};")
    out += _stream_legend(xstage(0) - 0.4, -0.85, STREAMS)
    out.append(rf"\draw[black!45,line width=2.4pt,opacity=0.35,line cap=round] ({f(xstage(0)+9.6)},{f(-0.97)})--({f(xstage(0)+9.6)},{f(-0.73)});")
    out.append(rf"\filldraw[fill=black!45,draw=white,line width=0.6pt] ({f(xstage(0)+9.6)},-0.85) circle (0.075);")
    out.append(rf"\node[tick,text=ink,anchor=west] at ({f(xstage(0)+9.72)},-0.85) {{median and IQR over scenes}};")
    out.append(END)
    return "".join(s + "\n" for s in out)


def fig_layers_opt2():
    """Option 2: interquartile ribbons along depth, one per stream, merging in CA3."""
    acc = _acc()
    W = 1.45; xstage = lambda s: 0.7 + s * W; x1 = xstage(7) + 0.75
    H = 4.2; ymap = lambda v: (v - 0.2) / 0.8 * H
    st = {n: STAGE[n][0] for n in ACC_ORDER}
    paths = [("vcC", ["Retina", "V1", "V2"]),
             ("spaceC", ["V2", "V4", "PHC", "MEC", "CA3"]),
             ("timeC", ["V2", "IT", "PRC", "LEC", "CA3"]),
             ("hcC", ["CA3", "CA1"])]
    out = [PREAMBLE]
    out += _acc_axes(x1, ymap)
    for col, seq in paths:
        top = [(xstage(st[n]), ymap(acc[n]["q3"])) for n in seq]
        bot = [(xstage(st[n]), ymap(acc[n]["q1"])) for n in reversed(seq)]
        ring = " -- ".join(f"({f(x)},{f(y)})" for x, y in top + bot)
        out.append(rf"\fill[{col},opacity=0.16] {ring} -- cycle;")
    for col, seq in paths:
        line = " -- ".join(f"({f(xstage(st[n]))},{f(ymap(acc[n]['med']))})" for n in seq)
        out.append(rf"\draw[{col},line width=1.6pt,line join=round,line cap=round] {line};")
    for n in ACC_ORDER:
        out.append(rf"\filldraw[fill={REGION[n]},draw=white,line width=0.7pt] ({f(xstage(st[n]))},{f(ymap(acc[n]['med']))}) circle (0.07);")
    # direct labels on the two streams where they separate, and the end value
    out.append(rf"\node[tick,text=spaceC!85!black,anchor=north west] at ({f(xstage(4)+0.05)},{f(ymap(acc['PHC']['med'])-0.05)}) {{PH $\to$ MEC}};")
    out.append(rf"\node[tick,text=timeC!85!black,anchor=south east] at ({f(xstage(4)-0.05)},{f(ymap(acc['PRC']['q3'])+0.02)}) {{PR $\to$ LEC}};")
    out.append(rf"\node[tick,text=ink,anchor=south] at ({f(xstage(7))},{f(ymap(acc['CA1']['q3'])+0.06)}) {{{acc['CA1']['med']:.2f}}};")
    out.append(rf"\node[tick,text=ink,anchor=south] at ({f(xstage(3))},{f(ymap(acc['IT']['q3'])+0.06)}) {{{acc['IT']['med']:.2f}}};")
    for i, lab in enumerate(STAGE_LABEL):
        out.append(rf"\node[lab,anchor=north] at ({f(xstage(i))},-0.12) {{{lab}}};")
    out.append(rf"\fill[black!45,opacity=0.16] ({f(xstage(0)-0.45)},-0.97) rectangle ({f(xstage(0)-0.15)},-0.73);")
    out.append(rf"\draw[black!55,line width=1.6pt] ({f(xstage(0)-0.45)},-0.85)--({f(xstage(0)-0.15)},-0.85);")
    out.append(rf"\node[tick,text=ink,anchor=west] at ({f(xstage(0)-0.1)},-0.85) {{line: median over scenes; band: interquartile range}};")
    out.append(END)
    return "".join(s + "\n" for s in out)


def fig_layers_opt3():
    """Option 3: one row per layer, a lollipop from chance to the median."""
    acc = _acc()
    W = 7.0; xmap = lambda v: (v - 0.2) / 0.8 * W
    rh = 0.42
    names = ACC_ORDER
    y = {n: (len(names) - 1 - i) * rh for i, n in enumerate(names)}
    top = (len(names) - 1) * rh + 0.35
    out = [PREAMBLE]
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(rf"\draw[gridC,line width=0.4pt] ({f(xmap(t))},-0.3)--({f(xmap(t))},{f(top)});")
        out.append(rf"\node[tick,anchor=north] at ({f(xmap(t))},-0.32) {{{t:.1f}}};")
    out.append(rf"\node[lab,anchor=north] at ({f(xmap(0.6))},-0.72) {{triplet accuracy}};")
    c = xmap(1 / 3)
    out.append(rf"\draw[muted,densely dashed,line width=0.5pt] ({f(c)},-0.3)--({f(c)},{f(top)});")
    out.append(rf"\node[tick,anchor=south] at ({f(c)},{f(top)}) {{chance}};")
    # group separators between visual cortex, cortex and hippocampus
    for after in ("IT", "LEC"):
        yy = y[after] - rh / 2
        out.append(rf"\draw[black!25,line width=0.4pt] (-1.25,{f(yy)})--({f(W)},{f(yy)});")
    for n in names:
        col = REGION[n]; yy = y[n]
        q1, med, q3 = (xmap(acc[n][k]) for k in ("q1", "med", "q3"))
        out.append(rf"\draw[{col},line width=0.9pt] ({f(c)},{f(yy)})--({f(med)},{f(yy)});")
        out.append(rf"\draw[{col},line width=3.2pt,opacity=0.3,line cap=round] ({f(q1)},{f(yy)})--({f(q3)},{f(yy)});")
        out.append(rf"\filldraw[fill={col},draw=white,line width=0.7pt] ({f(med)},{f(yy)}) circle (0.085);")
        out.append(rf"\node[lab,anchor=east] at (-0.12,{f(yy)}) {{{n}}};")
        out.append(rf"\node[tick,text=ink,anchor=west] at ({f(q3+0.12)},{f(yy)}) {{{acc[n]['med']:.2f}}};")
    out += _stream_legend(-1.1, -1.45, STREAMS)
    out.append(END)
    return "".join(s + "\n" for s in out)


# ------------------------- allocentricity, with CA1 units' arena maps: three treatments ---

ALLO_ORDER = ["PHC", "PRC", "MEC", "LEC", "CA3", "CA1"]


def _allo():
    d = load("figS4_allocentricity.json")["layers"]
    return {k: d[k] for k in ALLO_ORDER}


def _maps():
    d = load("fig4_single_cells.json")["ratemaps"]
    return d["allo_n1"], d["allo_n2"]


def _map_pair(x0, ytop, cw, ch, gap, vertical=True):
    """The two CA1 example units' arena maps, labelled; returns code and bounding box."""
    m1, m2 = _maps()
    out = []
    w, h = 15 * cw, 15 * ch
    for k, m in enumerate((m1, m2)):
        if vertical:
            xx, yy = x0, ytop - k * (h + gap)
        else:
            xx, yy = x0 + k * (w + gap), ytop
        out += cells(xx, yy, m, cw, ch)
        out.append(rf"\node[tick,text=ink,anchor=south west,inner sep=1pt] at ({f(xx)},{f(yy)}) {{unit {k+1}}};")
    if vertical:
        box = (x0, ytop - 2 * h - gap, x0 + w, ytop)
    else:
        box = (x0, ytop - h, x0 + 2 * w + gap, ytop)
    return out, box


def _arena_note(x, y):
    return [rf"\node[tick,anchor=north,align=center] at ({f(x)},{f(y)}) {{response by object position\\in the arena ($15{{\times}}15$ bins)}};"]


def fig_allo_opt1():
    """Option 1: the trajectory plot, with a call-out from CA1 to two of its units."""
    a = _allo()
    W = 1.6; xs = {"PHC": (0, -0.14), "PRC": (0, 0.14), "MEC": (1, -0.14), "LEC": (1, 0.14), "CA3": (2, 0), "CA1": (3, 0)}
    xstage = lambda s: 0.7 + s * W; x1 = xstage(3) + 0.7
    H = 4.4; ymap = lambda v: (v + 7.5) / 17.5 * H
    out = [PREAMBLE]
    out += y_axis(0, x1, ymap, [-5, 0, 5, 10], lambda t: f"{t:d}".replace("-", "$-$"), label="allocentricity", label_x=-0.72)
    z = ymap(0)
    out.append(rf"\draw[black!45,line width=0.5pt] (0,{f(z)})--({f(x1)},{f(z)});")
    out.append(rf"\node[tick,anchor=south east] at ({f(x1)},{f(z)}) {{view-invariant $\uparrow$}};")
    out.append(rf"\node[tick,anchor=north east] at ({f(x1)},{f(z)}) {{view-dependent $\downarrow$}};")
    xy = {n: (xstage(xs[n][0]) + xs[n][1], ymap(a[n]["med"])) for n in ALLO_ORDER}
    for p, q, col in (("PHC", "MEC", "spaceC"), ("PRC", "LEC", "timeC"), ("MEC", "CA3", "spaceC"), ("LEC", "CA3", "timeC"), ("CA3", "CA1", "hcC")):
        out.append(rf"\draw[{col}!70,line width=0.9pt] ({f(xy[p][0])},{f(xy[p][1])})--({f(xy[q][0])},{f(xy[q][1])});")
    for n in ALLO_ORDER:
        x, y = xy[n]; col = REGION[n]
        out.append(rf"\draw[{col},line width=2.4pt,line cap=round,opacity=0.35] ({f(x)},{f(ymap(a[n]['q1']))})--({f(x)},{f(ymap(a[n]['q3']))});")
        out.append(rf"\filldraw[fill={col},draw=white,line width=0.7pt] ({f(x)},{f(y)}) circle (0.09);")
    for i, lab in enumerate(["PH $\\cdot$ PR", "MEC $\\cdot$ LEC", "CA3", "CA1"]):
        out.append(rf"\node[lab,anchor=north] at ({f(xstage(i))},-0.12) {{{lab}}};")
    # call-out
    mx = x1 + 1.3
    p, (bx0, by0, bx1, by1) = _map_pair(mx, H - 0.05, 0.13, 0.105, 0.42)
    out += p
    ca1 = xy["CA1"]
    out.append(rf"\draw[hcC,line width=0.6pt] ({f(ca1[0]+0.1)},{f(ca1[1])}) .. controls ({f(ca1[0]+0.9)},{f(ca1[1])}) and ({f(bx0-0.9)},{f((by0+by1)/2)}) .. ({f(bx0-0.25)},{f((by0+by1)/2)});")
    out.append(rf"\draw[hcC,line width=0.6pt] ({f(bx0-0.12)},{f(by0)})--({f(bx0-0.25)},{f(by0)})--({f(bx0-0.25)},{f(by1)})--({f(bx0-0.12)},{f(by1)});")
    out += _arena_note((bx0 + bx1) / 2, by0 - 0.08)
    out += colourbar(bx1 + 0.25, by0 + 0.6, 1.3)
    out.append(END)
    return "".join(s + "\n" for s in out)


def fig_allo_opt2():
    """Option 2: ranked rows, CA1's row runs on into its units' maps."""
    a = _allo()
    W = 5.0; xmap = lambda v: (v + 7.5) / 17.5 * W
    rh = 0.5
    y = {n: (len(ALLO_ORDER) - 1 - i) * rh for i, n in enumerate(ALLO_ORDER)}
    top = (len(ALLO_ORDER) - 1) * rh + 0.35
    out = [PREAMBLE]
    for t in (-5, 0, 5, 10):
        out.append(rf"\draw[{'black!45' if t == 0 else 'gridC'},line width={'0.5' if t == 0 else '0.4'}pt] ({f(xmap(t))},-0.3)--({f(xmap(t))},{f(top)});")
        out.append(rf"\node[tick,anchor=north] at ({f(xmap(t))},-0.32) {{{str(t).replace('-', '$-$')}}};")
    out.append(rf"\node[lab,anchor=north] at ({f(xmap(1.25))},-0.72) {{allocentricity}};")
    out.append(rf"\node[tick,anchor=south east] at ({f(xmap(0)-0.05)},{f(top)}) {{$\leftarrow$ view-dependent}};")
    out.append(rf"\node[tick,anchor=south west] at ({f(xmap(0)+0.05)},{f(top)}) {{view-invariant $\rightarrow$}};")
    out.append(rf"\draw[black!25,line width=0.4pt] (-1.0,{f(y['LEC']-rh/2)})--({f(W)},{f(y['LEC']-rh/2)});")
    z = xmap(0)
    for n in ALLO_ORDER:
        col = REGION[n]; yy = y[n]
        q1, med, q3 = (xmap(a[n][k]) for k in ("q1", "med", "q3"))
        out.append(rf"\draw[{col},line width=0.9pt] ({f(z)},{f(yy)})--({f(med)},{f(yy)});")
        out.append(rf"\draw[{col},line width=3.2pt,opacity=0.3,line cap=round] ({f(q1)},{f(yy)})--({f(q3)},{f(yy)});")
        out.append(rf"\filldraw[fill={col},draw=white,line width=0.7pt] ({f(med)},{f(yy)}) circle (0.09);")
        out.append(rf"\node[lab,anchor=east] at (-0.12,{f(yy)}) {{{n}}};")
        txt = f"${a[n]['med']:.1f}$"
        out.append(rf"\node[tick,text=ink,anchor=east] at ({f(W+0.55)},{f(yy)}) {{{txt}}};")
    # CA1 row continues into its units
    cy = y["CA1"]
    mx = W + 1.1
    cw, ch = 0.135, 0.11
    h = 15 * ch
    p, (bx0, by0, bx1, by1) = _map_pair(mx, cy + h - 0.25, cw, ch, 0.3, vertical=False)
    out += p
    out.append(rf"\draw[hcC,line width=0.6pt,densely dotted] ({f(W+0.6)},{f(cy)})--({f(bx0-0.08)},{f(cy)});")
    out += _arena_note((bx0 + bx1) / 2, by0 - 0.08)
    out += colourbar(bx1 + 0.25, by0 + 0.1, h - 0.1)
    for k, (col, txt) in enumerate([("spaceC", "averaged over views (PH, MEC)"), ("timeC", "averaged over space (PR, LEC)"), ("hcC", "hippocampus")]):
        out.append(rf"\filldraw[fill={col},draw=white,line width=0.6pt] (-0.9,{f(-1.25-k*0.3)}) circle (0.075);")
        out.append(rf"\node[tick,text=ink,anchor=west] at (-0.8,{f(-1.25-k*0.3)}) {{{txt}}};")
    out.append(END)
    return "".join(s + "\n" for s in out)


def fig_allo_opt3():
    """Option 3: the circuit itself, each area coloured by its allocentricity."""
    a = _allo()
    # diverging: view-dependent (warm grey-brown) <- 0 -> view-invariant (purple)
    neg, pos = (166, 97, 26), (107, 91, 149)
    def colour(v, vmax=6.0):
        t = max(-1.0, min(1.0, v / vmax))
        base, w = (pos, t) if t > 0 else (neg, -t)
        c = [round(255 + (b - 255) * w) for b in base]
        return f"{{rgb,255:red,{c[0]};green,{c[1]};blue,{c[2]}}}"
    pos_xy = {"PHC": (0, 1.0), "PRC": (0, -0.4), "MEC": (1.9, 1.0), "LEC": (1.9, -0.4), "CA3": (3.8, 0.3), "CA1": (5.5, 0.3)}
    out = [PREAMBLE]
    for p, q, col in (("PHC", "MEC", "spaceC"), ("PRC", "LEC", "timeC"), ("MEC", "CA3", "spaceC"), ("LEC", "CA3", "timeC"), ("CA3", "CA1", "hcC")):
        (x0, y0), (x1, y1) = pos_xy[p], pos_xy[q]
        out.append(rf"\draw[->,{{Stealth[length=1.7mm]}},{col},line width=0.8pt,shorten <=0.5cm,shorten >=0.5cm] ({f(x0)},{f(y0)})--({f(x1)},{f(y1)});".replace("->,{Stealth[length=1.7mm]}", "-{Stealth[length=1.7mm]}"))
    for n, (x, y) in pos_xy.items():
        v = a[n]["med"]
        txt = f"${v:.1f}$"
        out.append(rf"\node[draw={REGION[n]}!85!black,line width=0.9pt,fill={colour(v)},rounded corners=2pt,minimum width=0.95cm,minimum height=0.78cm,align=center,font=\footnotesize] at ({f(x)},{f(y)}) {{\textbf{{{n.replace('PHC','PH').replace('PRC','PR')}}}\\[-1pt]{{\scriptsize {txt}}}}};")
    # scale for the node fill
    sx, sy = 0.0, -1.55
    for k in range(41):
        v = -6 + 12 * k / 40
        out.append(rf"\fill[fill={colour(v)}] ({f(sx+k*0.11)},{f(sy)}) rectangle ({f(sx+(k+1)*0.11+0.002)},{f(sy+0.16)});")
    out.append(rf"\draw[black!35,line width=0.3pt] ({f(sx)},{f(sy)}) rectangle ({f(sx+41*0.11)},{f(sy+0.16)});")
    out.append(rf"\node[tick,anchor=north] at ({f(sx)},{f(sy)}) {{$-6$}};")
    out.append(rf"\node[tick,anchor=north] at ({f(sx+20.5*0.11)},{f(sy)}) {{0}};")
    out.append(rf"\node[tick,anchor=north] at ({f(sx+41*0.11)},{f(sy)}) {{6}};")
    out.append(rf"\node[tick,text=ink,anchor=north] at ({f(sx+20.5*0.11)},{f(sy-0.3)}) {{allocentricity, median over scenes}};")
    out.append(rf"\node[tick,anchor=north west] at ({f(sx)},{f(sy-0.62)}) {{view-dependent}};")
    out.append(rf"\node[tick,anchor=north east] at ({f(sx+41*0.11)},{f(sy-0.62)}) {{view-invariant}};")
    # CA1's units
    x1, y1 = pos_xy["CA1"]
    mx = x1 + 1.35
    p, (bx0, by0, bx1, by1) = _map_pair(mx, y1 + 1.55, 0.13, 0.105, 0.35)
    out += p
    out.append(rf"\draw[hcC,line width=0.6pt] ({f(x1+0.5)},{f(y1)})--({f(bx0-0.3)},{f(y1)});")
    out.append(rf"\draw[hcC,line width=0.6pt] ({f(bx0-0.18)},{f(by0)})--({f(bx0-0.3)},{f(by0)})--({f(bx0-0.3)},{f(by1)})--({f(bx0-0.18)},{f(by1)});")
    out += _arena_note((bx0 + bx1) / 2, by0 - 0.08)
    out += colourbar(bx1 + 0.25, by0 + 0.6, 1.3)
    out.append(END)
    return "".join(s + "\n" for s in out)


# ------------------------------------------------------------ figure 4 -----

def polar(cx, cy, R, theta, r, ring_label):
    out = []
    for k in range(1, 5):
        out.append(rf"\draw[gridC,line width=0.4pt] ({f(cx)},{f(cy)}) circle ({f(R*k/5)});")
    out.append(rf"\draw[black!45,line width=0.4pt] ({f(cx)},{f(cy)}) circle ({f(R)});")
    for a, lab, anch in ((0, r"0\textdegree", "west"), (90, r"90\textdegree", "south"),
                         (180, r"180\textdegree", "east"), (270, r"270\textdegree", "north")):
        x = cx + (R + 0.06) * math.cos(math.radians(a)); y = cy + (R + 0.06) * math.sin(math.radians(a))
        out.append(rf"\node[tick,anchor={anch},inner sep=1pt] at ({f(x)},{f(y)}) {{{lab}}};")
    pts = [(t, v) for t, v in zip(theta, r) if v is not None]
    path = " -- ".join(f"({f(cx+R*v*math.cos(math.radians(t)))},{f(cy+R*v*math.sin(math.radians(t)))})" for t, v in pts)
    out.append(rf"\filldraw[fill=hcC!14,draw=hcC,line width=1.0pt,line join=round] {path} -- cycle;")
    out.append(rf"\node[tick,anchor=south west,inner sep=1pt] at ({f(cx+R*0.72)},{f(cy+R*0.72)}) {{${ring_label}$}};".replace("-", "-"))
    return out



# One scene, used for every schematic panel so they agree with each other:
# four objects in an arena, a grass disc of radius 15 and a rock rim to 20,
# and a distal ridge whose silhouette changes with the view (the global cue).
OBJECTS = [("A", -10.0, -5.0, "objA"), ("B", 6.0, 7.0, "objB"), ("C", 9.0, -4.0, "objC"), ("D", -2.0, 9.5, "objD")]
FOCUS = "A"
ARENA, RIM, CONE_R, CONE_H = 15.0, 20.0, 2.3, 8.0
CAM_D, CAM_H, FOCAL = 31.0, 14.0, 1.85
SCREEN_PX = 128


def camera(az):
    a = math.radians(az)
    P = (CAM_D * math.cos(a), CAM_D * math.sin(a), CAM_H)
    fw = (-P[0], -P[1], -P[2] + 2.0)
    n = math.sqrt(sum(v * v for v in fw)); fw = tuple(v / n for v in fw)
    up = (0.0, 0.0, 1.0)
    r = (fw[1] * up[2] - fw[2] * up[1], fw[2] * up[0] - fw[0] * up[2], fw[0] * up[1] - fw[1] * up[0])
    n = math.sqrt(sum(v * v for v in r)); r = tuple(v / n for v in r)
    u = (r[1] * fw[2] - r[2] * fw[1], r[2] * fw[0] - r[0] * fw[2], r[0] * fw[1] - r[1] * fw[0])
    def proj(X, Y, Z):
        d = (X - P[0], Y - P[1], Z - P[2])
        xc = sum(a * b for a, b in zip(d, r)); yc = sum(a * b for a, b in zip(d, u)); zc = sum(a * b for a, b in zip(d, fw))
        return 0.5 + FOCAL * xc / zc / 2, 0.42 + FOCAL * yc / zc / 2, zc
    return proj


def ring(proj, R, z=0.0, n=96):
    return [proj(R * math.cos(2 * math.pi * k / n), R * math.sin(2 * math.pi * k / n), z)[:2] for k in range(n)]


def poly(pts, x0, y0, s):
    return " -- ".join(f"({f(x0+s*u)},{f(y0+s*v)})" for u, v in pts) + " -- cycle"


def view_panel(az, x0, y0, s):
    """The camera image: sky, distal ridge, rim, grass, cones back to front."""
    proj = camera(az)
    out = [r"\begin{scope}", rf"\clip ({f(x0)},{f(y0)}) rectangle ({f(x0+s)},{f(y0+s)});",
           rf"\fill[sky] ({f(x0)},{f(y0)}) rectangle ({f(x0+s)},{f(y0+s)});"]
    ridge = []
    for k in range(240):
        p = 2 * math.pi * k / 240
        h = 3.0 + 2.2 * math.sin(3 * p + 0.6) + 1.3 * math.sin(7 * p + 1.9) + 0.8 * math.sin(13 * p)
        ridge.append((p, h))
    # only the far half of the ridge is visible behind the arena
    vis = [proj(26 * math.cos(p), 26 * math.sin(p), h) for p, h in ridge]
    vis = [(u, v) for u, v, z in vis if z > CAM_D * 0.75]
    vis.sort()
    if vis:
        pts = [(vis[0][0], -0.2)] + vis + [(vis[-1][0], -0.2)]
        out.append(rf"\fill[ridgeC] {poly(pts, x0, y0, s)};")
    out.append(rf"\fill[rimC] {poly(ring(proj, RIM), x0, y0, s)};")
    out.append(rf"\fill[grassC] {poly(ring(proj, ARENA), x0, y0, s)};")
    cones = []
    for name, X, Y, col in OBJECTS:
        bx, by, z = proj(X, Y, 0); ax, ay, _ = proj(X, Y, CONE_H)
        hw = FOCAL * CONE_R / z / 2
        cones.append((z, col, (bx - hw, by), (ax, ay), (bx + hw, by)))
    for z, col, l, a, r in sorted(cones, reverse=True):
        out.append(rf"\fill[{col}] {poly([l, a, r], x0, y0, s)};")
        out.append(rf"\fill[{col}!70!black,opacity=0.35] {poly([a, r, ((l[0]+r[0])/2, l[1])], x0, y0, s)};")
    out.append(r"\end{scope}")
    out.append(rf"\draw[black!55,line width=0.4pt] ({f(x0)},{f(y0)}) rectangle ({f(x0+s)},{f(y0+s)});")
    return out


def ego_panel(az, x0, y0, s):
    """The same image reduced to what an egocentric code sees: screen position."""
    proj = camera(az)
    out = [r"\begin{scope}", rf"\clip ({f(x0)},{f(y0)}) rectangle ({f(x0+s)},{f(y0+s)});",
           rf"\fill[white] ({f(x0)},{f(y0)}) rectangle ({f(x0+s)},{f(y0+s)});",
           rf"\fill[rimC!45] {poly(ring(proj, RIM), x0, y0, s)};",
           rf"\fill[grassC!45] {poly(ring(proj, ARENA), x0, y0, s)};"]
    feet = []
    for name, X, Y, col in OBJECTS:
        pts = ring(lambda a, b, c: proj(X + a, Y + b, 0), CONE_R * 1.15, n=32)
        cx, cy, z = proj(X, Y, 0)
        feet.append((z, name, col, pts, cx, cy))
    for z, name, col, pts, cx, cy in sorted(feet, reverse=True):
        out.append(rf"\filldraw[fill={col},draw={col}!70!black,line width=0.4pt] {poly(pts, x0, y0, s)};")
    out.append(r"\end{scope}")
    out.append(rf"\draw[black!65,line width=0.5pt] ({f(x0)},{f(y0)}) rectangle ({f(x0+s)},{f(y0+s)});")
    _, _, _, _, cx, cy = [t for t in feet if t[1] == FOCUS][0]
    X, Y = x0 + s * cx, y0 + s * cy
    out.append(rf"\draw[ink,line width=0.6pt] ({f(x0)},{f(Y)})--({f(X)},{f(Y)})--({f(X)},{f(y0)});")
    xa = "south east" if X > x0 + s - 0.9 else "south west"
    out.append(rf"\node[font=\scriptsize,text=ink,anchor={xa},inner sep=1.5pt] at ({f(X)},{f(y0)}) {{$x={round(cx*SCREEN_PX)}$\,px}};")
    out.append(rf"\node[font=\scriptsize,text=ink,anchor=south west,inner sep=1.5pt] at ({f(x0)},{f(Y)}) {{$y={round(cy*SCREEN_PX)}$\,px}};")
    return out


def allo_panel(az, cx0, cy0, s):
    """Top-down arena: identical for every view; only the camera moves."""
    k = s / (2 * (RIM + 6))
    out = [rf"\fill[rimC!45] ({f(cx0)},{f(cy0)}) circle ({f(RIM*k)});",
           rf"\fill[grassC!45] ({f(cx0)},{f(cy0)}) circle ({f(ARENA*k)});"]
    for name, X, Y, col in OBJECTS:
        out.append(rf"\filldraw[fill={col},draw={col}!70!black,line width=0.4pt] ({f(cx0+X*k)},{f(cy0+Y*k)}) circle ({f(CONE_R*1.15*k)});")
    fx, fy = [(X, Y) for n, X, Y, c in OBJECTS if n == FOCUS][0]
    out.append(rf"\draw[ink,line width=0.6pt] ({f(cx0)},{f(cy0)})--({f(cx0+fx*k)},{f(cy0)})--({f(cx0+fx*k)},{f(cy0+fy*k)});")
    out.append(rf"\fill[white] ({f(cx0)},{f(cy0)}) circle (0.045); \draw[ink,line width=0.4pt] ({f(cx0)},{f(cy0)}) circle (0.045);")
    out.append(rf"\node[font=\scriptsize,text=ink,anchor=south,inner sep=1.5pt] at ({f(cx0+fx*k/2)},{f(cy0)}) {{$X={int(fx)}$}};")
    out.append(rf"\node[font=\scriptsize,text=ink,anchor=east,inner sep=1.5pt] at ({f(cx0+fx*k)},{f(cy0+fy*k/2)}) {{$Y={int(fy)}$}};")
    # the camera, outside the rim, looking at the centre
    a = math.radians(az); d = (RIM + 4.5) * k
    px, py = cx0 + d * math.cos(a), cy0 + d * math.sin(a)
    for sgn in (-1, 1):
        b = a + math.pi + sgn * math.radians(24)
        out.append(rf"\draw[black!45,line width=0.4pt] ({f(px)},{f(py)})--++({f(math.cos(b)*0.75)},{f(math.sin(b)*0.75)});")
    out.append(rf"\filldraw[fill=ink,draw=white,line width=0.5pt] ({f(px)},{f(py)}) circle (0.075);")
    return out


def fig_single_cells():
    d = load("fig4_single_cells.json")
    out = [PREAMBLE.replace(r"\begin{document}", r"""\definecolor{sky}{HTML}{A9CDE6}
\definecolor{ridgeC}{HTML}{8E8A80}
\definecolor{rimC}{HTML}{B5A58A}
\definecolor{grassC}{HTML}{7FB77E}
\definecolor{objA}{HTML}{E0607E}
\definecolor{objB}{HTML}{F2C14E}
\definecolor{objC}{HTML}{9B6FC4}
\definecolor{objD}{HTML}{29B6C6}
\begin{document}""")]
    S = 2.75
    cols = [0.0, 3.55, 7.1]
    out.append(r"\node[ttl] at (-1.0,7.55) {One scene, three ways to describe where an object is};")
    heads = ["camera view", "egocentric: on screen", "allocentric: in the arena"]
    for x, h in zip(cols, heads):
        out.append(rf"\node[lab,anchor=south] at ({f(x+S/2)},6.95) {{{h}}};")
    for r, az in enumerate((0, 80)):
        y0 = 6.85 - (r + 1) * S - r * 0.3
        out.append(rf"\node[lab,rotate=90,anchor=south] at (-0.15,{f(y0+S/2)}) {{azimuth {az}\textdegree}};")
        out += view_panel(az, cols[0], y0, S)
        out += ego_panel(az, cols[1], y0, S)
        out += allo_panel(az, cols[2] + S / 2, y0 + S / 2, S)
    out.append(rf"\node[tick,anchor=north,text width={f(S+0.6)}cm,align=center] at ({f(cols[1]+S/2)},0.6) {{changes with every step the camera takes}};")
    out.append(rf"\node[tick,anchor=north,text width={f(S+0.6)}cm,align=center] at ({f(cols[2]+S/2)},0.6) {{the same for every view: only the camera (\tikz\fill[ink] (0,0) circle (0.06);) moves}};")

    # single-unit responses, one column per frame
    out.append(r"\node[ttl] at (-1.0,-0.45) {What single units in the trained model respond to};")
    heads2 = ["azimuth tuning", "by screen position", "by arena position"]
    for x, h in zip(cols, heads2):
        out.append(rf"\node[lab,anchor=south] at ({f(x+S/2)},-1.05) {{{h}}};")
    for r, unit in enumerate(("n1", "n2")):
        top = -1.15 - r * 3.0
        out.append(rf"\node[lab,rotate=90,anchor=south] at (-0.15,{f(top-1.2)}) {{unit {r+1}}};")
        p = d["polar"][unit]
        out += polar(cols[0] + S / 2, top - 1.2, 1.0, p["theta"], p["r"], f"{d['outer_ring_label'][unit]}")
        out += cells(cols[1] + 0.03, top - 0.05, d["ratemaps"][f"ego_{unit}"], 0.09, 0.075)
        out += cells(cols[2] + 0.03, top - 0.05, d["ratemaps"][f"allo_{unit}"], 0.18, 0.15)
    out.append(rf"\node[tick,anchor=north] at ({f(cols[1]+S/2)},-7.55) {{$30{{\times}}30$ screen bins}};")
    out.append(rf"\node[tick,anchor=north] at ({f(cols[2]+S/2)},-7.55) {{$15{{\times}}15$ arena bins}};")
    out += colourbar(cols[2] + S + 0.35, -6.9, 1.5)
    out.append(END)
    return "".join(s + "\n" for s in out)


FIGS = {"results_layers.tex": fig_layers,
        "results_layers_option1.tex": fig_layers_opt1,
        "results_layers_option2.tex": fig_layers_opt2,
        "results_layers_option3.tex": fig_layers_opt3,
        "results_allocentricity_option1.tex": fig_allo_opt1,
        "results_allocentricity_option2.tex": fig_allo_opt2,
        "results_allocentricity_option3.tex": fig_allo_opt3,
        "results_scene_discrimination.tex": fig_scene_discrimination,
        "results_representations.tex": fig_representations,
        "results_lesion_remapping.tex": fig_lesion_remapping,
        "results_single_cells.tex": fig_single_cells}

if __name__ == "__main__":
    for name, fn in FIGS.items():
        header = ("% Generated by make_results_figures.py from media/data/*.json.\n"
                  "% Those numbers are DIGITIZED from the published figures, not the\n"
                  "% original data: see media/data/README.md.\n")
        with open(os.path.join(HERE, name), "w") as fh:
            fh.write(header + fn())
        print("wrote", name)
