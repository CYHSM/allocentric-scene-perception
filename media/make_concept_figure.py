"""Egocentric vs allocentric reference frames, drawn with the same scene model
as the architecture and results figures (make_results_figures.py).

Two observers stand at the rim of the same arena. For each: what they see,
the scene in a body-centred frame (forward is always up), and the scene in a
world-centred frame (north is always up). The egocentric maps differ between
the observers; the allocentric maps are identical.

    python make_concept_figure.py && pdflatex concept_ego_allo.tex
"""
import math
import os

import make_results_figures as M
from make_results_figures import f

# close-up camera, as in the triplet figure, so the cones read at slide size
M.CONE_R, M.CONE_H, M.FOCAL = 3.0, 10.0, 2.35

OBSERVERS = [("1", -90.0), ("2", 35.0)]   # azimuth of the observer around the arena
NAMES = {"A": "pink", "B": "yellow", "C": "purple", "D": "cyan"}
S = 3.3                                   # panel size, cm
K = S / (2 * (M.CAM_D + 5.5))               # cm per world unit, same in both maps
EGO_Y = 0.11 * S                          # observer's height inside the ego panel


def observer(az):
    a = math.radians(az)
    P = (M.CAM_D * math.cos(a), M.CAM_D * math.sin(a))
    h = (-math.cos(a), -math.sin(a))       # heading: towards the arena centre
    r = (h[1], -h[0])                      # right-hand side
    return P, h, r


def to_ego(az, X, Y):
    P, h, r = observer(az)
    d = (X - P[0], Y - P[1])
    return d[0] * r[0] + d[1] * r[1], d[0] * h[0] + d[1] * h[1]


def frame(x0, y0):
    return [r"\begin{scope}", rf"\clip ({f(x0)},{f(y0)}) rectangle ({f(x0+S)},{f(y0+S)});",
            rf"\fill[white] ({f(x0)},{f(y0)}) rectangle ({f(x0+S)},{f(y0+S)});"]


def unframe(x0, y0):
    return [r"\end{scope}", rf"\draw[black!55,line width=0.4pt] ({f(x0)},{f(y0)}) rectangle ({f(x0+S)},{f(y0+S)});"]


def wedge(px, py, ang, rad=0.75, spread=26):
    """The observer: a dot and a light field-of-view wedge pointing along ang (deg)."""
    return [rf"\fill[ink,opacity=0.16] ({f(px)},{f(py)}) -- ++({f(ang-spread)}:{f(rad)}) "
            rf"arc[start angle={f(ang-spread)},end angle={f(ang+spread)},radius={f(rad)}] -- cycle;",
            rf"\filldraw[fill=ink,draw=white,line width=0.6pt] ({f(px)},{f(py)}) circle (0.085);"]


def ego_map(az, name, x0, y0):
    ox, oy = x0 + S / 2, y0 + EGO_Y
    pt = lambda X, Y: (ox + to_ego(az, X, Y)[0] * K, oy + to_ego(az, X, Y)[1] * K)
    cx, cy = pt(0, 0)
    out = frame(x0, y0)
    out += [rf"\fill[rimC!40] ({f(cx)},{f(cy)}) circle ({f(M.RIM*K)});",
            rf"\fill[grassC!45] ({f(cx)},{f(cy)}) circle ({f(M.ARENA*K)});"]
    # body axes, under the objects
    out.append(rf"\draw[egoC,line width=0.9pt,-{{Stealth[length=4pt]}}] ({f(ox)},{f(oy)})--({f(ox)},{f(y0+S-0.12)});")
    out.append(rf"\draw[egoC,line width=0.9pt,{{Stealth[length=4pt]}}-{{Stealth[length=4pt]}}] ({f(x0+0.12)},{f(oy)})--({f(x0+S-0.12)},{f(oy)});")
    for n, X, Y, col in M.OBJECTS:
        x, y = pt(X, Y)
        out.append(rf"\filldraw[fill={col},draw={col}!70!black,line width=0.4pt] ({f(x)},{f(y)}) circle ({f(M.CONE_R*K)});")
    out.append(rf"\node[font=\scriptsize,text=egoC,anchor=north west,inner sep=2pt] at ({f(ox)},{f(y0+S-0.05)}) {{ahead}};")
    out.append(rf"\node[font=\scriptsize,text=egoC,anchor=south east,inner sep=2pt] at ({f(x0+S-0.08)},{f(oy)}) {{right}};")
    out.append(rf"\node[font=\scriptsize,text=egoC,anchor=south west,inner sep=2pt] at ({f(x0+0.08)},{f(oy)}) {{left}};")
    # the focus object, in body coordinates
    fx, fy = [(X, Y) for n, X, Y, c in M.OBJECTS if n == M.FOCUS][0]
    x, y = pt(fx, fy)
    out.append(rf"\draw[ink,line width=0.5pt,densely dashed] ({f(ox)},{f(y)})--({f(x)},{f(y)});")
    out.append(rf"\draw[ink,line width=0.5pt,densely dashed] ({f(x)},{f(oy)})--({f(x)},{f(y)});")
    out += wedge(ox, oy, 90)
    out += unframe(x0, y0)
    return out


def allo_map(az, name, x0, y0):
    cx, cy = x0 + S / 2, y0 + S / 2
    out = frame(x0, y0)
    out += [rf"\fill[rimC!40] ({f(cx)},{f(cy)}) circle ({f(M.RIM*K)});",
            rf"\fill[grassC!45] ({f(cx)},{f(cy)}) circle ({f(M.ARENA*K)});"]
    # world axes through the arena centre, under the objects
    out.append(rf"\draw[alloC,line width=0.9pt,-{{Stealth[length=4pt]}}] ({f(cx)},{f(y0+0.12)})--({f(cx)},{f(y0+S-0.12)});")
    out.append(rf"\draw[alloC,line width=0.9pt,-{{Stealth[length=4pt]}}] ({f(x0+0.12)},{f(cy)})--({f(x0+S-0.12)},{f(cy)});")
    for n, X, Y, col in M.OBJECTS:
        out.append(rf"\filldraw[fill={col},draw={col}!70!black,line width=0.4pt] ({f(cx+X*K)},{f(cy+Y*K)}) circle ({f(M.CONE_R*K)});")
    out.append(rf"\node[font=\scriptsize,text=alloC,anchor=north west,inner sep=2pt] at ({f(cx)},{f(y0+S-0.05)}) {{north}};")
    out.append(rf"\node[font=\scriptsize,text=alloC,anchor=south east,inner sep=2pt] at ({f(x0+S-0.08)},{f(cy)}) {{east}};")
    fx, fy = [(X, Y) for n, X, Y, c in M.OBJECTS if n == M.FOCUS][0]
    out.append(rf"\draw[ink,line width=0.5pt,densely dashed] ({f(cx)},{f(cy+fy*K)})--({f(cx+fx*K)},{f(cy+fy*K)});")
    out.append(rf"\draw[ink,line width=0.5pt,densely dashed] ({f(cx+fx*K)},{f(cy)})--({f(cx+fx*K)},{f(cy+fy*K)});")
    P, h, r = observer(az)
    out += wedge(cx + P[0] * K, cy + P[1] * K, math.degrees(math.atan2(h[1], h[0])))
    out += unframe(x0, y0)
    return out


def describe_ego(az):
    fx, fy = [(X, Y) for n, X, Y, c in M.OBJECTS if n == M.FOCUS][0]
    rx, ay = to_ego(az, fx, fy)
    side = "right" if rx > 0 else "left"
    return rf"{abs(round(rx))} {side}, {round(ay)} ahead"


def describe_allo():
    fx, fy = [(X, Y) for n, X, Y, c in M.OBJECTS if n == M.FOCUS][0]
    ew = "east" if fx > 0 else "west"
    ns = "north" if fy > 0 else "south"
    return rf"{abs(round(fx))} {ew}, {abs(round(fy))} {ns}"


def figure():
    pre = M.PREAMBLE.replace(r"\begin{document}", r"""\definecolor{sky}{HTML}{A9CDE6}
\definecolor{ridgeC}{HTML}{8E8A80}
\definecolor{rimC}{HTML}{B5A58A}
\definecolor{grassC}{HTML}{7FB77E}
\definecolor{objA}{HTML}{E0607E}
\definecolor{objB}{HTML}{F2C14E}
\definecolor{objC}{HTML}{9B6FC4}
\definecolor{objD}{HTML}{29B6C6}
\definecolor{egoC}{HTML}{D9822B}
\definecolor{alloC}{HTML}{3B6EA5}
\begin{document}""")
    out = [pre]
    gap, colgap = 0.95, 0.95
    xs = [0.0, S + colgap, 2 * S + colgap + gap]
    top = 0.0
    # column heads
    out.append(rf"\node[lab,anchor=south,text width={f(S+0.4)}cm,align=center] at ({f(xs[0]+S/2)},{f(top+0.12)}) "
               r"{\textbf{what the observer sees}\\[1pt]{\scriptsize\color{muted}the input}};")
    out.append(rf"\node[lab,anchor=south,text width={f(S+0.4)}cm,align=center] at ({f(xs[1]+S/2)},{f(top+0.12)}) "
               r"{\textbf{\color{egoC}egocentric}\\[1pt]{\scriptsize\color{muted}relative to the observer}};")
    out.append(rf"\node[lab,anchor=south,text width={f(S+0.4)}cm,align=center] at ({f(xs[2]+S/2)},{f(top+0.12)}) "
               r"{\textbf{\color{alloC}allocentric}\\[1pt]{\scriptsize\color{muted}relative to the world}};")
    rowgap = 0.85
    for i, (name, az) in enumerate(OBSERVERS):
        y0 = top - (i + 1) * S - i * rowgap
        out.append(rf"\node[lab,rotate=90,anchor=south] at ({f(xs[0]-0.12)},{f(y0+S/2)}) {{\textbf{{observer {name}}}}};")
        out += M.view_panel(az, xs[0], y0, S)
        out += ego_map(az, name, xs[1], y0)
        out += allo_map(az, name, xs[2], y0)
        # where the pink cone is, in words
        lab = NAMES[M.FOCUS]
        out.append(rf"\node[font=\scriptsize,text=ink,anchor=north] at ({f(xs[1]+S/2)},{f(y0-0.06)}) "
                   rf"{{{lab} cone: {describe_ego(az)}}};")
        out.append(rf"\node[font=\scriptsize,text=ink,anchor=north] at ({f(xs[2]+S/2)},{f(y0-0.06)}) "
                   rf"{{{lab} cone: {describe_allo()}}};")
        # arrow from the view to the two codes
        yy = y0 + S / 2
        out.append(rf"\draw[vcC,line width=0.9pt,-{{Stealth[length=5pt]}}] ({f(xs[0]+S+0.12)},{f(yy)})--({f(xs[1]-0.12)},{f(yy)});")
        # the transformation between the two frames
        for dy, tip in ((0.12, "-"), (-0.12, "<-")):
            out.append(rf"\draw[ink!70,line width=0.9pt,{tip}{{Stealth[length=5pt]}}] ({f(xs[1]+S+0.14)},{f(yy+dy)})--({f(xs[2]-0.14)},{f(yy+dy)});"
                       if tip == "-" else
                       rf"\draw[ink!70,line width=0.9pt,{{Stealth[length=5pt]}}-] ({f(xs[1]+S+0.14)},{f(yy+dy)})--({f(xs[2]-0.14)},{f(yy+dy)});")
    bot = top - 2 * S - rowgap - 0.55
    out.append(rf"\node[font=\footnotesize,text=egoC,anchor=north,text width={f(S+0.6)}cm,align=center] at ({f(xs[1]+S/2)},{f(bot)}) "
               r"{changes whenever the observer moves};")
    out.append(rf"\node[font=\footnotesize,text=alloC,anchor=north,text width={f(S+0.6)}cm,align=center] at ({f(xs[2]+S/2)},{f(bot)}) "
               r"{the same from every viewpoint};")
    out.append(M.END)
    return "".join(s + "\n" for s in out)


if __name__ == "__main__":
    path = os.path.join(M.HERE, "concept_ego_allo.tex")
    with open(path, "w") as fh:
        fh.write("% Generated by make_concept_figure.py.\n" + figure())
    print("wrote", path)
