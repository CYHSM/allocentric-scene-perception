#!/usr/bin/env python3
"""
The paper's tables, from paper/*.csv.

    python3 bench/make_tables.py          # markdown to stdout
    python3 bench/make_tables.py --latex  # also write paper/table*.tex

Reads only what `collate.py` wrote. It does not open a result file, so a table
and a figure cannot disagree about an accuracy.

Table 1 is the headline: one row per observer, accuracy split at the appearance
gate (delta = 0) and the rotated trials (delta >= 45). The gate column is not
decoration -- it is the denominator of every claim in the paper. An observer
that cannot do the task with no viewpoint change has nothing to be
viewpoint-invariant about, and its rotated accuracy means nothing.

Table 2 is the mode breakdown, restricted to rotated trials, because at delta=0
several observers are at ceiling and the modes are indistinguishable there.

Table 3 is the viewpoint curve. It was a line plot and is better as numbers: the
reader wants to compare five cells per row exactly, and the shape carries one
fact -- that 135 degrees is the worst cell, not 180 -- which a table states and a
plot of ten crossing lines buries.

Table 4 is the foil ladder, the same trials at three foil difficulties. Also a
grid of numbers a reader compares cell by cell.
"""

import argparse
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_spec as SPEC
import stats as S

PAPER = os.path.join(SPEC.REPO, "paper")


def _rows(name):
    with open(os.path.join(PAPER, name)) as f:
        return list(csv.DictReader(f))


def _order(o):
    """Human first, then by accuracy."""
    return (0 if o["access"] == "human" else 1, -float(o["acc"]))


def _blocks(observers):
    """
    Observers in two blocks: the human, then every model by accuracy.

    Ascending, so the frontier models sit at the bottom of the table next to
    the human row they are compared against, and the reader's eye ends on the
    comparison the paper is making rather than on a 1B model.
    """
    hum = [o for o in observers if o["access"] == "human"]
    mod = sorted((o for o in observers if o["access"] != "human"),
                 key=lambda o: (float(o["acc"]), o["label"]))
    return [b for b in (hum, mod) if b]


def _flatten(blocks, keep=None):
    """(rows, rules): rules maps a body index to the rule drawn above it."""
    if keep is not None:
        blocks = [[o for o in b if keep(o)] for b in blocks]
        blocks = [b for b in blocks if b]
    rows, rules = [], {}
    for b, block in enumerate(blocks):
        if b:
            # The human is the reference, not a competitor. A rule says so.
            rules[len(rows)] = r"\midrule"
        rows.extend(block)
    return rows, rules


def _bold_worst(body, cols, solo=(), rows=None, tie="last"):
    """
    Bold each row's weakest cell.

    Column-wise emphasis is wrong for these tables: the question is not which
    observer is worst in c3, it is whether a given observer has a cell it falls
    down on, and that comparison runs along the row.

    Ties are the awkward case, and the two tables want opposite things. In the
    mode table a tie is broken in favour of the last column, so the mark lands
    in one place rather than two. In the viewpoint table a tie means the row has
    no single worst viewpoint, which is the thing being claimed, so nothing is
    marked.
    """
    for i, row in enumerate(body):
        if rows is not None and i not in rows:
            continue
        if i in solo:
            for c in cols:
                if _pct(row[c]) is not None:
                    row[c] = f"**{row[c]}**"
            continue
        got = [(c, _pct(row[c])) for c in cols]
        got = [(c, v) for c, v in got if v is not None]
        if not got:
            continue
        worst = min(v for _, v in got)
        hits = [c for c, v in got if v == worst]
        if len(hits) > 1 and tie == "none":
            continue
        for c in ([hits[-1]] if tie == "last" else hits):
            row[c] = f"**{row[c]}**"


def _human_rows(rows):
    return [i for i, o in enumerate(rows) if o["access"] == "human"]


def _pct(cell):
    # Tolerant of the emphasis markers, so a cell can be read back after a
    # bolding pass has run over it.
    m = re.match(r"^(?:\*\*)?(-?\d+(?:\.\d+)?)%", str(cell))
    return float(m.group(1)) if m else None


def _bold_best(body, rules, cols, solo=()):
    """
    Bold the best cell of each column, within each block.

    Blocks of one row are skipped: a single row is trivially its own best, and
    bolding it whole says something the table does not mean. `solo` is the
    exception, for the human row, which is a reference rather than a competitor
    and is marked for that reason.
    """
    bounds = [b for b in sorted(rules) if b] + [len(body)]
    start = 0
    for end in bounds:
        if end - start > 1:
            for c in cols:
                got = {i: _pct(body[i][c]) for i in range(start, end)}
                got = {i: v for i, v in got.items() if v is not None}
                if got:
                    best = max(got.values())
                    for i, v in got.items():
                        if v == best:
                            body[i][c] = f"**{body[i][c]}**"
        start = end
    for i in solo:
        for c in cols:
            if _pct(body[i][c]) is not None and not str(body[i][c]).startswith("**"):
                body[i][c] = f"**{body[i][c]}**"


def table1(observers, include_ineligible=True):
    blocks = _blocks(observers)
    rows, rules = _flatten(blocks)
    head = ["Observer", "Params", "Accuracy", "95% CI",
            "\u0394=0 gate", "\u0394\u226545", "own prior", "vs prior", "Notes"]
    body = []
    for o in rows:
        if not include_ineligible and o["blocking"]:
            continue
        body.append([
            o["label"] + (" *" if o["blocking"] else ""),
            (f"{float(o['params_b']):.0f}B" if o["params_b"] else "\u2014"),
            f"{float(o['acc']):.0%}",
            f"[{float(o['acc_lo']):.0%}, {float(o['acc_hi']):.0%}]",
            (f"{float(o['acc_d0']):.0%}" if int(o["n_d0"] or 0) else "\u2014"),
            (f"{float(o['acc_drot']):.0%}" if int(o["n_drot"] or 0) else "\u2014"),
            f"{float(o['prior_rot']):.0%}" if o["prior_rot"] else "\u2014",
            f"{float(o['excess_rot']):+.0%}" if o["excess_rot"] else "\u2014",
            o["blocking"] or "",
        ])
    # Accuracy, gate, rotated. Not the interval, and not the prior columns,
    # where a high number is not a good one.
    _bold_best(body, rules, [2, 4, 5], solo=_human_rows(rows))
    return head, body, rules


def table2(cells, observers):
    """Rotated-trial accuracy by stimulus mode."""
    # Partial runs stay in, marked. Leaving the human out of the mode table
    # because the arm is 40/100 would remove the only row a reader can compare
    # the models against.
    agg = {}
    for c in cells:
        if int(c["delta"]) == 0:
            continue
        agg.setdefault((c["key"], c["mode"]), [0, 0])
        agg[(c["key"], c["mode"])][0] += int(c["k"])
        agg[(c["key"], c["mode"])][1] += int(c["n"])
    seen = {k for k, _ in agg}
    rows, rules = _flatten(_blocks(observers), keep=lambda o: o["key"] in seen)
    head = ["Observer"] + [SPEC.MODE_LABELS[m].split("  ")[0]
                           for m in SPEC.MODES]
    body = []
    for o in rows:
        row = [o["label"] + (" *" if o["blocking"] else "")]
        for m in SPEC.MODES:
            kk, nn = agg.get((o["key"], m), (0, 0))
            row.append(f"{kk / nn:.0%}" if nn else "\u2014")
        body.append(row)
    _bold_worst(body, list(range(1, len(head))), solo=_human_rows(rows),
                tie="last")

    # The pooled row is the point of the table as much as any single row: it is
    # where c1 separates from the rest. Added after the emphasis pass, so it
    # carries its own bold and not a "worst mode" mark.
    pooled = [[0, 0] for _ in SPEC.MODES]
    n_models = 0
    for o in rows:
        if o["access"] == "human":
            continue
        n_models += 1
        for j, m in enumerate(SPEC.MODES):
            kk, nn = agg.get((o["key"], m), (0, 0))
            pooled[j][0] += kk
            pooled[j][1] += nn
    if n_models and any(n for _, n in pooled):
        rules[len(body)] = r"\midrule"
        body.append([f"**All {n_models} models pooled**"]
                    + [f"**{k / n:.0%}**" if n else "\u2014" for k, n in pooled])
    return head, body, rules


def table3(cells, observers):
    """
    Accuracy by viewpoint change, and where each observer's floor is.

    The floor is the point of the table. If rotation simply degraded a
    representation, accuracy would fall monotonically to its minimum at 180
    degrees, the largest change. It does not. Pooled over the models the minimum
    is at 135 degrees (13%, below the 25% chance level) and accuracy *recovers*
    to 27% at a half turn; six of the nine models have their own worst cell at
    135, and the three that do not bottom out at 90 -- in no case at 180.

    A half-turn view of a roughly bilateral layout can be matched as a mirror
    image: the right answer at 180, an actively wrong one at 135. That is a
    shortcut, not a degradation, and it is what the non-monotonicity is
    evidence for.
    """
    ordered, rules = _flatten(_blocks(observers))
    agg = collections.defaultdict(lambda: [0, 0])
    for c in cells:
        agg[(c["key"], int(c["delta"]))][0] += int(c["k"])
        agg[(c["key"], int(c["delta"]))][1] += int(c["n"])

    head = ["Observer"] + [f"Δ={d}°" for d in SPEC.DELTAS]
    body = []
    pooled = collections.defaultdict(lambda: [0, 0])
    for o in ordered:
        row, vals = [o["label"] + (" *" if o["blocking"] else "")], {}
        for d in SPEC.DELTAS:
            k, n = agg.get((o["key"], d), (0, 0))
            if n:
                vals[d] = k / n
                row.append(f"{k / n:.0%}")
                if o["access"] != "human":
                    pooled[d][0] += k
                    pooled[d][1] += n
            else:
                row.append("—")
        body.append(row)

    if pooled:
        row, vals = ["**All models pooled**"], {}
        for d in SPEC.DELTAS:
            k, n = pooled[d]
            vals[d] = k / n
            row.append(f"**{k / n:.0%}**")
        rules[len(body)] = r"\midrule"
        # Emphasis before the pooled row is added: the pooled row is bold
        # throughout, and a second mark inside it would mean something else.
        _bold_worst(body, list(range(1, len(head))),
                    rows={i for i, o in enumerate(ordered)
                          if o["access"] != "human"}, tie="none")
        body.append(row)
    return head, body, rules


def table4(ladder, observers):
    """Rotated accuracy at each foil difficulty. Empty where a run has not landed."""
    banks = SPEC.BANKS
    rows = collections.defaultdict(dict)
    for r in ladder:
        if r["complete"] == "1" and r["acc_drot"]:
            rows[r["key"]][r["bank"]] = float(r["acc_drot"])
    if not rows:
        return None, None, None
    ordered, rules = _flatten(_blocks(observers),
                              keep=lambda o: o["key"] in rows)
    head = ["Observer"] + [f"{b['key']}  ({b['median_foil_m']:.0f} m)"
                           for b in banks]
    body = []
    for o in ordered:
        row = [o["label"] + (" *" if o["blocking"] else "")]
        for b in banks:
            v = rows[o["key"]].get(b["key"])
            row.append(f"{v:.0%}" if v is not None else "—")
        body.append(row)
    return head, body, rules


def table5():
    """
    One model across instruction styles: is the result about the model or the
    prompt?

    Rotated trials only. Six of the nine styles assert that the test views come
    from a different viewpoint, which is true at every delta >= 45 and false at
    delta = 0; restricting to rotated trials is the subset where every style
    asks the same question. Returns (None, None) until at least two styles have
    finished.
    """
    import json as _json
    rows = []
    for style, blurb in SPEC.SWEEP_STYLES:
        path = os.path.join(SPEC.REPO, SPEC.SWEEP_PATH.format(style=style))
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = _json.load(f)
        trials = d["results"]
        if len(trials) < SPEC.ARM["n_trials"]:
            continue
        rot = [t for t in trials if int(t["delta"]) > 0]
        k = sum(bool(t["is_correct"]) for t in rot)
        p, lo, hi = S.wilson(k, len(rot))
        rows.append([blurb, f"{p:.0%}", f"[{lo:.0%}, {hi:.0%}]",
                     sum(1 for t in trials if t.get("model_choice") is None)])
    if len(rows) < 2:
        return None, None
    return ["Instruction", "Rotated accuracy", "95% CI", "Unparsed"], rows


def md(head, body):
    w = [max(len(str(h)), *(len(str(r[i])) for r in body)) if body else len(str(h))
         for i, h in enumerate(head)]
    out = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(head)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    for r in body:
        out.append("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def latex(head, body, caption, label, rules=None, images=None):
    def esc(v):
        v = str(v)
        # The delta headers go to maths whole, so "=" and the degree sign sit
        # inside one group. Escaping them piecemeal leaves two maths groups
        # abutting, which typesets with a visible seam.
        v = re.sub(r"\u0394\s*=\s*(\d+)\u00b0", r"$\\Delta\\!=\\!\1^\\circ$", v)
        v = re.sub(r"\u0394\s*\u2265\s*(\d+)", r"$\\Delta\\!\\geq\\!\1$", v)
        v = re.sub(r"\u0394\s*=\s*(\d+)", r"$\\Delta\\!=\\!\1$", v)
        v = (v.replace("%", r"\%").replace("\u2265", r"$\geq$")
             .replace("\u0394", r"$\Delta$").replace("\u2014", "--")
             .replace("\u00b0", r"$^\circ$"))
        return re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", v)
    # @{} trims the padding off the outer edges, so the rules line up with the
    # text block instead of hanging into the margin. Columns carrying a picture
    # are centred rather than right aligned: the image is several times wider
    # than "25\%", and a right-aligned number under a centred image reads as a
    # misalignment even though both are inside their column.
    cols = "@{}l" + ("c" if images else "r") * (len(head) - 1) + "@{}"
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\setlength{\tabcolsep}{6pt}",
           r"\begin{tabular}{" + cols + "}", r"\toprule"]
    # An optional strip of pictures above the column headers. A reader should
    # not have to hold "135 degrees" in their head as an abstraction while
    # reading the number underneath it.
    if images:
        cells = [""] + [r"\includegraphics[width=" + w + "]{" + f + "}"
                        for f, w in images]
        out.append(" & ".join(cells) + r" \\[1pt]")
    out += [" & ".join(r"\textbf{" + esc(h) + "}" for h in head) + r" \\",
            r"\midrule"]
    for i, r in enumerate(body):
        if rules and i in rules:
            out.append(rules[i])
        out.append(" & ".join(esc(c) for c in r) + r" \\")
    # A bare % in the caption comments out the closing brace and the whole
    # table fails to compile. The cell escaper does not run on the caption, so
    # escape it here -- but only the percent, since captions carry deliberate
    # LaTeX markup like \emph and $\Delta$.
    caption = re.sub(r"(?<!\\)%", r"\\%", caption)
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{" + caption + "}", r"\label{tab:" + label + "}",
            r"\end{table}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    observers = _rows("observers.csv")
    cells = _rows("cells.csv")
    meta = json.load(open(os.path.join(PAPER, "runs.json")))
    arm = meta["arm"]

    h1, b1, r1 = table1(observers)
    h2, b2, r2 = table2(cells, observers)
    # The star footnote is only true while some run is actually partial. Left
    # unconditional it outlives the run it described and tells a reader to hunt
    # for a mark that is not on the page.
    star = " * marks a partial run." if any(o["blocking"] for o in observers) else ""

    print(f"### Table 1 — {arm['n_options']}AFC, hard foils, "
          f"chance {arm['chance']:.0%}, same {arm['n_trials']} trials\n")
    print(md(h1, b1))
    print(f"\n*own prior* = what this observer's own answer distribution scores "
          f"on the rotated trials with the images removed. *vs prior* is the "
          f"Δ≥45 column minus it, and is the number to read: the answer key of "
          f"the 100-trial slice is 11/22/26/21 on rotated trials rather than "
          f"balanced, so raw accuracy carries a few points of position luck.\n")
    print("### Table 2 — rotated trials (Δ≥45) by stimulus mode\n")
    print(md(h2, b2))
    print("\n\\* partial run; see paper/runs.json for what is missing.")

    h3, b3, r3 = table3(cells, observers)
    print("\n### Table 3 — accuracy by viewpoint change\n")
    print(md(h3, b3))
    def _floor(row):
        vals = [(i, _pct(c)) for i, c in enumerate(row[1:], 1)]
        vals = [(i, v) for i, v in vals if v is not None]
        lo = min(v for _, v in vals)
        hits = [i for i, v in vals if v == lo]
        return SPEC.DELTAS[hits[0] - 1] if len(hits) == 1 else None
    floors = [_floor(r) for r in b3[1:-1]]
    n_135 = sum(1 for f in floors if f == 135)
    n_tied = sum(1 for f in floors if f is None)
    print(f"\nPooled, the floor is Δ=135° — below chance — and accuracy "
          f"*recovers* at a half turn. {n_135} of {len(floors)} models have "
          f"their own worst cell at 135°, and {n_tied} tie for worst across "
          f"two viewpoints.\n")

    h5, b5 = table5()
    if h5:
        print(f"### Table 5 — {SPEC.SWEEP_MODEL} across instruction styles "
              f"(rotated trials, chance 25%)\n")
        print(md(h5, b5))
        print()

    lad = _rows("ladder.csv")
    h4, b4, r4 = table4(lad, observers)
    if h4:
        print("### Table 4 — rotated accuracy at three foil difficulties\n")
        print(md(h4, b4))
        print("\nThe same trials throughout: same study scene, same azimuths, "
              "same answer slot. Only the foils differ.\n")

    gaps = [r for r in meta["runs"] if r.get("provenance_gaps")]
    if gaps:
        print(f"\n{len(gaps)} run(s) carry provenance gaps; see paper/runs.json.")

    if args.latex:
        # The LaTeX main-text table drops three columns the CSV keeps: Notes
        # (prose, wide, and the * marker says the same thing) and the two prior
        # columns. Every observer's positional prior lands between 22% and 27%,
        # so per-row prior and excess columns cost two columns to say what one
        # sentence in the methods says: nobody's answer distribution moves the
        # chance level appreciably.
        def _drop_notes(head, body):
            drop = {"Notes", "own prior", "vs prior"}
            keep = [i for i, h in enumerate(head) if h not in drop]
            return ([head[i] for i in keep],
                    [[r[i] for i in keep] for r in body])
        bold = (" Rows are ordered by overall accuracy. Bold marks the "
                "human row and the best model in each column.")
        with open(f"{PAPER}/table3.tex", "w") as f:
            strip = [(f"delta_strip/delta_{d:03d}", "0.122\\textwidth")
                     for d in SPEC.DELTAS
                     if os.path.exists(f"{SPEC.REPO}/figures/delta_strip/"
                                       f"delta_{d:03d}.pdf")]
            f.write(latex(h3, b3, images=strip if len(strip) == 5 else None,
                          caption="Accuracy by viewpoint change. The images "
                          "above the columns are one c0 scene at each viewpoint "
                          "change, from the same study view. Pooled over the "
                          "models the floor is $\\Delta=135^\\circ$, below chance, "
                          "and accuracy recovers at a half turn. Bold marks "
                          "each model's own worst viewpoint; rows that tie for "
                          "worst are left unmarked." + star,
                          label="delta", rules=r3))
        with open(f"{PAPER}/table1.tex", "w") as f:
            h, b = _drop_notes(h1, b1)
            f.write(latex(h, b,
                          caption=f"Four-alternative forced choice with hard "
                          f"foils ({arm['n_trials']} trials, chance "
                          f"{arm['chance']:.0%}: 20 unrotated and 80 rotated). "
                          f"The $\\Delta=0$ column is the appearance gate. "
                          f"Qwen3-VL-235B (i) is the \\texttt{{instruct}} "
                          f"variant, (t) the \\texttt{{thinking}} one."
                          + bold + star,
                          label="main", rules=r1))
        with open(f"{PAPER}/table2.tex", "w") as f:
            modes = [(f"mode_strip/{m.split('_')[0]}", "0.122\\textwidth")
                     for m in SPEC.MODES
                     if os.path.exists(f"{SPEC.REPO}/figures/mode_strip/"
                                       f"{m.split('_')[0]}.pdf")]
            f.write(latex(h2, b2,
                          images=modes if len(modes) == 5 else None,
                          caption="Rotated-trial accuracy by stimulus mode, 16 "
                          "trials per cell, with one place shown in each mode "
                          "above its column. c0 shape and colour, c1 shape "
                          "only, c2 colour only, c3 bare peaks, c4 valley. "
                          "Rows are ordered by overall accuracy. Bold marks the "
                          "human row and, for each model, its weakest mode; "
                          "ties go to the rightmost." + star,
                          label="modes", rules=r2))
        print("\nwrote paper/table1.tex, table2.tex, table3.tex")


if __name__ == "__main__":
    main()
