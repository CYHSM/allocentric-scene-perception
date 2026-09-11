#!/usr/bin/env python3
"""
The exact instruction text, straight out of the runner, as `paper/prompts.tex`.

The six styles in the sweep are not one sentence apart. Each is a different
instruction of its own, and three of them hand the model a strategy. A reader
cannot judge "no instruction rescues the rotation" from six two-word labels in
a figure, so the appendix carries the wording verbatim.

Generated rather than pasted: a prompt that drifts from the one the runs used
is worse than no appendix at all. Long lines are wrapped to fit the page and
the wrap points are the only thing here that is not literal.
"""

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_spec as SPEC
import evaluate_vlm as E

WIDTH = 84

# The sweep styles, then the wording the human arm and the main model arm read.
BLOCKS = [(s, b) for s, b in SPEC.SWEEP_STYLES] + [
    ("neutral_anyview", "the human arm, and the only wording a person saw")]


def wrap(text):
    out = []
    for line in text.split("\n"):
        if not line.strip():
            out.append("")
            continue
        indent = " " * (len(line) - len(line.lstrip()) + 2)
        out += textwrap.wrap(line, WIDTH, subsequent_indent=indent) or [""]
    return "\n".join(out)


def main():
    n = SPEC.ARM["n_options"]
    parts = [r"\subsection{The instructions, verbatim}",
             r"\label{app:prompts}", "",
             "Every model saw one of these, followed by the study image and "
             "the four options.", "The sweep in Figure~\\ref{fig:landmarks} "
             "runs the first six on Qwen2.5-VL-32B;", "every other run in the "
             "paper uses \\texttt{cot\\_anyview}. Lines are wrapped to fit "
             "the page.", ""]
    for style, blurb in BLOCKS:
        parts += [r"\paragraph{\texttt{" + style.replace("_", r"\_") + "}}"
                  + f" {blurb}.", "",
                  r"\begin{footnotesize}", r"\begin{verbatim}",
                  wrap(E.get_prompt_text(n, style=style)),
                  r"\end{verbatim}", r"\end{footnotesize}", ""]
    out = os.path.join(SPEC.REPO, "paper", "prompts.tex")
    with open(out, "w") as f:
        f.write("\n".join(parts) + "\n")
    print(f"wrote paper/prompts.tex ({len(BLOCKS)} prompts)")


if __name__ == "__main__":
    main()
