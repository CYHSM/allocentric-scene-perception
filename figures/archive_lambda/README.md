# Superseded: the lambda framing, on the 40-anchor bank

Everything in this directory was produced from `data/scenes/` -- 40 anchors plus
a 120-probe displacement ladder, so a retrieval gallery of 40 and a chance
Recall@1 of **2.5%**. The current results come from `data/scenes_100/`, a
100-anchor gallery with chance **1.0%**.

The two disagree, and not by a little. Recall@1 at 45 degrees for DINOv2 on
`c0_shape_colour` is **24.7%** here and **0.56%** on the 100-scene bank -- about
ten times chance against exactly chance. The 40-anchor bank was drawn before the
canonical landmark set landed, so its scenes differed in *which* landmarks they
held and a model could identify a place by recognising an object in it. That is
the effect these figures are measuring.

The axis labels are the same in both, so these plots are easy to mis-cite. They
are kept only for provenance. Nothing here should appear in the paper.

- `fig2_lambda.png`, `fig3_curves.png` -- lambda(delta), the displacement that
  costs a model as much as a turn. Every entry was off the top of the ladder
  (`> 24 m`), which is why the framing was dropped: a measure that saturates on
  every model and every mode ranks nothing.
- `table1.tex` -- the LaTeX version, c0-c3 only; c4 was never scored on this bank.
- `results.md` -- the summary written against it, including a "c4 valley: (In
  progress on GPU 1)" line for a job that never finished.
