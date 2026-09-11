# Review response — 4MT-VLM workshop submission

**Start here when the reviews arrive.** This file is the entry point; read it
first, then [HANDOVER.md](HANDOVER.md) for how the pipeline and the cluster
work. Nothing else needs reading to get moving.

**Submitted** 11 September 2026, NeurIPS 2026 workshop (2nd Workshop on Embodied
Spatial Reasoning), double-blind, non-archival.
**Title** *4MT-VLM: How Coarse Is a Vision--Language Model's Cognitive Map?*

---

## 0. Read this before touching the manuscript

⚠️ **`paper/tex/main.tex` in this repo is NOT the submitted version.** The last
round of fixes was applied by hand in Overleaf and never came back here. The
repo copy is missing, at least: the delayed-match-to-sample concession, the
c3/c4 inventory fix, the κ and foil-ranking corrections, the bank-labelling
fixes and the rewritten Limitations paragraph.

**First action when you return:** download the final `main.tex` from Overleaf
and overwrite `paper/tex/main.tex`, then `git commit`. Until that happens, every
line number below refers to a file that does not match what the reviewers read.

Everything else in the repo — figures, tables, prompts appendix, collated data —
was regenerated from the code and *is* current.

---

## 1. What was submitted

Four figures, three tables, a prompt appendix. Built by one command:

```bash
bash bench/build_paper.sh      # collate -> tables -> figures -> prompts
```

| output | source |
|---|---|
| `figures/fig0_stimuli.pdf` | Fig 1, five modes + two example trials |
| `figures/fig1_gate_and_scaling.pdf` | Fig 2, gate scatter + scaling ladder |
| `figures/fig2_landmarks_and_prompts.pdf` | Fig 3, instruction sweep + landmark count |
| `figures/fig4_agreement_and_errors.pdf` | Fig 4 (appendix), κ matrix + foil ranks |
| `paper/table1.tex` | main results, 16 models + human |
| `paper/table2.tex` | rotated accuracy by stimulus mode |
| `paper/table3.tex` | accuracy by viewpoint change |
| `paper/prompts.tex` | all seven instructions, generated from `evaluate_vlm.get_prompt_text` |

The headline claim as submitted: models identify a place from the studied
viewpoint and lose it once the camera moves; scale raises the appearance gate
and not rotated accuracy; distractor separation is the only manipulation that
recovers performance, and only for the two closed-weight models.

---

## 2. Numbers already verified — do not recompute

A detailed pre-submission review raised the points below. Each was checked
against `paper/trials.csv` and `results/`. **These are settled; reuse them.**

### Confirmed errors that were fixed before submission

| claim | truth |
|---|---|
| human at 135° | **85%** (17/20). The 82% figure is pooled rotated accuracy, not 135° |
| Gemini 39→84 / GPT 31→52 | averages of the 31.4 m **and** 42.8 m banks |
| Gemini 39→85 / GPT 31→55 | the 31.4 m bank alone — the paper now uses these throughout |
| Qwen ladder "24→27→26→28" | also a two-bank average; mid bank alone is **25→25→28→24** |
| 1092 vs 1112 errors | 1112 total model errors, **20 had no parsable choice** and are excluded from the foil-rank analysis |
| InternVL3.5 gate "25→65%" | actually **35, 25, 35, 55, 50, 65** from 1B to 38B — non-monotonic |
| c3 "no objects" vs "all modes share an inventory" | only c0–c2 carry objects; c3/c4 layout is carried by terrain peaks |

### Analyses run in response to the review

**κ is marginal-bounded, and normalising reverses the claim.** Per-pair κ/κ_max:

| | mean κ | mean κ_max | κ/κ_max |
|---|---|---|---|
| model–model (120 pairs) | 0.182 | 0.812 | **0.218** |
| human–model (16 pairs) | 0.046 | 0.137 | **0.358** |

Relative to its ceiling the human agrees with models *more* than models agree
with each other. The "models resemble each other more than any resembles the
human" claim was removed. Do not reinstate it.

**The foil-rank split is significant, in the unexpected direction.**
329/405/358 over 1092 parsed errors, χ² = 8.08, df = 2, **p = 0.018** — models
over-pick the *middle* foil. In the hard bank the three foils sit at median
6.8 / 9.0 / 11.2, so the ranking has little range. The human's 14 errors picked
the nearest foil 10 times.

**Position bias does NOT explain the 135° dip** (this rebuts the reviewer).
Expected accuracy from each model's own answer frequencies, computed *within*
each Δ:

| Δ | observed | expected from answer bias alone |
|---|---|---|
| 0 | 60.6% | 25.2% |
| 45 | 32.5% | 24.5% |
| 90 | 21.2% | 24.4% |
| **135** | **15.0%** | **24.3%** |
| 180 | 23.1% | 23.9% |

**Rotation direction is single-signed**: every trial is `study_azimuth + Δ`, all
100 trials, no ± ambiguity.

**Appearance gate by mode**, pooled over 16 models: c0 59.4%, c1 68.8%,
c2 62.5%, c3 62.5%, c4 50.0%.

**Distractor bands as percentiles** of all 4,950 pairwise layout distances
(identical across modes): hard = **2nd** percentile, mid = **44th**,
easy = **92nd**. Use this framing if a reviewer objects that metres are
arbitrary in a synthetic world — the manipulation is percentile-defined, so
every comparison is invariant to scene scale.

---

## 3. Open experiments, by value per unit cost

### 3.1 Model arm with the human prompt — do this first

The single strongest objection is that the human received a different wording:
`neutral_anyview` tells the participant that the distractors contain *the same
landmarks arranged differently* and are photographed from the correct option's
direction. The models are told only that the other options are different
landscapes. That is strategy-relevant information sitting on the main dependent
variable.

```bash
bash bench/run_openrouter.sh --model google/gemini-3.8-flash \
  --benchmark data/vlm_benchmark_4afc_hard.json \
  --prompt_style neutral_anyview --n 1 --budget_usd 0.05    # smoke test first
```

~100 calls, on the order of $0.10 at the rates in `results/`. Answers the
objection outright, either way.

### 3.2 Frontier tier

Both closed models are economy tiers (GPT-5.6 **Luna**, Gemini 3.8 **Flash**).
Add GPT-5.6 Sol or Terra and one Claude model. Note Luna already reaches 75%
rotated in the six-landmark bank, so "too coarse" is softer than it looks.

### 3.3 Input-resolution control

All renders are 640×440. Re-render one ladder at 2× and rerun one open model —
needs GPU time, no API spend. Rules out "the input is too coarse" as an
explanation for "the representation is too coarse".

### 3.4 Response-indexing sanity check

Below-chance accuracy is also the signature of a format failure. Add a control
trial where one option *is* the study image, or ask the model to describe each
option before choosing, to show it maps "Option 3" to the third image.

### 3.5 The mirror hypothesis is testable with data in hand

Check whether errors at 135° preferentially select the mirror-consistent foil.
Needs per-scene landmark coordinates; `data/layout_distance.npz` holds only the
pairwise matrix, so this means re-deriving layouts from the Blender bank.

### 3.6 Cheap leftovers

- The other 400 trials were never run (only the 100-trial subset was).
- The instruction sweep ran only on Qwen2.5-VL-32B, which is at chance under
  every condition and so has no power. Rerun on Gemini/GPT, where there is
  headroom.
- Set-size sweep on dgx2 was left at 40/52 cells: InternVL3.5-14B and -30B-A3B
  at N=4, and ten models at N=6. Nothing in the paper depends on it.

---

## 4. Conceded in the submitted Limitations

Do not re-argue these; they are already in the paper.

- Human arm is n=1 with a different prompt, so the gap is an upper bound.
- The clinical 4MT is **delayed** match-to-sample, and the delayed condition is
  the hippocampal one. Our task shows study and options together, so it measures
  viewpoint-invariant matching, not delayed recall. The paper no longer claims
  to index hippocampal function in models.
- Both closed models are economy tiers.
- Untested alternatives: input resolution, response indexing, and the fact that
  separating layouts also separates them in image space.
- Landmark count and layout separation co-vary.
- Stimuli are synthetic.

---

## 5. Bibliography items to check

- `chan2016four` — second author is **Laura Marie Gallaher**, not "Lucy M."
- `zhang2025spinbench` — the ICLR camera-ready may be titled "SpinBench: 3D
  Rotation as a Lens on Spatial Reasoning in VLMs"; arXiv v2 differs.
- `li2025viewspatial` — their own numbers are 33.2% camera frame vs 35.7% other
  frame, both near chance, so do not describe them as competent in the camera
  frame.
- `qiu2026fragility`, `yamamoto2026symmetry` — added late from a literature
  search; **verify both arXiv IDs before any camera-ready.**

---

## 6. Repo state

- `human_task*/images/` deleted (394 MB of copies). Regenerate with
  `bench/build_human_task.py`; the response JSONs are kept and are the only
  irreplaceable part.
- `bench/figure_*.py` are **superseded** by `make_figures.py` and kept only for
  provenance. The live pipeline is `build_paper.sh`.
- `bench/make_prompts.py` is now wired into `build_paper.sh`.
- Two OpenRouter API keys were pasted in chat during development and **must be
  treated as compromised**. Rotate them if not already done.
