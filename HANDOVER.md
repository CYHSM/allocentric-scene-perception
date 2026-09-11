# Handover — allocentric scene perception (4 Mountains for VLMs)

**Branch** `vlm-paper-clean` · **Last worked** 11 September 2026
**Status** Workshop paper submitted 11 September 2026.

> **Coming back after the reviews? Read [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md)
> first.** It carries the submission state, every number already verified
> against a detailed pre-submission review, the open experiments in priority
> order, and the warning that `paper/tex/main.tex` here is *not* the submitted
> version. This file is the infrastructure reference underneath it: how the
> pipeline, the benchmarks, the cluster and the human arm work.

**Claim as submitted:** VLMs identify a place from the studied viewpoint and
lose it once the camera moves. Scale raises the appearance gate and leaves
rotated accuracy flat. Distractor separation is the only manipulation that
recovers it, and only for the two closed-weight models: Gemini 3.8 Flash goes
from 39% to 85% rotated accuracy on the *same trials* when the foils move from
the 2nd to the 44th percentile of layout distance (6.8 m to 31.4 m). A model
with no spatial representation cannot show a 46-point foil effect, so the limit
is the resolution of the representation rather than its absence.

---

## 0. Build the paper

```bash
bash bench/build_paper.sh
```

That is the whole pipeline. It runs three scripts in a fixed order and nothing
else should be run by hand:

| script | does | writes |
|---|---|---|
| `bench/collate.py` | the **only** script that opens a result file. Validates every run against the locked arm, joins each trial's foil geometry, computes accuracy, Wilson intervals, d', VII and the positional-prior controls | `paper/runs.json`, `observers.csv`, `cells.csv`, `trials.csv` |
| `bench/make_tables.py` | Tables 1-3, markdown and LaTeX | `paper/tables.md`, `table[1-3].tex` |
| `bench/make_figures.py` | the four paper figures and the two table image strips | `figures/fig{0,1,2,4}_*.{png,pdf}`, `figures/{delta,mode}_strip/` |
| `bench/make_prompts.py` | the prompt appendix, read straight from the runner | `paper/prompts.tex` |

Supporting modules: `bench/paper_spec.py` (the locked arm and the observer
roster — **the single source of truth**), `bench/stats.py` (Wilson, McNemar,
permutation, Fisher), `bench/vii.py` (d' and VII), `bench/figstyle.py` (one
look for every figure).

The tables and figures never open a result file, so they cannot disagree with
each other about an accuracy. Adding an observer means adding a row to
`OBSERVERS` in `paper_spec.py`; a result file that matches the arm but is in no
row is reported as *unclaimed* rather than silently included, because a
leftover smoke test next to the real run is how a dead result gets back into a
figure. Superseded and smoke-test runs live in `results/archive/`.

`bench/collate.py --check` validates and exits non-zero without writing.

---

## 1. The locked run configuration

Every number that goes in the paper must come from a run with **exactly** these
settings. Nine runs already satisfy them and sit in one table; anything else is
a separate, labelled condition.

| setting | value | why it is not negotiable |
|---|---|---|
| benchmark | `data/vlm_benchmark_4afc_hard.json` | identity-matched hard foils (§3) |
| `--prompt_style` | `cot_anyview` (models), `neutral_anyview` (people) | `cot`/`neutral` assert the answer is rotated, which is **false at Δ=0** (§2.1) |
| `--max_tokens` | `8000` | at 4000, 27 % of Gemini replies truncated mid-thought and scored as errors (§2.2) |
| `--max_trials` | `100`, stratified, seed 0 | the *same* 100 trials for every model and for p01 |
| `--workers` | `8` (API only) | no effect on results; temperature 0, independent calls |

**Known drift in older files.** `results/or_*randomfoils*` ran at `max_tokens
4000`; `results/or_*hardfoils*` used the pre-fix distance metric. Neither is
comparable to the locked set. `data/vlm_benchmark_4afc_hard_v1slice.json`
preserves the exact 100 trials those two runs used, so they remain scoreable.

**Result files do not yet record `max_tokens`, `workers`, or a benchmark hash.**
That is the top open task (§7).

---

## 1b. The paper

`paper/tex/main.tex` + `refs.bib`; compile with `bash paper/tex/build.sh`
(pdflatex + bibtex, three passes). Currently 12 pages: 7.5 of main body,
references, four appendix pages.

**Target venue**: 2nd Workshop on Embodied Spatial Reasoning, NeurIPS 2026.
4--8 pages, references and appendix excluded, so 7.5 fits. Double-blind,
non-archival. It uses `paper/tex/neurips_2026.sty` with the
`dblblindworkshop` option, which is what produces the anonymous author block
and the review line numbers -- both are correct, do not "fix" them. The
workshop site said 5 Sep and the tracker said 11 Sep 11:59 UTC; confirm on
OpenReview.

It `\input`s `paper/table[1-5].tex` and pulls `figures/fig[1-8].pdf`, both of
which `bash bench/build_paper.sh` regenerates -- so **run that first** whenever a
run has landed, then recompile. No number is typed into the .tex by hand except
in the prose, and every one of those appears in a generated table.

**Rounding**: `12.5%` and `22.5%` round *down* in the generated tables (Python
banker's rounding). The prose has to match the table, not round half up, or a
reader comparing the two finds a disagreement. This bit twice.

**Prose style** (the author's standing instruction): no em-dashes, plain words,
short sentences, no filler. It was rewritten once from a first draft that read
as AI-generated; do not reintroduce "moreover", "notably", "underscore",
"comprehensive" or three-part lists.

Every reference in `refs.bib` was checked against the primary source, most
recently on 10 September 2026, when four problems turned up: SpinBench is ICLR
2026 and not a preprint; `yang2024thinking` reports that chain-of-thought,
self-consistency *and* tree-of-thoughts all fail, not just CoT; and three 2026
papers on building allocentric maps were missing (`gu2026spacemind`,
`ruan2026world2mind`, `zhang2026multiview`). Those three motivate the "Building
allocentric representations" paragraph: they all *supply* the representation
externally, which is what makes measuring the internal one the contribution.
Search also confirmed no prior 4MT adaptation for VLMs. Do not add a reference
that has not been checked this way.

Section map: 1 introduction, 2 related work (4MT and mental rotation, VLM
spatial benchmarks, hard negatives, present contribution), 3 methods, 4 results,
5 discussion and limitations, appendix (mode table, agreement, error structure,
mode heatmap).

Two things stay out of the paper deliberately. The prompt bug and the token
ceiling are development history, not results -- the prompt choice is stated once
in Methods as a design decision and the token budget as a parameter. And the
per-observer positional priors are one sentence in Methods ("between 19% and
27% for all seventeen observers"), not two table columns: they are all close to
chance, so per-row prior and excess columns cost space to say nothing. The
columns remain in `paper/observers.csv` for internal checks.

---

## 1c. The landmark-count scene banks (10 September 2026)

`data/scenes_100` holds **four** landmarks per scene, not five -- `n_objects: 4`
in every `bank.json`, in all five modes. An earlier draft of the paper said five
and was wrong.

Four banks now vary that count, all in `c0_shape_colour` only, all otherwise
identical to `scenes_100`:

| tree | landmarks | anchor separation | scenes | size |
|---|---|---|---|---|
| `data/scenes_100` | 4 | 12 m (field absent) | 100 | 4.4 GB, five modes |
| `data/scenes_100_n01` | 1 | 4 m | 100 | 745 MB |
| `data/scenes_100_n02` | 2 | 10 m | 100 | 762 MB |
| `data/scenes_100_n06` | 6 | 12 m | 100 | 802 MB |

Render with `blender/four_mountains/render_n_sweep.sh` (about 20 minutes per
bank, four shards on four GPUs). `data/DATASET.md` is the description written
for a Hugging Face upload and is the file to read first.

### Four constraints, each of which cost a failed launch

**1. Six landmarks is the hard ceiling for c0.** Every scene in a bank is built
from the same canonical landmark set -- that is what makes the task about place
rather than about objects -- and in c0 each landmark must be unique in *both*
shape and colour. The inventory is six shapes and six colours, so
`fm_stimulus.canonical_objects` raises above six. N=8 was attempted and is not
renderable in c0 without either a larger inventory or a weaker rule (unique
`(shape, colour)` pairs, which allows 36 but lets two landmarks share a shape).
The peak modes c3 and c4 have no ceiling; their identity is a continuous form
vector and they sample cleanly at n = 12.

**2. Anchor capacity fails at both ends, not monotonically.** Two anchors must
be at least `separation_m` apart in layout distance to count as different
places. Maximum reachable at the project default of 12 m, measured with the same
greedy packer `fm_bank.sample_anchors` uses:

| N | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| max anchors @ 12 m | 20 | 92 | 100 | 100 | 61 |

N=8 reaches 100 anchors once the floor drops to 10 m (92 at 11 m), so its
capacity is not what blocks it -- the c0 six-landmark ceiling above is.

Few landmarks give too small a layout space; many shrink and crowd onto the
ring so layouts concentrate. `render_bank.py --separation_m` now exposes the
floor and it is recorded per bank as `anchor_separation_m` in `bank.json`.
`scenes_100` predates the field; its value is 12 m. **Read that field before
comparing distances across banks** -- it sets the floor of the scale any
distance-based foil band is measured on.

**3. OptiX is broken on dgx2.** Cycles prefers OPTIX when both backends are
exposed and it fails with `OPTIX_ERROR_INTERNAL_COMPILER_ERROR` loading
`kernel_optix.ptx.zst`. `data/scenes_100` was rendered on CUDA -- `bank100_c4_0.log`
says so -- so the launcher pins `CYCLES_GPU=CUDA`. The first CUDA render of a
session spends about five minutes compiling kernels before the first image
appears; that is not a hang.

**4. Never run two copies of the sweep.** Two copies write the same scene
directories and the same per-shard bank files, and the merge then sees half of
each and writes a tree that looks complete and is not. This happened twice, from
a retry loop that fired more than once. `render_n_sweep.sh` now takes
`flock logs/.n_sweep.lock` and a second copy exits. Do not defeat it.

### Set size is confounded with landmark size

Landmarks shrink to pack onto the ring, so an across-N comparison is partly an
across-object-size comparison:

| N | 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| mean height (m) | 18.7 | 17.7 | 17.9 | 15.1 |
| min spacing (m) | -- | 75 | 47 | 33 |

This is a property of putting more landmarks in a fixed valley, not an artefact
of the sampler, but any claim about set size has to state it.

### The set-size benchmarks (built 10 September 2026)

Done, superseding the note that used to sit here. One benchmark per bank:

    data/vlm_benchmark_setsize_n01.json   n02   n04   n06

Each is 100 trials, c0 only, 1 mode x 5 deltas x 20, so a 100-trial run uses the
whole bank with no subsampling. The answer key is exactly 25/25/25/25, which is
better balanced than the main arm's 16/29/29/26. Distance matrices are
`data/layout_distance_n0*.npz`, one per bank, because the distribution changes
with landmark count.

**Foils are uniform random draws, not a percentile band.** This is the design
decision to preserve. Layout distance is a mean over landmarks, so its
distribution shifts with landmark count and no percentile band means the same
thing at two sizes. No absolute band works either: at N=6, 95 of 100 scenes have
fewer than three candidate foils within 12 m, and the ranges barely overlap
(N=1 tops out at 8.9 m, N=6 starts at 4.7 m). Fixing the *sampling rule* is what
makes the banks comparable. Two banded variants were built and deleted; do not
rebuild them expecting a matched difficulty axis.

The consequence, which must be reported with every result from these banks:
landmark count and layout separation cannot be varied independently. Median
nearest foil is 0.9 / 7.6 / 23.2 / 28.6 m at N = 1 / 2 / 4 / 6. "N=1 is hard"
and "the N=1 foils are 0.9 m away" are one fact.

Collation is `paper/setsize.csv`, written by `collate.setsize_rows()`. It is
deliberately separate from `ladder.csv` (the foil ladder is within-items, these
are different scenes) and these runs stay out of `observers.csv` and Table 1,
because they are not on the locked benchmark. Drawn in the left half of
`fig2_landmarks_and_prompts`.

Results, gate / rotated in percent:

| model | N=1 | N=2 | N=4 | N=6 |
|---|---|---|---|---|
| GPT-5.6 Luna | 100 / 15 | 100 / 26 | 100 / 52 | 100 / 75 |
| Qwen2.5-VL-72B | 30 / 12 | 70 / 18 | 95 / 16 | 100 / 19 |
| InternVL3.5-38B | 55 / 15 | 70 / 18 | 75 / 22 | 75 / 21 |

At N=6 Luna and Qwen2.5-VL-72B have identical 100 % gates and 75 % vs 19 %
rotated on the same 80 trials (exact McNemar p = 4e-12). This is the paper's
strongest single comparison. ### State of the sweep at handover (11 September 2026, early morning)

**39 of 52 cells done, 13 outstanding.** All three models in the paper's figure
(Luna, Qwen2.5-VL-72B, InternVL3.5-38B) are complete across all four banks, so
nothing the paper says is waiting on this. The rest is dataset completeness.

Outstanding: InternVL3.5-8B, -14B and -30B-A3B need N=4 and N=6; the other seven
open models need N=6 only.

    ssh dgx2 'tail -f /raid/nbe_tmp/markus_frey/asp/logs/setsize.log'
    bash bench/sync_results.sh      # pull finished runs, show what is running

`run_setsize.sh` skips any run whose output already exists, so re-running it
after an interruption is safe and resumes where it stopped. If the box was
rebooted, relaunch with:

    ssh dgx2 'cd /raid/nbe_tmp/markus_frey/asp && setsid nohup ./run_setsize.sh \
        > logs/setsize.log 2>&1 < /dev/null'

When it finishes, `bash bench/build_paper.sh` regenerates everything. A 13x4
appendix table is then worth adding: with 12 open models the gate column climbs
with landmark count and the rotated column does not, which is a stronger
statement than the three curves in Figure 4. The two completed columns already
show it:

| | gate (mean, range) | rotated (mean, range) |
|---|---|---|
| N=1, 12 open models | 37 % (15-60) | 20 % (12-26) |
| N=2, 12 open models | 55 % (25-80) | 21 % (16-34) |

One number to be careful with: InternVL3.5-2B scored 34 % rotated at N=2, the
highest of any open model in the sweep. It does not clear chance (27/80, Wilson
[24 %, 45 %], one-sided binomial p = 0.050 uncorrected across 28 comparisons)
and its N=4 cell came back at 25 %. Do not report it as an effect.

---

## 1d. Where the data is, and what draws from it (11 September 2026)

**Everything the paper needs is in this repo.** dgx2 holds the scene renders and
is where runs execute, but every result file the paper reads is local.

### The chain

    results/*.json          one file per (model, benchmark) run, written by evaluate_vlm.py
    human_task_*/           the human arm: index.html, task.json, the saved result
         |
         |  bench/collate.py   <- the ONLY script that opens a result file
         v
    paper/observers.csv     one row per roster entry, on the locked arm
    paper/cells.csv         one row per observer x mode x delta
    paper/trials.csv        long form, one row per trial, foil geometry joined on
    paper/ladder.csv        observer x foil bank (hard/mid/easy), within-items
    paper/setsize.csv       model x landmark-count bank, between-banks
    paper/runs.json         provenance, deviations, unclaimed files
         |
         |  bench/make_figures.py    bench/make_tables.py --latex
         v
    figures/fig*.pdf        paper/table*.tex

`bash bench/build_paper.sh` runs the whole chain. **Run it before recompiling
the paper** whenever a result lands. No figure or table script opens a result
file, so a figure and a table cannot disagree.

### Which output reads which csv

| output | reads |
|---|---|
| fig0_stimuli | the benchmark json + PNGs under `data/scenes_100/` (no csv) |
| (retired) fig3_modes, fig6_item_difficulty, the foil-ladder table and the prompt-sweep table: each said what a surviving table or figure already said |
| fig1_gate_and_scaling | observers.csv, ladder.csv |
| fig4_agreement_and_errors | observers.csv, trials.csv |
| fig2_landmarks_and_prompts | setsize.csv + the six sweep files named in `SPEC.SWEEP_PATH` |
| table1 main | observers.csv |
| table2 modes (appendix) | cells.csv, observers.csv |
| table3 viewpoint | cells.csv, observers.csv |

### Benchmarks

| file | what it is |
|---|---|
| `data/vlm_benchmark_4afc_hard.json` | the locked arm, 500 trials, 5 modes, foils in the 0-10th percentile |
| `..._4afc_mid_m.json`, `..._4afc_easy_m.json` | the same 500 questions, foils redrawn at 40-60th and 90-100th (`--match`) |
| `data/vlm_benchmark_setsize_n0{1,2,4,6}.json` | landmark-count banks, c0 only, 100 trials each, uniform random foils |
| `data/layout_distance.npz`, `..._n0*.npz` | pairwise rotation-optimal layout distance, one per bank |

Everything else in `data/vlm_benchmark_*.json` is superseded (2AFC, `_ctrl`,
`_vhard`, `_v1slice`, the unmatched `_easy`). They are not in `paper_spec.py`
and nothing reads them.

### Syncing from dgx2

    bash bench/sync_results.sh

Pulls `*_n100.json` only and prints what is still running. Smoke tests and
dev-era probes deliberately stay on the box: they carry the arm's benchmark
field, so `collate.unclaimed()` would list all ~40 of them and bury the real
warnings. As of 11 Sep the only remote-only files are smoke tests, a `corrupt/`
directory and pre-paper probes; nothing the paper needs.

### Human tasks

| directory | trials | status |
|---|---|---|
| `human_task_hard/` | 100, locked arm | p01 complete, 86 % |
| `human_task_hard_c34/` | 40, c3+c4 re-collection | merged into the above |
| `human_task_n01/` | 100, the N=1 set-size bank | **built, not yet run** |
| `human_task_hard/superseded/` | earlier partial saves and the stale task | archive, not scanned |

`human_task_n01/index.html` opens in any browser, no server. It is the same 100
trials the models answered on `vlm_benchmark_setsize_n01.json` (verified: 0
items differ, sha `1bba64d10184` stamped into task.json). Save the JSON at the
end and drop it in that directory.

---

## 2. Where the numbers are

Everything comes out of `bash bench/build_paper.sh` (§0). The generated
artefacts:

| file | what it is |
|---|---|
| `paper/tables.md` | Tables 1-4, ready to paste: main, modes, viewpoint curve, foil ladder |
| `paper/table1.tex` ... `table3.tex` | the same, LaTeX |
| `paper/ladder.csv` | one row per observer x foil bank |
| `paper/observers.csv` | one row per observer: accuracy, CI, gate, rotated, d', VII, priors, deviations |
| `paper/cells.csv` | one row per observer x mode x delta |
| `paper/trials.csv` | the long form, with each trial's foil distance joined on — this is what the paired tests read |
| `paper/runs.json` | provenance: config diff, unclaimed files, API spend, git commit |
| `figures/fig1_gate_and_scaling` | left: gate against rotated accuracy. right: Qwen 3B->235B, the gate rises and the walk does not |
| `figures/fig2_landmarks_and_prompts` | left: landmark count, three models. right: the six instruction styles |
| `figures/fig4_agreement_and_errors` | left: Cohen's kappa, rows ordered by similarity. right: of the errors, which of the three foils was picked |

Three earlier figures were retired on 9 September 2026 and are not coming back:

* the **viewpoint curve** is Table 3 -- ten crossing lines over five points
  buried the one fact it carried, that the floor is at 135 degrees and not 180;
* the **foil ladder** is Table 4, for the same reason;
* the **within-band foil-distance panel** showed nothing and could not: that arm
  draws every foil from the hardest decile, so its whole contrast is a few
  metres. Table 4 is the version of that question that works;
* the **positional-prior panel** is the `prior` column of Table 1.

Older one-off figure scripts (`figure_double_dissociation.py`,
`figure_modes_full.py`, `figure_mode_detail.py`, `analyze_vii.py`) still work
and were the exploration; they each open result files directly and each carry
their own copy of `wilson`. They are not part of the paper build. Prefer the
pipeline above; if one of them shows something the pipeline does not, that is a
figure to port, not a script to keep running.

To view the figures: `python3 -m http.server 8765 --directory .` then open
`/figures/`.

### Current table (4AFC, hard foils, chance 25 %, locked config, same 100 trials)

Regenerate rather than trusting this copy — it is a snapshot of
`paper/tables.md` on 9 September 2026.

| observer | acc | 95 % CI | Δ=0 gate | Δ≥45 | prior |
|---|---|---|---|---|---|
| **Human p01** (40 of 100) | **0.88** | [0.74, 0.95] | 100 % (12) | 82 % (28) | 26 % |
| GPT-5.6 Luna | 0.45 | [0.36, 0.55] | **100 % (20)** | 31 % (80) | 25 % |
| Gemini 3.8 Flash | 0.44 | [0.35, 0.54] | 65 % (20) | 39 % (80) | 24 % |
| Qwen2.5-VL-7B | 0.36 | [0.27, 0.46] | 55 % (20) | 31 % (80) | 26 % |
| Qwen3-VL-235B-instruct | 0.33 | [0.25, 0.43] | 90 % (20) | 19 % (80) | 23 % |
| Qwen3-VL-235B-**thinking** | 0.30 | [0.22, 0.40] | 75 % (20) | 19 % (80) | 26 % |
| Qwen2.5-VL-3B | 0.29 | [0.21, 0.39] | 30 % (20) | 29 % (80) | 27 % |
| Qwen2.5-VL-32B *(running)* | 0.28 | [0.19, 0.41] | 83 % (12) | 15 % (48) | 24 % |

*prior* = what the observer's own answer distribution scores with the images
removed. Every observer is at its own prior on the rotated trials except the
human and, marginally, Gemini.

---

## 2b. The 32B/72B audit, and the answer-key defect

Larger Qwen2.5-VL models scoring *below* their own answer prior on rotated
trials looked like a pipeline failure. It is not:

* no reply truncated -- longest is 2 704 characters against an 8 000-token
  ceiling, and the median is 1 670 for 32B, the longest of any model;
* no parse failures; 100/100 replies from 32B and 72B matched the strongest
  extraction pattern (an explicit `Final Answer: Option N`), and hand-checked
  samples parse faithfully;
* the appearance gate is intact -- both are 15/20 at delta = 0, which a broken
  pipeline would have depressed;
* 32B *improves* on the mid foil bank (14/80 -> 22/80 rotated), so model and
  pipeline both work and it is the near foils that defeat it.

**What the audit did find: the 100-trial slice's answer key is not balanced.**
16/29/29/26 overall and 11/22/26/21 on rotated trials, although the 500-trial
bank is exactly 125 each -- `_stratified` balances mode and delta and ignores
the answer slot. Qwen2.5-VL-7B answers "4" on 55% of rotated trials and is
rewarded for it. Table 1's **vs prior** column is the free fix and is the number
to quote; a slice balanced on the answer slot is the real fix and is an open
task.

---

## 3. Two bugs that invalidated earlier results

### 2.1 The prompt asserted a rotation that does not exist at Δ=0

`cot`, `neutral`, `direct`, `anchor`, `birdseye` and `elimination` all say the
matching option is *"viewed from a different viewpoint"*. At Δ=0 the target sits
at the **study's own azimuth**, so the clause is false and instructs the model
to eliminate the correct answer. Gemini did exactly that, in so many words:

> "Option 2 merely reproduces the initial study perspective rather than the
> required rotated viewpoint."

Controlled fix, one clause changed, everything else identical: **5/10 → 10/10**
at Δ=0 (one-sided exact p = 0.031). Use `cot_anyview` / `neutral_anyview`.
`mental_rotation`, `elevation` and `hybrid` were always clean.

This is why the first Gemini run appeared to get *better* with rotation. It was
an instruction-following artifact, and it penalised the *better* model — Qwen is
not strong enough at instruction-following to be hurt by it.

### 2.2 `morphologies` is a stale label field — do not use it as identity

`bank.shard*.json` records `layout["morphologies"]`, written by `sample_layout`
**before** `fm_stimulus.assign_objects` replaces the landforms, and never
updated. It lists 106 distinct strings over 100 scenes describing forms that
were **never rendered**.

All five modes are landmark-inventory controlled. The peaks' actual identity is
`obj` (c0/c1/c2) or the `form` vector (c3/c4), and both resolve to **one shared
set per mode**. Reading `morphologies` produced a false "c3/c4 leak landmark
identity" finding, and worse, made `layout_distance.py` minimise over all 24
permutations for those modes — compressing their distances to a 7.3 m median
against 32.4 m for c0–c2, **from identical peak coordinates**. Fixed: identity
matching everywhere, one distance matrix for all five modes.

---

## 4. The hard-foil benchmark

`build_vlm_benchmark.py` drew distractors with `rng.sample(other_scenes, ...)`,
so a trial's difficulty was a lottery: median random foil **32.4 m** of layout
displacement, with the occasional near-duplicate at 3 m. Gemini's errors were
almost exactly those near-duplicates — failures sat at a mean 17.8° from the
nearest foil against 36.2° for successes (permutation p = 0.0001), and in 12 of
16 failures it chose the single most confusable foil (binomial p = 7.9e-04).

So difficulty is now **selected, not sampled**:

```
python3 bench/layout_distance.py                                  # 4 s
python3 bench/build_hard_benchmark.py --pct 0 10 --out data/vlm_benchmark_4afc_hard.json
python3 bench/build_hard_benchmark.py --pct 0 2  --out data/vlm_benchmark_4afc_vhard.json
```

`D(i,j) = min over φ of mean_k ‖R(φ)p_ik − p_jk‖`, in metres, minimised over
global rotation **because the camera rotates too** — a layout that is the target
turned by φ is the hardest possible foil, not a different place. Peaks
correspond by landmark identity in every mode.

Bands are **per-mode percentiles**, not absolute metres, so a rung means the
same thing everywhere. Every trial records `layout_distance_m` per option and
`min_foil_distance_m`, so difficulty is a regressor rather than a label.

**This gives the probe ladder without Blender.** `probe_ladder: [1,2,4,8,16,24]`
was configured and never rendered; selecting existing pairs by D covers
2.7–48 m, so a psychometric threshold can be fitted per observer from frames
already on disk.

| | random foils | hard (pct 0–10) | vhard (pct 0–2) |
|---|---|---|---|
| nearest foil, median | 32.4 m | 6.8 m | 4.5 m |
| Gemini 3.8 Flash | 61 %* | 44 % | not run |

\* not comparable — that run used `max_tokens 4000` (§0).

---

## 5. What the results say so far

**Every model has a working appearance gate and none keep it through a turn.**
Luna is the sharpest case: **20/20 at Δ=0, 25/80 beyond**. Qwen3-VL-235B-instruct
is 18/20 → 15/80. That dissociation *is* the paper.

**Test-time compute buys nothing here.** Same weights, reasoning on vs off:
Qwen3-VL-235B-**thinking** 0.30 vs **instruct** 0.33, with 2.7× the completion
tokens. Luna matches Gemini's accuracy on a quarter of Gemini's reasoning
tokens.

**The human and the model fail on orthogonal axes** (`figures/fig_double_dissociation.png`).
p01 pays for rotation and not for foil similarity (100 % at Δ≤45 → 5/9 at 180°,
Fisher p = 0.047; flat across foil distance). Gemini pays for foil similarity and
not for rotation (at chance below the 2nd percentile, 90 % above; no delta
effect, p = 0.65). Paired McNemar at 40 trials: p = 0.0013.

**Failure structure separates model families.** Gemini's errors track foil
geometry (§3). *No* open model shows the effect — Qwen 3B/7B/32B/72B and
InternVL3-38B all have permutation p between 0.28 and 0.86 on the random-foil
runs. Their errors carry no information about the stimulus.

**Models are reliably *below* chance at Δ = 135°, and recover at 180°.** Pooled
over the seven models: 18/132 at 135° (13.6 % against 25 % chance, one-sided
binomial p = 0.001), rising to 37/132 (28 %) at 180°. Every individual model is
at or below chance at 135°; the human is 5/5. Below chance is a systematic wrong
answer, not weakness, and the recovery at exactly 180° suggests the models are
matching a mirrored or half-turned appearance — right at 180°, actively wrong at
135°. This is `figures/fig2_delta_curve` and is the sharpest single piece of
evidence that the failure is a heuristic rather than noise.

**Paired against the human's 40 trials, every model is worse** by exact
McNemar: Gemini p = 0.0002, Luna p = 0.0015, Qwen3-VL-235B-instruct p = 1e-6,
Qwen2.5-VL-7B p = 1e-5. Restricting to rotated trials only (n = 28) leaves all
four below p = 0.005.

**The foil-distance effect is the headline, and it splits the field.** Paired
within-items, hard (6.8 m) -> mid (31.4 m), rotated trials only:

| observer | hard | mid | McNemar |
|---|---|---|---|
| Gemini 3.8 Flash | 39 % | 85 % | p = 1.5e-10 |
| GPT-5.6 Luna | 31 % | 55 % | p = 0.002 |
| Qwen2.5-VL-3B | 29 % | 25 % | p = 0.65 |
| Qwen2.5-VL-7B | 31 % | 25 % | p = 0.36 |

**Kappa is symmetric under relabelling**, so kappa-on-correct and
kappa-on-error are the same matrix to machine precision. fig7 is therefore
labelled in the positive direction -- high kappa = this pair succeeds and fails
on the same trials -- and asking "are they *right* together" needs a different
measure, not a different plot of the same one. fig10 is that measure: sort the
80 rotated trials by how many models solved each and compare against a shuffled
null. 11 trials defeat all nine models where 6 are expected; spread of trial
difficulty is 2.0x independence, permutation p = 0.0002.

**They agree with each other about which trials are solvable.** Cohen's kappa on trial-level
correctness: model-model 0.24 mean over 36 pairs (max 0.61, Qwen2.5-VL-72B vs
InternVL3-38B); human-model 0.07 over 9 pairs, none above 0.15; permutation
p = 0.006. Gemini sits outside the block at mean kappa 0.06.

**Modes do not separate.** Once foil selection is identical across modes,
c0–c4 span 0.35–0.50 with every interval overlapping (n=20/mode). The large
per-mode differences seen earlier were the broken distance metric.

---

## 6. The human arm

```
python3 bench/build_human_task.py --benchmark data/vlm_benchmark_4afc_hard.json \
    --match_run 100 --match_seed 0 --prompt_style neutral_anyview --out human_task_hard
python3 -m http.server 8765 --directory .   # open /human_task_hard/index.html
```

`--match_run N` reproduces `evaluate_vlm.py --max_trials N` exactly, so the
person and the models answer **trial for trial the same slice**. Without it the
builder chose its own subset and the pairing was silently broken.

The page autosaves to `localStorage` after every trial and offers **Resume**;
**Export progress** writes a partial file in the model-result schema with
`in_progress: true`. `human_task_hard/human_p01_4afc_partial40.json` is the
current human data — 40 of 100, on the **v1** benchmark (see §0).

**p01 built the scenes and is not naive.** 40/40 → 35/40 establishes the task is
*doable* where a frontier model is at chance; it is not a human ceiling. Naive
participants are needed for that.

---

## 7. Machines

**dgx2** — no scheduler, 20+ users, `/raid` was **99 % full (483 GB)** on
9 Sep 2026. Repo lives at `/raid/nbe_tmp/markus_frey/asp` (**not** a git
checkout — rsync `bench/` and `data/*.json` into it).

```
ssh dgx2 'cd /raid/nbe_tmp/markus_frey/asp && tail -f logs/hard_queue.log'
bash bench/launch_hard_queue.sh    # 5 cached models, shared GPU lock, smoke-tested
```

The queue takes the shared `logs/.vlm_gpu.lock`, smoke-tests 3 trials per model
and skips any model that parses nothing, continues past failures, and skips
already-complete outputs so it is restartable.

**Never edit a shell script while it is executing.** Bash re-reads from a byte
offset; rewriting the file mid-run killed two runs today with
`thon3: command not found`. Python is safe (loaded once at start).

---

## 8. Open tasks, in priority order

**The paper is submitted.** The task list that used to live here has moved to
[REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) §3, which ranks the open experiments by
value per unit cost and records which objections they answer. What remains here
is infrastructure background: §1b the paper, §1c the landmark-count banks, §1d
where every piece of data lives and what draws from it.

0. **Pull the final `main.tex` out of Overleaf** and commit it. The repo copy
   predates the last round of fixes and does not match what was submitted.

0b. **Rotate the OpenRouter keys.** Two were pasted into session transcripts
   (10 and 11 Sep). Total spend across both sessions was about $1.06.

0c. **Run the N=1 human task** if you want a human floor for the set-size axis.
   The `images/` folders were deleted to keep the repo small, so rebuild the
   bundle first with `bench/build_human_task.py`, then open
   `human_task_n01/index.html` and save the JSON into that directory. It is the
   same 100 trials the models answered.


1. ~~Record the full config in every result summary.~~ **Done.**
   `evaluate_vlm.py` now stamps `summary.run_config` with the benchmark path
   and sha256, a hash of the literal prompt text, `max_tokens`, `workers`,
   the trial slice, the reasoning effort, the git commit, and whether the run
   resumed from a pre-existing output file. Runs made before this carry a
   *provenance gap* in `paper/runs.json`; `paper/provenance.json` records what
   the launch script used, with the source named. Each gap closes when that
   observer is next re-run. **Dgx2 has not been synced with this change** — do
   it once the queue is idle, never while it is running.
2. **Finish p01's remaining 60 trials.** The v1/v2 question is settled: the
   human's 40 trials and every model's 100 are the *same* trial ids (the
   100-trial slice of `vlm_benchmark_4afc_hard.json`, verified by
   `collate.py`'s trial-identity check), so the human arm is a proper paired
   subset and no rebuild is needed.
3. ~~Re-run the random-foil condition.~~ **Superseded and done better.** The
   random bank was never comparable to the hard bank: they were built
   independently and 483 of 500 study scenes differ. `build_hard_benchmark.py
   --match` now copies the trial spine and redraws only the foils, so
   `4afc_mid_m` (40-60th pct) and `4afc_easy_m` (90-100th pct) ask the *same*
   questions as `4afc_hard`. Difficulty is within-items and the ladder is
   7 / 31 / 43 m. Runs are in flight for all observers.
4. **Naive human participants** — the ceiling claim needs them.
4b. **Balance the 100-trial slice on the answer slot** (see §2b), and re-run.
    Until then quote Table 1's *vs prior* column, not raw accuracy.
4d. **Build benchmarks on the landmark-count banks** (see §1c). Needs
    `layout_distance.py` then `build_hard_benchmark.py` run per bank, and a
    roster entry per bank in `paper_spec.py`. Nothing exists yet.
4c. **The prompt sweep** (`bench/launch_prompt_sweep.sh`, queued behind the
    ladder on dgx2): Qwen2.5-VL-32B x six prompt styles on the locked
    benchmark. Report rotated trials only -- six of the nine styles assert a
    rotation that is false at delta = 0. The old `results/calib_7b_*.json`
    ablation is unusable: superseded benchmark, unfixed prompt, no control.
5. ~~Clean the data folders.~~ **Done for `results/`**: smoke tests and
   superseded runs moved to `results/archive/` with a README saying why each is
   there, and `collate.py` now reports any file matching the arm that is in no
   roster row. `data/` is deliberately several banks — `4afc` (random foils),
   `4afc_easy` (90-100th pct), `4afc_hard` (0-10th, the paper's arm),
   `4afc_vhard` (0-2nd), `4afc_ctrl` (c0-c2 only) — which is the difficulty
   axis, not clutter. `4afc_hard_v1slice` is the explicit 100-trial slice and
   is what the human task was built from.
6. Optional: a psychometric ladder (`--pct 0 2 / 2 5 / 5 10 / 90 100`) fitted
   per observer, which turns "did you pick a hard enough condition" into a
   measurement.

## 9. OpenRouter

Key is in the shell history, not the repo. `$7.41 of $40` monthly used as of
9 Sep. `bash bench/run_openrouter.sh <model> <n> <budget>`; `WORKERS`,
`MAX_TOKENS`, `PROMPT_STYLE`, `BENCHMARK` are env overrides. Output names now
include the benchmark tag — before that fix, two benchmarks collided on one
filename.

Frontier models are ~$27/100 trials (`gpt-6-astra`, `claude-fable-5.1`) versus
$0.20 for Luna. Eight labs at ~$10.50 total is better value than one frontier
point: DeepSeek has exactly one vision model
(`deepseek-v4-flash-vision-exp`, $0.40); Kimi has `k2.5` ($1.20), `k2.6`
($2.21), `k3` ($7.98).
