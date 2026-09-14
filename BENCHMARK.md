# The benchmark

## The question

Two images of a place can differ for exactly two reasons: **the place changed**,
or **you moved**. A model that represents *place* should be sensitive to the
first and indifferent to the second. Most image benchmarks cannot separate them,
because "the same place" and "the same picture" coincide whenever the viewpoint
is held fixed.

The task is the clinical **Four Mountains Test** (Burgess et al. 2002; Hartley
et al. 2007) turned into a forced choice a model can take:

> A **study** image of a place. Then four **options**, all photographed from one
> bearing and under one lighting condition. Exactly one is the same place. Which?

Two factors are crossed on top of it:

* **Δ, the viewpoint change** — 0°, 45°, 90°, 135°, 180° between the study
  bearing and the option bearing. Δ=0 is the **appearance gate**: same place,
  same bearing, resampled weather. It measures what the model can do when no
  viewpoint change is asked of it.
* **Foil similarity** — how close the wrong options are to the target in layout
  (§3). This used to be an accident; it is now the axis.

The headline claim the design supports: **scaling buys the gate, not the map.**

---

## 1. The dataset: one set of layouts, rendered five ways

100 anchor layouts × 5 stimulus modes × 2 appearances × 8 azimuths = **8 000
frames** (plus instance masks) in `data/scenes_100/`.

| mode | landmarks | world |
|---|---|---|
| `c0_shape_colour` | 4 shapes × colours | bare plane |
| `c1_shape` | 4 shapes, one colour | bare plane |
| `c2_colour` | one shape, 4 colours | bare plane |
| `c3_peaks_bare` | 4 parametric landforms | bare plane |
| `c4_valley` | the same landforms | terrain, lake, vegetation, haze |

**Scene `a007` is the same place in all five modes**, at the same landmark
positions from the same camera — verified: `a000`'s peak coordinates are
byte-identical across c0 and c3. Mode is a within-scene factor, so every
cross-mode comparison is paired. Landmark **size is uniform in every mode**
(`UNIFORM_HEIGHT`, `UNIFORM_RADIUS`), so apparent size stays a pure distance cue.

**Every mode is landmark-inventory controlled.** All 100 scenes in a mode are
built from *one* set of four landmarks, permuted per scene, so only the
arrangement separates two places. `fm_stimulus.assign_objects` applies
`canonical_objects` (c0–c2) or `canonical_forms` (c3–c4).

> ⚠️ **`layout["morphologies"]` and `peak["type"]` are stale labels.**
> `sample_layout` writes them *before* `assign_objects` replaces the landforms,
> and never updates them — they list 106 distinct strings over 100 scenes
> describing forms that were never rendered. **Identity is `obj` (c0–c2) or the
> `form` vector (c3–c4).** Reading the labels instead produced a false "c3/c4
> leak landmark identity" finding and a broken distance metric.

---

## 2. The prompt

`bench/evaluate_vlm.py::get_prompt_text` holds every wording; `build_human_task.py`
reads the same function, so people and models cannot drift apart.

**Use `cot_anyview` for models and `neutral_anyview` for people.**

`cot`, `neutral`, `direct`, `anchor`, `birdseye` and `elimination` all state that
the matching option is *"viewed from a different viewpoint"*. **That is false for
every Δ=0 trial** — the target sits at the study's own azimuth — and it tells the
model to eliminate the correct answer. Gemini 3.8 Flash did so explicitly
("Option 2 merely reproduces the initial study perspective rather than the
required rotated viewpoint") and scored 5/10 at Δ=0. With one clause changed and
nothing else: **10/10** (one-sided exact p = 0.031).

`--max_tokens 8000`. At 4000, 27 % of Gemini's replies were cut off mid-thought
and scored as errors rather than as missing data; those trials scored 0.37
against 0.70 for well-formed ones.

Reasoning models are asked for `reasoning: {effort, exclude: false}` and the
answer is read from the trace if `content` never arrives. Disabling reasoning is
not always allowed — Gemini 3.8 Flash returns *"Reasoning is mandatory for this
endpoint"*.

---

## 3. Foil selection: difficulty is chosen, not sampled

`build_vlm_benchmark.py` drew distractors uniformly at random, so a trial's
difficulty was a lottery: the median random foil is **32.4 m** of layout
displacement away, but the occasional draw lands at 3 m. Those near-duplicates
were almost exactly where the strongest model failed — Gemini's errors sat at a
mean 17.8° from the nearest foil against 36.2° for its successes (permutation
p = 0.0001), and it chose the *single* most confusable foil in 12 of 16 failures
(binomial p = 7.9e-04).

### The distance

```
D(i, j) = min over φ  of  mean over peaks k  ‖ R(φ) p_ik − p_jk ‖      [metres]
```

* **Minimised over global rotation φ** because the camera rotates too: a layout
  that is the target turned by φ is the *hardest possible* foil, not a different
  place.
* **Peaks correspond by landmark identity in every mode.** Minimising over
  permutations instead — as an earlier version did for c3/c4 — finds alignments
  identity matching forbids, and compressed those modes to a 7.3 m median
  against 32.4 m **from identical coordinates**.
* All five modes therefore share **one** distance matrix.

`python3 bench/layout_distance.py` → `data/layout_distance.npz` (4 s).

### The bands

`bench/build_hard_benchmark.py --pct LO HI` selects foils from a percentile band
of each mode's own distance distribution. **Percentiles, not metres**, so a rung
means the same thing in every mode.

| file | band | nearest foil, median |
|---|---|---|
| `data/vlm_benchmark_4afc.json` | random (legacy) | 32.4 m |
| `data/vlm_benchmark_4afc_hard.json` | pct 0–10 | 6.8 m |
| `data/vlm_benchmark_4afc_vhard.json` | pct 0–2 | 4.5 m |

Every trial records `layout_distance_m` per option and `min_foil_distance_m`, so
difficulty is a **regressor**, not a label. The answer key is exactly balanced
(125/125/125/125), so the best-fixed-answer baseline is exactly 0.250.

**Feasibility.** 4AFC needs three in-band foils for the *same* target. At the
2nd percentile ~32 scenes per mode qualify (20 distinct targets per cell, no
reuse); at the 1st percentile only ~10, which is too few.

**This replaces the unrendered probe ladder.** `probe_ladder: [1,2,4,8,16,24]`
metres was configured and never rendered. Selecting existing pairs by D covers
2.7–48 m, so a psychometric threshold can be fitted per observer with no Blender
time at all.

---

## 4. Metrics

**Report two numbers, not one.** The claim is a *dissociation*, and no single
scalar can express one.

* **d′(0) — the gate.** Can the model tell places apart when nothing rotates?
  Comparable across 2AFC/4AFC/retrieval, which raw accuracy is not.
* **VII(Δ) = d′(Δ) / d′(0) — the invariance.** How much of the gate survives the
  turn. `bench/vii.py`, m-AFC d′ by Gauss–Hermite inversion.

**Cohen's κ is not used.** It corrects for chance but still collapses "can it
tell places apart" and "does that survive rotation" into one number.

**Two floors that must be stated wherever VII appears:**

1. **VII is undefined when d′(0) = 0.** Gemini scored exactly chance at Δ=0 on
   one run, so the model with the most apparent invariance was the one VII could
   not score. VII assumes the gate is the ceiling and rotation degrades it.
2. **d′ clips at 0 for p ≤ chance**, so "at chance" and "reliably below chance"
   both collate as VII = 0 and the retinotopic trap vanishes from that axis.
   Report signed accuracy−chance with a binomial interval beside it.

**Always report the positional-prior controls.** `constant_answer_rate` gives
`best_fixed` (always answer the most common option) and `distribution` (sample
the model's own answer distribution, ignoring the images). Qwen2.5-VL-3B and 7B
do **not** beat their own best-fixed baseline on the 4AFC arm.

---

## 5. Running it

```bash
# distances and benchmarks (fast, local)
python3 bench/layout_distance.py
python3 bench/build_hard_benchmark.py --pct 0 10 --out data/vlm_benchmark_4afc_hard.json

# an API model  (WORKERS parallelises; it is pure network latency)
WORKERS=8 MAX_TOKENS=8000 PROMPT_STYLE=cot_anyview \
BENCHMARK=data/vlm_benchmark_4afc_hard.json OPENROUTER_API_KEY=sk-or-... \
  bash bench/run_openrouter.sh google/gemini-3.8-flash 100 3.00

# open models on dgx2 (shared GPU lock, smoke-tested, restartable)
bash bench/launch_hard_queue.sh

# a person, on the SAME 100 trials
python3 bench/build_human_task.py --benchmark data/vlm_benchmark_4afc_hard.json \
    --match_run 100 --match_seed 0 --prompt_style neutral_anyview --out human_task/hard

# read it
bash bench/build_paper.sh
# (historical one-off figure scripts are preserved in bench/archive/)
```

Always smoke-test a new model at **n=1** before widening.

---

## 6. What must hold before a number is read

* **The run used the locked configuration** (`HANDOVER.md` §0). `max_tokens` and
  `workers` are not yet recorded in result files — check provenance by hand.
* **Accuracy clears both positional-prior baselines.** If it does not, it is not
  evidence of anything.
* **The truncation rate is low.** Count replies with no `Final Answer` line; at
  `max_tokens 4000` this reached 27 % and depressed whole modes unevenly.
* **The comparison is paired.** `--match_run N` for people; the same stratified
  seed for models. Trial ids collide between benchmark files, so a result must be
  scored against the benchmark it was actually run on.
* **Per-cell n is stated.** A 50-trial run has 2 trials per (mode × delta) cell;
  every such cell is one of {0 %, 50 %, 100 %} and its interval spans most of the
  axis.

---

## 7. Known limits

* **Synthetic stimuli.** Landmark identity is a shape/colour pair or a form
  vector, not an object category. Frontier models may be familiar with the
  *rendering style* (c0 is essentially CLEVR) independently of the spatial task —
  testable by re-rendering the same layouts in a different visual style.
* **Azimuth only.** The camera orbits at fixed elevation and radius.
* **Modes do not separate** under matched foil selection: c0–c4 span 0.35–0.50
  with every interval overlapping at n=20/mode. Larger n needed to say more.
* **The human arm is one non-naive participant.** p01 built the scenes.
* **`visible_px` in the masks is absolute area**, so it conflates "occluded" with
  "far away". It is a floor on visibility, not an occlusion rate.
