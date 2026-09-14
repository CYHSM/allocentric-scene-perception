# Allocentric scene perception

**Does scaling a vision-language model buy a cognitive map, or only a better
appearance gate?**

This repository adapts the clinical **Four Mountains Test** into a forced choice
that models and people take on identical trials, and separates the two reasons
two images of a place can differ — *the place changed* or *you moved*.

**→ [`BENCHMARK.md`](BENCHMARK.md) is the specification.**
**→ [`HANDOVER.md`](HANDOVER.md) is the current state, the locked run config, and the open tasks.**

---

## The result in one table

4AFC, chance 25 %, hard foils, the identical 100 trials for every observer.

| observer | accuracy | Δ = 0° (the gate) | Δ ≥ 45° (the map) |
|---|---|---|---|
| **Human p01** *(40/100 done, not naive)* | **0.88** | 12/12 | 23/28 |
| GPT-5.6 Luna | 0.45 | **20/20** | 25/80 |
| Gemini 3.8 Flash | 0.44 | 13/20 | 31/80 |
| Qwen3-VL-235B-instruct | 0.33 | 18/20 | 15/80 |
| Qwen3-VL-235B-**thinking** | 0.30 | 15/20 | 15/80 |

Every model has a working appearance gate. None of them keep it through a turn.
Luna is the clearest case: **perfect at Δ=0, 31 % once the viewpoint moves.**

Two controls that matter:

* **Test-time compute buys nothing.** Same weights, reasoning on versus off:
  0.30 vs 0.33, at 2.7× the completion tokens.
* **The human and the model fail on orthogonal axes.** The person pays for
  rotation and not for foil similarity; the model pays for foil similarity and
  not for rotation. Paired McNemar at 40 trials: **p = 0.0013**.
  (`figures/fig_double_dissociation.png`)

---

## The dataset

100 anchor layouts × 5 stimulus modes × 2 appearances × 8 azimuths = **8 000
frames** in `data/scenes_100/`. Scene `a007` is the same place in all five modes,
from the same camera; only the cues that tell landmarks apart, and the world
around them, differ. Mode is therefore a within-scene factor and every
cross-mode comparison is paired.

| mode | landmarks | world |
|---|---|---|
| `c0_shape_colour` | 4 shapes × colours | bare plane |
| `c1_shape` | 4 shapes, one colour | bare plane |
| `c2_colour` | one shape, 4 colours | bare plane |
| `c3_peaks_bare` | 4 parametric landforms | bare plane |
| `c4_valley` | the same landforms | terrain, lake, vegetation, haze |

Every mode is landmark-inventory controlled: all 100 scenes in a mode use *one*
set of four landmarks, permuted per scene, so only the arrangement separates two
places.

## Difficulty is selected, not sampled

Foils used to be drawn at random, which made each trial's difficulty a lottery —
and the strongest model's errors turned out to be almost exactly the trials where
that lottery produced a near-duplicate. Foils are now chosen from a percentile
band of a **rotation-optimal, identity-matched layout distance**, in metres:

```
D(i, j) = min over φ  of  mean over peaks k  ‖ R(φ) p_ik − p_jk ‖
```

Minimised over global rotation because the camera rotates too, so a layout that
is the target turned by φ is the *hardest* foil rather than a different place.

| benchmark | nearest foil, median |
|---|---|
| `vlm_benchmark_4afc.json` (random, legacy) | 32.4 m |
| `vlm_benchmark_4afc_hard.json` | 6.8 m |
| `vlm_benchmark_4afc_vhard.json` | 4.5 m |

This also supplies the displacement ladder that was configured but never
rendered — existing scene pairs span 2.7–48 m, so a psychometric threshold can be
fitted per observer with no Blender time.

## The landmarks are a parametric shape space

A landform is a six-number **form** vector plus a size and a pose:

| attribute | range | what it does |
|---|---|---|
| `peak_angle` | 0.70–2.00 | sharp summit … straight cone … broad-topped |
| `flank` | 1.15–2.60 | flared into a talus apron; above 1 the surface meets the ground tangentially |
| `offset` | 0.00–0.20 | summit displacement, as a fraction of width |
| `crest` | 0.00–0.55 | summit ridge length, as a fraction of width |
| `roughness` | 0.10–0.55 | relief amplitude |
| `ruggedness` | 0.55–1.70 | relief frequency |

Form is separable from size, which lets identity vary without touching apparent
size — and apparent size is the image's distance cue. Size is held **uniform
across every landmark and every mode**, so a landmark is the same landmark in all
five renders.

> ⚠️ `layout["morphologies"]` and `peak["type"]` are **stale labels**, written
> before `assign_objects` replaces the landforms. Identity is `obj` (c0–c2) or
> the `form` vector (c3–c4). See `BENCHMARK.md` §1.

## Repository

```
blender/four_mountains/     the renderer (Blender 5.2 + Cycles)
├── fm_layout.py            layouts, framing validity, camera geometry
├── fm_peak.py              the parametric landform shape space
├── fm_stimulus.py          the five modes; canonical landmark sets
├── fm_bank.py              what is in the bank — no Blender, so it is testable
└── render_bank.py          the only renderer

bench/
├── evaluate_vlm.py         the runner: local GPU + any OpenAI-compatible API
├── layout_distance.py      the rotation-optimal layout distance
├── build_hard_benchmark.py foil selection by percentile band
├── build_human_task.py     the browser task, on the SAME trials
├── vii.py                  d′, VII, positional-prior controls
├── agents.py               one discovery layer every figure reads
├── make_figures.py         the paper figures (historical ones in archive/)
├── run_openrouter.sh       API runs      launch_hard_queue.sh   dgx2 runs
└── tests/

human_task/                 the browser tasks + collected human data (hard, n01, pilot_2afc)
paper/tex/main.tex          the write-up
```

## Getting started

```bash
pip install -e .
pytest bench/tests
python3 -m http.server 8765 --directory .    # then open /figures/

# score a model on the hard benchmark (see HANDOVER.md §0 for the locked config)
WORKERS=8 MAX_TOKENS=8000 PROMPT_STYLE=cot_anyview \
BENCHMARK=data/vlm_benchmark_4afc_hard.json OPENROUTER_API_KEY=sk-or-... \
  bash bench/run_openrouter.sh google/gemini-3.8-flash 100 3.00
```

Rendering needs Blender 5.2 with Cycles and a CUDA GPU; local model scoring needs
`torch` and `transformers`. See [`CLUSTER.md`](CLUSTER.md) for the multi-GPU
setup.

## Status

The 4AFC arm, the hard-foil benchmark, the human arm and nine model runs are
built and working. Open: finishing p01's remaining 60 trials, naive
participants, and recording the full run configuration inside every result file.
See [`HANDOVER.md`](HANDOVER.md) §7.

## The paper

```bash
bash bench/build_paper.sh     # tables + figures from the result files
bash paper/tex/build.sh       # -> paper/tex/main.pdf
```

Draft in `paper/tex/main.tex`. It inputs the generated tables and figures, so
the two commands must run in that order.
