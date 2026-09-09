# The benchmark

## The question

Two images of a place can differ for exactly two reasons: **the place changed**,
or **you moved**. A model that represents *place* should be sensitive to the
first and indifferent to the second. Most image benchmarks cannot separate them,
because "the same place" and "the same picture" coincide whenever the viewpoint
is held fixed.

This dataset varies both independently and measures them in one currency —
cosine distance in a frozen encoder's embedding — which lets them be traded
against each other:

> **How far must the landmarks physically move to confuse a model as much as
> walking Δ degrees around the scene does?**

That distance is **λ(Δ)**, in metres, and **small λ is the good end**. A model
with a genuine allocentric representation barely notices the walk, so only a
small displacement is needed to cost it the same — λ of a metre or two. A model
matching appearance is thrown by the walk, and the landmarks would have to be
rebuilt somewhere else entirely before a scene change cost as much — λ of tens
of metres, or off the top of the ladder.

## The design: one set of scenes, rendered five ways

Every scene is sampled once — landmark positions, uniform sizes, one camera rig
— and then rendered in each of five stimulus modes. Scene `a007` in `c0` and in
`c4` is **the same place from the same spot**; only the cues that tell landmarks
apart, and the world around them, differ.

| mode | landmarks | world |
|---|---|---|
| `c0_shape_colour` | 6 shapes × 6 colours | bare plane |
| `c1_shape` | 6 shapes, one colour | bare plane |
| `c2_colour` | one shape, 6 colours | bare plane |
| `c3_peaks_bare` | parametric landforms | bare plane |
| `c4_valley` | the same landforms | terrain, lake, vegetation, haze |

Mode is therefore a **within-scene factor**, so every cross-mode comparison is
paired. This matters more than it looks: sampling layouts independently per mode
entangles "mode" with "which places happened to be drawn", and nothing in the
rendered images reveals it.

Landmark **size is uniform in every mode**. A landmark 14 m wide in `c3` and
17 m wide in `c0` would not be the same landmark seen two ways, and apparent size
is the image's distance cue.

## Three axes

| axis | varies | |
|---|---|---|
| **viewpoint** | camera azimuth, 8 bearings | how much does moving cost? |
| **scene** | landmark position, `d_pos` metres | how much does the place changing cost? |
| **cue** | c0 → c4 | which cues does the model rely on? |

Probes perturb **positions only**, which is mode-independent, so the probe
geometry is identical in all five modes. Identity is varied by the cue ladder,
which is what it is for.

## Scene distance

A scene is a set of (identity, position) pairs. `fm_scenedist` measures three
coordinates, each under an optimal assignment between landmark sets:

| | | units | translation | permutation | substitution |
|---|---|---|---|---|---|
| `d_pos` | largest distance a landmark must move | **metres** | = displacement | **0** | 0 |
| `d_bind` | fraction of places whose occupant changed | 0–1 | 0 | **> 0** | > 0 |
| `d_id` | how different the landmark *sets* are | 0–1 / form L2 | 0 | **0** | **> 0** |

Only `d_pos` is swept, because it is the one with units.

**Why this replaced the old foil families.** The previous displacement measure
matched landmarks *by slot name*, so a permutation — which leaves the set of
occupied positions exactly as it found it — recorded 50–65 m, more than an
explicit 20 m translation. A second helper matched *by identity* and scored the
same permutation ~0 m. The two disagreed precisely on the case that separated the
families, so every binding-vs-metric contrast depended on which convention the
caller happened to use. Here the matching is stated once and the three
coordinates are orthogonal by construction.

`d_pos` is the **bottleneck** (largest) matched displacement, not the mean, so
"one object moved 20 m" measures 20 m. It is still a metric: symmetry and the
triangle inequality both hold.

## The bank

* **40 anchors** — mutually distinct places, by rejection rather than distinct
  seeds. Two independently sampled layouts can land on the same place, and a
  retrieval miss between them would be the pool's fault, not the model's.
* **120 probes** — 20 anchors × `d_pos ∈ {1, 2, 4, 8, 16, 24}` m. One landmark
  per anchor moves, so the anchor's ladder is a single trajectory.
* Every scene, anchor or probe, rendered **identically**: 2 appearances × 8
  azimuths, plus an instance mask. **Nothing is a distractor at render time**;
  roles are metadata and every analysis is post hoc.

## Read-outs

**λ** (`bench/exchange.py`) — the headline. `D_view(Δ)` from an anchor against
itself Δ degrees away; `D_scene(d)` from an anchor against its probe at the same
bearing; λ where the curves cross, bootstrap CI over anchors. Both curves use the
same appearance contrast, or λ would be wrong by that margin with nothing
downstream to catch it. Where the crossing falls outside the probe ladder, λ is
reported as **unmeasured** — clamping would turn "we did not measure this" into a
number.

**Retrieval, NVM, RSA** (`bench/metrics.py`) — cross-view Recall@1/@5/mAP over
the anchor gallery, the invariance margin, and the Spearman correlation between
scene×scene distance matrices at two bearings. The gallery is anchors only:
probes are the same place nudged, so including them would make Recall@1 depend on
how many near-duplicates the pool happened to hold.

## Running it

```bash
# 1. render one mode (repeat for c0..c4; shard with --shard/--shards)
blender -b -P blender/four_mountains/render_bank.py -- \
    --out data/scenes/c0_shape_colour --mode c0_shape_colour \
    --anchors 40 --probes_per_anchor 20 --azimuth_step 45 --samples 24

# 2. merge shards and CHECK the dataset before scoring it
python bench/merge_bank.py --root data/scenes

# 3. look at it
python bench/figure_dataset.py --root data/scenes --out figures/fig1_dataset.png

# 4. score
python bench/metrics.py  --bank data/scenes/c0_shape_colour --model <timm id>
python bench/exchange.py --bank data/scenes/c0_shape_colour --model <timm id>
```

## What must hold before a number is read

* `merge_bank.py` exits non-zero unless the five modes hold **the same scenes at
  the same positions**, every scene has all 16 frames, and every probe has
  `d_id = d_bind = 0`.
* At Δ0° with appearance unchanged the query is the byte-identical file, so
  **Recall@1 must be exactly 1.000**.
* Look at Figure 1 before scoring. It prints how much of the frame each rung
  actually changes. Measured on `a000` at 0°, the smallest rung moves **5.5% of
  the pixels** and the ladder is monotone (5.5, 9.9, 14.5, 20.0, 33.0, 38.4%),
  so the bottom of the ladder is above the renderer's floor: a flat
  `D_scene(1 m)` would be the model's doing, not the stimulus's.
* Read the **occlusion census** `merge_bank.py` prints. See the known limit
  below.

## Known limits

* **Synthetic stimuli.** Landmark identity is a shape/colour pair or a form
  vector, not an object category.
* **Azimuth only.** The camera orbits at fixed elevation and radius; elevation
  and distance are not varied.
* **λ assumes the two curves cross once.** `D_view` is not always monotone in Δ
  — near-symmetric layouts can make 180° easier than 135° — so λ can be
  multi-valued. Report the surface when that happens.
* **No 4AFC arm at present.** The human-comparable forced-choice path was tied to
  the previous bank schema and has not yet been rebuilt against this one.
* **The top of the probe ladder occludes.** Displacing a landmark can move it in
  front of another. On `a000` at 0°, the second landmark falls from 4913 px to
  575 px at `d = 16` m and 237 px at `d = 24` m — hidden behind the one that
  moved. `d_id` and `d_bind` are still exactly 0, so the *scene* changed only in
  position, but the *image* lost an object, and `D_scene` at the top rungs is
  therefore not purely a displacement. `merge_bank.py` prints the rate per rung
  over the whole bank; treat a large λ that rests only on `d = 16`–`24` m with
  suspicion, and read Figure 3 to see which rungs it was interpolated between.
* `visible_px` in the masks is absolute area, so it conflates "occluded" with
  "far away". It is a floor on visibility, not an occlusion rate.
