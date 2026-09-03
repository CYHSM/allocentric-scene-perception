# Four Mountains Task: Procedural Alpine World & Dataset Generator

A fully procedural, photorealistic alpine environment and dataset generator for
**Allocentric Scene Perception** and the **Four Mountains Task (FMT)**, built for
**Blender 5.2+**.

The scene is a real landscape, not a set of cones on a plane: an alpine tarn in a
forested valley, layered foothills and a hazed snow range on the horizon, a
physical sky with a cumulus deck, and four freely movable landmark peaks drawn
from a library of **twelve distinct landforms**.

---

## Visual Showcase

![Four Mountains preview](blender/four_mountains/preview.png)

### 1. 360 degree multi-viewpoint orbit and instance segmentation

![Four viewpoints and instance masks](blender/four_mountains/four_views_collage.png)

*Top row: Cycles renders at azimuth 0/90/180/270 degrees. Bottom row:
pixel-exact instance masks. Each peak carries its own colour, its forest and
boulder field are labelled with it, and the terrain, the lake and the sky are
separate classes.*

### 2. The landform library

![Landform gallery](blender/four_mountains/morphology_gallery.png)

![Hillshaded heightfields](blender/four_mountains/heightmaps.png)

*All twelve morphologies at identical summit height and base radius.*

### 3. Allocentric position manipulation (distractor foils)

![Position swap comparison](blender/four_mountains/position_comparison.png)

*Left: canonical arrangement from azimuth 45 degrees. Right: distractor foil
with M1 and M2 swapped, from the identical viewpoint. Vegetation and boulders
are parented to each peak, so a moved mountain takes its forest with it.*

---

## How the world is generated

Everything is built procedurally through `bpy` and numpy. No add-ons, no
external assets, no image textures.

### 1. Twelve landform morphologies

Each peak is a polar heightfield evaluated with domain-warped ridged
multifractal noise, then normalised so `height` is the true summit altitude and
clipped so `Z = 0` exactly at the base radius (no flat overlapping skirts, no
Z-fighting against the valley floor).

| Morphology | Landform |
| --- | --- |
| `horn` | Matterhorn-style pyramid: four razor arêtes, glacial cirques between them |
| `ridge` | Elongated massif with a knife-edge spine and a saddled crest |
| `mesa` | Flat-topped tableland: sheer cliff band under a level summit, talus apron |
| `dome` | Stratovolcano with concave-up flanks and radiating fluvial ravines |
| `caldera` | Collapsed volcano: raised, breached rim around a sunken basin |
| `butte` | Isolated rock tower on a wide debris cone |
| `twin` | Two summits of unequal height joined by a col |
| `sawtooth` | Serrated crest: a row of pinnacles and notches along one strike |
| `cuesta` | Tilted plateau: cliff scarp on one side, long dip slope opposite |
| `cone` | Symmetric ash cone with a small summit crater |
| `whaleback` | Roche moutonnée: ice-smoothed stoss side, steep plucked lee |
| `massif` | Broad complex block with several subsidiary summits |

Adding a thirteenth is one function plus one line in
`fm_morphology.MORPHOLOGIES`.

### 2. One seamless world, from the shoreline to the horizon

A single radial terrain sheet (out to 1.4 km, radially graded resolution)
carries every distance band: the carved lake basin, the valley floor, the
foothills, the mid range and the horizon skyline. Relief is **centred** rather
than piled onto a monotonic ramp, so ridges alternate with real basins and the
skyline reads as layered ridgelines instead of one continuous wall.

### 3. Slope- and altitude-driven biome shader

The terrain material stacks biomes, each overriding the previous where its mask
is 1: shore gravel -> alpine meadow (with sun-bleached patches) -> conifer belt
-> scree -> cliff rock (mineral-banded, iron-stained, stratified) -> snow. The
tree line and snow line are jittered by noise, and snow is inhibited on walls
too steep to hold it. An ambient-occlusion term darkens couloirs and crevices,
which is most of what stops a procedural mountain looking like poured concrete.

### 4. Atmosphere

- **Nishita multiple-scattering sky** with low aerosol density for a deep blue.
- **Sky-dome cumulus**: each view ray is intersected with a virtual cloud plane
  in the world shader, so clouds get true perspective from every camera angle.
  (A modelled cloud plane degenerates into a solid ceiling when viewed edge-on.)
- **Aerial perspective**: an analytic distance term in the terrain shader fades
  far ranges into the sky colour. No volumetrics, no render cost.
- **Low sun** (22 degrees) cross-lighting the canonical viewpoint, so topography
  reads in relief.

### 5. Vegetation and boulders

Conifers are scattered into clumped stands between the shoreline and the tree
line, off the cliffs; boulders are biased toward steeper ground near cliff
bases. Both are emitted in the **local frame of their host object** and parented
to it, so a mountain carries its forest and its talus when you move it.

---

## Repository structure

```
game/
└── odd_one_out.html      # browser game: three plates, find the odd valley
blender/four_mountains/
├── fm_noise.py           # numpy value / fBm / ridged-multifractal primitives
├── fm_morphology.py      # the twelve landform heightfields
├── fm_materials.py       # terrain, water, foliage and sky-cloud shaders
├── fm_scatter.py         # mesh building, surface sampling, vegetation scatter
├── generate_scene.py     # scene assembly -> four_mountains.blend
├── render_dataset.py     # controller API + batch dataset renderer
├── make_figures.py       # regenerates every image in this README
└── four_mountains.blend  # self-contained saved scene
```

`fm_noise.py` and `fm_morphology.py` import no `bpy`, so every heightfield can be
generated, inspected and unit-tested with plain Python:

```python
import sys; sys.path.append("blender/four_mountains")
from fm_morphology import build_heightfield, morphology_names
X, Y, Z = build_heightfield("caldera", height=34.0, base_radius=17.0)
```

---

## Getting started

### Prerequisites

- **Blender 4.5+ or 5.0+** (developed on **5.2.1 LTS**). Apple Silicon Metal GPU
  acceleration is detected and enabled automatically.
- Python 3.10+ with numpy, Pillow and matplotlib for the figure pass.

### Build the scene

```bash
blender -b -P blender/four_mountains/generate_scene.py
```

Options (after `--`):

```bash
blender -b -P blender/four_mountains/generate_scene.py -- \
    --types horn,caldera,sawtooth,butte \
    --seed 7 --samples 128 --resolution 1600 1000
```

`--types` accepts any number of landforms; with more than four they are spread
evenly around the valley.

### Render a dataset

```bash
blender -b -P blender/four_mountains/render_dataset.py -- \
    --num_views 16 --output_dir data/four_mountains --samples 96
```

Per sample:

- `sample_XXXX_azYYY_rgb.png` — Cycles render
- `sample_XXXX_azYYY_mask.png` — instance segmentation mask
- `sample_XXXX_azYYY_meta.json` — camera pose, sun angles, peak positions,
  landform types and the mask colour legend

### Regenerate the README figures

```bash
blender -b -P blender/four_mountains/make_figures.py   # render panels
python  blender/four_mountains/make_figures.py         # compose sheets
```

---

## Python API

```python
import sys
sys.path.append("blender/four_mountains")
from render_dataset import FourMountainsRenderer

renderer = FourMountainsRenderer("blender/four_mountains/four_mountains.blend")

# Move any peak; its forest and boulder field come along
renderer.set_mountain_position("M1", x=-10.0, y=8.0, rot_z=0.4)

# Or build a distractor foil in one call
renderer.swap_mountains("M1", "M2")

# Camera orbit (azimuth degrees, elevation, radius, look-at height)
renderer.set_camera_orbit(azimuth_deg=45.0, elevation_deg=11.0, radius=88.0)

# Time of day: azimuth is the bearing the light comes from
renderer.set_sun(elevation_deg=18.0, azimuth_deg=210.0, energy=75.0)

# RGB + instance mask + metadata
renderer.render_sample(output_dir="output/", sample_idx=0, azimuth_deg=45.0)
```

`renderer.mountains` is discovered from the scene, so scenes with more than four
peaks work unchanged, and `renderer.mask_legend()` returns the class colours for
decoding the masks.

---

## Play it in the browser

`game/odd_one_out.html` is a self-contained, no-build version of the task:
three plates per trial, two showing one valley from different bearings and one
showing somewhere else. Pick the odd one. Difficulty ramps by peak count — one
peak, then two, then three, up to eight.

Terrain is generated live in the browser: the noise primitives and all twelve
landforms from `fm_noise.py` / `fm_morphology.py` are ported to JavaScript and
meshed with three.js, so every trial is a fresh valley rather than a fixed image
bank.

Foils follow the clinical design — they keep the target's landforms and change
the arrangement, so local feature matching fails:

| Level | Foil |
| --- | --- |
| 1–2 | different landforms at the same positions |
| 3–4 | one peak relocated |
| 5+ | two peaks trade places (same shapes, same locations, different assignment) |

From level 4 the light and season are re-rolled *independently per plate*, as in
the original test, so appearance never marks the answer. Answering reveals a
plan view under each plate showing the layout and the camera bearing.

Open the file directly, or play the published version:
<https://claude.ai/code/artifact/5eb3353b-a3ef-4cb1-a511-7aa03fd46d8c>

---

## Roadmap

1. **Weather and season presets** — autumn larch, winter snow line, overcast
   diffuse lighting to test invariance to shadow cues.
2. **Hydraulic erosion** — sediment transport passes for dendritic drainage and
   alluvial fans.
3. **Ground-level walkthrough camera** — first-person egocentric trajectories
   through the valley and passes.
4. **More ground truth** — 32-bit EXR depth, surface normals, optical flow for
   video sequences.
5. **Geometry-nodes instancing** for the vegetation, to trade the baked meshes
   for lighter scenes at much higher tree counts.
