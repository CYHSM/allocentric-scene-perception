# Wrong Valley · Four Mountains Task Game

A standalone, browser-based implementation of the **Four Mountains Task** (Hartley et al. 2007; Chan et al. 2016) powered by photorealistic **Blender Cycles 360° raytraced renders**.

---

## Overview

In each trial:
1. **Three plates** are presented simultaneously (Plates I, II, III).
2. **Two plates** photograph the **same valley** (Canonical Four Mountains) from two different viewpoints along a 360° circular orbit.
3. **One plate** photographs a **distractor foil valley** from a third viewpoint. The foil features the identical four landmark landforms (Matterhorn pyramid, knife-edge ridge, tableland mesa, and volcanic dome), but their spatial positions relative to each other and the central tarn have been altered.
4. The participant clicks the odd plate out (or presses keys `1`, `2`, `3`).
5. Feedback reveals the 2D orbital plan views showing the exact peak positions and camera line-of-sight bearings.

---

## Features

- **Blender Cycles Raytracing**: Every plate is an authentic Blender Cycles raytraced image with realistic atmospheric lighting, soft shadows, conifer forests, talus fields, snow caps, and water reflections.
- **Fixed Constant-Radius Orbit ($17^\circ$ Elevation)**:
  - Radius: $R = 94.0$
  - Elevation: $\phi = 17.0^\circ$
  - Target: $(0, 0, 6.0)$ (center of the tarn)
  - All four peaks and the lake are visible from every single bearing without any artificial zoom variation.
- **Interactive 360° Turntable Scrubbing**:
  - Drag horizontally across any plate image, or scrub the range slider beneath it, to rotate smoothly around that valley in 15° steps across all 24 angles ($0^\circ \dots 345^\circ$).
- **Orbital 2D Plan Views**:
  - Dynamic top-down maps displaying the exact positions and labels of M1 (Matterhorn), M2 (Knife Ridge), M3 (Table Mesa), and M4 (Ash Dome).
  - Shows the circular camera orbit ring and sightline bearing arrow.
- **Clinical FMT Foils**:
  - `foil_swap12`: M1 and M2 traded places across the northern rim.
  - `foil_swap34`: M3 and M4 traded places across the southern rim.
  - `foil_swap13`: M1 and M3 traded places along the western rim.

---

## Running the Game

Run Python's local HTTP server from the project root:

```bash
python3 -m http.server 8000 --directory game
```

Then open:
**[http://localhost:8000/odd_one_out.html](http://localhost:8000/odd_one_out.html)**

*(On macOS, you can also run `open game/odd_one_out.html` directly).*

---

## Regenerating the Viewpoint Bank

To re-render all 96 frames (4 valleys $\times$ 24 viewpoints) with Blender 5.2+:

```bash
blender -b blender/four_mountains/four_mountains.blend -P blender/four_mountains/render_game_bank.py
```

Renders are written to `game/renders/bank/` along with `manifest.json`.
Raw render PNGs are excluded from version control via `.gitignore`.
