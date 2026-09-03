"""
Regenerate every showcase image in this folder.

Two passes, because Blender ships numpy but no Pillow/matplotlib:

    blender -b -P blender/four_mountains/make_figures.py     # render panels
    python  blender/four_mountains/make_figures.py           # compose sheets

The first pass writes raw panels into `_figtmp/`; the second stitches them into
the collages and draws the heightfield sheet from the pure-numpy morphology
library. Running the second pass alone re-composes without re-rendering.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

TMP = os.path.join(_HERE, "_figtmp")
HERO = dict(elevation_deg=11.0, radius=88.0, target_z=6.0)


# --------------------------------------------------------------------------- #
# Pass 1 (inside Blender): render the panels
# --------------------------------------------------------------------------- #

def render_panels():
    from fm_morphology import morphology_names
    from generate_scene import build_four_mountains_scene
    from render_dataset import FourMountainsRenderer

    os.makedirs(TMP, exist_ok=True)
    blend = os.path.join(_HERE, "four_mountains.blend")

    def shot(r, name, az, samples=128, res=(800, 500), **orbit):
        r.scene.render.resolution_x, r.scene.render.resolution_y = res
        r.set_camera_orbit(az, **dict(HERO, **orbit))
        return r.render_rgb(os.path.join(TMP, name), samples=samples)

    r = FourMountainsRenderer(blend)
    shot(r, "hero.png", 55.0, samples=192, res=(1600, 1000))

    for i, az in enumerate((0.0, 90.0, 180.0, 270.0)):
        shot(r, f"v{i}_rgb.png", az)
        r.render_mask(os.path.join(TMP, f"v{i}_mask.png"))

    r = FourMountainsRenderer(blend)
    shot(r, "canonical.png", 45.0, res=(900, 600))
    r.swap_mountains("M1", "M2")
    r.render_rgb(os.path.join(TMP, "foil.png"), samples=128)

    gallery = os.path.join(TMP, "gallery.blend")
    build_four_mountains_scene(gallery, types=morphology_names(), valley_trees=9000)
    g = FourMountainsRenderer(gallery)
    for i, az in enumerate((30.0, 210.0)):
        shot(g, f"g{i}.png", az, res=(1600, 620), elevation_deg=8.5,
             radius=170.0, target_z=16.0)
    print("Panels written to", TMP)


# --------------------------------------------------------------------------- #
# Pass 2 (project Python): compose the sheets
# --------------------------------------------------------------------------- #

def _grid(names, out_name, cols, pad=10, bg=(248, 248, 248)):
    from PIL import Image
    tiles = [Image.open(os.path.join(TMP, n)).convert("RGB") for n in names]
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad,
                              rows * h + (rows + 1) * pad), bg)
    for i, tile in enumerate(tiles):
        row, col = divmod(i, cols)
        sheet.paste(tile, (pad + col * (w + pad), pad + row * (h + pad)))
    out = os.path.join(_HERE, out_name)
    sheet.save(out)
    print("wrote", out)


def _heightmap_sheet():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LightSource

    from fm_morphology import build_heightfield, morphology_names

    names = morphology_names()
    fig, axes = plt.subplots(3, 4, figsize=(15, 11.4))
    ls = LightSource(azdeg=315, altdeg=35)

    for ax, name in zip(axes.ravel(), names):
        X, Y, Z = build_heightfield(name, height=34.0, base_radius=17.0,
                                    n_r=200, n_theta=420, seed=101)
        # Resample the polar field onto a square grid for display.
        n = 420
        gx = np.linspace(-17, 17, n)
        GX, GY = np.meshgrid(gx, gx)
        R = np.clip(np.hypot(GX, GY), 0, 17)
        T = np.mod(np.arctan2(GY, GX), 2 * np.pi)
        ri = np.clip((R / 17.0 * (Z.shape[0] - 1)).astype(int), 0, Z.shape[0] - 1)
        ti = np.mod((T / (2 * np.pi) * Z.shape[1]).astype(int), Z.shape[1])
        grid = np.where(np.hypot(GX, GY) <= 17, Z[ri, ti], np.nan)

        cell = 34.0 / n
        shaded = ls.shade(np.nan_to_num(grid), cmap=plt.get_cmap("gist_earth"),
                          vmin=-22.0, vmax=36.0, vert_exag=1.2,
                          dx=cell, dy=cell, blend_mode="soft")
        shaded[np.isnan(grid)] = 1.0
        ax.imshow(shaded, origin="lower", extent=(-17, 17, -17, 17))
        ax.set_title(name, fontsize=13, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Four Mountains landform library - hillshaded heightfields "
                 "(34 m summit, 17 m base radius, seed 101)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(_HERE, "heightmaps.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


def compose():
    from PIL import Image
    Image.open(os.path.join(TMP, "hero.png")).save(os.path.join(_HERE, "preview.png"))
    print("wrote", os.path.join(_HERE, "preview.png"))
    _grid([f"v{i}_rgb.png" for i in range(4)] + [f"v{i}_mask.png" for i in range(4)],
          "four_views_collage.png", cols=4)
    _grid(["canonical.png", "foil.png"], "position_comparison.png", cols=2)
    _grid(["g0.png", "g1.png"], "morphology_gallery.png", cols=1)
    _heightmap_sheet()


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
    except ImportError:
        compose()
    else:
        render_panels()
