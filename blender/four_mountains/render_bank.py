"""
Render the scene bank: one set of layouts, rendered in every stimulus mode.

    blender -b -P render_bank.py -- --out data/scenes --mode c0_shape_colour

The layouts are a deterministic function of `--seed0` and `--anchors` alone, so
running this five times with five different `--mode` values produces five banks
that are **paired scene for scene**: `s007` in c0 and `s007` in c4 are the same
place, the same landmark positions, the same camera, differing only in what
tells the landmarks apart and what surrounds them. That pairing is the point of
the dataset, and `bench/tests/test_pairing.py` checks it rather than trusting it.

Every scene -- anchor or probe -- is rendered identically: both appearances, all
azimuths, one instance mask per frame. There is no "target" and no "foil" at
render time. Which scenes serve as gallery, query or distractor is decided by
the analysis, from `role` and `d_pos` recorded here.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import fm_bank
import fm_layout as layoutlib
import fm_stimulus as stim
from render_dataset import FourMountainsRenderer

def _read_rgb_bytes(path):
    import bpy
    img = bpy.data.images.load(path)
    try:
        buf = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(buf)
        return (buf.reshape(-1, 4)[:, :3] * 255.0).round().astype(int)
    finally:
        bpy.data.images.remove(img)


def mask_visibility(mask_path, legend, tolerance=12):
    a = _read_rgb_bytes(mask_path)
    return {name: int(np.all(np.abs(a - np.array(col)) <= tolerance, axis=1).sum())
            for name, col in legend.items() if name.startswith("M")}


def render_scene(renderer, out_dir, scene_id, azimuths, samples, write_mask):
    """Both appearances at every azimuth. Identical for anchors and probes."""
    frames, legend = {}, renderer.mask_legend()
    for app in ("A", "B"):
        renderer.set_appearance(**stim.APPEARANCES[app])
        for az in azimuths:
            base = f"{scene_id}_{app}_az{int(round(az)):03d}"
            rgb = os.path.join(out_dir, f"{base}.png")
            mask = os.path.join(out_dir, f"{base}_mask.png")
            if not os.path.exists(rgb):
                renderer.set_camera_orbit(az, layoutlib.CAM_ELEVATION,
                                          layoutlib.CAM_RADIUS,
                                          layoutlib.CAM_TARGET_Z)
                renderer.render_rgb(rgb, samples=samples)
                if write_mask:
                    renderer.render_mask(mask)
            entry = {"image": os.path.relpath(rgb, out_dir)}
            if write_mask and os.path.exists(mask):
                entry["visible_px"] = mask_visibility(mask, legend)
            frames[f"{app}_{int(round(az))}"] = entry
    return frames, legend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scenes/c0_shape_colour")
    ap.add_argument("--mode", default="c0_shape_colour", choices=list(stim.MODES))
    ap.add_argument("--anchors", type=int, default=40)
    ap.add_argument("--probes_per_anchor", type=int, default=20,
                    help="How many anchors also get the probe ladder.")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--n_peaks", type=int, default=4)
    ap.add_argument("--azimuth_step", type=int, default=45)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--resolution", type=int, nargs=2, default=(640, 440))
    ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--separation_m", type=float, default=fm_bank.SEPARATION_M,
                    help="Minimum layout distance between two anchors, in "
                         "metres: how far apart two scenes must be to count as "
                         "different places. The default packs 100 anchors for "
                         "n_peaks >= 3, but the layout space shrinks with the "
                         "peak count and saturates: at n_peaks=1 only 20 "
                         "anchors fit at 12 m and at n_peaks=2 only 88, so "
                         "those banks need 4 m and 10 m respectively to reach "
                         "100. Lower it only for that reason, and read the "
                         "value back from bank.json before comparing distances "
                         "across banks -- it sets the floor of the scale every "
                         "foil band is measured on.")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    azimuths = layoutlib.benchmark_azimuths(args.azimuth_step)

    # Sampled in full on every shard, then sliced. The layout set must not
    # depend on how the render was parallelised.
    all_scenes = fm_bank.plan_bank(
        seed0=args.seed0, n_anchors=args.anchors, n_peaks=args.n_peaks,
        azimuth_step=args.azimuth_step,
        probes_per_anchor=args.probes_per_anchor,
        separation_m=args.separation_m)
    mine = all_scenes[args.shard::args.shards]

    bank = {"camera": {"elevation_deg": layoutlib.CAM_ELEVATION,
                       "radius": layoutlib.CAM_RADIUS,
                       "target_z": layoutlib.CAM_TARGET_Z,
                       "lens_mm": layoutlib.CAM_LENS_MM,
                       "azimuths": list(azimuths)},
            "resolution": list(args.resolution),
            "cycles_samples": args.samples,
            "stimulus_mode": args.mode,
            "stimulus_label": stim.MODES[args.mode]["label"],
            "n_objects": int(args.n_peaks),
            "anchor_separation_m": float(args.separation_m),
            "probe_ladder": list(fm_bank.PROBE_LADDER),
            "appearances": {k: dict(v) for k, v in stim.APPEARANCES.items()},
            "scenes": {}}

    t0 = time.time()
    for k, entry in enumerate(mine):
        lid = entry["layout"]["layout_id"]
        sdir = os.path.join(args.out, lid)
        os.makedirs(sdir, exist_ok=True)

        # Keyed on the anchor, not the scene: a probe must be its anchor with
        # one landmark moved, not a different set of objects. crc32, not hash(),
        # because Python randomises string hashing per process.
        rng = np.random.default_rng(fm_bank.identity_seed(entry["anchor"]))
        layout = stim.assign_objects(entry["layout"], rng, args.mode)
        blend = os.path.join(sdir, "scene.blend")
        stim.build_scene(blend, layout, args.mode, samples=args.samples,
                         resolution=tuple(args.resolution))
        renderer = FourMountainsRenderer(blend)
        frames, legend = render_scene(renderer, sdir, lid, azimuths,
                                      args.samples, not args.no_mask)

        bank["scenes"][lid] = {
            "layout_id": lid, "dir": lid, "role": entry["role"],
            "anchor": entry["anchor"], "d_pos": entry["d_pos"],
            "requested_d_pos": entry.get("requested_d_pos"),
            "d_bind": entry.get("d_bind", 0.0), "d_id": entry.get("d_id", 0.0),
            "n_objects": int(args.n_peaks), "stimulus_mode": args.mode,
            "layout": layout, "frames": frames, "mask_legend": legend,
        }
        if os.path.exists(blend):
            os.remove(blend)
        print(f"[{args.mode}] {k + 1}/{len(mine)}  {lid}  "
              f"{entry['role']}  d_pos={entry['d_pos']:.1f}m  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    with open(os.path.join(args.out, f"bank.shard{args.shard}.json"), "w") as f:
        json.dump(bank, f, indent=1)
    print(f"[{args.mode}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
