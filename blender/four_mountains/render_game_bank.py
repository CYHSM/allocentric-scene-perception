"""
Render 360-degree viewpoint bank for the Four Mountains web game.

Renders the canonical valley and 3 distinct distractor foils across 24 bearings
(15-degree steps) using Cycles on GPU at 17 degrees elevation, ensuring all
four landmark mountains and the central tarn are always visible.

Outputs:
  game/renders/bank/
    canonical_az000.png ... canonical_az345.png
    foil_swap12_az000.png ... foil_swap12_az345.png
    foil_swap34_az000.png ... foil_swap34_az345.png
    foil_swap13_az000.png ... foil_swap13_az345.png
    manifest.json
"""

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bpy
from render_dataset import FourMountainsRenderer

CONFIGS = [
    {
        "id": "canonical",
        "name": "Canonical Four Mountains",
        "type": "target",
        "description": "Original landmark arrangement",
        "modifications": []
    },
    {
        "id": "foil_swap12",
        "name": "Foil: Peaks 1 & 2 Swapped",
        "type": "foil",
        "description": "M1 (Matterhorn) and M2 (Ridge) traded places across the northern valley.",
        "modifications": [("swap", "M1", "M2")]
    },
    {
        "id": "foil_swap34",
        "name": "Foil: Peaks 3 & 4 Swapped",
        "type": "foil",
        "description": "M3 (Tableland mesa) and M4 (Dome) traded places across the southern valley.",
        "modifications": [("swap", "M3", "M4")]
    },
    {
        "id": "foil_swap13",
        "name": "Foil: Peaks 1 & 3 Swapped",
        "type": "foil",
        "description": "M1 (Matterhorn) and M3 (Tableland mesa) traded places across the western valley.",
        "modifications": [("swap", "M1", "M3")]
    }
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", default=os.path.join(_HERE, "four_mountains.blend"))
    parser.add_argument("--output_dir", default=os.path.abspath(os.path.join(_HERE, "../../game/renders/bank")))
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 440])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--elevation", type=float, default=17.0)
    parser.add_argument("--radius", type=float, default=94.0)
    parser.add_argument("--target_z", type=float, default=6.0)
    parser.add_argument("--step_deg", type=int, default=15)

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    angles = list(range(0, 360, args.step_deg))
    total_images = len(CONFIGS) * len(angles)

    print("=" * 70)
    print(f"Four Mountains Game Bank Renderer (Cycles GPU)")
    print(f"Configurations : {len(CONFIGS)} ({', '.join(c['id'] for c in CONFIGS)})")
    print(f"Viewpoints     : {len(angles)} bearings ({args.step_deg}° step)")
    print(f"Total Frames   : {total_images}")
    print(f"Camera Orbit   : elev={args.elevation}°, radius={args.radius}, target_z={args.target_z}")
    print(f"Resolution     : {args.resolution[0]}x{args.resolution[1]}, samples={args.samples}")
    print(f"Output         : {args.output_dir}")
    print("=" * 70)

    renderer = FourMountainsRenderer(blend_path=args.blend)
    renderer.scene.render.resolution_x = args.resolution[0]
    renderer.scene.render.resolution_y = args.resolution[1]

    # Save original mountain positions so we can reset cleanly
    base_positions = renderer.get_mountain_positions()
    mountain_types = dict(zip(sorted(renderer.mountains), renderer.types))

    manifest = {
        "camera": {
            "elevation_deg": args.elevation,
            "radius": args.radius,
            "target_z": args.target_z,
            "fov_lens_mm": float(renderer.cam_obj.data.lens),
            "step_deg": args.step_deg,
            "angles": angles
        },
        "mountain_types": mountain_types,
        "valleys": {}
    }

    start_time = time.time()
    frame_count = 0

    for cfg in CONFIGS:
        cid = cfg["id"]
        print(f"\n>>> Setting up valley: {cid} ({cfg['name']})")

        # Reset to base positions
        for mname, pos in base_positions.items():
            renderer.set_mountain_position(mname, pos["x"], pos["y"], pos["rot_z"], pos["z"])

        # Apply modifications for this valley
        for mod in cfg["modifications"]:
            if mod[0] == "swap":
                renderer.swap_mountains(mod[1], mod[2])
                print(f"    Swapped {mod[1]} <-> {mod[2]}")

        current_positions = renderer.get_mountain_positions()
        valley_data = {
            "id": cid,
            "name": cfg["name"],
            "type": cfg["type"],
            "description": cfg["description"],
            "mountains": current_positions,
            "images": {}
        }

        for az in angles:
            fname = f"{cid}_az{az:03d}.png"
            out_path = os.path.join(args.output_dir, fname)
            rel_path = f"renders/bank/{fname}"

            # Only render if not already rendered
            if not os.path.exists(out_path):
                t0 = time.time()
                renderer.set_camera_orbit(az, args.elevation, args.radius, args.target_z)
                renderer.render_rgb(out_path, samples=args.samples)
                dt = time.time() - t0
                frame_count += 1
                elapsed = time.time() - start_time
                avg = elapsed / max(frame_count, 1)
                rem = (total_images - frame_count) * avg
                print(f"  [{frame_count:02d}/{total_images:02d}] {fname} ({dt:.2f}s | rem: {rem/60:.1f}m)")
            else:
                frame_count += 1
                print(f"  [{frame_count:02d}/{total_images:02d}] {fname} (cached)")

            valley_data["images"][str(az)] = rel_path

        manifest["valleys"][cid] = valley_data

        # Write progressive manifest
        manifest_path = os.path.join(args.output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nAll {total_images} frames rendered in {total_time/60:.2f} minutes!")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
