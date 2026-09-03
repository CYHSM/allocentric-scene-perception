"""
Four Mountains dataset generator and controller API for Blender 5.2+.

Wraps the generated .blend in a small Python API for moving peaks, orbiting the
camera, changing the time of day, and writing out RGB, instance masks, depth
and camera/landmark metadata.

    blender -b -P blender/four_mountains/render_dataset.py -- \
        --num_views 16 --output_dir data/four_mountains

Any number of mountains is supported; they are discovered from the scene rather
than hard-coded, and each peak's forest and boulder field are parented to it, so
they travel with it when it is moved.
"""

import argparse
import colorsys
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bpy
from mathutils import Euler, Vector

# Mirrors generate_scene.SUN_ROT_OFFSET; duplicated so this module can be used
# against a .blend without importing the generator.
SUN_ROT_OFFSET = 90.0

TERRAIN_MASK_COLOR = (0.15, 0.32, 0.15, 1.0)
WATER_MASK_COLOR = (0.10, 0.25, 0.55, 1.0)
TERRAIN_OBJECTS = ("Landscape", "Valley_Forest", "Valley_Rocks")


def _eevee_engine_id():
    """EEVEE's identifier moved between Blender versions; pick whatever exists."""
    items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if name in items:
            return name
    return "BLENDER_WORKBENCH"


def _mask_palette(n):
    """n visually separable, saturated colours for instance masks."""
    base = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.35, 1.0), (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
    if n <= len(base):
        return [(*c, 1.0) for c in base[:n]]
    return [(*colorsys.hsv_to_rgb(i / n, 0.95, 1.0), 1.0) for i in range(n)]


class FourMountainsRenderer:
    """Controller for the Four Mountains procedural environment."""

    def __init__(self, blend_path=None):
        if blend_path is None:
            blend_path = os.path.join(_HERE, "four_mountains.blend")
        self.blend_path = os.path.abspath(blend_path)
        if not os.path.exists(self.blend_path):
            raise FileNotFoundError(f"Scene blend file not found: {self.blend_path}")

        bpy.ops.wm.open_mainfile(filepath=self.blend_path)
        self.scene = bpy.context.scene
        self.cam_obj = bpy.data.objects.get("Camera")
        self.target_obj = bpy.data.objects.get("CameraTarget")
        self.sun_obj = bpy.data.objects.get("Sun")

        names = list(self.scene.get("fm_mountains", []))
        if not names:  # older .blend files, or one edited by hand
            names = sorted(o.name for o in bpy.data.objects
                           if o.name.startswith("M") and o.name[1:].isdigit())
        self.mountains = {n: bpy.data.objects[n] for n in names if n in bpy.data.objects}
        self.types = list(self.scene.get("fm_types", []))

        self._setup_mask_materials()

    # ------------------------------------------------------------------ #
    # Scene control
    # ------------------------------------------------------------------ #

    def set_mountain_position(self, name, x, y, rot_z=None, z=None):
        """Move a peak on the valley floor. Its forest and boulders follow."""
        if name not in self.mountains:
            raise ValueError(f"Unknown mountain {name!r}. Have: {sorted(self.mountains)}")
        obj = self.mountains[name]
        obj.location.x = float(x)
        obj.location.y = float(y)
        if z is not None:
            obj.location.z = float(z)
        if rot_z is not None:
            obj.rotation_euler.z = float(rot_z)
        bpy.context.view_layer.update()

    def swap_mountains(self, name_a, name_b):
        """Exchange two peaks' positions -- the classic FMT distractor foil."""
        a, b = self.mountains[name_a], self.mountains[name_b]
        pa, pb = a.location.copy(), b.location.copy()
        a.location, b.location = pb, pa
        bpy.context.view_layer.update()

    def get_mountain_positions(self):
        return {
            name: {
                "x": float(obj.location.x),
                "y": float(obj.location.y),
                "z": float(obj.location.z),
                "rot_z": float(obj.rotation_euler.z),
            }
            for name, obj in self.mountains.items()
        }

    def set_camera_orbit(self, azimuth_deg, elevation_deg=11.0, radius=88.0, target_z=6.0):
        """Place the camera on a spherical orbit around the valley centre."""
        phi = math.radians(elevation_deg)
        theta = math.radians(azimuth_deg)
        self.cam_obj.location = Vector((
            radius * math.cos(phi) * math.cos(theta),
            radius * math.cos(phi) * math.sin(theta),
            radius * math.sin(phi) + target_z,
        ))
        if self.target_obj:
            self.target_obj.location.z = target_z
        bpy.context.view_layer.update()

    def set_sun(self, elevation_deg=22.0, azimuth_deg=150.0, energy=None):
        """
        Set the time of day. `azimuth_deg` is the bearing the light comes from;
        the sky node and the sun lamp are kept in step.
        """
        elev = math.radians(elevation_deg)
        rot = math.radians(azimuth_deg + SUN_ROT_OFFSET)
        if self.sun_obj:
            if energy is not None:
                self.sun_obj.data.energy = energy
            self.sun_obj.rotation_euler = Euler((math.pi / 2 - elev, 0.0, rot), "XYZ")
        if self.scene.world and self.scene.world.node_tree:
            for node in self.scene.world.node_tree.nodes:
                if node.type == "TEX_SKY":
                    node.sun_elevation = elev
                    node.sun_rotation = rot
        bpy.context.view_layer.update()

    # ------------------------------------------------------------------ #
    # Ground truth
    # ------------------------------------------------------------------ #

    def _setup_mask_materials(self):
        """Flat emission materials, one colour per instance."""
        self.mask_materials = {}
        palette = _mask_palette(len(self.mountains))

        assignments = {"__terrain__": TERRAIN_MASK_COLOR, "__water__": WATER_MASK_COLOR}
        for name, col in zip(sorted(self.mountains), palette):
            assignments[name] = col

        for key, col in assignments.items():
            mat = bpy.data.materials.new(name=f"Mask_{key.strip('_')}")
            mat.use_nodes = True
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            nodes.clear()
            out = nodes.new("ShaderNodeOutputMaterial")
            emit = nodes.new("ShaderNodeEmission")
            emit.inputs["Color"].default_value = col
            emit.inputs["Strength"].default_value = 1.0
            links.new(emit.outputs["Emission"], out.inputs["Surface"])
            self.mask_materials[key] = mat

        # Which mask colour every renderable object should take.
        self.mask_assignment = {}
        for name, obj in self.mountains.items():
            self.mask_assignment[obj.name] = name
            for child in obj.children:  # forest + boulder fields
                self.mask_assignment[child.name] = name
        for name in TERRAIN_OBJECTS:
            if name in bpy.data.objects:
                self.mask_assignment[name] = "__terrain__"
        if "Lake" in bpy.data.objects:
            self.mask_assignment["Lake"] = "__water__"

    def mask_legend(self):
        """{object class -> RGB 0-255} so masks can be decoded downstream."""
        legend = {}
        for key, mat in self.mask_materials.items():
            col = mat.node_tree.nodes["Emission"].inputs["Color"].default_value
            label = {"__terrain__": "Landscape", "__water__": "Water"}.get(key, key)
            legend[label] = [int(round(c * 255)) for c in col[:3]]
        return legend

    def render_rgb(self, filepath, samples=None):
        self.scene.render.engine = "CYCLES"
        if samples:
            self.scene.cycles.samples = samples
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        return filepath

    def render_mask(self, filepath):
        """Flat-shaded instance segmentation, rendered in EEVEE."""
        originals = {}
        for obj_name, key in self.mask_assignment.items():
            obj = bpy.data.objects.get(obj_name)
            if obj and obj.data and obj.data.materials:
                originals[obj_name] = obj.data.materials[0]
                obj.data.materials[0] = self.mask_materials[key]

        r = self.scene.render
        vs = self.scene.view_settings
        saved = (r.engine, vs.view_transform, vs.look, vs.exposure, self.scene.world)

        # A flat colour response, and a black background for "no object".
        r.engine = _eevee_engine_id()
        vs.view_transform = "Standard"
        vs.look = "None"
        vs.exposure = 0.0
        black = bpy.data.worlds.new("MaskWorld")
        black.use_nodes = True
        black.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
        self.scene.world = black

        r.filepath = filepath
        bpy.ops.render.render(write_still=True)

        r.engine, vs.view_transform, vs.look, vs.exposure, self.scene.world = saved
        for obj_name, mat in originals.items():
            bpy.data.objects[obj_name].data.materials[0] = mat
        bpy.data.worlds.remove(black)
        return filepath

    def render_sample(self, output_dir, sample_idx, azimuth_deg, elevation_deg=11.0,
                      radius=88.0, target_z=6.0, samples=None, write_mask=True):
        """Render one RGB frame, its instance mask, and a metadata sidecar."""
        os.makedirs(output_dir, exist_ok=True)
        self.set_camera_orbit(azimuth_deg, elevation_deg, radius, target_z)

        base = f"sample_{sample_idx:04d}_az{int(round(azimuth_deg)):03d}"
        rgb_path = os.path.join(output_dir, f"{base}_rgb.png")
        mask_path = os.path.join(output_dir, f"{base}_mask.png")
        meta_path = os.path.join(output_dir, f"{base}_meta.json")

        self.render_rgb(rgb_path, samples=samples)
        if write_mask:
            self.render_mask(mask_path)

        sun_rot = self.sun_obj.rotation_euler if self.sun_obj else Euler((0, 0, 0))
        metadata = {
            "sample_index": sample_idx,
            "camera": {
                "azimuth_deg": float(azimuth_deg),
                "elevation_deg": float(elevation_deg),
                "radius": float(radius),
                "target_z": float(target_z),
                "lens_mm": float(self.cam_obj.data.lens),
                "position": [float(c) for c in self.cam_obj.location],
                "rotation_quaternion": [float(q) for q in self.cam_obj.matrix_world.to_quaternion()],
            },
            "sun": {
                "elevation_deg": float(90.0 - math.degrees(sun_rot.x)),
                "azimuth_deg": float(math.degrees(sun_rot.z) - SUN_ROT_OFFSET),
            },
            "mountains": self.get_mountain_positions(),
            "mountain_types": dict(zip(sorted(self.mountains), self.types)),
            "mask_legend": self.mask_legend(),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[FourMountains] sample {sample_idx}: az={azimuth_deg:.0f} -> {rgb_path}")
        return rgb_path, (mask_path if write_mask else None), meta_path


def main():
    parser = argparse.ArgumentParser(description="Render a Four Mountains dataset")
    parser.add_argument("--blend", default=None)
    parser.add_argument("--output_dir", default=os.path.join(_HERE, "renders"))
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--elevation", type=float, default=11.0)
    parser.add_argument("--radius", type=float, default=88.0)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=None)
    parser.add_argument("--no_mask", action="store_true")

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    renderer = FourMountainsRenderer(blend_path=args.blend)
    if args.resolution:
        renderer.scene.render.resolution_x, renderer.scene.render.resolution_y = args.resolution

    print(f"Rendering {args.num_views} viewpoints of "
          f"{len(renderer.mountains)} mountains ({', '.join(renderer.types)})")
    for idx, az in enumerate(np.linspace(0, 360, args.num_views, endpoint=False)):
        renderer.render_sample(
            output_dir=args.output_dir, sample_idx=idx, azimuth_deg=float(az),
            elevation_deg=args.elevation, radius=args.radius,
            samples=args.samples, write_mask=not args.no_mask)
    print("All renders complete.")


if __name__ == "__main__":
    main()
