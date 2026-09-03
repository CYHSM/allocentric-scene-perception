"""
Four Mountains Dataset Generator and Renderer for Blender 5.2+
Provides a complete Python API to manipulate mountain positions, camera orbits,
sun/weather parameters, and render RGB, depth maps, and instance masks.
"""

import sys
import os
import math
import json
import argparse
import numpy as np
import bpy
from mathutils import Vector, Euler


class FourMountainsRenderer:
    """Controller for the Four Mountains procedural environment."""

    def __init__(self, blend_path="blender/four_mountains/four_mountains.blend"):
        self.blend_path = os.path.abspath(blend_path)
        if not os.path.exists(self.blend_path):
            raise FileNotFoundError(f"Scene blend file not found: {self.blend_path}")

        bpy.ops.wm.open_mainfile(filepath=self.blend_path)
        self.scene = bpy.context.scene
        self.cam_obj = bpy.data.objects.get("Camera")
        self.target_obj = bpy.data.objects.get("CameraTarget")
        self.sun_obj = bpy.data.objects.get("Sun")
        self.mountains = {f"M{i}": bpy.data.objects.get(f"M{i}") for i in range(1, 5)}

        # Setup emission material for instant segmentation masks
        self._setup_mask_materials()

    def _setup_mask_materials(self):
        """Create flat color emission materials for ground truth instance segmentation."""
        self.mask_materials = {}
        # Colors mapped by pass_id:
        # 1: Valley (dark green)
        # 2: Backdrop (blue-gray)
        # 3: M1 (Red)
        # 4: M2 (Green)
        # 5: M3 (Blue)
        # 6: M4 (Yellow)
        palette = {
            "Landscape": (0.2, 0.4, 0.2, 1.0),
            "M1": (1.0, 0.0, 0.0, 1.0),
            "M2": (0.0, 1.0, 0.0, 1.0),
            "M3": (0.0, 0.0, 1.0, 1.0),
            "M4": (1.0, 1.0, 0.0, 1.0),
        }
        for name, col in palette.items():
            mat = bpy.data.materials.new(name=f"Mask_{name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            out = nodes.new("ShaderNodeOutputMaterial")
            emit = nodes.new("ShaderNodeEmission")
            emit.inputs["Color"].default_value = col
            emit.inputs["Strength"].default_value = 1.0
            links.new(emit.outputs["Emission"], out.inputs["Surface"])
            self.mask_materials[name] = mat

    def set_mountain_position(self, name, x, y, rot_z=None):
        """Move a mountain to coordinate (x, y). Optionally adjust rotation."""
        if name not in self.mountains or not self.mountains[name]:
            raise ValueError(f"Unknown mountain: {name}")
        obj = self.mountains[name]
        obj.location.x = float(x)
        obj.location.y = float(y)
        if rot_z is not None:
            obj.rotation_euler.z = float(rot_z)
        bpy.context.view_layer.update()

    def get_mountain_positions(self):
        """Return dict of current mountain coordinates."""
        return {
            name: {
                "x": float(obj.location.x),
                "y": float(obj.location.y),
                "z": float(obj.location.z),
                "rot_z": float(obj.rotation_euler.z),
            }
            for name, obj in self.mountains.items()
            if obj
        }

    def set_camera_orbit(self, azimuth_deg, elevation_deg=28.0, radius=52.0, target_z=4.0):
        """Set camera along a spherical orbit around the valley center."""
        phi = math.radians(elevation_deg)
        theta = math.radians(azimuth_deg)

        x = radius * math.cos(phi) * math.cos(theta)
        y = radius * math.cos(phi) * math.sin(theta)
        z = radius * math.sin(phi) + target_z

        self.cam_obj.location = Vector((x, y, z))
        if self.target_obj:
            self.target_obj.location.z = target_z
        bpy.context.view_layer.update()

    def set_sun(self, elevation_deg=24.0, azimuth_deg=52.0, energy=2.5):
        """Set sun position and Nishita sky lighting."""
        sun_elev = math.radians(elevation_deg)
        sun_rot = math.radians(azimuth_deg)

        if self.sun_obj:
            self.sun_obj.data.energy = energy
            self.sun_obj.rotation_euler = Euler((math.pi / 2 - sun_elev, 0.0, sun_rot), "XYZ")

        # Update Nishita sky node in World
        if self.scene.world and self.scene.world.node_tree:
            for node in self.scene.world.node_tree.nodes:
                if node.type == "TEX_SKY":
                    node.sun_elevation = sun_elev
                    node.sun_rotation = sun_rot

        bpy.context.view_layer.update()

    def render_rgb(self, filepath, samples=48):
        """Render photorealistic RGB image with Cycles."""
        self.scene.render.engine = "CYCLES"
        self.scene.cycles.samples = samples
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        return filepath

    def render_mask(self, filepath):
        """Render crisp instance segmentation mask."""
        # Temporarily assign emission materials to objects
        original_mats = {}
        for name, obj in self.mountains.items():
            if obj and obj.data.materials:
                original_mats[name] = obj.data.materials[0]
                obj.data.materials[0] = self.mask_materials[name]

        landscape_obj = bpy.data.objects.get("Landscape")
        if landscape_obj and landscape_obj.data.materials:
            original_mats["Landscape"] = landscape_obj.data.materials[0]
            landscape_obj.data.materials[0] = self.mask_materials["Landscape"]

        # Fast render for masks
        orig_engine = self.scene.render.engine
        self.scene.render.engine = "BLENDER_EEVEE"
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

        # Restore original materials and engine
        for name, orig_mat in original_mats.items():
            obj = bpy.data.objects.get(name)
            if obj:
                obj.data.materials[0] = orig_mat
        self.scene.render.engine = orig_engine
        return filepath

    def render_sample(self, output_dir, sample_idx, azimuth_deg, elevation_deg=28.0, radius=52.0):
        """Render a full sample with RGB, mask, and metadata JSON."""
        os.makedirs(output_dir, exist_ok=True)
        self.set_camera_orbit(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, radius=radius)

        base_name = f"sample_{sample_idx:04d}_az{int(azimuth_deg):03d}"
        rgb_path = os.path.join(output_dir, f"{base_name}_rgb.png")
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        meta_path = os.path.join(output_dir, f"{base_name}_meta.json")

        self.render_rgb(rgb_path)
        self.render_mask(mask_path)

        metadata = {
            "sample_index": sample_idx,
            "camera": {
                "azimuth_deg": azimuth_deg,
                "elevation_deg": elevation_deg,
                "radius": radius,
                "position": [float(c) for c in self.cam_obj.location],
                "rotation_quaternion": [float(q) for q in self.cam_obj.matrix_world.to_quaternion()],
            },
            "mountains": self.get_mountain_positions(),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[FourMountains] Rendered sample {sample_idx}: azimuth={azimuth_deg}° -> {rgb_path}")
        return rgb_path, mask_path, meta_path


def main():
    parser = argparse.ArgumentParser(description="Render Four Mountains Dataset")
    parser.add_argument("--blend", default="blender/four_mountains/four_mountains.blend")
    parser.add_argument("--output_dir", default="blender/four_mountains/renders")
    parser.add_argument("--num_views", type=int, default=4, help="Number of viewpoints around 360° orbit")
    parser.add_argument("--elevation", type=float, default=28.0)
    parser.add_argument("--radius", type=float, default=52.0)

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    renderer = FourMountainsRenderer(blend_path=args.blend)
    angles = np.linspace(0, 360, args.num_views, endpoint=False)

    print(f"Rendering {args.num_views} viewpoints around the Four Mountains scene...")
    for idx, az in enumerate(angles):
        renderer.render_sample(
            output_dir=args.output_dir,
            sample_idx=idx,
            azimuth_deg=az,
            elevation_deg=args.elevation,
            radius=args.radius
        )
    print("All renders complete!")


if __name__ == "__main__":
    main()
