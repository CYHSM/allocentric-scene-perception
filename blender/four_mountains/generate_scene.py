"""
Procedural Four Mountains Scene Generator for Blender 5.2+.

Builds a full alpine world: movable Four-Mountains-Task peaks drawn from a
library of twelve landforms, a lake-bearing valley, layered foothills, a
distant snow range on the horizon, a physical sky with a cumulus deck, conifer
forests, boulder fields and atmospheric aerial perspective.

    blender -b -P blender/four_mountains/generate_scene.py
    blender -b -P blender/four_mountains/generate_scene.py -- --types horn,caldera,sawtooth,butte
"""

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bpy
from mathutils import Euler, Vector

from fm_materials import (add_sky_clouds, create_alpine_material,
                          create_tree_material, create_water_material)
from fm_morphology import build_heightfield, morphology_names
from fm_noise import fbm, ridged_fbm, smoothstep
from fm_scatter import (mesh_from_arrays, mesh_from_grid, parent_keep_local,
                        scatter_boulders, scatter_conifers)

# --------------------------------------------------------------------------- #
# World constants
# --------------------------------------------------------------------------- #

WATER_LEVEL = 0.0        # the lake surface is the vertical datum of the world
VALLEY_FLOOR = 1.7       # meadow height above the waterline
LAKE_RADIUS = 9.5
VALLEY_RADIUS = 56.0
WORLD_RADIUS = 1400.0
SNOW_LINE = 21.0
TREE_LINE = 13.0

# Pass indices: 1 = terrain, 2 = water, 3.. = mountains (and their vegetation).
PASS_TERRAIN = 1
PASS_WATER = 2
PASS_MOUNTAIN_BASE = 3

DEFAULT_TYPES = ["horn", "ridge", "mesa", "dome"]
DEFAULT_AZIMUTHS = [140.0, 40.0, 229.0, 318.0]
DEFAULT_HEIGHTS = [34.0, 27.0, 24.0, 30.0]
DEFAULT_RADII = [17.0, 19.0, 15.0, 18.0]
MOUNTAIN_RING = 34.0
MOUNTAIN_SINK = 0.45     # how deep each peak's rim is bedded into the meadow


def clean_scene():
    """Start from an empty file so re-runs are deterministic."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def configure_render_engine(scene, engine="CYCLES", samples=96, resolution=(1024, 1024)):
    """Cycles with Metal GPU when available, and a neutral filmic response."""
    scene.render.engine = engine
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.preview_samples = 32
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 8
        scene.cycles.transmission_bounces = 8
        scene.cycles.transparent_max_bounces = 12

        try:
            cpref = bpy.context.preferences.addons["cycles"].preferences
            cpref.get_devices()
            metal = [d for d in cpref.devices if d.type == "METAL"]
            if metal:
                cpref.compute_device_type = "METAL"
                for d in metal:
                    d.use = True
                scene.cycles.device = "GPU"
                print(f"[Blender] Metal GPU: {[d.name for d in metal]}")
            else:
                scene.cycles.device = "CPU"
        except Exception as exc:  # pragma: no cover - depends on local hardware
            print(f"[Blender] Cycles device setup skipped: {exc}")

    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "AgX - Punchy"
    scene.view_settings.exposure = -2.6


# Blender's sky node and a sun lamp both take a rotation that leads the sun's
# compass azimuth by 90 degrees; SUN_ROT_OFFSET keeps the two in step.
SUN_ROT_OFFSET = 90.0


def setup_lighting(scene, sun_elevation_deg=22.0, sun_azimuth_deg=150.0,
                   sun_energy=75.0, haze=0.35):
    """
    Nishita multiple-scattering sky plus a matched directional sun.

    `sun_azimuth_deg` is the compass bearing the sunlight comes *from*. The
    default cross-lights the canonical viewpoint so topography reads in relief.
    """
    world = bpy.data.worlds.new("AlpineWorld")
    scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    out = tree.nodes.new("ShaderNodeOutputWorld")
    out.location = (400, 0)
    bg = tree.nodes.new("ShaderNodeBackground")
    bg.location = (200, 0)
    sky = tree.nodes.new("ShaderNodeTexSky")
    sky.location = (-100, 0)

    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(sun_elevation_deg)
    sky.sun_rotation = math.radians(sun_azimuth_deg + SUN_ROT_OFFSET)
    sky.air_density = 1.0
    sky.aerosol_density = haze
    sky.ozone_density = 2.6
    sky.sun_intensity = 1.0
    # The sun disc is provided by a real Sun lamp (cleaner, less noisy shadows).
    if hasattr(sky, "sun_disc"):
        sky.sun_disc = False

    bg.inputs["Strength"].default_value = 1.0
    sky_with_clouds = add_sky_clouds(tree, sky.outputs["Color"])
    tree.links.new(sky_with_clouds, bg.inputs["Color"])
    tree.links.new(bg.outputs["Background"], out.inputs["Surface"])

    sun_data = bpy.data.lights.new(name="SunLight", type="SUN")
    sun_data.energy = sun_energy
    sun_data.color = (1.0, 0.94, 0.85)
    sun_data.angle = math.radians(0.7)

    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = Euler(
        (math.pi / 2 - math.radians(sun_elevation_deg), 0.0,
         math.radians(sun_azimuth_deg + SUN_ROT_OFFSET)), "XYZ")
    sun.location = (0, 0, 200)
    return sun


# --------------------------------------------------------------------------- #
# Terrain
# --------------------------------------------------------------------------- #

_KNOTS_R = [0.0, 45.0, VALLEY_RADIUS, 85.0, 130.0, 190.0, 290.0, 450.0, 750.0,
            1080.0, WORLD_RADIUS]


def _terrain_profile(R):
    """Base elevation envelope: valley -> foothills -> mid range -> horizon wall."""
    knots_z = [VALLEY_FLOOR, VALLEY_FLOOR + 0.5, 3.4, 8.0, 15.0, 21.0,
               26.0, 34.0, 44.0, 54.0, 58.0]
    return np.interp(R, _KNOTS_R, knots_z)


def _terrain_relief_amp(R):
    """How much ridged relief rides on top of the envelope at each distance."""
    knots_a = [0.22, 0.55, 2.20, 7.00, 16.0, 28.0, 38.0, 50.0, 62.0, 72.0, 78.0]
    return np.interp(R, _KNOTS_R, knots_a)


def generate_landscape(material, n_theta=420, seed=77):
    """
    One seamless radial terrain sheet carrying every distance band:
    lake basin, valley floor, foothills, mid range and the horizon skyline.
    """
    r_near = np.linspace(0.0, VALLEY_RADIUS, 96)
    r_far = np.geomspace(VALLEY_RADIUS, WORLD_RADIUS, 190)[1:]
    r_vals = np.concatenate([r_near, r_far])
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(r_vals, theta, indexing="ij")

    X = R * np.cos(T)
    Y = R * np.sin(T)

    # Three ridged octave families: range structure, spurs, and gully detail.
    coarse = ridged_fbm(X, Y, freq=0.0034, octaves=6, seed=seed)
    medium = ridged_fbm(X, Y, freq=0.017, octaves=5, seed=seed + 300)
    fine = ridged_fbm(X, Y, freq=0.055, octaves=4, seed=seed + 600)
    rolling = fbm(X, Y, freq=0.05, octaves=4, seed=seed + 900) - 0.5

    relief = 0.42 * coarse + 0.42 * medium + 0.16 * fine * np.clip(600.0 / (R + 80.0), 0.0, 1.0)
    # Centre the relief so ridges alternate with real basins instead of piling
    # onto one monotonic ramp -- that is what produces layered ridgelines.
    Z = _terrain_profile(R) + _terrain_relief_amp(R) * (relief - 0.42)
    Z += rolling * np.clip(R / 35.0, 0.12, 1.0) * 0.8

    # Lake basin: an irregular shoreline carved into the valley floor.
    shore = LAKE_RADIUS * (0.82 + 0.36 * fbm(X * 3.2, Y * 3.2, freq=0.09,
                                             octaves=3, seed=seed + 41))
    u = R / shore
    Z -= (VALLEY_FLOOR + 3.4) * (1.0 - smoothstep(0.52, 1.15, u))

    obj = mesh_from_grid("Landscape", X, Y, Z)
    obj.data.materials.append(material)
    obj.pass_index = PASS_TERRAIN
    return obj, X, Y, Z


def create_lake(material, radius=14.0, segments=160):
    """Flat water disc filling the carved basin."""
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    verts = [(0.0, 0.0, WATER_LEVEL)]
    verts += [(radius * math.cos(a), radius * math.sin(a), WATER_LEVEL) for a in ang]
    faces = [(0, 1 + j, 1 + (j + 1) % segments) for j in range(segments)]
    obj = mesh_from_arrays("Lake", verts, faces)
    obj.data.materials.append(material)
    obj.pass_index = PASS_WATER
    return obj


# --------------------------------------------------------------------------- #
# Mountains
# --------------------------------------------------------------------------- #

def build_mountain(cfg, alpine_mat, tree_mat, rng, n_r=140, n_theta=320):
    """Create one peak plus the forest and boulder field that travel with it."""
    X, Y, Z = build_heightfield(
        cfg["type"], height=cfg["height"], base_radius=cfg["radius"],
        n_r=n_r, n_theta=n_theta, seed=cfg["seed"], relief=cfg.get("relief", 1.0))

    obj = mesh_from_grid(cfg["name"], X, Y, Z)
    obj.location = Vector(cfg["pos"])
    obj.rotation_euler = Euler((0.0, 0.0, cfg["rot_z"]), "XYZ")
    obj.data.materials.append(alpine_mat)
    obj.pass_index = cfg["pass_id"]

    children = []
    base_z = cfg["pos"][2]
    trees = scatter_conifers(
        f"{cfg['name']}_Forest", X, Y, Z,
        count=cfg.get("trees", 2200), rng=rng,
        z_min=max(0.9 - base_z, 0.4), z_max=TREE_LINE - base_z,
        min_slope=0.56, height_range=(0.75, 1.65))
    if trees:
        trees.data.materials.append(tree_mat)
        trees.pass_index = cfg["pass_id"]
        children.append(trees)

    rocks = scatter_boulders(
        f"{cfg['name']}_Rocks", X, Y, Z,
        count=cfg.get("boulders", 280), rng=rng,
        z_min=0.3, z_max=cfg["height"] * 0.75, min_slope=0.42,
        size_range=(0.10, 0.38))
    if rocks:
        rocks.data.materials.append(alpine_mat)
        rocks.pass_index = cfg["pass_id"]
        children.append(rocks)

    return obj, children


def default_mountain_configs(types=None, seed=0):
    """Four-peak FMT layout by default; any of the twelve landforms can be used."""
    types = list(types or DEFAULT_TYPES)
    n = len(types)
    azimuths = DEFAULT_AZIMUTHS if n == 4 else list(np.linspace(0, 360, n, endpoint=False) + 25.0)
    rng = np.random.default_rng(seed + 5150)

    configs = []
    for i, kind in enumerate(types):
        az = math.radians(azimuths[i % len(azimuths)])
        height = DEFAULT_HEIGHTS[i] if n == 4 else float(rng.uniform(12.5, 19.0))
        radius = DEFAULT_RADII[i] if n == 4 else float(rng.uniform(11.0, 15.0))
        ring = MOUNTAIN_RING if n <= 6 else MOUNTAIN_RING + 12.0
        configs.append({
            "name": f"M{i + 1}",
            "type": kind,
            "height": height,
            "radius": radius,
            "pos": (ring * math.cos(az), ring * math.sin(az), VALLEY_FLOOR - MOUNTAIN_SINK),
            "rot_z": float(rng.uniform(-math.pi, math.pi)),
            "seed": 101 + 97 * i + seed,
            "pass_id": PASS_MOUNTAIN_BASE + i,
            "trees": 2200,
            "boulders": 280,
        })
    return configs


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #

def setup_camera_rig(scene, lens=28.0, target_z=6.0):
    target = bpy.data.objects.new("CameraTarget", None)
    target.location = (0.0, 0.0, target_z)
    scene.collection.objects.link(target)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens
    cam_data.clip_start = 0.1
    cam_data.clip_end = 12000.0

    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    track = cam.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return cam, target


def position_camera_orbit(cam, radius=88.0, elevation_deg=11.0, azimuth_deg=55.0,
                          target_z=6.0):
    """Spherical orbit around the valley centre. Low elevation keeps sky in frame."""
    phi = math.radians(elevation_deg)
    theta = math.radians(azimuth_deg)
    cam.location = Vector((
        radius * math.cos(phi) * math.cos(theta),
        radius * math.cos(phi) * math.sin(theta),
        radius * math.sin(phi) + target_z,
    ))
    bpy.context.view_layer.update()


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_four_mountains_scene(output_blend_path=None, types=None, seed=0,
                               samples=96, resolution=(1024, 1024),
                               valley_trees=14000, valley_boulders=340):
    if output_blend_path is None:
        output_blend_path = os.path.join(_HERE, "four_mountains.blend")

    print("=" * 68)
    print("Building the Four Mountains alpine world (Blender 5.2)")
    print("=" * 68)

    clean_scene()
    scene = bpy.context.scene
    rng = np.random.default_rng(seed + 20250903)

    configure_render_engine(scene, samples=samples, resolution=resolution)
    setup_lighting(scene)

    alpine_mat = create_alpine_material(snow_line=SNOW_LINE, tree_line=TREE_LINE,
                                       water_level=WATER_LEVEL)
    water_mat = create_water_material()
    tree_mat = create_tree_material()

    print("  Terrain: valley, foothills, mid range and horizon skyline...")
    landscape, LX, LY, LZ = generate_landscape(alpine_mat, seed=77 + seed)
    scene.collection.objects.link(landscape)

    print("  Alpine tarn...")
    scene.collection.objects.link(create_lake(water_mat))

    configs = default_mountain_configs(types=types, seed=seed)
    names = []
    for cfg in configs:
        print(f"  {cfg['name']}: {cfg['type']:9s} h={cfg['height']:.1f}m r={cfg['radius']:.1f}m")
        peak, children = build_mountain(cfg, alpine_mat, tree_mat, rng)
        scene.collection.objects.link(peak)
        for child in children:
            scene.collection.objects.link(child)
            parent_keep_local(child, peak)
        names.append(cfg["name"])

    print("  Valley forest and boulder fields...")
    valley_forest = scatter_conifers(
        "Valley_Forest", LX, LY, LZ, count=valley_trees, rng=rng,
        z_min=VALLEY_FLOOR - 0.6, z_max=TREE_LINE, min_slope=0.62,
        height_range=(0.75, 1.75), radius_limit=270.0)
    if valley_forest:
        valley_forest.data.materials.append(tree_mat)
        valley_forest.pass_index = PASS_TERRAIN
        scene.collection.objects.link(valley_forest)

    valley_rocks = scatter_boulders(
        "Valley_Rocks", LX, LY, LZ, count=valley_boulders, rng=rng,
        z_min=WATER_LEVEL - 0.2, z_max=9.0, min_slope=0.40, size_range=(0.12, 0.42),
        radius_limit=115.0)
    if valley_rocks:
        valley_rocks.data.materials.append(alpine_mat)
        valley_rocks.pass_index = PASS_TERRAIN
        scene.collection.objects.link(valley_rocks)

    cam, _ = setup_camera_rig(scene)
    position_camera_orbit(cam)

    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True
    view_layer.use_pass_mist = True

    scene["fm_mountains"] = names
    scene["fm_types"] = [c["type"] for c in configs]
    scene["fm_water_level"] = WATER_LEVEL

    os.makedirs(os.path.dirname(output_blend_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)
    print(f"Saved scene -> {output_blend_path}")
    return output_blend_path


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Generate the Four Mountains scene")
    p.add_argument("--out", default=None, help="Output .blend path")
    p.add_argument("--types", default=None,
                   help=f"Comma-separated landforms. Available: {','.join(morphology_names())}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=96)
    p.add_argument("--resolution", type=int, nargs=2, default=(1024, 1024))
    p.add_argument("--valley_trees", type=int, default=14000)
    return p.parse_args(argv)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = _parse_args(argv)
    build_four_mountains_scene(
        output_blend_path=args.out,
        types=args.types.split(",") if args.types else None,
        seed=args.seed,
        samples=args.samples,
        resolution=tuple(args.resolution),
        valley_trees=args.valley_trees,
    )
    print("Done!")
