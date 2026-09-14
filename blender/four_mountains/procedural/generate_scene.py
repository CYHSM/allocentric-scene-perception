import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_FM_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) in ("procedural", "layout", "rendering") else _HERE
for _sub in (_FM_ROOT, os.path.join(_FM_ROOT, "layout"), os.path.join(_FM_ROOT, "procedural"), os.path.join(_FM_ROOT, "rendering")):
    if _sub not in sys.path:
        sys.path.insert(0, _sub)

"""
Procedural Four Mountains Scene Generator for Blender 5.2+.

Builds a full alpine world: movable Four-Mountains-Task peaks drawn from a
parametric shape space (see `fm_peak`), a lake-bearing valley, layered
foothills, a distant snow range on the horizon, a physical sky with a cumulus
deck, conifer forests, boulder fields and atmospheric aerial perspective.

    blender -b -P blender/four_mountains/generate_scene.py
    blender -b -P blender/four_mountains/generate_scene.py -- --n_peaks 5 --seed 3
"""

import argparse
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

from fm_materials import (add_sky_clouds, create_alpine_material,
                          create_broadleaf_material, create_deadwood_material,
                          create_trail_material, create_tree_material,
                          create_water_material)
from fm_noise import fbm, ridged_fbm, smoothstep
from fm_scatter import (mesh_from_arrays, mesh_from_grid, parent_keep_local,
                        distance_to_path, meander_path, ribbon_mesh,
                        scatter_boulders, scatter_broadleaf, scatter_conifers,
                        scatter_logs, scatter_reeds, scatter_shrubs,
                        scatter_snags, scatter_talus)

# --------------------------------------------------------------------------- #
# World constants
# --------------------------------------------------------------------------- #

WATER_LEVEL = 0.0        # the lake surface is the vertical datum of the world
VALLEY_FLOOR = 1.7       # meadow height above the waterline
LAKE_RADIUS = 9.5
VALLEY_RADIUS = 56.0
WORLD_RADIUS = 1400.0
# Biome lines, in metres above the waterline. Sampled peaks run ~17-30 m, so a
# 21 m snow line left them bare and a 13 m tree line ran forest most of the way
# up them.
SNOW_LINE = 18.0
TREE_LINE = 10.0

# Pass indices: 1 = terrain, 2 = water, 3.. = mountains (and their vegetation).
PASS_TERRAIN = 1
PASS_WATER = 2
PASS_MOUNTAIN_BASE = 3

DEFAULT_AZIMUTHS = [140.0, 40.0, 229.0, 318.0]
DEFAULT_HEIGHTS = [34.0, 27.0, 24.0, 30.0]
DEFAULT_RADII = [17.0, 19.0, 15.0, 18.0]
MOUNTAIN_RING = 34.0
MOUNTAIN_SINK = 0.45     # how deep each peak's rim is bedded into the meadow


def clean_scene():
    """Start from an empty file so re-runs are deterministic."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _enable_gpu(scene):
    """
    Point Cycles at whatever accelerator this machine has.

    Apple Silicon exposes METAL; the A100 boxes expose OPTIX and CUDA for the
    same devices, and OPTIX is preferred because Cycles uses the RT cores.
    Tried in order, first hit wins, CPU if none -- the same scene therefore
    renders on the laptop and on the cluster without a code change.

    `CYCLES_GPU` overrides the choice ("OPTIX", "CUDA", "METAL", "CPU"), which
    matters on a shared box where another user may already hold a device.
    """
    import os

    forced = os.environ.get("CYCLES_GPU", "").strip().upper()
    order = [forced] if forced else ["METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"]

    try:
        cpref = bpy.context.preferences.addons["cycles"].preferences
        if forced == "CPU":
            scene.cycles.device = "CPU"
            print("[Blender] Cycles: CPU (forced)")
            return
        for backend in order:
            try:
                cpref.compute_device_type = backend
            except TypeError:
                continue                      # this build has no such backend
            cpref.get_devices()
            devices = [d for d in cpref.devices if d.type == backend]
            if not devices:
                continue
            for d in cpref.devices:
                d.use = (d.type == backend)
            scene.cycles.device = "GPU"
            print(f"[Blender] Cycles {backend}: {[d.name for d in devices]}")
            return
        scene.cycles.device = "CPU"
        print("[Blender] Cycles: no GPU backend found, using CPU")
    except Exception as exc:  # pragma: no cover - depends on local hardware
        print(f"[Blender] Cycles device setup skipped: {exc}")


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
        # The bank renders many azimuths of a *static* scene, so re-syncing and
        # re-building the BVH for every frame is pure waste. Persistent data
        # keeps the device-side scene between renders; it costs memory, which is
        # abundant here (a few GB against 80).
        scene.render.use_persistent_data = True
        scene.cycles.max_bounces = 8
        scene.cycles.transmission_bounces = 8
        scene.cycles.transparent_max_bounces = 12

        _enable_gpu(scene)

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
                   sun_energy=75.0, haze=0.35, clouds=True,
                   cloud_altitude=1.0, cloud_coverage=(0.40, 0.61)):
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
    if clouds:
        sky_color = add_sky_clouds(tree, sky.outputs["Color"],
                                   altitude=cloud_altitude, coverage=cloud_coverage)
    else:
        sky_color = sky.outputs["Color"]
    tree.links.new(sky_color, bg.inputs["Color"])
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


STREAM_COUNT = 3
STREAM_WIDTH = 2.1        # metres, half-width of the carved channel
STREAM_DEPTH = 1.15
TRAIL_COUNT = 2
TRAIL_WIDTH = 0.85


def _carve_channels(X, Y, Z, paths, width, depth):
    """
    Cut a smooth V into the terrain along each path.

    A stream drawn as a ribbon laid on flat ground reads as a painted stripe.
    Carving first means the banks actually fall toward the water and the
    vegetation masks follow the valley, which is most of what sells it.
    """
    for path in paths:
        d = distance_to_path(X, Y, path)
        Z = Z - depth * np.exp(-((d / width) ** 2))
    return Z


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

    # Streams: sourced high on the surrounding slopes, draining to the tarn.
    srng = np.random.default_rng(seed + 555)
    stream_paths = []
    for k in range(STREAM_COUNT):
        a = 2 * np.pi * (k + srng.uniform(0.15, 0.85)) / STREAM_COUNT
        start = (78.0 * np.cos(a), 78.0 * np.sin(a))
        end = (LAKE_RADIUS * 0.7 * np.cos(a), LAKE_RADIUS * 0.7 * np.sin(a))
        stream_paths.append(meander_path(start, end, srng, steps=72, wobble=7.5))
    Z = _carve_channels(X, Y, Z, stream_paths, STREAM_WIDTH, STREAM_DEPTH)

    # Trails: from the shore out into the meadow. Barely incised, but they clear
    # the vegetation, which is what makes a path legible from the air.
    trail_paths = []
    for k in range(TRAIL_COUNT):
        a = 2 * np.pi * (k + srng.uniform(0.2, 0.8)) / TRAIL_COUNT + 0.9
        start = (LAKE_RADIUS * 1.25 * np.cos(a), LAKE_RADIUS * 1.25 * np.sin(a))
        end = (62.0 * np.cos(a + srng.uniform(-0.7, 0.7)),
               62.0 * np.sin(a + srng.uniform(-0.7, 0.7)))
        trail_paths.append(meander_path(start, end, srng, steps=64, wobble=9.0))
    Z = _carve_channels(X, Y, Z, trail_paths, TRAIL_WIDTH * 1.6, 0.16)

    obj = mesh_from_grid("Landscape", X, Y, Z)
    obj.data.materials.append(material)
    obj.pass_index = PASS_TERRAIN
    return obj, X, Y, Z, stream_paths, trail_paths


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
    if cfg.get("form") is not None:
        # Parametric peak: identity is an explicit 5-vector (see fm_peak), so a
        # substitution moves a known distance through shape space instead of
        # swapping one noise field for another.
        from fm_peak import build_peak
        # rot_z is left at 0 here: the mesh is built in its canonical
        # orientation and the *object* transform below applies rot_z, so the
        # summit lean and the aretes are oriented exactly once.
        X, Y, Z = build_peak(cfg["form"], height=cfg["height"],
                             width=cfg["radius"], rot_z=0.0,
                             seed=cfg["seed"], n_r=n_r, n_theta=n_theta)
    else:
        raise ValueError(
            f"peak {cfg.get('name')!r} has no 'form'. Peaks became a parametric "
            f"shape space (see fm_peak); the twelve named landforms and their "
            f"heightfield builder were removed. Every caller must supply a form "
            f"vector -- fm_layout.layout_to_configs and "
            f"default_mountain_configs both do.")

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


def default_mountain_configs(n_peaks=4, seed=0):
    """
    A standalone demo layout, for `blender -b -P generate_scene.py`.

    Forms are drawn from `fm_peak`'s shape space with the same mutual-separation
    rule the benchmark layouts use, so the demo scene shows peaks that are
    individually recognisable rather than four variations on a cone. The
    benchmark itself does not come through here -- it builds configs from a
    sampled layout via `fm_layout.layout_to_configs`.
    """
    import fm_peak as peaklib

    n = int(n_peaks)
    azimuths = DEFAULT_AZIMUTHS if n == 4 else list(np.linspace(0, 360, n, endpoint=False) + 25.0)
    rng = np.random.default_rng(seed + 5150)
    forms = peaklib.sample_distinct_forms(rng, n)

    configs = []
    for i, form in enumerate(forms):
        kind = peaklib.describe(form)
        az = math.radians(azimuths[i % len(azimuths)])
        height = DEFAULT_HEIGHTS[i] if n == 4 else float(rng.uniform(12.5, 19.0))
        radius = DEFAULT_RADII[i] if n == 4 else float(rng.uniform(11.0, 15.0))
        ring = MOUNTAIN_RING if n <= 6 else MOUNTAIN_RING + 12.0
        configs.append({
            "name": f"M{i + 1}",
            "type": kind,
            "form": dict(form),
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

def build_four_mountains_scene(output_blend_path=None, n_peaks=4, seed=0,
                               samples=96, resolution=(1024, 1024),
                               valley_trees=16000, valley_boulders=340,
                               valley_shrubs_n=7000, valley_logs_n=520,
                               valley_reeds_n=3200, valley_broadleaf_n=3000,
                               valley_snags_n=420, valley_talus_n=2600,
                               mountain_configs=None, snow_line=None,
                               tree_line=None, sun_elevation_deg=24.0,
                               sun_azimuth_deg=150.0, sun_energy=75.0,
                               distant_lift=220.0, haze_strength=0.78,
                               exposure=-2.15, clouds=True, cloud_altitude=3.0,
                               cloud_coverage=(0.50, 0.68), sky_haze=0.35,
                               snow_slope=(0.12, 0.45), rock_warmth=0.35,
                               meadow_tint=None, haze_start=None, haze_scale=None,
                               rock_palette=None, bump_strength=2.2,
                               bump_distance=0.75, tree_patch_sharpness=3.6,
                               tree_patch_bias=1.30, tree_height_range=(0.9, 2.6),
                               cavity_strength=0.95, cavity_distance=1.4,
                               mountain_relief=2.0):
    """
    Build the alpine world and save it.

    `mountain_configs` overrides the built-in layout with an explicit list of
    peak configs (see `default_mountain_configs` for the shape), which is how
    `render_bench.py` builds a sampled layout rather than the hand-tuned one.
    """
    if output_blend_path is None:
        output_blend_path = os.path.join(_HERE, "four_mountains.blend")

    print("=" * 68)
    print("Building the Four Mountains alpine world (Blender 5.2)")
    print("=" * 68)

    clean_scene()
    scene = bpy.context.scene
    rng = np.random.default_rng(seed + 20250903)

    configure_render_engine(scene, samples=samples, resolution=resolution)
    scene.view_settings.exposure = exposure
    setup_lighting(scene, sun_elevation_deg=sun_elevation_deg,
                   sun_azimuth_deg=sun_azimuth_deg, sun_energy=sun_energy,
                   haze=sky_haze, clouds=clouds, cloud_altitude=cloud_altitude,
                   cloud_coverage=cloud_coverage)

    alpine_mat = create_alpine_material(
        snow_line=SNOW_LINE if snow_line is None else snow_line,
        tree_line=TREE_LINE if tree_line is None else tree_line,
        water_level=WATER_LEVEL, distant_lift=distant_lift,
        haze_strength=haze_strength, snow_slope=tuple(snow_slope),
        rock_warmth=rock_warmth,
        meadow_tint=(tuple(meadow_tint[0])[:3] + (1.0,),
                     tuple(meadow_tint[1])[:3] + (1.0,)) if meadow_tint else None,
        haze_start=haze_start, haze_scale=haze_scale, rock_palette=rock_palette,
        bump_strength=bump_strength, bump_distance=bump_distance,
        cavity_strength=cavity_strength, cavity_distance=cavity_distance)
    water_mat = create_water_material()
    tree_mat = create_tree_material()
    leaf_mat = create_broadleaf_material()
    dead_mat = create_deadwood_material()
    trail_mat = create_trail_material()

    print("  Terrain: valley, foothills, mid range and horizon skyline...")
    landscape, LX, LY, LZ, stream_paths, trail_paths = generate_landscape(
        alpine_mat, seed=77 + seed)
    scene.collection.objects.link(landscape)

    print("  Alpine tarn...")
    scene.collection.objects.link(create_lake(water_mat))

    configs = (list(mountain_configs) if mountain_configs is not None
               else default_mountain_configs(n_peaks=n_peaks, seed=seed))
    if mountain_relief is not None:
        configs = [dict(c, relief=mountain_relief) for c in configs]
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
    # Water surfaces in the carved channels, and bare earth along the trails.
    def _terrain_height_along(path):
        """Sample the carved terrain under a path (nearest grid vertex)."""
        out = []
        for px, py in path:
            i = np.argmin((LX - px) ** 2 + (LY - py) ** 2)
            out.append(float(LZ.ravel()[i]))
        return np.array(out)

    for i, path in enumerate(stream_paths):
        zs = _terrain_height_along(path) + 0.10
        # Widen downstream, the way a stream gathers.
        w = np.linspace(0.45, 1.5, len(path))
        rib = ribbon_mesh(f"Stream_{i}", path, zs, w, smooth=True)
        if rib:
            rib.data.materials.append(water_mat)
            rib.pass_index = PASS_WATER
            scene.collection.objects.link(rib)

    for i, path in enumerate(trail_paths):
        zs = _terrain_height_along(path) + 0.045
        rib = ribbon_mesh(f"Trail_{i}", path, zs, TRAIL_WIDTH, smooth=False)
        if rib:
            rib.data.materials.append(trail_mat)
            rib.pass_index = PASS_TERRAIN
            scene.collection.objects.link(rib)

    # Nothing grows in the streambed or on a trodden path. Without this the
    # forest closes straight over both and neither is visible from the air.
    clear = np.ones_like(LZ)
    for path in stream_paths:
        clear *= np.clip(distance_to_path(LX, LY, path) / (STREAM_WIDTH * 1.5),
                         0.0, 1.0)
    for path in trail_paths:
        clear *= np.clip(distance_to_path(LX, LY, path) / (TRAIL_WIDTH * 2.6),
                         0.0, 1.0)

    valley_forest = scatter_conifers(
        "Valley_Forest", LX, LY, LZ, count=valley_trees, rng=rng,
        z_min=VALLEY_FLOOR - 0.6,
        z_max=TREE_LINE if tree_line is None else tree_line, min_slope=0.62,
        height_range=tuple(tree_height_range), radius_limit=270.0,
        patch_sharpness=tree_patch_sharpness, patch_bias=tree_patch_bias,
        avoid=clear)
    if valley_forest:
        valley_forest.data.materials.append(tree_mat)
        valley_forest.pass_index = PASS_TERRAIN
        scene.collection.objects.link(valley_forest)

    # Layers a real valley has and a single tree scatter does not: a shrub belt
    # softening the forest edge (and the only green above the tree line),
    # deadfall on the forest floor, and reed beds breaking the waterline.
    valley_shrubs = scatter_shrubs(
        "Valley_Shrubs", LX, LY, LZ, count=valley_shrubs_n, rng=rng,
        z_min=VALLEY_FLOOR - 1.0,
        z_max=(TREE_LINE if tree_line is None else tree_line) + 3.5,
        radius_limit=200.0, avoid=clear)
    if valley_shrubs:
        valley_shrubs.data.materials.append(tree_mat)
        valley_shrubs.pass_index = PASS_TERRAIN
        scene.collection.objects.link(valley_shrubs)

    valley_logs = scatter_logs(
        "Valley_Logs", LX, LY, LZ, count=valley_logs_n, rng=rng,
        z_min=VALLEY_FLOOR - 0.4,
        z_max=(TREE_LINE if tree_line is None else tree_line) - 1.0,
        radius_limit=150.0, avoid=clear)
    if valley_logs:
        valley_logs.data.materials.append(dead_mat)
        valley_logs.pass_index = PASS_TERRAIN
        scene.collection.objects.link(valley_logs)

    valley_reeds = scatter_reeds(
        "Valley_Reeds", LX, LY, LZ, count=valley_reeds_n, rng=rng,
        water_z=WATER_LEVEL + 0.25, band=0.7, radius_limit=LAKE_RADIUS * 2.6)
    if valley_reeds:
        valley_reeds.data.materials.append(tree_mat)
        valley_reeds.pass_index = PASS_TERRAIN
        scene.collection.objects.link(valley_reeds)

    valley_broadleaf = scatter_broadleaf(
        "Valley_Broadleaf", LX, LY, LZ, count=valley_broadleaf_n, rng=rng,
        z_min=VALLEY_FLOOR - 0.8, z_max=(TREE_LINE if tree_line is None
                                         else tree_line) - 2.0,
        radius_limit=190.0, avoid=clear)
    if valley_broadleaf:
        valley_broadleaf.data.materials.append(leaf_mat)
        valley_broadleaf.pass_index = PASS_TERRAIN
        scene.collection.objects.link(valley_broadleaf)

    valley_snags = scatter_snags(
        "Valley_Snags", LX, LY, LZ, count=valley_snags_n, rng=rng,
        z_min=VALLEY_FLOOR - 0.4, z_max=(TREE_LINE if tree_line is None
                                         else tree_line) - 0.5,
        height_range=tuple(tree_height_range), radius_limit=170.0, avoid=clear)
    if valley_snags:
        valley_snags.data.materials.append(dead_mat)
        valley_snags.pass_index = PASS_TERRAIN
        scene.collection.objects.link(valley_snags)

    valley_talus = scatter_talus(
        "Valley_Talus", LX, LY, LZ, count=valley_talus_n, rng=rng,
        radius_limit=200.0)
    if valley_talus:
        valley_talus.data.materials.append(alpine_mat)
        valley_talus.pass_index = PASS_TERRAIN
        scene.collection.objects.link(valley_talus)

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

    # Persist the *full* layout, not just names and types. Heights and base
    # radii used to live only in module constants, so they never reached the
    # per-sample metadata -- which left the egocentric text serialisation and
    # the map channel unable to reconstruct the scene they describe.
    scene["fm_mountains"] = names
    scene["fm_types"] = [c["type"] for c in configs]
    scene["fm_heights"] = [float(c["height"]) for c in configs]
    scene["fm_base_radii"] = [float(c["radius"]) for c in configs]
    scene["fm_layout_seed"] = int(seed)
    scene["fm_mountain_relief"] = float(
        configs[0].get("relief", 1.0) if configs else 1.0)
    scene["fm_mountain_seeds"] = [int(c["seed"]) for c in configs]
    if configs and configs[0].get("form") is not None:
        # Persist the form vectors so a reopened .blend can restore a peak
        # exactly; `replace_mountain` needs them to put one back.
        scene["fm_forms"] = json.dumps([c.get("form") for c in configs])
    scene["fm_water_level"] = WATER_LEVEL
    scene["fm_valley_floor"] = VALLEY_FLOOR
    scene["fm_mountain_sink"] = MOUNTAIN_SINK

    os.makedirs(os.path.dirname(output_blend_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)
    print(f"Saved scene -> {output_blend_path}")
    return output_blend_path


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Generate the Four Mountains scene")
    p.add_argument("--out", default=None, help="Output .blend path")
    p.add_argument("--n_peaks", type=int, default=4,
                   help="How many peaks to place in the demo scene")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=96)
    p.add_argument("--resolution", type=int, nargs=2, default=(1024, 1024))
    p.add_argument("--valley_trees", type=int, default=22000)
    return p.parse_args(argv)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = _parse_args(argv)
    build_four_mountains_scene(
        output_blend_path=args.out,
        n_peaks=args.n_peaks,
        seed=args.seed,
        samples=args.samples,
        resolution=tuple(args.resolution),
        valley_trees=args.valley_trees,
    )
    print("Done!")
