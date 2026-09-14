"""
A complexity ladder for the Four Mountains task.

The v4 bank asks a hard question in a hard scene at the same time: recognise a
place from a new viewpoint, where the landmarks are rocky peaks that differ only
subtly and sit in a cluttered, lit, vegetated valley. Frozen encoders score at
or below chance on it. That single number cannot say *which* part they fail --
the allocentric transformation, or telling four grey cones apart.

This module builds the same task out of progressively simpler stimuli, holding
everything else fixed: the same ring geometry, the same camera orbit, the same
framing validity test, the same three foil families, the same item builder.
Only what the landmarks look like changes.

    c0_shape_colour   four primitives, each a different shape AND colour
    c1_shape          four primitives, all one colour -- shape is the only cue
    c2_colour         four identical cones in different colours -- colour only
    c3_peaks_bare     the real fm_peak landforms, grey, on a bare plane

with `bench_v4` itself as the top rung. Read against Delta-azimuth, the ladder
answers a question the headline number cannot: if a model can match four
coloured blocks across a 180 degree rotation but not four grey cones, the
failure is about landmark distinctiveness; if it fails on the blocks too, the
failure is viewpoint invariance itself, and no amount of stimulus simplification
will rescue it.

Sizes are deliberately held EQUAL across objects in c0-c2. Apparent size is a
strong identity cue as well as the image's distance cue, so leaving the sampled
size variation in would mean "colour only" was never colour only.

The bpy-dependent scene builder is at the bottom; everything above it is plain
Python and is unit-tested.
"""

import copy
import math

import numpy as np

# --------------------------------------------------------------------------- #
# The discrete object space used by the c0-c2 rungs
# --------------------------------------------------------------------------- #

SHAPES = ("cube", "dome", "cone", "cylinder", "prism", "pyramid")

# Chosen for roughly equal luminance so colour does not smuggle in a brightness
# cue, and far apart in hue so they survive a sun-angle change.
COLOURS = {
    "red":    (0.72, 0.13, 0.11),
    "green":  (0.16, 0.55, 0.20),
    "blue":   (0.13, 0.32, 0.75),
    "yellow": (0.85, 0.70, 0.10),
    "purple": (0.50, 0.20, 0.62),
    "cyan":   (0.10, 0.60, 0.62),
}
COLOUR_NAMES = tuple(COLOURS)

NEUTRAL = (0.55, 0.54, 0.52)          # the one colour used when colour is not a cue

# Uniform size for the discrete rungs. 19 m / 17 m is mid-range for the layout
# sampler, so framing validity behaves as it does in the full bank.
UNIFORM_HEIGHT = 19.0
UNIFORM_RADIUS = 17.0

# c4 rebuilds the world for every variant, as every other rung does. The full
# bank's scatter counts (16k trees, 3.2k reeds, ...) were tuned for a bank that
# built each scene once; at ~135 builds per cell they dominate the render. These
# are the same populations thinned to roughly a third, which keeps the valley
# legible as a valley -- the thing c4 is testing -- at a build cost the grid can
# actually afford. Measured before the grid was launched, not guessed.
VALLEY_BUDGET = {"valley_trees": 5200, "valley_boulders": 120,
                 "valley_shrubs_n": 2300, "valley_logs_n": 170,
                 "valley_reeds_n": 1100, "valley_broadleaf_n": 1000,
                 "valley_snags_n": 140, "valley_talus_n": 900}

MODES = {
    "c0_shape_colour": {"shape_varies": True,  "colour_varies": True,
                        "geometry": "primitive", "ground": "contrast",
                        "world": "bare",
                        "label": "distinct shape and colour"},
    "c1_shape":        {"shape_varies": True,  "colour_varies": False,
                        "geometry": "primitive", "ground": "contrast",
                        "world": "bare",
                        "label": "distinct shape, one colour"},
    "c2_colour":       {"shape_varies": False, "colour_varies": True,
                        "geometry": "primitive", "ground": "contrast",
                        "world": "bare",
                        "label": "one shape, distinct colours"},
    # A landform meets the ground tangentially (`fm_peak.flank` >= 1.15), so it
    # ends in a wide, almost flat apron. Against a ground of a different colour
    # that apron renders as a light disc around every peak -- the same "the base
    # is clearly visible" artefact the valley had, and worse here because it
    # also marks each footprint's centre, which is information the task is
    # supposed to withhold. Matching the ground to the landmark makes the two
    # surfaces shade identically and the seam disappears.
    "c3_peaks_bare":   {"shape_varies": True,  "colour_varies": False,
                        "geometry": "peak", "ground": "match",
                        "world": "bare",
                        "label": "real landforms, bare plane"},
    # The top rung: the same landmarks as c3, in the full alpine world --
    # terrain relief, lake, vegetation, haze and cloud. `geometry` stays "peak"
    # so identity, foils, the ladder and the text channels are byte-for-byte
    # what c3 uses; only `world` differs, which is the whole point. It is a
    # rung of this pipeline, reached with `--mode c4_valley`, not a separate
    # script with its own conventions.
    "c4_valley":       {"shape_varies": True,  "colour_varies": False,
                        "geometry": "peak", "ground": "terrain",
                        "world": "valley",
                        "label": "real landforms, full alpine valley"},
}
MODE_ORDER = ("c0_shape_colour", "c1_shape", "c2_colour",
              "c3_peaks_bare", "c4_valley")


def object_cue_count(mode):
    """How many distinct identities the mode's object space can express."""
    spec = MODES[mode]
    n = 1
    if spec["shape_varies"]:
        n *= len(SHAPES)
    if spec["colour_varies"]:
        n *= len(COLOURS)
    return n


CANONICAL_SEED = 20260908      # fixes the landmark set for the whole bank


def canonical_objects(n):
    """
    The `n` landmarks every scene in the bank is built from.

    **This is what makes the benchmark about place rather than about objects.**
    If each scene drew its own subset of shapes and colours, two scenes would
    differ in *which landmarks are present*, and a model could tell them apart
    by recognising the purple sphere from any angle -- with no representation of
    an arrangement at all. Cross-view retrieval over such a gallery measures
    viewpoint-invariant object recognition and reports it as spatial memory.

    Holding the set fixed removes that route. Every scene contains the same `n`
    landmarks; scenes differ only in *where each one stands*, so recognising a
    place requires encoding the configuration. It also makes `d_id` zero between
    any two scenes, which is what lets `fm_scenedist.same_place` enforce the
    positional separation floor it was written for -- with varying identities
    that test short-circuited and the floor never bound.
    """
    if n > len(SHAPES):
        raise ValueError(f"only {len(SHAPES)} shapes available for {n} landmarks")
    names = list(COLOURS)
    if n > len(names):
        raise ValueError(f"only {len(names)} colours available for {n} landmarks")
    rng = np.random.default_rng(CANONICAL_SEED)
    shapes = [int(i) for i in rng.permutation(len(SHAPES))[:n]]
    colours = [int(i) for i in rng.permutation(len(names))[:n]]
    return [{"shape": s, "colour": c} for s, c in zip(shapes, colours)]


def canonical_forms(n):
    """The `n` landforms c3 and c4 are built from -- the peak-mode counterpart
    of `canonical_objects`, fixed for the same reason."""
    import fm_peak as peaklib
    rng = np.random.default_rng(CANONICAL_SEED + 1)
    # `sample_distinct_forms` yields dicts keyed by parameter name; list() on a
    # dict returns its keys, which silently turns a form into nonsense.
    return [dict(f) for f in peaklib.sample_distinct_forms(rng, n)]


def assign_objects(layout, rng, mode):
    """
    Give every peak a discrete identity, in place of (or beside) its form.

    Returns a NEW layout; the input is untouched. Peak modes (c3, c4) keep the
    form vector the layout was sampled with as their identity.

    **Size is uniform in every mode.** The same layout is rendered in all five,
    so anything that varies between them other than the identity cue and the
    world would break the pairing: a landmark that is 14 m wide in c3 and 17 m
    wide in c0 is not the same landmark seen two ways. Holding size fixed also
    makes apparent size a pure distance cue rather than a mode-dependent one.
    """
    spec = MODES[mode]
    out = copy.deepcopy(layout)
    n = len(out["peaks"])

    # The serialisers cannot infer the world from the landmarks: c3 and c4 use
    # identical peaks and differ only in what surrounds them. Record it.
    out["world"] = spec.get("world", "bare")

    for p in out["peaks"]:
        p["height"] = UNIFORM_HEIGHT
        p["base_radius"] = UNIFORM_RADIUS

    # One permutation, drawn per scene, decides which landmark stands where.
    # The *set* is fixed; only the binding of landmark to place varies, which is
    # the object-place binding the task is about.
    order = [int(i) for i in rng.permutation(n)]

    if spec["geometry"] == "peak":
        forms = canonical_forms(n)
        for p, k in zip(out["peaks"], order):
            p["obj"] = None
            p["form"] = dict(forms[k])
        out["forms"] = [dict(p["form"]) for p in out["peaks"]]
        return out

    # The canonical set, permuted by the *same* `order` the peak modes use, so a
    # landmark standing at a given place is the same landmark in all five modes.
    # Shape and colour move together: permuting them independently would make
    # the *set* of objects differ between scenes (a red cube here, a blue cube
    # there) and reinstate the identity shortcut this is here to remove.
    objs = canonical_objects(n)
    shape_draw = [objs[k]["shape"] for k in order]
    colour_draw = [objs[k]["colour"] for k in order]
    shapes = shape_draw if spec["shape_varies"] else [SHAPES.index("cone")] * n
    colours = colour_draw if spec["colour_varies"] else [None] * n
    for p, s, c in zip(out["peaks"], shapes, colours):
        p["obj"] = {"shape": int(s), "colour": None if c is None else int(c)}
        p["type"] = describe_object(p["obj"])
        # The form vector is meaningless once the landmark is a primitive, and
        # leaving it in would make `fm_text` describe a landform that is not
        # there. Dropping it makes the serialisers fall back to `type`, which is
        # exactly the object identity.
        p.pop("form", None)
    out["morphologies"] = [p["type"] for p in out["peaks"]]
    out.pop("forms", None)
    return out


def describe_object(obj):
    if obj is None:
        return "peak"
    name = SHAPES[obj["shape"]]
    if obj["colour"] is None:
        return name
    return f"{COLOUR_NAMES[obj['colour']]} {name}"


APPEARANCES = {
    "A": {"hue": 0.50, "saturation": 1.00, "value": 1.00,
          "sun_elevation_deg": 24.0, "sun_azimuth_deg": 150.0},
    "B": {"hue": 0.56, "saturation": 0.86, "value": 1.10,
          "sun_elevation_deg": 33.0, "sun_azimuth_deg": 285.0},
}


# --------------------------------------------------------------------------- #
# Blender scene construction (needs bpy)
# --------------------------------------------------------------------------- #

def _primitive_mesh(shape, radius, height, segments=48):
    """(verts, faces) for a unit-ish primitive scaled to `radius` x `height`."""
    r, h = float(radius), float(height)

    if shape == "cube":
        # Half-side, not radius: a square of half-side s has corners at
        # s*sqrt(2), and framing validity is computed from `base_radius`, so
        # s = 0.80r put the corners 13% outside the footprint the layout
        # sampler had guaranteed was in frame.
        s = r * 0.68
        v = [(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0),
             (-s, -s, h), (s, -s, h), (s, s, h), (-s, s, h)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return v, f

    if shape == "prism":                       # triangular prism
        v, f = [], []
        for k in range(3):
            a = 2 * math.pi * k / 3 + math.pi / 2
            v.append((r * math.cos(a), r * math.sin(a), 0.0))
        for k in range(3):
            a = 2 * math.pi * k / 3 + math.pi / 2
            v.append((r * math.cos(a), r * math.sin(a), h))
        f = [(0, 2, 1), (3, 4, 5),
             (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
        return v, f

    if shape == "cone":
        v = [(r * math.cos(2 * math.pi * k / segments),
              r * math.sin(2 * math.pi * k / segments), 0.0)
             for k in range(segments)]
        v.append((0.0, 0.0, h))
        apex = len(v) - 1
        f = [(k, (k + 1) % segments, apex) for k in range(segments)]
        v.append((0.0, 0.0, 0.0))              # base centre, for a triangle fan
        c = len(v) - 1
        f += [(c, (k + 1) % segments, k) for k in range(segments)]
        return v, f

    if shape == "cylinder":
        v = [(r * math.cos(2 * math.pi * k / segments),
              r * math.sin(2 * math.pi * k / segments), 0.0)
             for k in range(segments)]
        v += [(x, y, h) for x, y, _ in v]
        f = [(k, (k + 1) % segments, segments + (k + 1) % segments, segments + k)
             for k in range(segments)]
        v.append((0.0, 0.0, 0.0))
        v.append((0.0, 0.0, h))
        cb, ct = len(v) - 2, len(v) - 1
        f += [(cb, (k + 1) % segments, k) for k in range(segments)]
        f += [(ct, segments + k, segments + (k + 1) % segments)
              for k in range(segments)]
        return v, f

    if shape == "dome":                        # a hemisphere sitting on the ground
        rings, v, f = 12, [], []
        for i in range(rings + 1):
            phi = 0.5 * math.pi * i / rings
            rr, zz = r * math.cos(phi), h * math.sin(phi)
            for k in range(segments):
                a = 2 * math.pi * k / segments
                v.append((rr * math.cos(a), rr * math.sin(a), zz))
        for i in range(rings):
            for k in range(segments):
                a0 = i * segments + k
                a1 = i * segments + (k + 1) % segments
                b0 = (i + 1) * segments + k
                b1 = (i + 1) * segments + (k + 1) % segments
                f.append((a0, a1, b1, b0))
        v.append((0.0, 0.0, 0.0))
        c = len(v) - 1
        f += [(c, (k + 1) % segments, k) for k in range(segments)]
        return v, f

    if shape == "pyramid":                     # square base, four flat faces
        s_ = r * 0.68                          # see the cube: corners at s*sqrt(2)
        v = [(-s_, -s_, 0), (s_, -s_, 0), (s_, s_, 0), (-s_, s_, 0), (0.0, 0.0, h)]
        f = [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
        return v, f

    raise ValueError(f"unknown shape {shape!r}")


def _flat_material(name, base_rgb, roughness=0.55):
    """
    A plain diffuse material carrying the appearance node.

    `FourMountainsRenderer.set_appearance` drives every material holding a node
    called `APPEARANCE_NODE`, so the ladder gets the same study/test nuisance
    variation the full bank does. Without it, the simple rungs would have no
    appearance axis and their Delta-0 condition would be trivially solvable by
    pixel matching -- exactly the v2 hole, reopened in the control.
    """
    import bpy

    from fm_materials import APPEARANCE_NODE

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (*base_rgb, 1.0)

    hsv = nodes.new("ShaderNodeHueSaturation")
    hsv.name = hsv.label = APPEARANCE_NODE
    hsv.inputs["Hue"].default_value = 0.5
    hsv.inputs["Saturation"].default_value = 1.0
    hsv.inputs["Value"].default_value = 1.0
    hsv.inputs["Fac"].default_value = 1.0

    links.new(rgb.outputs["Color"], hsv.inputs["Color"])
    links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _mesh_object(name, verts, faces, material, pass_index, location, rot_z,
                 smooth=False):
    import bpy

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.validate()
    mesh.update()
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    obj.pass_index = int(pass_index)
    obj.location = (float(location[0]), float(location[1]), float(location[2]))
    obj.rotation_euler.z = float(rot_z)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def build_scene(output_blend_path, layout, mode, samples=24,
                       resolution=(640, 440), ground_rgb=(0.38, 0.40, 0.36),
                       sky_rgb=(0.52, 0.66, 0.86)):
    """
    Build one rung's scene as a .blend `FourMountainsRenderer` can drive.

    The renderer discovers landmarks from the `fm_mountains` scene property and
    builds its instance-mask materials from object names, so a scene that
    follows those two conventions works with the existing camera, appearance,
    mask and orbit code unchanged -- which is the point: the ladder must differ
    from the full bank in the stimulus and in nothing else.

    Every variant gets a fresh scene rather than an in-place edit. On the full
    bank that would be prohibitive (a terrain sheet plus 30k plants); here a
    scene is four primitives and a plane, and rebuilding sidesteps the entire
    class of bug that `replace_mountain` produced in v3.
    """
    import bpy

    import fm_peak as peaklib
    import generate_scene as gs
    from generate_scene import PASS_MOUNTAIN_BASE

    # c4 is this pipeline's top rung, not a different pipeline: the same
    # layout, identities, foils and ladder, handed to the full alpine builder
    # instead of the bare-plane one. `build_four_mountains_scene` sets the same
    # `fm_mountains` / pass-index conventions, so the renderer, camera orbit,
    # appearance and mask code downstream are unchanged.
    if MODES[mode].get("world") == "valley":
        import fm_layout as layoutlib
        return gs.build_four_mountains_scene(
            output_blend_path=output_blend_path,
            mountain_configs=layoutlib.layout_to_configs(layout),
            seed=int(layout["seed"]),
            samples=samples,
            resolution=resolution,
            **VALLEY_BUDGET)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    gs.configure_render_engine(scene, samples=samples, resolution=resolution)

    # A plain gradient world: enough to light the scene and give the horizon a
    # direction, with none of the cloud and haze structure of the full bank.
    world = bpy.data.worlds.new("SimpleWorld")
    world.use_nodes = True
    wnodes, wlinks = world.node_tree.nodes, world.node_tree.links
    wnodes.clear()
    wout = wnodes.new("ShaderNodeOutputWorld")
    bg = wnodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (*sky_rgb, 1.0)
    # Sky fill and sun are both several times the Blender defaults: with the
    # view transform the full bank uses, a 1 W/m2 sun renders these scenes at a
    # mean pixel value of ~50/255, and a benchmark stimulus that dark measures
    # the exposure as much as the layout.
    bg.inputs["Strength"].default_value = 2.2
    wlinks.new(bg.outputs["Background"], wout.inputs["Surface"])
    scene.world = world

    sun_data = bpy.data.lights.new(name="SunLight", type="SUN")
    sun_data.energy = 22.0
    sun_data.color = (1.0, 0.96, 0.90)
    sun_data.angle = math.radians(2.0)
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.location = (0, 0, 200)

    spec = MODES[mode]

    # Named "Landscape" so the renderer's mask code labels it as terrain.
    if MODES[mode].get("ground") == "match":
        ground_rgb = NEUTRAL
    ground_mat = _flat_material("SimpleGround", ground_rgb, roughness=0.85)
    g = 900.0
    ground = _mesh_object("Landscape",
                          [(-g, -g, 0), (g, -g, 0), (g, g, 0), (-g, g, 0)],
                          [(0, 1, 2, 3)], ground_mat, 0, (0, 0, 0), 0.0)
    ground.hide_render = False

    names = []
    for i, p in enumerate(layout["peaks"]):
        obj_spec = p.get("obj")
        if spec["geometry"] == "peak":
            X, Y, Z = peaklib.build_peak(p["form"], p["height"], p["base_radius"],
                                         rot_z=0.0,
                                         seed=101 + 97 * i + int(layout["seed"]))
            verts, faces = gs.grid_to_mesh(X, Y, Z) if hasattr(gs, "grid_to_mesh") \
                else _grid_to_mesh(X, Y, Z)
            colour, smooth = NEUTRAL, True
        else:
            verts, faces = _primitive_mesh(SHAPES[obj_spec["shape"]],
                                           p["base_radius"], p["height"])
            colour = (NEUTRAL if obj_spec["colour"] is None
                      else COLOURS[COLOUR_NAMES[obj_spec["colour"]]])
            smooth = SHAPES[obj_spec["shape"]] in ("dome", "cone", "cylinder")

        mat = _flat_material(f"Mat_{p['name']}", colour)
        _mesh_object(p["name"], verts, faces, mat, PASS_MOUNTAIN_BASE + i,
                     (p["x"], p["y"], p["z"]), p["rot_z"], smooth=smooth)
        names.append(p["name"])

    import fm_layout as layoutlib
    gs.setup_camera_rig(scene, lens=layoutlib.CAM_LENS_MM,
                        target_z=layoutlib.CAM_TARGET_Z)

    scene["fm_mountains"] = names
    scene["fm_types"] = [p["type"] for p in layout["peaks"]]
    scene["fm_heights"] = [float(p["height"]) for p in layout["peaks"]]
    scene["fm_base_radii"] = [float(p["base_radius"]) for p in layout["peaks"]]
    scene["fm_layout_seed"] = int(layout["seed"])
    scene["fm_mode"] = mode

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(output_blend_path))
    return output_blend_path


def _grid_to_mesh(X, Y, Z):
    """Polar heightfield grid -> (verts, quads), matching generate_scene."""
    n_r, n_t = Z.shape
    verts = [(float(X[i, j]), float(Y[i, j]), float(Z[i, j]))
             for i in range(n_r) for j in range(n_t)]
    faces = []
    for i in range(n_r - 1):
        for j in range(n_t):
            j1 = (j + 1) % n_t
            a, b = i * n_t + j, i * n_t + j1
            c, d = (i + 1) * n_t + j1, (i + 1) * n_t + j
            faces.append((a, b, c, d))
    return verts, faces


import os  # noqa: E402  (kept at the bottom: the bpy path is the only user)
