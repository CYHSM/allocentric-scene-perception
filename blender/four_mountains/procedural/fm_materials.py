"""
Procedural PBR materials for the Four Mountains environment.

The terrain shader is a slope- and altitude-driven biome stack (shore gravel ->
alpine meadow -> conifer belt -> scree -> cliff rock -> snow) with a cheap
analytic aerial-perspective term that fades distant ranges into the sky colour
the way real atmosphere does.
"""

import bpy

# Shared atmospheric tint. Keep in sync with the sky so the horizon dissolves
# into the backdrop instead of ending in a hard cut-out edge.
HAZE_COLOR = (0.30, 0.43, 0.66, 1.0)
# The benchmark camera orbits at 94 m and the landmark peaks sit 66-123 m away,
# so an onset of 120 m put the peaks themselves inside the haze ramp and
# bleached them to the colour of plaster. Aerial perspective should start
# beyond the far side of the valley.
# Name of the Hue/Saturation node the renderer drives to restyle a built scene.
APPEARANCE_NODE = "AppearanceHSV"

HAZE_START = 300.0      # distance before haze begins to accumulate
HAZE_SCALE = 800.0      # distance over which it saturates


class _Tree:
    """Thin convenience wrapper over a material node tree."""

    def __init__(self, mat):
        self.nodes = mat.node_tree.nodes
        self.links = mat.node_tree.links
        self.nodes.clear()

    def new(self, kind, location=(0, 0), **kwargs):
        node = self.nodes.new(kind)
        node.location = location
        for key, value in kwargs.items():
            if key.startswith("in_"):
                node.inputs[key[3:].replace("_", " ")].default_value = value
            else:
                setattr(node, key, value)
        return node

    def link(self, a, b):
        self.links.new(a, b)

    def mix_rgb(self, fac, color_a, color_b, location=(0, 0)):
        """ShaderNodeMix in RGBA mode: index 0 = Factor, 6 = A, 7 = B, out 2 = Result."""
        node = self.nodes.new("ShaderNodeMix")
        node.data_type = "RGBA"
        node.location = location
        if hasattr(fac, "bl_idname") or hasattr(fac, "node"):
            self.links.new(fac, node.inputs[0])
        else:
            node.inputs[0].default_value = fac
        for socket_index, value in ((6, color_a), (7, color_b)):
            if hasattr(value, "node"):
                self.links.new(value, node.inputs[socket_index])
            else:
                node.inputs[socket_index].default_value = value
        return node.outputs[2]

    def band(self, value_socket, lo, hi, location=(0, 0), clamp=True):
        """Map Range: 0 below `lo`, 1 above `hi`."""
        node = self.new("ShaderNodeMapRange", location, clamp=clamp)
        node.inputs["From Min"].default_value = lo
        node.inputs["From Max"].default_value = hi
        node.inputs["To Min"].default_value = 0.0
        node.inputs["To Max"].default_value = 1.0
        self.link(value_socket, node.inputs["Value"])
        return node.outputs["Result"]

    def math(self, operation, a, b=None, location=(0, 0), c=None):
        node = self.new("ShaderNodeMath", location, operation=operation)
        for socket_index, value in ((0, a), (1, b), (2, c)):
            if value is None:
                continue
            if hasattr(value, "node"):
                self.link(value, node.inputs[socket_index])
            else:
                node.inputs[socket_index].default_value = value
        return node.outputs["Value"]

    def noise(self, vector, scale, detail=8.0, roughness=0.55, location=(0, 0)):
        node = self.new("ShaderNodeTexNoise", location)
        node.inputs["Scale"].default_value = scale
        node.inputs["Detail"].default_value = detail
        node.inputs["Roughness"].default_value = roughness
        self.link(vector, node.inputs["Vector"])
        return node.outputs["Fac"]


def _aerial_perspective(t, color_socket, location=(0, 0), strength=0.78,
                        haze_color=None, haze_start=None, haze_scale=None):
    """
    Blend a colour toward the atmospheric haze as distance from camera grows.

    `haze_start` matters more than it looks. The benchmark camera orbits at 94 m
    and the peaks sit 66-123 m away, so the stock 120 m onset puts the landmark
    peaks *inside* the haze ramp -- which is what bleached them to the colour of
    plaster. Pushing the onset past the far side of the valley keeps the peaks
    crisp and leaves aerial perspective to do its real job on the backdrop.
    """
    cam = t.new("ShaderNodeCameraData", (location[0] - 400, location[1] - 200))
    start = HAZE_START if haze_start is None else haze_start
    scale = HAZE_SCALE if haze_scale is None else haze_scale
    fog = t.band(cam.outputs["View Distance"], start, start + scale,
                 (location[0] - 200, location[1] - 200))
    # Ease the ramp so the near field stays crisp and only the far range washes out.
    fog = t.math("POWER", fog, 0.75, (location[0] - 60, location[1] - 200))
    fog = t.math("MULTIPLY", fog, strength, (location[0] - 60, location[1] - 320))
    return t.mix_rgb(fog, color_socket, haze_color or HAZE_COLOR, location), fog


def create_alpine_material(snow_line=13.5, tree_line=9.5, water_level=0.0,
                           distant_lift=220.0, lift_start=90.0, lift_end=620.0,
                           haze_strength=0.78, haze_color=None,
                           snow_slope=(0.12, 0.45), rock_warmth=0.35,
                           meadow_tint=None, haze_start=None, haze_scale=None,
                           rock_palette=None, bump_strength=2.2,
                           bump_distance=0.75, cavity_strength=0.95,
                           cavity_distance=1.4):
    """
    Slope + altitude biome shader.

    Layer order (each one overrides the previous where its mask is 1):
      shore gravel -> alpine meadow -> conifer belt -> scree -> cliff rock -> snow

    `distant_lift` raises both the snow line and the tree line with horizontal
    distance from the valley centre, ramping in between `lift_start` and
    `lift_end` metres.

    That is not how snow lines work in the real world -- but the backdrop is not
    a real world. `_terrain_profile` climbs to 58 m and carries up to 78 m of
    ridged relief on top, so the horizon range stands ~100 m tall purely as a
    stylistic device. Against a fixed 21 m snow line every distant ridge is
    therefore *entirely* above the snow line and renders solid white, which is
    what turned the skyline into a bank of cotton wool. Lifting both biome lines
    with distance puts rock and forest back on the far ranges and leaves snow to
    the summits, which is what actually reads as depth.
    """
    mat = bpy.data.materials.new(name="AlpineTerrain")
    mat.use_nodes = True
    t = _Tree(mat)

    out = t.new("ShaderNodeOutputMaterial", (1900, 0))
    bsdf = t.new("ShaderNodeBsdfPrincipled", (1600, 0))
    t.link(bsdf.outputs["BSDF"], out.inputs["Surface"])

    geom = t.new("ShaderNodeNewGeometry", (-1500, 200))
    sep = t.new("ShaderNodeSeparateXYZ", (-1300, 380))
    t.link(geom.outputs["Position"], sep.inputs["Vector"])
    height = sep.outputs["Z"]

    # Slope as N . Z: 1 = flat, 0 = vertical wall.
    slope = t.new("ShaderNodeVectorMath", (-1300, 120), operation="DOT_PRODUCT")
    t.link(geom.outputs["Normal"], slope.inputs[0])
    slope.inputs[1].default_value = (0.0, 0.0, 1.0)
    slope = slope.outputs["Value"]

    pos = geom.outputs["Position"]

    # Horizontal distance from the valley centre, used to lift the biome lines
    # over the stylised backdrop (see the docstring).
    flat = t.new("ShaderNodeCombineXYZ", (-1300, 560))
    t.link(sep.outputs["X"], flat.inputs["X"])
    t.link(sep.outputs["Y"], flat.inputs["Y"])
    flat.inputs["Z"].default_value = 0.0
    ground_dist = t.new("ShaderNodeVectorMath", (-1140, 560), operation="LENGTH")
    t.link(flat.outputs["Vector"], ground_dist.inputs[0])
    lift = t.math("MULTIPLY",
                  t.band(ground_dist.outputs["Value"], lift_start, lift_end, (-980, 560)),
                  distant_lift, (-820, 560))

    macro = t.noise(pos, 0.030, 6.0, 0.60, (-1300, -120))    # regional variation
    meso = t.noise(pos, 0.185, 8.0, 0.55, (-1300, -320))     # patch / outcrop scale
    micro = t.noise(pos, 2.60, 8.0, 0.72, (-1300, -520))     # surface breakup

    # Vertically squashed noise -> sedimentary strata on the cliff faces.
    strata_map = t.new("ShaderNodeMapping", (-1300, -720))
    strata_map.inputs["Scale"].default_value = (1.0, 1.0, 3.2)
    t.link(pos, strata_map.inputs["Vector"])
    strata = t.noise(strata_map.outputs["Vector"], 0.42, 7.0, 0.62, (-1080, -720))

    # -- Palette -----------------------------------------------------------
    shore_col = (0.300, 0.268, 0.212, 1.0)
    grass_dk = (0.045, 0.105, 0.030, 1.0)
    grass_lt = (0.165, 0.240, 0.062, 1.0)
    forest_dk = (0.012, 0.032, 0.014, 1.0)
    forest_lt = (0.038, 0.080, 0.030, 1.0)
    scree_col = (0.175, 0.158, 0.132, 1.0)
    # Higher contrast than a literal rock sample: at benchmark resolution the
    # peaks are only a few hundred pixels tall, and a low-contrast palette
    # collapses the strata and gullies into flat pale shapes.
    rock_dk = (0.014, 0.014, 0.018, 1.0)
    rock_lt = (0.155, 0.140, 0.122, 1.0)
    rock_warm = (0.190, 0.112, 0.070, 1.0)
    rock_pale = (0.300, 0.280, 0.250, 1.0)
    if rock_palette is not None:
        rock_dk, rock_lt, rock_warm, rock_pale = (
            tuple(c)[:3] + (1.0,) for c in rock_palette)
    snow_col = (0.880, 0.915, 0.975, 1.0)

    if meadow_tint is not None:
        grass_dk, grass_lt = meadow_tint
    grass_dry = (0.215, 0.195, 0.080, 1.0)
    meadow = t.mix_rgb(micro, grass_dk, grass_lt, (-800, -260))
    # Sun-bleached patches drifting across the meadow at the regional scale.
    meadow = t.mix_rgb(t.band(macro, 0.42, 0.72, (-1000, -60)),
                       meadow, grass_dry, (-620, -260))

    forest = t.mix_rgb(micro, forest_dk, forest_lt, (-800, -460))

    rock = t.mix_rgb(strata, rock_dk, rock_lt, (-800, -700))
    # Outcrop-scale mineral banding, then broad iron staining across faces.
    rock = t.mix_rgb(t.band(meso, 0.40, 0.66, (-1000, -640)), rock, rock_pale, (-800, -560))
    rock = t.mix_rgb(t.math("MULTIPLY", macro, rock_warmth, (-1000, -820)),
                     rock, rock_warm, (-620, -700))

    # -- Masks -------------------------------------------------------------
    # Shore: only the narrow strip just above the waterline is washed gravel.
    shore_jitter = t.math("MULTIPLY_ADD", meso, 0.45, (-1000, 620), c=-0.22)
    shore_jitter = t.math("ADD", shore_jitter, height, (-820, 620))
    shore_mask = t.math("SUBTRACT", 1.0,
                        t.band(shore_jitter, water_level + 0.15,
                               water_level + 0.95, (-640, 620)), (-460, 620))

    # Conifer belt: a noisy band that stops dead at the tree line and thins on cliffs.
    tl_jitter = t.math("MULTIPLY_ADD", macro, 3.4, (-1000, 460), c=-1.7)
    tl_jitter = t.math("ADD", tl_jitter, height, (-820, 460))
    tl_lifted = t.math("SUBTRACT", tl_jitter, lift, (-720, 460))
    forest_alt = t.math("MULTIPLY",
                        t.band(tl_jitter, water_level + 0.9,
                               water_level + 2.6, (-640, 520)),
                        t.math("SUBTRACT", 1.0,
                               t.band(tl_lifted, tree_line - 2.2, tree_line + 1.6,
                                      (-640, 380)), (-460, 380)),
                        (-300, 460))
    forest_patch = t.band(meso, 0.36, 0.60, (-640, 240))
    forest_slope = t.band(slope, 0.52, 0.78, (-640, 100))
    forest_mask = t.math("MULTIPLY", forest_alt, forest_patch, (-120, 460))
    forest_mask = t.math("MULTIPLY", forest_mask, forest_slope, (60, 460))

    # Scree: loose talus above the tree line, mostly on moderate slopes.
    scree_mask = t.math("MULTIPLY",
                        t.band(tl_lifted, tree_line - 2.5, tree_line + 2.0, (-640, -20)),
                        t.band(macro, 0.30, 0.62, (-640, -160)), (-120, -20))

    # Cliff rock: purely slope-driven, so it works at any altitude.
    rock_mask = t.math("SUBTRACT", 1.0, t.band(slope, 0.56, 0.86, (-640, -320)), (-460, -320))

    # Snow: altitude with a noisy line, suppressed on faces too steep to hold it.
    snow_alt_in = t.math("MULTIPLY_ADD", macro, 3.2, (-1000, 300), c=-1.6)
    snow_alt_in = t.math("ADD", snow_alt_in, height, (-820, 300))
    snow_alt_in = t.math("SUBTRACT", snow_alt_in, lift, (-720, 300))
    snow_alt = t.band(snow_alt_in, snow_line, snow_line + 4.0, (-640, 300))
    # Alpine summits are steep; only near-vertical walls shed their snow.
    # Steep faces shed their snow. The default band only lets snow sit on
    # ground flatter than ~57 deg, which leaves a sharp alpine horn -- exactly
    # what this library is full of -- completely bare. Widening it puts snow
    # back on ledges and couloirs the way real spires carry it.
    snow_slope = t.band(slope, snow_slope[0], snow_slope[1], (-640, 160))
    snow_mask = t.math("MULTIPLY", snow_alt, snow_slope, (-300, 300))

    # -- Composite ---------------------------------------------------------
    col = t.mix_rgb(shore_mask, meadow, shore_col, (300, -200))
    col = t.mix_rgb(forest_mask, col, forest, (460, -200))
    col = t.mix_rgb(scree_mask, col, scree_col, (620, -200))
    col = t.mix_rgb(rock_mask, col, rock, (780, -200))
    col = t.mix_rgb(snow_mask, col, snow_col, (940, -200))

    # Ambient-occlusion cavity shading: gullies, couloirs and crevices between
    # boulders darken the way they do in real rock, which is most of what makes
    # a procedural mountain stop looking like poured concrete.
    ao = t.new("ShaderNodeAmbientOcclusion", (1000, 60))
    ao.samples = 8
    ao.inputs["Distance"].default_value = cavity_distance
    # 0 out in the open, 1 deep inside a couloir or between boulders.
    cavity = t.math("SUBTRACT", 1.0, ao.outputs["AO"], (1160, 60))
    cavity = t.math("MULTIPLY", cavity, cavity_strength, (1300, 60))
    shaded = t.mix_rgb(cavity, col, (0.0, 0.0, 0.0, 1.0), (1300, -140))

    # Named appearance control. The clinical 4MT changes colour and texture
    # between sample and test so the task cannot be solved by pixel matching --
    # v2 lacked this, and frozen encoders duly scored 196/196 on the
    # same-viewpoint condition by matching an identical file. Putting a
    # Hue/Saturation node here lets the renderer restyle a built scene in
    # milliseconds; doing it by editing the biome palette would need a rebuild.
    appearance = t.new("ShaderNodeHueSaturation", (1380, -200))
    appearance.name = APPEARANCE_NODE
    appearance.label = APPEARANCE_NODE
    appearance.inputs["Hue"].default_value = 0.5        # 0.5 = unchanged
    appearance.inputs["Saturation"].default_value = 1.0
    appearance.inputs["Value"].default_value = 1.0
    appearance.inputs["Fac"].default_value = 1.0
    t.link(shaded, appearance.inputs["Color"])

    hazed, fog = _aerial_perspective(t, appearance.outputs["Color"], (1600, -200),
                                     strength=haze_strength, haze_color=haze_color,
                                     haze_start=haze_start, haze_scale=haze_scale)
    t.link(hazed, bsdf.inputs["Base Color"])

    # Roughness: snow is satin, rock and turf are matt; haze flattens the far field.
    rough = t.math("MULTIPLY_ADD", micro, 0.12, (940, 300), c=0.82)
    rough = t.mix_rgb(snow_mask, rough, (0.42, 0.42, 0.42, 1.0), (1260, 300))
    rough = t.mix_rgb(fog, rough, (0.62, 0.62, 0.62, 1.0), (1420, 300))
    t.link(rough, bsdf.inputs["Roughness"])

    bump = t.new("ShaderNodeBump", (1300, -560))
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = bump_distance
    rock_h = t.mix_rgb(0.55, strata, meso, (940, -640))
    detail_h = t.mix_rgb(rock_mask, micro, rock_h, (1100, -560))
    t.link(detail_h, bump.inputs["Height"])
    t.link(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def create_water_material():
    """Still alpine tarn: mirror-flat with a whisper of wind ripple."""
    mat = bpy.data.materials.new(name="LakeWater")
    mat.use_nodes = True
    t = _Tree(mat)

    out = t.new("ShaderNodeOutputMaterial", (700, 0))
    bsdf = t.new("ShaderNodeBsdfPrincipled", (400, 0))
    bsdf.inputs["Base Color"].default_value = (0.020, 0.055, 0.070, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.030
    bsdf.inputs["IOR"].default_value = 1.333
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    t.link(bsdf.outputs["BSDF"], out.inputs["Surface"])

    geom = t.new("ShaderNodeNewGeometry", (-400, -200))
    ripple = t.noise(geom.outputs["Position"], 6.0, 6.0, 0.5, (-200, -200))
    bump = t.new("ShaderNodeBump", (120, -200))
    bump.inputs["Strength"].default_value = 0.05
    bump.inputs["Distance"].default_value = 0.02
    t.link(ripple, bump.inputs["Height"])
    t.link(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def create_tree_material():
    """Conifer canopy: dark blue-green with per-position hue drift."""
    mat = bpy.data.materials.new(name="ConiferFoliage")
    mat.use_nodes = True
    t = _Tree(mat)

    out = t.new("ShaderNodeOutputMaterial", (900, 0))
    bsdf = t.new("ShaderNodeBsdfPrincipled", (600, 0))
    bsdf.inputs["Roughness"].default_value = 0.88
    t.link(bsdf.outputs["BSDF"], out.inputs["Surface"])

    geom = t.new("ShaderNodeNewGeometry", (-500, 0))
    var = t.noise(geom.outputs["Position"], 0.55, 4.0, 0.5, (-300, 0))
    col = t.mix_rgb(var, (0.016, 0.042, 0.020, 1.0), (0.048, 0.092, 0.036, 1.0), (0, 0))
    hazed, _ = _aerial_perspective(t, col, (350, 0))
    t.link(hazed, bsdf.inputs["Base Color"])
    return mat


def create_broadleaf_material():
    """
    Larch / broadleaf canopy: warmer and lighter than the conifers.

    A forest of one species reads as a texture rather than a wood. Giving the
    lower slopes a second, yellower tree separates the belts visually and gives
    the appearance axis something real to act on.
    """
    mat = bpy.data.materials.new(name="BroadleafFoliage")
    mat.use_nodes = True
    t = _Tree(mat)
    out = t.new("ShaderNodeOutputMaterial", (900, 0))
    bsdf = t.new("ShaderNodeBsdfPrincipled", (600, 0))
    bsdf.inputs["Roughness"].default_value = 0.84
    t.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    geom = t.new("ShaderNodeNewGeometry", (-500, 0))
    var = t.noise(geom.outputs["Position"], 0.75, 4.0, 0.5, (-300, 0))
    col = t.mix_rgb(var, (0.043, 0.062, 0.021, 1.0), (0.086, 0.104, 0.034, 1.0), (0, 0))
    hazed, _ = _aerial_perspective(t, col, (350, 0))
    t.link(hazed, bsdf.inputs["Base Color"])
    return mat


def create_deadwood_material():
    """Bleached standing snags and fallen trunks: pale silver-grey timber."""
    mat = bpy.data.materials.new(name="Deadwood")
    mat.use_nodes = True
    t = _Tree(mat)
    out = t.new("ShaderNodeOutputMaterial", (900, 0))
    bsdf = t.new("ShaderNodeBsdfPrincipled", (600, 0))
    bsdf.inputs["Roughness"].default_value = 0.95
    t.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    geom = t.new("ShaderNodeNewGeometry", (-500, 0))
    var = t.noise(geom.outputs["Position"], 1.4, 3.0, 0.5, (-300, 0))
    col = t.mix_rgb(var, (0.088, 0.074, 0.055, 1.0), (0.168, 0.150, 0.124, 1.0), (0, 0))
    hazed, _ = _aerial_perspective(t, col, (350, 0))
    t.link(hazed, bsdf.inputs["Base Color"])
    return mat


def create_trail_material():
    """Trodden earth: the bare, paler ground of a footpath."""
    mat = bpy.data.materials.new(name="TrailEarth")
    mat.use_nodes = True
    t = _Tree(mat)
    out = t.new("ShaderNodeOutputMaterial", (900, 0))
    bsdf = t.new("ShaderNodeBsdfPrincipled", (600, 0))
    bsdf.inputs["Roughness"].default_value = 0.97
    t.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    geom = t.new("ShaderNodeNewGeometry", (-500, 0))
    var = t.noise(geom.outputs["Position"], 2.2, 3.0, 0.5, (-300, 0))
    col = t.mix_rgb(var, (0.126, 0.104, 0.078, 1.0), (0.196, 0.170, 0.132, 1.0), (0, 0))
    hazed, _ = _aerial_perspective(t, col, (350, 0))
    t.link(hazed, bsdf.inputs["Base Color"])
    return mat


def add_sky_clouds(world_tree, sky_color_socket, altitude=1.0, scale=0.55,
                   coverage=(0.40, 0.61), softness=(0.13, 0.33)):
    """
    Composite a cumulus layer onto the sky background.

    Each view ray is intersected with a virtual cloud plane at `altitude`, so
    the clouds get true perspective: big and separated overhead, compressed
    into a band as they approach the horizon. No geometry, no extra ray cost,
    and it works from every camera angle -- unlike a modelled cloud plane,
    which degenerates into a solid ceiling when viewed edge-on.
    """
    nodes, links = world_tree.nodes, world_tree.links

    def new(kind, loc):
        n = nodes.new(kind)
        n.location = loc
        return n

    coord = new("ShaderNodeTexCoord", (-1600, -300))
    sep = new("ShaderNodeSeparateXYZ", (-1420, -300))
    links.new(coord.outputs["Generated"], sep.inputs["Vector"])

    # t = altitude / ray.z  -> the plane-intersection parameter.
    up = new("ShaderNodeMath", (-1240, -420))
    up.operation = "MAXIMUM"
    up.inputs[1].default_value = 0.035        # keep the division well behaved
    links.new(sep.outputs["Z"], up.inputs[0])

    t = new("ShaderNodeMath", (-1060, -420))
    t.operation = "DIVIDE"
    t.inputs[0].default_value = altitude
    links.new(up.outputs["Value"], t.inputs[1])

    uv = new("ShaderNodeVectorMath", (-880, -300))
    uv.operation = "SCALE"
    links.new(coord.outputs["Generated"], uv.inputs[0])
    links.new(t.outputs["Value"], uv.inputs["Scale"])

    mapping = new("ShaderNodeMapping", (-700, -300))
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    links.new(uv.outputs["Vector"], mapping.inputs["Vector"])

    def noise(sc, detail, rough, loc):
        n = new("ShaderNodeTexNoise", loc)
        n.inputs["Scale"].default_value = sc
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    def ramp(socket, lo, hi, loc):
        n = new("ShaderNodeMapRange", loc)
        n.inputs["From Min"].default_value = lo
        n.inputs["From Max"].default_value = hi
        links.new(socket, n.inputs["Value"])
        return n

    clump = ramp(noise(1.05, 4.0, 0.52, (-520, -460)).outputs["Fac"],
                 coverage[0], coverage[1], (-340, -460))
    billow = noise(3.30, 10.0, 0.62, (-520, -180))

    body = new("ShaderNodeMath", (-160, -300))
    body.operation = "MULTIPLY"
    links.new(billow.outputs["Fac"], body.inputs[0])
    links.new(clump.outputs["Result"], body.inputs[1])

    alpha = ramp(body.outputs["Value"], softness[0], softness[1], (20, -300))

    # Fade the deck out along the horizon, where the projection degenerates.
    horizon = ramp(sep.outputs["Z"], 0.015, 0.13, (-1240, -620))
    alpha_h = new("ShaderNodeMath", (200, -300))
    alpha_h.operation = "MULTIPLY"
    links.new(alpha.outputs["Result"], alpha_h.inputs[0])
    links.new(horizon.outputs["Result"], alpha_h.inputs[1])

    # Sunlit tops vs shaded undersides.
    shade = new("ShaderNodeMix", (200, -60))
    shade.data_type = "RGBA"
    shade.inputs[6].default_value = (1.55, 1.62, 1.95, 1.0)
    shade.inputs[7].default_value = (4.30, 4.40, 4.60, 1.0)
    links.new(billow.outputs["Fac"], shade.inputs[0])

    composite = new("ShaderNodeMix", (420, -160))
    composite.data_type = "RGBA"
    links.new(alpha_h.outputs["Value"], composite.inputs[0])
    links.new(sky_color_socket, composite.inputs[6])
    links.new(shade.outputs[2], composite.inputs[7])
    return composite.outputs[2]
