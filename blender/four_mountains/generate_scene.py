"""
Procedural Four Mountains Scene Generator for Blender 5.2+
Generates four geomorphologically distinct mountain peaks with polar boundary clipping,
continuous alpine valley, panoramic backdrop mountain range,
procedural alpine PBR materials, and Nishita sky lighting.
"""

import sys
import os
import math
import numpy as np
import bpy
from mathutils import Vector, Euler


def clean_scene():
    """Remove all default objects, materials, and collections."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if "Scene Collection" not in bpy.data.collections and len(bpy.data.collections) == 0:
        col = bpy.data.collections.new("FourMountains")
        bpy.context.scene.collection.children.link(col)


def configure_render_engine(scene, engine="CYCLES", samples=64, resolution=(1024, 1024)):
    """Configure render settings with Metal GPU acceleration and balanced physical exposure."""
    scene.render.engine = engine
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100

    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.preview_samples = 32
        scene.cycles.use_denoising = True

        try:
            cpref = bpy.context.preferences.addons["cycles"].preferences
            cpref.get_devices()
            metal_devices = [d for d in cpref.devices if d.type == "METAL"]
            if metal_devices:
                cpref.compute_device_type = "METAL"
                for d in metal_devices:
                    d.use = True
                scene.cycles.device = "GPU"
                print(f"[Blender] Using Metal GPU: {[d.name for d in metal_devices]}")
            else:
                scene.cycles.device = "CPU"
        except Exception as e:
            print(f"[Blender] Warning configuring Cycles device: {e}")

    # Color management: AgX or Filmic
    scene.view_settings.view_transform = "AgX" if "AgX" in [v.name for v in scene.display_settings.bl_rna.properties["display_device"].enum_items] else "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -4.0


def setup_lighting(scene, sun_elevation=math.radians(24), sun_rotation=math.radians(52)):
    """Set up Nishita physical sky with low golden-hour sun for dramatic topography shadows."""
    world = bpy.data.worlds.new("AlpineWorld")
    scene.world = world
    world_tree = world.node_tree
    world_tree.nodes.clear()

    out_node = world_tree.nodes.new("ShaderNodeOutputWorld")
    sky_node = world_tree.nodes.new("ShaderNodeTexSky")
    bg_node = world_tree.nodes.new("ShaderNodeBackground")

    sky_node.sky_type = "MULTIPLE_SCATTERING"
    sky_node.sun_elevation = sun_elevation
    sky_node.sun_rotation = sun_rotation
    sky_node.air_density = 1.0
    sky_node.aerosol_density = 1.4
    sky_node.ozone_density = 2.0
    sky_node.sun_intensity = 1.0

    bg_node.inputs["Strength"].default_value = 1.0
    world_tree.links.new(sky_node.outputs["Color"], bg_node.inputs["Color"])
    world_tree.links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])

    # Synchronized directional sun
    sun_data = bpy.data.lights.new(name="SunLight", type="SUN")
    sun_data.energy = 2.5
    sun_data.color = (1.0, 0.95, 0.88)
    sun_data.angle = math.radians(1.0)

    sun_obj = bpy.data.objects.new(name="Sun", object_data=sun_data)
    scene.collection.objects.link(sun_obj)

    sun_rot_x = math.pi / 2 - sun_elevation
    sun_rot_z = sun_rotation
    sun_obj.rotation_euler = Euler((sun_rot_x, 0.0, sun_rot_z), "XYZ")
    sun_obj.location = (0, 0, 60)


def create_alpine_material():
    """
    Procedural Alpine Tri-Planar PBR Material:
    - Steep slopes (>38°): Deep charcoal/slate granite rock with sharp vertical fissures.
    - Lowland/valley (<28° and Z < 7m): Vibrant emerald alpine grass & meadow moss.
    - Transition scree: Gravel talus banks at the mountain bases.
    - High summits (Z > 9.5m): Snow caps clinging to crests, arêtes, and north-facing hollows.
    """
    mat = bpy.data.materials.new(name="AlpineMaterial")
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (1500, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (1200, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    geom = nodes.new("ShaderNodeNewGeometry")
    geom.location = (-950, 200)

    sep_pos = nodes.new("ShaderNodeSeparateXYZ")
    sep_pos.location = (-750, 350)
    links.new(geom.outputs["Position"], sep_pos.inputs["Vector"])

    # Slope: Dot product of Normal with (0, 0, 1)
    dot_normal = nodes.new("ShaderNodeVectorMath")
    dot_normal.operation = "DOT_PRODUCT"
    dot_normal.location = (-750, 100)
    links.new(geom.outputs["Normal"], dot_normal.inputs[0])
    dot_normal.inputs[1].default_value = (0.0, 0.0, 1.0)

    # Multi-frequency procedural noise textures
    macro_noise = nodes.new("ShaderNodeTexNoise")
    macro_noise.location = (-750, -100)
    macro_noise.inputs["Scale"].default_value = 0.35
    macro_noise.inputs["Detail"].default_value = 5.0
    macro_noise.inputs["Roughness"].default_value = 0.65
    links.new(geom.outputs["Position"], macro_noise.inputs["Vector"])

    micro_noise = nodes.new("ShaderNodeTexNoise")
    micro_noise.location = (-750, -300)
    micro_noise.inputs["Scale"].default_value = 2.5
    micro_noise.inputs["Detail"].default_value = 8.0
    micro_noise.inputs["Roughness"].default_value = 0.75
    links.new(geom.outputs["Position"], micro_noise.inputs["Vector"])

    # Vertical rock strata noise
    strata_map = nodes.new("ShaderNodeMapping")
    strata_map.location = (-750, -500)
    strata_map.inputs["Scale"].default_value = (0.6, 0.6, 4.0)
    links.new(geom.outputs["Position"], strata_map.inputs["Vector"])

    strata_noise = nodes.new("ShaderNodeTexNoise")
    strata_noise.location = (-550, -500)
    strata_noise.inputs["Scale"].default_value = 1.2
    strata_noise.inputs["Detail"].default_value = 6.0
    links.new(strata_map.outputs["Vector"], strata_noise.inputs["Vector"])

    # Slope Ramp: 1 = Steep cliff (Rock), 0 = Flat (Grass/Meadow)
    slope_ramp = nodes.new("ShaderNodeValToRGB")
    slope_ramp.location = (-480, 100)
    slope_ramp.color_ramp.elements[0].position = 0.60  # Steep cliff
    slope_ramp.color_ramp.elements[0].color = (1, 1, 1, 1)  # Factor 1 -> Rock
    slope_ramp.color_ramp.elements[1].position = 0.85  # Gentle slope
    slope_ramp.color_ramp.elements[1].color = (0, 0, 0, 1)  # Factor 0 -> Meadow
    links.new(dot_normal.outputs["Value"], slope_ramp.inputs["Fac"])

    # Rock Colors: High-contrast dark granite and slate
    rock_dark = (0.08, 0.08, 0.09, 1.0)
    rock_light = (0.24, 0.22, 0.20, 1.0)
    rock_mix = nodes.new("ShaderNodeMix")
    rock_mix.data_type = "RGBA"
    rock_mix.location = (-220, -400)
    rock_mix.inputs[6].default_value = rock_dark
    rock_mix.inputs[7].default_value = rock_light
    links.new(strata_noise.outputs["Fac"], rock_mix.inputs[0])

    # Meadow Colors: Vibrant alpine moss and sunlit grass
    grass_dark = (0.10, 0.19, 0.05, 1.0)
    grass_light = (0.26, 0.36, 0.10, 1.0)
    grass_mix = nodes.new("ShaderNodeMix")
    grass_mix.data_type = "RGBA"
    grass_mix.location = (-220, -200)
    grass_mix.inputs[6].default_value = grass_dark
    grass_mix.inputs[7].default_value = grass_light
    links.new(micro_noise.outputs["Fac"], grass_mix.inputs[0])

    # Scree / Talus gravel (transition slopes)
    scree_col = (0.20, 0.18, 0.15, 1.0)
    meadow_scree_mix = nodes.new("ShaderNodeMix")
    meadow_scree_mix.data_type = "RGBA"
    meadow_scree_mix.location = (40, -200)
    links.new(macro_noise.outputs["Fac"], meadow_scree_mix.inputs[0])
    links.new(grass_mix.outputs[2], meadow_scree_mix.inputs[6])
    meadow_scree_mix.inputs[7].default_value = scree_col

    # Blend Meadow/Scree with Cliff Rock based on Slope
    terrain_base_mix = nodes.new("ShaderNodeMix")
    terrain_base_mix.data_type = "RGBA"
    terrain_base_mix.location = (300, -100)
    links.new(slope_ramp.outputs["Color"], terrain_base_mix.inputs[0])
    links.new(meadow_scree_mix.outputs[2], terrain_base_mix.inputs[6])
    links.new(rock_mix.outputs[2], terrain_base_mix.inputs[7])

    # Snow Mask: Altitude-dependent with slope inhibition
    snow_z = nodes.new("ShaderNodeMath")
    snow_z.operation = "ADD"
    snow_z.location = (-480, 350)
    links.new(sep_pos.outputs["Z"], snow_z.inputs[0])

    z_noise_scale = nodes.new("ShaderNodeMath")
    z_noise_scale.operation = "MULTIPLY"
    z_noise_scale.location = (-480, 200)
    z_noise_scale.inputs[1].default_value = 2.0
    links.new(macro_noise.outputs["Fac"], z_noise_scale.inputs[0])
    links.new(z_noise_scale.outputs["Value"], snow_z.inputs[1])

    # Snow begins at ~10.0m, fully covering by ~14.0m
    snow_alt_ramp = nodes.new("ShaderNodeMapRange")
    snow_alt_ramp.location = (-220, 350)
    snow_alt_ramp.inputs["From Min"].default_value = 10.0
    snow_alt_ramp.inputs["From Max"].default_value = 14.0
    snow_alt_ramp.inputs["To Min"].default_value = 0.0
    snow_alt_ramp.inputs["To Max"].default_value = 1.0
    links.new(snow_z.outputs["Value"], snow_alt_ramp.inputs["Value"])

    # Snow slope factor: snow slides off sheer cliffs
    snow_slope = nodes.new("ShaderNodeMapRange")
    snow_slope.location = (-220, 150)
    snow_slope.inputs["From Min"].default_value = 0.40
    snow_slope.inputs["From Max"].default_value = 0.70
    snow_slope.inputs["To Min"].default_value = 0.0
    snow_slope.inputs["To Max"].default_value = 1.0
    links.new(dot_normal.outputs["Value"], snow_slope.inputs["Value"])

    snow_factor = nodes.new("ShaderNodeMath")
    snow_factor.operation = "MULTIPLY"
    snow_factor.location = (40, 250)
    links.new(snow_alt_ramp.outputs["Result"], snow_factor.inputs[0])
    links.new(snow_slope.outputs["Result"], snow_factor.inputs[1])

    # Final Color: Blend Terrain with Snow
    snow_color = (0.96, 0.98, 1.0, 1.0)
    final_color_mix = nodes.new("ShaderNodeMix")
    final_color_mix.data_type = "RGBA"
    final_color_mix.location = (580, 0)
    links.new(snow_factor.outputs["Value"], final_color_mix.inputs[0])
    links.new(terrain_base_mix.outputs[2], final_color_mix.inputs[6])
    final_color_mix.inputs[7].default_value = snow_color
    links.new(final_color_mix.outputs[2], bsdf.inputs["Base Color"])

    # Bump mapping for rock & scree surface detail
    bump = nodes.new("ShaderNodeBump")
    bump.location = (900, -250)
    bump.inputs["Strength"].default_value = 0.75
    bump.inputs["Distance"].default_value = 0.25
    links.new(micro_noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Roughness"].default_value = 0.85
    return mat


def generate_mountain_mesh(name, morphology_type, height=15.0, base_radius=12.0, n_r=70, n_theta=180, seed=42):
    """
    Generate an authentic mountain peak using polar concentric rings.
    At R = base_radius, elevation Z touches 0.00 exactly, eliminating any flat overlapping skirts!
    """
    np.random.seed(seed)
    r_vals = np.linspace(0.0, base_radius, n_r)
    theta_vals = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    R, Theta = np.meshgrid(r_vals, theta_vals, indexing="ij")

    # 1. 2D Domain warping
    def warp(px, py, f=0.14, s=2.0):
        wx = np.sin(px * f * 1.4 + 1.1) * np.cos(py * f * 1.2 + 0.7) + \
             0.5 * np.sin(px * f * 2.8 - 1.5) * np.cos(py * f * 2.4 + 1.2)
        wy = np.cos(px * f * 1.3 + 0.8) * np.sin(py * f * 1.5 + 2.1) + \
             0.5 * np.cos(px * f * 2.6 + 1.9) * np.sin(py * f * 2.7 - 0.9)
        return s * wx, s * wy

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    Wx, Wy = warp(X, Y, f=0.14, s=2.0)
    Xw = X + Wx
    Yw = Y + Wy
    Rw = np.sqrt(Xw**2 + Yw**2)
    Tw = np.arctan2(Yw, Xw)

    norm_r = np.clip(R / base_radius, 0.0, 1.0)
    # Cosine taper: exactly 0 at boundary R = base_radius
    taper = 0.5 * (1.0 + np.cos(np.pi * norm_r))

    # 2. Multi-octave sharp ridge fractal
    def ridged_noise(px, py, octaves=5, base_f=0.20):
        val = np.zeros_like(px)
        amp = 1.0
        f = base_f
        for _ in range(octaves):
            phase_x = np.random.uniform(0, 100)
            phase_y = np.random.uniform(0, 100)
            angle = np.random.uniform(0, 2 * np.pi)
            rx = px * np.cos(angle) - py * np.sin(angle)
            ry = px * np.sin(angle) + py * np.cos(angle)
            n_raw = np.sin(rx * f + phase_x) * np.cos(ry * f + phase_y)
            r_val = (1.0 - np.abs(n_raw)) ** 2.2
            val += amp * r_val
            f *= 2.1
            amp *= 0.48
        return val / 1.7

    r_fbm = ridged_noise(Xw, Yw, octaves=5, base_f=0.22)

    if morphology_type == "horn":
        arête_mask = (np.abs(np.cos(2.0 * Tw + 0.3))) ** 3.5
        cirque_carve = 0.35 * (1.0 - arête_mask) * (norm_r ** 1.3)
        profile = (1.0 - norm_r) ** 1.9
        Z = height * (profile * (0.50 + 0.50 * arête_mask) - cirque_carve + 0.35 * r_fbm * (1.0 - norm_r)) * taper

    elif morphology_type == "ridge":
        aspect_x = 1.55
        aspect_y = 0.70
        ang = np.radians(20)
        Xr = Xw * np.cos(ang) - Yw * np.sin(ang)
        Yr = Xw * np.sin(ang) + Yw * np.cos(ang)
        R_ellip = np.sqrt((Xr / aspect_x)**2 + (Yr / aspect_y)**2)
        norm_ellip = np.clip(R_ellip / base_radius, 0.0, 1.0)
        taper_ellip = np.where(R_ellip > base_radius, 0.0, 0.5 * (1.0 + np.cos(np.pi * norm_ellip)))
        saddle = 0.75 + 0.28 * np.cos(2.7 * Xr / base_radius)
        spine = np.exp(-((Yr / (base_radius * 0.22)) ** 2.0))
        profile = (1.0 - norm_ellip) ** 1.35
        Z = height * (profile * saddle + 0.25 * spine + 0.32 * r_fbm * (1.0 - norm_ellip)) * taper_ellip

    elif morphology_type == "mesa_crag":
        profile = (1.0 - norm_r) ** 1.15
        tiers = 0.12 * np.sin(4.5 * np.pi * profile)
        faults = 0.10 * np.sin(3.0 * Tw)
        Z = height * (profile + tiers + faults + 0.40 * r_fbm * (1.0 - norm_r)) * taper

    elif morphology_type == "dome":
        profile = 1.0 / (1.0 + 3.8 * (norm_r ** 1.9))
        ravines = 0.12 * np.sin(8.0 * Tw) * (norm_r ** 1.2)
        Z = height * (profile + ravines + 0.22 * r_fbm * (1.0 - norm_r)) * taper

    else:
        Z = height * (1.0 - norm_r) * taper

    Z = np.maximum(Z, 0.0)

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    obj = bpy.data.objects.new(name, mesh)

    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).tolist()
    faces = []
    for i in range(n_r - 1):
        r1 = i * n_theta
        r2 = (i + 1) * n_theta
        for j in range(n_theta):
            next_j = (j + 1) % n_theta
            faces.append((r1 + j, r1 + next_j, r2 + next_j, r2 + j))

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    for poly in mesh.polygons:
        poly.use_smooth = True

    return obj


def generate_landscape(material, valley_radius=55.0, ring_radius=115.0, ring_height=36.0):
    """
    Generate a seamless terrain mesh:
    1. Rolling alpine valley floor.
    2. Panoramic 360° backdrop mountain horizon.
    """
    n_r = 90
    n_theta = 180

    r_vals = np.concatenate([
        np.linspace(0.0, valley_radius, 55),
        np.linspace(valley_radius, ring_radius, 35)[1:]
    ])
    theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R_grid, T_grid = np.meshgrid(r_vals, theta_vals, indexing="ij")

    X = R_grid * np.cos(T_grid)
    Y = R_grid * np.sin(T_grid)

    # Gentle valley undulation
    np.random.seed(77)
    valley_z = 0.35 * np.sin(0.14 * X) * np.cos(0.12 * Y) + 0.20 * np.sin(0.28 * X + 0.22 * Y)
    valley_z *= np.clip(R_grid / 16.0, 0.0, 1.0)

    # Backdrop mountain skyline
    r_factor = np.clip((R_grid - valley_radius) / (ring_radius - valley_radius), 0.0, 1.0) ** 1.35
    skyline = (
        0.48 * np.sin(3.0 * T_grid + 0.5) +
        0.34 * np.cos(7.0 * T_grid - 1.2) +
        0.26 * np.sin(13.0 * T_grid + 2.0) +
        0.18 * np.cos(27.0 * T_grid) +
        0.10 * np.sin(42.0 * T_grid)
    )
    skyline = (skyline - skyline.min()) / (skyline.max() - skyline.min())
    backdrop_z = ring_height * r_factor * (0.35 + 0.65 * skyline)

    Z = np.where(R_grid <= valley_radius, valley_z, valley_z + backdrop_z)

    mesh = bpy.data.meshes.new("Landscape_Mesh")
    obj = bpy.data.objects.new("Landscape", mesh)

    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).tolist()
    faces = []
    for i in range(len(r_vals) - 1):
        r1 = i * n_theta
        r2 = (i + 1) * n_theta
        for j in range(n_theta):
            next_j = (j + 1) % n_theta
            faces.append((r1 + j, r1 + next_j, r2 + next_j, r2 + j))

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    for poly in mesh.polygons:
        poly.use_smooth = True

    obj.data.materials.append(material)
    return obj


def setup_camera_rig(scene):
    """Set up an orbit camera tracking an empty target at scene center."""
    target = bpy.data.objects.new("CameraTarget", None)
    target.location = (0, 0, 4.0)
    scene.collection.objects.link(target)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 45  # Natural 45mm perspective
    cam_data.clip_start = 0.1
    cam_data.clip_end = 450.0

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    track = cam_obj.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    return cam_obj, target


def position_camera_orbit(cam_obj, radius=54.0, elevation_deg=28.0, azimuth_deg=55.0):
    """Place camera on a spherical orbit around (0, 0, 4.0)."""
    phi = math.radians(elevation_deg)
    theta = math.radians(azimuth_deg)

    x = radius * math.cos(phi) * math.cos(theta)
    y = radius * math.cos(phi) * math.sin(theta)
    z = radius * math.sin(phi) + 4.0

    cam_obj.location = Vector((x, y, z))
    bpy.context.view_layer.update()


def build_four_mountains_scene(output_blend_path="blender/four_mountains/four_mountains.blend"):
    """Main scene assembly function."""
    print("=" * 60)
    print("Building Photorealistic Four Mountains Scene in Blender 5.2...")
    print("=" * 60)

    clean_scene()
    scene = bpy.context.scene

    # 1. Configure render engine (Cycles + Metal GPU)
    configure_render_engine(scene, engine="CYCLES", samples=64, resolution=(1024, 1024))

    # 2. Setup Nishita Sky & Sun lighting
    setup_lighting(scene)

    # 3. Create procedural alpine PBR material
    alpine_mat = create_alpine_material()

    # 4. Generate the 4 distinct mountain peaks with polar meshes
    mountain_configs = [
        {"name": "M1", "type": "horn", "height": 16.0, "radius": 12.0, "pos": (-12.0, 10.0, -0.2), "rot_z": 0.40, "seed": 101, "pass_id": 3},
        {"name": "M2", "type": "ridge", "height": 13.0, "radius": 13.5, "pos": (12.5, 10.5, -0.2), "rot_z": -0.50, "seed": 202, "pass_id": 4},
        {"name": "M3", "type": "mesa_crag", "height": 11.5, "radius": 11.5, "pos": (-10.0, -11.5, -0.2), "rot_z": 1.15, "seed": 303, "pass_id": 5},
        {"name": "M4", "type": "dome", "height": 14.0, "radius": 14.0, "pos": (11.0, -10.0, -0.2), "rot_z": 0.0, "seed": 404, "pass_id": 6},
    ]

    mountains = []
    for cfg in mountain_configs:
        print(f"  Generating {cfg['name']}: {cfg['type']} (height={cfg['height']}m, radius={cfg['radius']}m)...")
        m_obj = generate_mountain_mesh(
            cfg["name"],
            morphology_type=cfg["type"],
            height=cfg["height"],
            base_radius=cfg["radius"],
            n_r=70,
            n_theta=180,
            seed=cfg["seed"]
        )
        m_obj.location = Vector(cfg["pos"])
        m_obj.rotation_euler = Euler((0, 0, cfg["rot_z"]), "XYZ")
        m_obj.pass_index = cfg["pass_id"]
        m_obj.data.materials.append(alpine_mat)
        scene.collection.objects.link(m_obj)
        mountains.append(m_obj)

    # 5. Generate Seamless Alpine Landscape (Valley + Horizon Backdrop Ring)
    print("  Generating Seamless Alpine Landscape & 360° Horizon...")
    landscape_obj = generate_landscape(alpine_mat)
    landscape_obj.pass_index = 1
    scene.collection.objects.link(landscape_obj)

    # 6. Setup Camera Rig
    cam_obj, target = setup_camera_rig(scene)
    position_camera_orbit(cam_obj, radius=54.0, elevation_deg=28.0, azimuth_deg=55.0)

    # 7. Enable Render Passes for depth / segmentation
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.view_layers["ViewLayer"].use_pass_object_index = True

    # Save .blend file
    os.makedirs(os.path.dirname(output_blend_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)
    print(f"Successfully generated and saved scene to: {output_blend_path}")

    return output_blend_path


if __name__ == "__main__":
    blend_path = build_four_mountains_scene()
    print("Done!")
