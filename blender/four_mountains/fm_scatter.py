"""
Mesh construction helpers: numpy -> Blender meshes, plus the vegetation and
boulder scattering used to give the valley and the lower slopes real silhouettes.

Scattered geometry is emitted in the *local* frame of its host object and then
parented to it, so moving a mountain carries its forest and its boulders along.
"""

import numpy as np
import bpy

from fm_noise import fbm


def mesh_from_grid(name, X, Y, Z, wrap_theta=True, smooth=True):
    """Build a quad mesh from a structured (n_u, n_v) heightfield grid."""
    n_u, n_v = Z.shape
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).tolist()

    faces = []
    v_limit = n_v if wrap_theta else n_v - 1
    for i in range(n_u - 1):
        row0 = i * n_v
        row1 = (i + 1) * n_v
        for j in range(v_limit):
            j1 = (j + 1) % n_v
            faces.append((row0 + j, row0 + j1, row1 + j1, row1 + j))

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    if smooth:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()
    return bpy.data.objects.new(name, mesh)


def mesh_from_arrays(name, verts, faces, smooth=False):
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(np.asarray(verts).tolist(), [], list(faces))
    mesh.update()
    if smooth and len(mesh.polygons):
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
        mesh.update()
    return bpy.data.objects.new(name, mesh)


# --------------------------------------------------------------------------- #
# Surface analysis
# --------------------------------------------------------------------------- #

def grid_slope(X, Y, Z):
    """cos(slope) per grid vertex: 1 = flat, 0 = vertical."""
    dzdu = np.gradient(Z, axis=0)
    dzdv = np.gradient(Z, axis=1)
    dxu = np.gradient(X, axis=0)
    dyu = np.gradient(Y, axis=0)
    dxv = np.gradient(X, axis=1)
    dyv = np.gradient(Y, axis=1)
    du = np.hypot(dxu, dyu) + 1e-6
    dv = np.hypot(dxv, dyv) + 1e-6
    grad = np.hypot(dzdu / du, dzdv / dv)
    return 1.0 / np.sqrt(1.0 + grad ** 2)


def sample_grid(X, Y, Z, weights, count, rng, jitter=0.0):
    """Draw `count` surface points with probability proportional to `weights`."""
    w = np.clip(weights, 0.0, None).ravel()
    total = w.sum()
    if total <= 0 or count <= 0:
        return np.zeros((0, 3))
    idx = rng.choice(w.size, size=int(count), replace=True, p=w / total)
    pts = np.stack([X.ravel()[idx], Y.ravel()[idx], Z.ravel()[idx]], axis=-1)
    if jitter > 0:
        pts[:, :2] += rng.normal(0.0, jitter, size=(len(idx), 2))
    return pts


# --------------------------------------------------------------------------- #
# Instanced props
# --------------------------------------------------------------------------- #

def _unit_conifer(tiers=3, segments=7):
    """A ~40-triangle spruce of unit height, origin at the base."""
    verts = [(0.0, 0.0, 0.0)]
    faces = []
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)

    # Trunk
    trunk_r, trunk_h = 0.022, 0.20
    base_ring = len(verts)
    verts += [(trunk_r * np.cos(a), trunk_r * np.sin(a), 0.0) for a in ang]
    top_ring = len(verts)
    verts += [(trunk_r * np.cos(a), trunk_r * np.sin(a), trunk_h) for a in ang]
    for j in range(segments):
        j1 = (j + 1) % segments
        faces.append((base_ring + j, base_ring + j1, top_ring + j1, top_ring + j))

    # Stacked canopy cones
    for k in range(tiers):
        f = k / max(tiers - 1, 1)
        z0 = 0.14 + 0.28 * k
        z1 = z0 + 0.42 - 0.06 * k
        radius = 0.185 * (1.0 - 0.52 * f)
        ring = len(verts)
        verts += [(radius * np.cos(a), radius * np.sin(a), z0) for a in ang]
        tip = len(verts)
        verts.append((0.0, 0.0, min(z1, 1.0)))
        for j in range(segments):
            j1 = (j + 1) % segments
            faces.append((ring + j, ring + j1, tip))
        faces.append(tuple(range(ring, ring + segments)))  # cone underside
    return np.array(verts), faces


def _unit_boulder(rng, rings=5, segments=8):
    """Irregular low-poly rock of unit radius, sunk slightly into the ground."""
    phi = np.linspace(0.15 * np.pi, 0.95 * np.pi, rings)
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    P, T = np.meshgrid(phi, theta, indexing="ij")
    r = 1.0 + rng.normal(0.0, 0.16, P.shape)
    X = r * np.sin(P) * np.cos(T)
    Y = r * np.sin(P) * np.sin(T) * 1.15
    Z = r * np.cos(P) * 0.62 + 0.25
    verts, faces = [], []
    for i in range(rings):
        for j in range(segments):
            verts.append((X[i, j], Y[i, j], Z[i, j]))
    for i in range(rings - 1):
        for j in range(segments):
            j1 = (j + 1) % segments
            faces.append((i * segments + j, i * segments + j1,
                          (i + 1) * segments + j1, (i + 1) * segments + j))
    return np.array(verts), faces


def _instance(base_verts, base_faces, points, scales, rot_z, tilt=None):
    """Bake transformed copies of one prop into a single combined mesh."""
    all_verts = []
    all_faces = []
    n_base = len(base_verts)
    for i, (p, s) in enumerate(zip(points, scales)):
        c, sn = np.cos(rot_z[i]), np.sin(rot_z[i])
        rot = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        v = base_verts * s
        if tilt is not None:
            tx = tilt[i]
            ct, st = np.cos(tx), np.sin(tx)
            v = v @ np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]]).T
        v = v @ rot.T + p
        offset = i * n_base
        all_verts.append(v)
        all_faces.extend([tuple(idx + offset for idx in f) for f in base_faces])
    if not all_verts:
        return np.zeros((0, 3)), []
    return np.concatenate(all_verts, axis=0), all_faces


def scatter_conifers(name, X, Y, Z, count, rng, z_min=0.9, z_max=8.5,
                     min_slope=0.60, height_range=(0.34, 0.72), radius_limit=None):
    """
    Populate the conifer belt of a heightfield: above the shoreline, below the
    tree line, off the cliffs. Returns a Blender object (or None if empty).
    """
    if count <= 0:
        return None
    cos_slope = grid_slope(X, Y, Z)

    belt = np.clip((Z - z_min) / 0.8, 0.0, 1.0) * np.clip((z_max - Z) / 1.6, 0.0, 1.0)
    walkable = np.clip((cos_slope - min_slope) / 0.25, 0.0, 1.0)
    # Patchiness so the forest reads as stands rather than an even lawn.
    patch = np.clip(fbm(X, Y, freq=0.045, octaves=5, seed=1234) * 2.6 - 0.95, 0.0, 1.0) ** 1.4
    weights = belt * walkable * patch

    if radius_limit is not None:
        weights = weights * (np.hypot(X, Y) < radius_limit)
    # Compensate for the polar grid packing far more vertices near the centre.
    weights = weights * (np.hypot(X, Y) + 1.0)

    pts = sample_grid(X, Y, Z, weights, count, rng, jitter=0.25)
    if len(pts) == 0:
        return None
    pts[:, 2] -= 0.03  # bed the trunks into the ground

    base_v, base_f = _unit_conifer()
    scales = rng.uniform(*height_range, size=len(pts))[:, None]
    rots = rng.uniform(0, 2 * np.pi, size=len(pts))
    tilt = rng.normal(0.0, 0.05, size=len(pts))
    verts, faces = _instance(base_v, base_f, pts, scales, rots, tilt)
    return mesh_from_arrays(name, verts, faces, smooth=False)


def scatter_boulders(name, X, Y, Z, count, rng, z_min=0.2, z_max=11.0,
                     min_slope=0.45, size_range=(0.10, 0.45), radius_limit=None):
    """Erratics and talus blocks, biased toward steeper ground near cliff bases."""
    if count <= 0:
        return None
    cos_slope = grid_slope(X, Y, Z)
    belt = np.clip((Z - z_min) / 0.6, 0.0, 1.0) * np.clip((z_max - Z) / 2.0, 0.0, 1.0)
    ok = np.clip((cos_slope - min_slope) / 0.30, 0.0, 1.0)
    rough = np.clip(1.25 - cos_slope, 0.0, 1.0) + 0.25
    weights = belt * ok * rough
    if radius_limit is not None:
        weights = weights * (np.hypot(X, Y) < radius_limit)
    weights = weights * (np.hypot(X, Y) + 1.0)

    pts = sample_grid(X, Y, Z, weights, count, rng, jitter=0.4)
    if len(pts) == 0:
        return None

    base_v, base_f = _unit_boulder(rng)
    scales = rng.uniform(*size_range, size=len(pts))[:, None]
    rots = rng.uniform(0, 2 * np.pi, size=len(pts))
    tilt = rng.normal(0.0, 0.18, size=len(pts))
    verts, faces = _instance(base_v, base_f, pts, scales, rots, tilt)
    pts[:, 2] -= scales[:, 0] * 0.25
    return mesh_from_arrays(name, verts, faces, smooth=True)


def parent_keep_local(child, parent):
    """Parent without altering the child's already-local vertex coordinates."""
    from mathutils import Matrix
    child.parent = parent
    child.matrix_parent_inverse = Matrix.Identity(4)
