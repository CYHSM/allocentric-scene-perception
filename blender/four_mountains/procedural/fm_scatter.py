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

def _unit_conifer(tiers=3, segments=10, rng=None):
    """
    A spruce of unit height, origin at the base.

    Ten segments rather than seven, and each canopy ring is jittered off a
    perfect circle: at seven segments a lit cone shows its facets plainly, and
    every tree showing the same facets in the same places is what reads as
    "polygonal" even when the trees themselves vary in size.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
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
        jit = 1.0 + rng.normal(0.0, 0.13, segments)
        droop = rng.uniform(-0.012, 0.004, segments)
        ring = len(verts)
        verts += [(radius * jit[j] * np.cos(a), radius * jit[j] * np.sin(a),
                   z0 + droop[j]) for j, a in enumerate(ang)]
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


# How hard the tree-size draw is biased toward young trees. 1.0 is uniform (a
# plantation); higher values give the many-small / few-large structure of a real
# stand. 2.4 was picked by eye against the rendered forest.
AGE_SKEW = 2.4

# Fraction of height a tree loses as it approaches the tree line.
TREELINE_DWARFING = 0.45


def _unit_shrub(rng, rings=4, segments=9):
    """
    A closed dome of unit radius sitting on the ground: the shrub layer.

    Built as an upper hemisphere from the apex down to a ring at Z == 0. An
    earlier version swept phi from 0.25*pi to 0.98*pi, which left a hole at the
    top and clipped the lower hemisphere into a rim -- the bushes rendered as
    green donuts lying in the grass.
    """
    verts = [(0.0, 0.0, 1.0)]                      # apex
    faces = []
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    phi = np.linspace(0.0, 0.5 * np.pi, rings + 1)[1:]   # apex handled above
    for i, ph in enumerate(phi):
        jitter = 1.0 + rng.normal(0.0, 0.10, segments)
        ring = len(verts)
        for j, th in enumerate(theta):
            r = jitter[j] * np.sin(ph)
            z = 0.0 if i == len(phi) - 1 else np.cos(ph) * jitter[j]
            verts.append((r * np.cos(th), r * np.sin(th), max(z, 0.0)))
        if i == 0:
            for j in range(segments):
                faces.append((0, ring + j, ring + (j + 1) % segments))
        else:
            prev = ring - segments
            for j in range(segments):
                j1 = (j + 1) % segments
                # Wound to match the apex fan. Reversed, these quads face
                # INWARD, and smooth shading then renders a dark crater in the
                # middle of every bush -- the "green donuts".
                faces.append((prev + j, ring + j, ring + j1, prev + j1))
    return np.array(verts), faces


def _unit_snag(segments=5, rng=None):
    """A dead standing tree: bare tapered trunk of unit height with stub limbs."""
    rng = rng or np.random.default_rng(0)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts, faces = [], []
    r0, r1 = 0.045, 0.016
    base = len(verts)
    verts += [(r0 * np.cos(a), r0 * np.sin(a), 0.0) for a in ang]
    top = len(verts)
    verts += [(r1 * np.cos(a), r1 * np.sin(a), 1.0) for a in ang]
    for j in range(segments):
        j1 = (j + 1) % segments
        faces.append((base + j, base + j1, top + j1, top + j))
    for k in range(3):                      # broken limbs
        h = 0.45 + 0.18 * k
        a = float(rng.uniform(0, 2 * np.pi))
        L = float(rng.uniform(0.10, 0.22))
        i0 = len(verts)
        verts += [(0.0, 0.0, h), (0.0, 0.0, h + 0.05),
                  (L * np.cos(a), L * np.sin(a), h + 0.09)]
        faces.append((i0, i0 + 1, i0 + 2))
    return np.array(verts), faces


def _unit_broadleaf(rng, segments=9):
    """A rounded larch/broadleaf crown on a short trunk, unit height."""
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts, faces = [], []
    tr, th = 0.035, 0.34
    base = len(verts)
    verts += [(tr * np.cos(a), tr * np.sin(a), 0.0) for a in ang]
    top = len(verts)
    verts += [(tr * np.cos(a), tr * np.sin(a), th) for a in ang]
    for j in range(segments):
        j1 = (j + 1) % segments
        faces.append((base + j, base + j1, top + j1, top + j))
    # crown: two stacked rings plus a cap, jittered so it is not a sphere
    rings = [(0.21, 0.44), (0.25, 0.66), (0.15, 0.87)]
    prev = None
    for rad, z in rings:
        jit = 1.0 + rng.normal(0.0, 0.13, segments)
        cur = len(verts)
        verts += [(rad * jit[j] * np.cos(a), rad * jit[j] * np.sin(a), z)
                  for j, a in enumerate(ang)]
        if prev is not None:
            for j in range(segments):
                j1 = (j + 1) % segments
                faces.append((prev + j, cur + j, cur + j1, prev + j1))
        prev = cur
    apex = len(verts)
    verts.append((0.0, 0.0, 1.0))
    for j in range(segments):
        faces.append((prev + j, apex, prev + (j + 1) % segments))
    return np.array(verts), faces


def meander_path(start_xy, end_xy, rng, steps=64, wobble=5.0):
    """
    A wandering polyline from start to end.

    Streams and trails both need a line that gets somewhere without being
    straight; lateral offsets are drawn from a smooth random walk so the result
    curves rather than jitters.
    """
    p0, p1 = np.asarray(start_xy, float), np.asarray(end_xy, float)
    t = np.linspace(0.0, 1.0, steps)[:, None]
    base = p0[None, :] + t * (p1 - p0)[None, :]
    d = p1 - p0
    n = np.array([-d[1], d[0]])
    n = n / max(np.linalg.norm(n), 1e-9)
    # smooth random walk, pinned to zero at both ends
    w = np.cumsum(rng.normal(0.0, 1.0, steps))
    w = np.convolve(w, np.ones(9) / 9.0, mode="same")
    w -= np.linspace(w[0], w[-1], steps)
    if np.abs(w).max() > 1e-9:
        w = w / np.abs(w).max()
    taper = np.sin(np.pi * t[:, 0]) ** 0.5
    return base + (w * taper * wobble)[:, None] * n[None, :]


def distance_to_path(X, Y, path):
    """Shortest distance from every grid point to a polyline. Vectorised."""
    P = np.asarray(path, float)
    best = np.full(X.shape, np.inf)
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0.0, 1.0)
        best = np.minimum(best, np.hypot(X - (a[0] + t * ab[0]),
                                         Y - (a[1] + t * ab[1])))
    return best


def ribbon_mesh(name, path, heights, width, smooth=False):
    """A flat strip following `path` at the given per-vertex heights."""
    P = np.asarray(path, float)
    if len(P) < 2:
        return None
    verts, faces = [], []
    for i in range(len(P)):
        j = min(i + 1, len(P) - 1)
        k = max(i - 1, 0)
        d = P[j] - P[k]
        nrm = np.array([-d[1], d[0]])
        nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
        w = width[i] if np.ndim(width) else width
        z = heights[i]
        verts.append((P[i, 0] - nrm[0] * w, P[i, 1] - nrm[1] * w, z))
        verts.append((P[i, 0] + nrm[0] * w, P[i, 1] + nrm[1] * w, z))
    for i in range(len(P) - 1):
        a = 2 * i
        faces.append((a, a + 1, a + 3, a + 2))
    return mesh_from_arrays(name, np.array(verts), faces, smooth=smooth)


def _unit_log(segments=5):
    """A fallen trunk of unit length lying along +X, radius ~0.05."""
    r = 0.05
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts, faces = [], []
    for x in (0.0, 1.0):
        for a in ang:
            verts.append((x, r * np.cos(a), r * np.sin(a) + r))
    for j in range(segments):
        j1 = (j + 1) % segments
        faces.append((j, j1, segments + j1, segments + j))
    return np.array(verts), faces


def _unit_reed(blades=3):
    """A tuft of thin blades of unit height, for the lake shore."""
    verts, faces = [], []
    for b in range(blades):
        a = 2 * np.pi * b / blades
        dx, dy = 0.06 * np.cos(a), 0.06 * np.sin(a)
        base = len(verts)
        verts += [(dx - 0.02, dy, 0.0), (dx + 0.02, dy, 0.0),
                  (dx + 0.10 * np.cos(a), dy + 0.10 * np.sin(a), 1.0)]
        faces.append((base, base + 1, base + 2))
    return np.array(verts), faces


def _instance(base_verts, base_faces, points, scales, rot_z, tilt=None):
    """Bake transformed copies of one prop into a single combined mesh."""
    all_verts = []
    all_faces = []
    n_base = len(base_verts)
    for i, (p, s) in enumerate(zip(points, scales)):
        c, sn = np.cos(rot_z[i]), np.sin(rot_z[i])
        rot = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        # `s` may be a scalar (uniform) or (3,) for independent girth/height --
        # a forest scaled uniformly has every tree the same shape, which is most
        # of why a scatter reads as clip art.
        v = base_verts * np.asarray(s).reshape(-1)[None, :] if np.size(s) == 3 \
            else base_verts * s
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
                     min_slope=0.60, height_range=(0.34, 0.72), radius_limit=None,
                     patch_sharpness=1.4, patch_bias=0.95, patch_freq=0.045, avoid=None):
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
    # Raising `patch_sharpness` (and the bias) contracts the forest into denser
    # stands with emptier gaps between them. A soft patch mask spreads the same
    # tree budget evenly, and individually resolvable conifers dotted across open
    # meadow read as scattered pins rather than woodland.
    patch = np.clip(fbm(X, Y, freq=patch_freq, octaves=5, seed=1234) * 2.6
                    - patch_bias, 0.0, 1.0) ** patch_sharpness
    weights = belt * walkable * patch
    if avoid is not None:
        weights = weights * avoid

    if radius_limit is not None:
        weights = weights * (np.hypot(X, Y) < radius_limit)
    # Compensate for the polar grid packing far more vertices near the centre.
    weights = weights * (np.hypot(X, Y) + 1.0)

    pts = sample_grid(X, Y, Z, weights, count, rng, jitter=0.25)
    if len(pts) == 0:
        return None
    pts[:, 2] -= 0.03  # bed the trunks into the ground
    n = len(pts)

    # Age structure. A uniform draw gives a plantation: every tree a different
    # size but all sizes equally common. Real stands are dominated by young
    # trees with a thin tail of old ones, so bias the draw hard toward small.
    lo, hi = height_range
    age = rng.random(n) ** AGE_SKEW
    scale_z = lo + (hi - lo) * age

    # Trees dwarf toward the tree line -- the same species is a spire in the
    # valley and stunted krummholz at its altitude limit.
    alt = np.clip((pts[:, 2] - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0)
    scale_z *= 1.0 - TREELINE_DWARFING * alt

    # Girth is only loosely tied to height: young conifers are narrow spires,
    # old ones broaden and flatten. Scaling all three axes together is what
    # makes a scatter read as one tree copy-pasted.
    girth = scale_z * rng.uniform(0.72, 1.30, size=n) * (0.80 + 0.45 * age)
    scales = np.stack([girth, girth, scale_z], axis=1)

    rots = rng.uniform(0, 2 * np.pi, size=n)
    tilt = rng.normal(0.0, 0.055, size=n)

    # Three crown shapes rather than one mesh for the whole forest.
    verts_all, faces_all, offset = [], [], 0
    which = rng.integers(0, 3, size=n)
    for k, tiers in enumerate((3, 4, 5)):
        m = which == k
        if not m.any():
            continue
        bv, bf = _unit_conifer(tiers=tiers, rng=rng)
        v, f = _instance(bv, bf, pts[m], scales[m], rots[m], tilt[m])
        verts_all.append(v)
        faces_all.extend([tuple(i + offset for i in face) for face in f])
        offset += len(v)
    if not verts_all:
        return None
    return mesh_from_arrays(name, np.concatenate(verts_all, axis=0), faces_all,
                            smooth=True)


def scatter_shrubs(name, X, Y, Z, count, rng, z_min=0.6, z_max=12.0,
                   min_slope=0.45, size_range=(0.18, 0.55), radius_limit=None, avoid=None):
    """
    Dwarf willow and heather: the layer between open meadow and closed forest,
    and the only green that survives above the tree line. Without it the forest
    edge is a hard line, which is the giveaway that vegetation was placed by a
    single rule.
    """
    if count <= 0:
        return None
    cos_slope = grid_slope(X, Y, Z)
    belt = np.clip((Z - z_min) / 0.7, 0.0, 1.0) * np.clip((z_max - Z) / 3.0, 0.0, 1.0)
    weights = belt * np.clip((cos_slope - min_slope) / 0.3, 0.0, 1.0)
    patch = np.clip(fbm(X, Y, freq=0.075, octaves=4, seed=771) * 2.2 - 0.75, 0.0, 1.0)
    weights = weights * patch
    if avoid is not None:
        weights = weights * avoid
    if radius_limit is not None:
        weights = weights * (np.hypot(X, Y) < radius_limit)
    weights = weights * (np.hypot(X, Y) + 1.0)
    pts = sample_grid(X, Y, Z, weights, count, rng, jitter=0.4)
    if len(pts) == 0:
        return None
    pts[:, 2] -= 0.02
    n = len(pts)
    r = rng.uniform(*size_range, size=n)
    scales = np.stack([r, r * rng.uniform(0.82, 1.18, size=n),
                       r * rng.uniform(0.55, 0.95, size=n)], axis=1)
    bv, bf = _unit_shrub(rng)
    verts, faces = _instance(bv, bf, pts, scales, rng.uniform(0, 2 * np.pi, size=n))
    return mesh_from_arrays(name, verts, faces, smooth=True)


def scatter_logs(name, X, Y, Z, count, rng, z_min=0.8, z_max=9.0,
                 min_slope=0.72, length_range=(0.9, 2.4), radius_limit=None, avoid=None):
    """Deadfall on the forest floor. Cheap, and forest floors are never bare."""
    if count <= 0:
        return None
    cos_slope = grid_slope(X, Y, Z)
    belt = np.clip((Z - z_min) / 0.6, 0.0, 1.0) * np.clip((z_max - Z) / 1.5, 0.0, 1.0)
    weights = belt * np.clip((cos_slope - min_slope) / 0.2, 0.0, 1.0)
    if avoid is not None:
        weights = weights * avoid
    if radius_limit is not None:
        weights = weights * (np.hypot(X, Y) < radius_limit)
    weights = weights * (np.hypot(X, Y) + 1.0)
    pts = sample_grid(X, Y, Z, weights, count, rng, jitter=0.5)
    if len(pts) == 0:
        return None
    n = len(pts)
    L = rng.uniform(*length_range, size=n)
    scales = np.stack([L, rng.uniform(0.8, 1.5, size=n),
                       rng.uniform(0.8, 1.5, size=n)], axis=1)
    bv, bf = _unit_log()
    verts, faces = _instance(bv, bf, pts, scales, rng.uniform(0, 2 * np.pi, size=n))
    return mesh_from_arrays(name, verts, faces, smooth=False)


def scatter_reeds(name, X, Y, Z, count, rng, water_z=0.0, band=0.55,
                  height_range=(0.30, 0.85), radius_limit=None):
    """
    Reed beds in the shallows. A lake that meets the shore as a clean line reads
    as a mirror dropped on the ground; reeds are what break that edge.
    """
    if count <= 0:
        return None
    # A narrow band straddling the waterline.
    weights = np.exp(-((Z - water_z) / band) ** 2) * (Z > water_z - band)
    if radius_limit is not None:
        weights = weights * (np.hypot(X, Y) < radius_limit)
    weights = weights * (np.hypot(X, Y) + 1.0)
    pts = sample_grid(X, Y, Z, weights, count, rng, jitter=0.3)
    if len(pts) == 0:
        return None
    n = len(pts)
    h = rng.uniform(*height_range, size=n)
    scales = np.stack([rng.uniform(0.6, 1.4, size=n),
                       rng.uniform(0.6, 1.4, size=n), h], axis=1)
    bv, bf = _unit_reed()
    verts, faces = _instance(bv, bf, pts, scales, rng.uniform(0, 2 * np.pi, size=n),
                             tilt=rng.normal(0.0, 0.16, size=n))
    return mesh_from_arrays(name, verts, faces, smooth=False)


def scatter_snags(name, X, Y, Z, count, rng, z_min=0.8, z_max=9.0,
                  min_slope=0.62, height_range=(1.0, 2.6), radius_limit=None,
                  avoid=None):
    """Dead standing trees. Every real wood has them; a forest without any
    reads as freshly planted."""
    if count <= 0:
        return None
    cos_slope = grid_slope(X, Y, Z)
    belt = np.clip((Z - z_min) / 0.7, 0.0, 1.0) * np.clip((z_max - Z) / 1.5, 0.0, 1.0)
    weights = belt * np.clip((cos_slope - min_slope) / 0.25, 0.0, 1.0)
    if avoid is not None:
        weights = weights * avoid
    if radius_limit is not None:
        weights = weights * (np.hypot(X, Y) < radius_limit)
    weights = weights * (np.hypot(X, Y) + 1.0)
    pts = sample_grid(X, Y, Z, weights, count, rng, jitter=0.4)
    if len(pts) == 0:
        return None
    pts[:, 2] -= 0.03
    n = len(pts)
    h = rng.uniform(*height_range, size=n) * rng.uniform(0.55, 1.0, size=n)
    scales = np.stack([rng.uniform(0.8, 1.3, size=n),
                       rng.uniform(0.8, 1.3, size=n), h], axis=1)
    bv, bf = _unit_snag(rng=rng)
    verts, faces = _instance(bv, bf, pts, scales, rng.uniform(0, 2 * np.pi, size=n),
                             tilt=rng.normal(0.0, 0.10, size=n))
    return mesh_from_arrays(name, verts, faces, smooth=False)


def scatter_broadleaf(name, X, Y, Z, count, rng, z_min=0.5, z_max=6.0,
                      min_slope=0.66, height_range=(0.9, 2.3), radius_limit=None,
                      avoid=None):
    """
    A second species on the warmer lower slopes, below the conifer belt.

    Real valleys stratify by altitude: broadleaf and larch low, spruce above,
    krummholz at the limit. One species everywhere is the strongest cue that a
    forest was generated.
    """
    if count <= 0:
        return None
    cos_slope = grid_slope(X, Y, Z)
    belt = np.clip((Z - z_min) / 0.6, 0.0, 1.0) * np.clip((z_max - Z) / 2.2, 0.0, 1.0)
    patch = np.clip(fbm(X, Y, freq=0.055, octaves=4, seed=4242) * 2.4 - 0.95, 0.0, 1.0)
    weights = belt * np.clip((cos_slope - min_slope) / 0.25, 0.0, 1.0) * patch
    if avoid is not None:
        weights = weights * avoid
    if radius_limit is not None:
        weights = weights * (np.hypot(X, Y) < radius_limit)
    weights = weights * (np.hypot(X, Y) + 1.0)
    pts = sample_grid(X, Y, Z, weights, count, rng, jitter=0.35)
    if len(pts) == 0:
        return None
    pts[:, 2] -= 0.03
    n = len(pts)
    age = rng.random(n) ** AGE_SKEW
    lo, hi = height_range
    hgt = lo + (hi - lo) * age
    girth = hgt * rng.uniform(0.62, 0.95, size=n)
    scales = np.stack([girth, girth, hgt], axis=1)
    bv, bf = _unit_broadleaf(rng)
    verts, faces = _instance(bv, bf, pts, scales, rng.uniform(0, 2 * np.pi, size=n),
                             tilt=rng.normal(0.0, 0.05, size=n))
    return mesh_from_arrays(name, verts, faces, smooth=True)


def scatter_talus(name, X, Y, Z, count, rng, size_range=(0.10, 0.55),
                  radius_limit=None, cliff_slope=0.55):
    """
    Talus fans: blocks pooling in cones BELOW cliff faces, not sprinkled evenly.

    Weighting purely by local slope puts rocks *on* the cliff. Real scree
    collects where the ground has just flattened out beneath one, so the weight
    is (gentle here) x (steep just uphill), with low-frequency noise to gather
    the blocks into distinct fans rather than an even apron.
    """
    if count <= 0:
        return None
    cos_slope = grid_slope(X, Y, Z)
    gentle = np.clip((cos_slope - 0.55) / 0.3, 0.0, 1.0)
    steep = np.clip((cliff_slope - cos_slope) / 0.3, 0.0, 1.0)
    # "just uphill" along the radial direction of the polar grid
    uphill = np.zeros_like(steep)
    uphill[:-3, :] = steep[3:, :]
    uphill[-3:, :] = steep[-1, :]
    fans = np.clip(fbm(X, Y, freq=0.11, octaves=3, seed=9091) * 2.3 - 0.85, 0.0, 1.0)
    weights = gentle * uphill * (0.35 + fans)
    if radius_limit is not None:
        weights = weights * (np.hypot(X, Y) < radius_limit)
    weights = weights * (np.hypot(X, Y) + 1.0)
    pts = sample_grid(X, Y, Z, weights, count, rng, jitter=0.35)
    if len(pts) == 0:
        return None
    n = len(pts)
    r = rng.uniform(*size_range, size=n) * (rng.random(n) ** 1.6 + 0.35)
    scales = np.stack([r, r * rng.uniform(0.7, 1.3, size=n),
                       r * rng.uniform(0.5, 0.9, size=n)], axis=1)
    bv, bf = _unit_boulder(rng)
    verts, faces = _instance(bv, bf, pts, scales, rng.uniform(0, 2 * np.pi, size=n))
    return mesh_from_arrays(name, verts, faces, smooth=True)


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
