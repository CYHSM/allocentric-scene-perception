"""
Layout sampler for the Four Mountains benchmark. Pure numpy -- no `bpy`.

`generate_scene.py` ships a single hand-tuned valley (DEFAULT_TYPES /
DEFAULT_AZIMUTHS / DEFAULT_HEIGHTS / DEFAULT_RADII). A benchmark needs a
*distribution* of valleys instead, and every one of them has to satisfy a
constraint the hand-tuned scene does not quite meet: every peak fully inside
the frame from every benchmark viewpoint. (The shipped canonical layout clips
M2 against the right edge at azimuth 90, 105 and 345.)

Sampling is rejection-based against two constraints:

1. **Framing** -- each peak's silhouette clears the frame, with a margin, at
   every azimuth the bank will render.
2. **Footprint** -- peak bases do not intersect, so no peak is embedded in
   another.
`distinctiveness` controls the shortcut available to a solver. With
``"distinct"`` every peak is a different landform, so a model can name the
shapes and match them. With ``"identical"`` all peaks share one morphology,
one height and one base radius, so nothing but the spatial arrangement
distinguishes the target from its foils.
"""

import math

import numpy as np

import fm_geom as geom
import fm_peak as peaklib

# Mirrors generate_scene.py, which is the authority on world geometry.
VALLEY_FLOOR = 1.7
MOUNTAIN_SINK = 0.45
BASE_Z = VALLEY_FLOOR - MOUNTAIN_SINK

# Camera rig used by the benchmark bank (the game bank's elevated orbit).
CAM_ELEVATION = 17.0
# Scaled with the valley: the ring grew from 29-35 m to 36-45 m and peaks
# from 18.5 m to 20 m, which brought the near side of the orbit close enough
# that a full-size peak ran off the frame. 122 m restores headroom at 28 mm.
CAM_RADIUS = 122.0
CAM_TARGET_Z = 6.0
CAM_LENS_MM = 28.0
RES_X, RES_Y = 640, 440

# Peaks are wider than they were (base radius 13-20 m, not 11-18.5) because
# the height:radius ratio had to come down to ~1 for them to read as
# mountains. A wider peak needs a wider ring: at 29-35 m a 30 m metric foil
# was geometrically impossible in half of all layouts.
RING_MIN, RING_MAX = 36.0, 45.0
HEIGHT_MIN, HEIGHT_MAX = 11.0, 27.0
# Summit height as a multiple of base radius. Sampling height and radius
# independently produces spires of aspect ratio 3+, on which every landform
# renders as the same rock needle and the morphology library stops meaning
# anything. The shipped hand-tuned valley sits at 1.4-2.0.
# Height / base radius. Above ~1.5 a peak is a spire, not a mountain: this is
# the ratio that decided whether the renders read as rock or as tipis.
ASPECT_MIN, ASPECT_MAX = 0.85, 1.35
RADIUS_MIN, RADIUS_HARD_MAX = 13.0, 20.0

FRAME_MARGIN = 0.03        # fraction of the half-FOV kept clear at the edges
FOOTPRINT_SLACK = 1.12     # centre distance >= this * (r_i + r_j)
ANGLE_JITTER = 0.28        # as a fraction of the even angular spacing


_FRAME_RADIUS_CACHE = {}


def max_radius_for_ring(ring, margin=FRAME_MARGIN, azimuth_step=15):
    """
    Largest base radius a peak at this ring distance can have and still clear
    the frame from every rendered azimuth.

    Framing depends only on a peak's own ring and radius, never on its
    neighbours, so this can be *solved* once per ring rather than rejected
    against thousands of times -- which is the difference between a sampler
    that returns and one that does not.
    """
    key = (round(float(ring), 3), round(float(margin), 4), int(azimuth_step))
    if key in _FRAME_RADIUS_CACHE:
        return _FRAME_RADIUS_CACHE[key]

    cams = _cameras(benchmark_azimuths(azimuth_step))

    def fits(rad):
        peak = [{"x": float(ring), "y": 0.0, "z": BASE_Z,
                 "height": HEIGHT_MAX, "base_radius": float(rad)}]
        xy, heights, radii = _peak_arrays(peak)
        _, _, h_ang, v_ang, depth = _project_all(xy, heights, radii, cams)
        return check_framing(h_ang, v_ang, depth, margin)

    if not fits(0.5):
        _FRAME_RADIUS_CACHE[key] = 0.0
        return 0.0
    lo, hi = 0.5, RADIUS_HARD_MAX
    if fits(hi):
        _FRAME_RADIUS_CACHE[key] = hi
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    _FRAME_RADIUS_CACHE[key] = lo
    return lo


def radius_bounds(n_peaks):
    """
    Base-radius range that lets `n_peaks` fit around the ring without their
    footprints intersecting.

    Peaks sit on a ring and must clear FOOTPRINT_SLACK * (r_i + r_j). Evenly
    spaced on the smallest ring that is 2 * RING_MIN * sin(pi / n) between
    neighbours, so the largest admissible radius shrinks as peaks are added --
    without this, four peaks of radius 18 simply cannot be placed and the
    sampler rejects every candidate.

    A single peak has no neighbour to clear, and the chord formula degenerates:
    sin(pi / 1) is 0, so the bound came out as radius 0.0 and the one object in
    the scene was invisible. Only the framing bound applies at n = 1, and
    `sample_layout` applies that separately per ring.
    """
    if n_peaks < 2:
        return RADIUS_MIN, RADIUS_HARD_MAX
    chord = 2.0 * RING_MIN * math.sin(math.pi / n_peaks)
    rmax = min(RADIUS_HARD_MAX, chord / (2.0 * FOOTPRINT_SLACK))
    rmin = min(RADIUS_MIN, rmax * 0.75)
    return rmin, rmax


def benchmark_azimuths(step_deg=15):
    return [float(a) for a in range(0, 360, step_deg)]


def _cameras(azimuths):
    return np.array([geom.camera_position(a, CAM_ELEVATION, CAM_RADIUS, CAM_TARGET_Z)
                     for a in azimuths])


def _target():
    return np.array([0.0, 0.0, CAM_TARGET_Z])


def _peak_arrays(peaks):
    xy = np.array([[p["x"], p["y"]] for p in peaks], dtype=float)
    return (xy,
            np.array([p["height"] for p in peaks], dtype=float),
            np.array([p["base_radius"] for p in peaks], dtype=float))


def check_footprints(xy, radii):
    """No two peak bases intersect."""
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    need = FOOTPRINT_SLACK * (radii[:, None] + radii[None, :])
    iu = np.triu_indices(len(radii), k=1)
    return bool(np.all(d[iu] >= need[iu]))


def _project_all(xy, heights, radii, cams):
    """
    Vectorised projection of every peak's silhouette into every camera.

    Returns (bearing, arad, h_ang, v_ang) with leading axis over cameras.
    Doing this with numpy rather than a Python double loop is what makes
    rejection sampling viable -- the loop version costs seconds per layout.
    """
    tgt = _target()
    n_cam, n_pk = len(cams), len(radii)
    p3 = np.concatenate([xy, np.full((n_pk, 1), BASE_Z)], axis=1)          # (P,3)

    fwd = tgt[None, :] - cams                                              # (C,3)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    right = np.cross(fwd, geom.WORLD_UP[None, :])
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    up = np.cross(right, fwd)

    d = p3[None, :, :] - cams[:, None, :]                                  # (C,P,3)
    depth = np.einsum("cpk,ck->cp", d, fwd)
    xr = np.einsum("cpk,ck->cp", d, right)
    yu = np.einsum("cpk,ck->cp", d, up)
    dist = np.linalg.norm(d, axis=-1)

    safe = np.maximum(depth, 1e-6)
    bearing = np.degrees(np.arctan2(xr, safe))
    arad = np.degrees(np.arctan2(radii[None, :], np.maximum(dist, 1e-6)))

    # Silhouette: lateral half-width at the base, plus the summit's rise.
    los = p3[None, :, :2] - cams[:, None, :2]
    n = np.linalg.norm(los, axis=-1, keepdims=True)
    perp = np.stack([-los[..., 1], los[..., 0]], axis=-1) / np.maximum(n, 1e-9)
    off = perp * radii[None, :, None]

    h_ang = np.empty((n_cam, n_pk))
    v_ang = np.empty((n_cam, n_pk))
    for sign in (1.0, -1.0):
        e = np.concatenate([p3[None, :, :2] + sign * off,
                            np.full((n_cam, n_pk, 1), BASE_Z)], axis=-1)
        de = e - cams[:, None, :]
        dep = np.maximum(np.einsum("cpk,ck->cp", de, fwd), 1e-6)
        a = np.abs(np.arctan2(np.einsum("cpk,ck->cp", de, right), dep))
        h_ang = a if sign == 1.0 else np.maximum(h_ang, a)

    summit = np.concatenate([xy, (BASE_Z + heights)[:, None]], axis=1)
    ds = summit[None, :, :] - cams[:, None, :]
    dsdep = np.maximum(np.einsum("cpk,ck->cp", ds, fwd), 1e-6)
    v_ang = np.maximum(np.abs(np.arctan2(np.einsum("cpk,ck->cp", ds, up), dsdep)),
                       np.abs(np.arctan2(yu, safe)))

    return bearing, arad, h_ang, v_ang, depth


def check_framing(h_ang, v_ang, depth, margin=FRAME_MARGIN):
    hfov, vfov = geom.half_fov(CAM_LENS_MM, RES_X, RES_Y)
    return bool(np.all(depth > 0)
                and np.all(h_ang <= hfov * (1.0 - margin))
                and np.all(v_ang <= vfov * (1.0 - margin)))


def layout_is_valid(peaks, cams=None, margin=FRAME_MARGIN):
    """
    Footprint and framing only.

    An earlier version also rejected layouts on predicted occlusion. Rendering
    the instance masks for the shipped canonical valley at all 24 azimuths
    settled it: every peak is visible from every viewpoint (the smallest is
    581 px, M2 at azimuth 285), so no analytic occlusion rule is needed and the
    one tried here disagreed badly with the renders. Visibility is instead
    *measured* from the instance mask at render time and recorded per frame, so
    the item factory can require a minimum visible area per peak.
    """
    cams = cams if cams is not None else _cameras(benchmark_azimuths())
    xy, heights, radii = _peak_arrays(peaks)
    if not check_footprints(xy, radii):
        return False
    _, _, h_ang, v_ang, depth = _project_all(xy, heights, radii, cams)
    return check_framing(h_ang, v_ang, depth, margin)


def sample_layout(seed, n_peaks=4, distinctiveness="distinct",
                  azimuth_step=15, max_tries=4000):
    """
    Draw one valid valley layout.

    Returns a dict with `seed`, `n_peaks`, `distinctiveness`, `morphologies`
    and `peaks` -- the last being a list of
    ``{name, type, x, y, z, rot_z, height, base_radius}``.

    Raises RuntimeError if no valid layout is found, which means the
    constraints and the ring/size ranges have drifted out of agreement.
    """
    if distinctiveness not in ("distinct", "identical"):
        raise ValueError(f"distinctiveness must be 'distinct' or 'identical', got {distinctiveness!r}")

    rng = np.random.default_rng(seed)
    cams = _cameras(benchmark_azimuths(azimuth_step))

    # A peak's identity is a form vector (see fm_peak), not a morphology name.
    # "distinct" enforces a minimum separation in that space so the peaks are
    # individually recognisable -- otherwise the `binding` family, which asks
    # which landmark stands where, has no answer.
    if distinctiveness == "distinct":
        forms = peaklib.sample_distinct_forms(rng, n_peaks)
    else:
        forms = [peaklib.sample_form(rng)] * n_peaks
    kinds = [peaklib.describe(f) for f in forms]

    for _ in range(max_tries):
        # Seed the angular positions from a jittered even spread; a purely
        # uniform draw almost never clears the occlusion test.
        even = 360.0 / n_peaks
        jitter = rng.uniform(-ANGLE_JITTER * even, ANGLE_JITTER * even, n_peaks)
        spread = np.sort((rng.uniform(0, 360)
                          + np.linspace(0, 360, n_peaks, endpoint=False)
                          + jitter) % 360)
        rings = rng.uniform(RING_MIN, RING_MAX, n_peaks)

        # Cap each radius by both the packing bound (how many peaks fit around
        # the ring) and the framing bound (how big a peak at this distance can
        # be before it runs off the edge of the frame).
        pack_min, pack_max = radius_bounds(n_peaks)
        caps = np.array([min(pack_max, max_radius_for_ring(r, azimuth_step=azimuth_step))
                         for r in rings])
        if np.any(caps <= pack_min):
            continue

        if distinctiveness == "identical":
            radii = np.full(n_peaks, float(rng.uniform(pack_min, caps.min())))
            aspects = np.full(n_peaks, float(rng.uniform(ASPECT_MIN, ASPECT_MAX)))
        else:
            radii = pack_min + rng.random(n_peaks) * (caps - pack_min)
            aspects = rng.uniform(ASPECT_MIN, ASPECT_MAX, n_peaks)
        heights = np.clip(radii * aspects, HEIGHT_MIN, HEIGHT_MAX)

        peaks = []
        for i in range(n_peaks):
            th = math.radians(float(spread[i]))
            peaks.append({
                "name": f"M{i + 1}",
                "form": dict(forms[i]),
                "type": kinds[i],       # human-readable label for text/figures
                "x": float(rings[i] * math.cos(th)),
                "y": float(rings[i] * math.sin(th)),
                "z": BASE_Z,
                "rot_z": float(rng.uniform(-math.pi, math.pi)),
                "height": float(heights[i]),
                "base_radius": float(radii[i]),
            })

        if layout_is_valid(peaks, cams):
            return {
                "seed": int(seed),
                "n_peaks": int(n_peaks),
                "distinctiveness": distinctiveness,
                "morphologies": kinds,
                "forms": [dict(f) for f in forms],
                "peaks": peaks,
            }

    raise RuntimeError(
        f"no valid layout for seed={seed} n_peaks={n_peaks} "
        f"distinctiveness={distinctiveness} in {max_tries} tries")


def layout_to_positions(layout):
    """{name: {x, y, z, rot_z}} -- the shape FourMountainsRenderer expects."""
    return {p["name"]: {"x": p["x"], "y": p["y"], "z": p["z"], "rot_z": p["rot_z"]}
            for p in layout["peaks"]}


def layout_to_configs(layout):
    """
    Mountain configs in the shape `generate_scene.build_four_mountains_scene`
    consumes, so a sampled layout can be built directly into a .blend.
    """
    from generate_scene import PASS_MOUNTAIN_BASE
    return [{
        "name": p["name"],
        "type": p["type"],
        "form": p.get("form"),
        "height": p["height"],
        "radius": p["base_radius"],
        "pos": (p["x"], p["y"], p["z"]),
        "rot_z": p["rot_z"],
        "seed": 101 + 97 * i + layout["seed"],
        "pass_id": PASS_MOUNTAIN_BASE + i,
        "trees": 2200,
        "boulders": 280,
    } for i, p in enumerate(layout["peaks"])]
