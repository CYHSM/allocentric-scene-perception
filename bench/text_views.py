"""
The same 2AFC trial, written out as numbers instead of drawn as a picture.

Every trial in `data/vlm_benchmark_2afc.json` shows three renders: a study view
of a place and two candidate views, one of which is the same place seen from a
different direction. This module writes each of those three views as a short
list of egocentric coordinates -- where each landmark sits relative to the
viewer who took that view.

Why bother
----------
The image arm cannot separate two failures. A model that scores at chance at
delta = 90 might be unable to *see* the arrangement (a perception failure), or
it might see it perfectly and be unable to *rotate* it (a transformation
failure). Handing it exact coordinates removes the first possibility entirely.
Whatever is left at chance in this channel is not about pixels.

The task is analytically trivial here, and that is the point: the inter-landmark
distance matrix is invariant to the viewer's heading, so matching two views is
comparing two 4x4 distance matrices. `bench/tests/test_text_views.py` checks
that this solver scores 100%, which is the solvability gate the image arm has
only for humans.

Identity
--------
Two variants ship, and they bracket the cue ladder:

``identity=True``  each landmark is named ("purple cone"). The canonical set is
                   the same in every scene, so the name never identifies the
                   *place* -- it only says which landmark in one view is which
                   landmark in the other. Correspondence is given; only the
                   rotation is left.
``identity=False`` landmarks are unnamed and listed left to right. The reader
                   must recover the correspondence as well as the rotation, and
                   the left-to-right index is explicitly disclaimed in the text
                   because it is a within-view ordering that does not carry
                   across a turn.

That is the c0 / c3 contrast of the rendered ladder, made exact: in text the
appearance cue is either fully present or fully absent, with nothing in between
for a model to read off a texture.

Numbers only, never labels
--------------------------
`M1..M4` are stable across a target and its foils. Quoting them, or any
allocentric quantity (world x/y, compass bearing, the camera's azimuth), hands
over the answer: the rotation would no longer have to be solved. Only
viewer-relative quantities appear below.
"""

import math
import os
import sys

import numpy as np

_BLENDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "blender", "four_mountains")
if _BLENDER not in sys.path:
    sys.path.insert(0, _BLENDER)

import fm_geom as geom  # noqa: E402  (pure numpy, no bpy)

# The bank records these per mode and they are identical in all five, but the
# serialiser takes them as an argument rather than assuming: a future bank with
# a different orbit radius would otherwise be described wrongly and silently.
DEFAULT_CAMERA = {"elevation_deg": 17.0, "radius": 122.0, "target_z": 6.0}


def view_rows(layout, azimuth_deg, camera=None):
    """
    Egocentric position of every landmark, ordered left to right in the view.

    The frame is the viewer's, flattened to the ground: `forward` is the
    horizontal direction the viewer faces, `right` is perpendicular to it, and
    both are metres. `bearing` is the signed angle from straight ahead (right
    positive) and `distance` is the horizontal range, so the polar and Cartesian
    forms are the same two numbers.

    Flattening is not cosmetic. The rendering camera sits 17 deg above the
    valley and looks down, so its own axes tilt the ground plane: coordinates
    taken straight from `fm_geom.to_camera_frame` change the inter-landmark
    distances by up to 3% as the viewer walks around the scene, and the task
    stops being exactly solvable. In the flattened frame a turn is a rigid
    rotation, the distance matrix is invariant, and matching two views is
    exactly the geometry problem the picture poses.
    """
    camera = camera or DEFAULT_CAMERA
    theta = math.radians(azimuth_deg)
    # `fm_geom.camera_position` puts the viewer at radius * (cos, sin) in the
    # horizontal plane, looking at the origin.
    viewer = np.array([camera["radius"] * math.cos(theta),
                       camera["radius"] * math.sin(theta)])
    forward = -viewer / np.linalg.norm(viewer)
    # Right-hand rule about +z, matched to the render: `fm_geom.camera_basis`
    # takes right = normalise(cross(forward, world_up)), whose horizontal part
    # is (forward_y, -forward_x). `test_bearing_sign_matches_the_render` pins it.
    right = np.array([forward[1], -forward[0]])

    rows = []
    for p in layout["peaks"]:
        d = np.array([p["x"], p["y"]]) - viewer
        rm, fm = float(d @ right), float(d @ forward)
        rows.append({"type": p["type"],
                     "bearing_deg": math.degrees(math.atan2(rm, fm)),
                     "distance_m": math.hypot(rm, fm),
                     "right_m": rm,
                     "forward_m": fm})
    rows.sort(key=lambda r: r["bearing_deg"])
    return rows


_HEADER_ID = (
    "Landmarks visible from this viewpoint. Bearing is degrees from straight\n"
    "ahead (negative to the left, positive to the right); distance is in metres;\n"
    "position is (right, forward) in metres from the viewpoint."
)

_HEADER_NOID = (
    "Landmarks visible from this viewpoint, listed left to right across the\n"
    "view. The numbering is the left-to-right order in THIS view only and does\n"
    "not tell you which landmark is which in any other view. Bearing is degrees\n"
    "from straight ahead (negative to the left, positive to the right); distance\n"
    "is in metres; position is (right, forward) in metres from the viewpoint."
)


def render_view(layout, azimuth_deg, camera=None, identity=True, jitter=0.0,
                rng=None):
    """
    One view as text. `identity` names the landmarks; without it they are numbered.

    `jitter` adds i.i.d. Gaussian error, in metres, to each landmark's
    egocentric position before it is written out. It exists for one reason: at
    delta = 0 the study view and the correct candidate come from the same layout
    at the same heading, so with exact coordinates their two blocks of text are
    byte-identical and the item is solved by string comparison. In the image arm
    that cell is the appearance gate -- same place, same bearing, a different
    lighting sample -- and it is not degenerate. `jitter` is the text channel's
    stand-in for that resample. It defaults to zero, because the exact-coordinate
    condition is the one that isolates the transformation, and the degenerate
    delta = 0 cell is flagged in the benchmark file rather than quietly averaged
    into a score.
    """
    rows = view_rows(layout, azimuth_deg, camera)
    if jitter:
        rng = rng or np.random.default_rng(0)
        for r in rows:
            r["right_m"] += float(rng.normal(0.0, jitter))
            r["forward_m"] += float(rng.normal(0.0, jitter))
            r["distance_m"] = float(math.hypot(r["right_m"], r["forward_m"]))
            r["bearing_deg"] = float(math.degrees(math.atan2(r["right_m"],
                                                             r["forward_m"])))
        rows.sort(key=lambda r: r["bearing_deg"])
    lines = [_HEADER_ID if identity else _HEADER_NOID]
    for i, r in enumerate(rows, 1):
        head = r["type"] if identity else f"landmark {i}"
        lines.append(f"- {head}: bearing {r['bearing_deg']:+.2f} deg, "
                     f"distance {r['distance_m']:.1f} m, "
                     f"position ({r['right_m']:+.1f}, {r['forward_m']:.1f})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The analytic solver -- the solvability gate, not a baseline to publish
# --------------------------------------------------------------------------- #

def distance_matrix(rows):
    """Pairwise inter-landmark distances, sorted. Invariant to the viewer's heading."""
    p = np.array([[r["right_m"], r["forward_m"]] for r in rows])
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    return np.sort(d[np.triu_indices(len(p), k=1)])


def solve(study_rows, option_rows):
    """
    Index (1-based) of the option whose landmark geometry matches the study's.

    Uses only the sorted pairwise-distance vector, so it needs neither the
    landmark names nor the camera heading: it is available in the no-identity
    variant exactly as it is in the identity one.
    """
    ref = distance_matrix(study_rows)
    err = [float(np.abs(distance_matrix(o) - ref).sum()) for o in option_rows]
    return int(np.argmin(err)) + 1, err


def view_rows_from_text(text):
    """
    Recover the coordinate rows from a rendered view.

    The solvability gate runs through this rather than off the layouts, so what
    it certifies is the text actually shipped in the benchmark file -- rounding
    included. A gate that reads the full-precision layout would pass on items
    whose printed coordinates no longer separate the foil.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- ") or "position (" not in line:
            continue
        name, _, rest = line[2:].partition(":")
        pos = rest.split("position (")[1].split(")")[0]
        right, forward = (float(v) for v in pos.split(","))
        rows.append({"type": name.strip(),
                     "bearing_deg": float(rest.split("bearing")[1].split("deg")[0]),
                     "distance_m": float(rest.split("distance")[1].split("m")[0]),
                     "right_m": right, "forward_m": forward})
    return rows
