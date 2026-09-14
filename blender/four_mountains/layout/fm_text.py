"""
Text serialisation of a Four Mountains view. Pure numpy -- no `bpy`.

The benchmark shows a human (or a VLM) a rendered image and gives a text-only
LLM a description of the *same* view. Because the scene is generated rather
than photographed, the two channels can be made information-equivalent by
construction instead of by an annotator's judgement.

Three levels, deliberately kept as a ladder rather than collapsing to one:

``L1`` allocentric world coordinates -- an overhead map in words. Comparing two
       layouts reduces to comparing coordinate lists, so this is the ceiling
       condition, not a fair stand-in for vision.
``L2`` egocentric bearing, distance and apparent width from the camera -- what
       the image actually affords. Recovering the layout from it requires the
       same mental rotation the picture does.
``L3`` qualitative prose, the way a person would describe the view aloud.

The level at which text stops beating vision is the measurement; that is why
all three ship.

Identity leakage
----------------
Peaks are never labelled ``M1..M4``. Those indices are stable across a target
and its foils, so quoting them would hand the text channel a correspondence the
image channel has to work out from shape. Peaks are instead ordered
left-to-right as seen from the camera and named by landform -- and under
``distinctiveness="identical"`` even the landform is uninformative, leaving
nothing but the spatial arrangement, exactly as in the rendered image.
"""

import math

import numpy as np

import fm_geom as geom

def form_phrase(form):
    """
    Describe a parametric peak in words, from its form vector.

    The text channels must carry the same identity information the image does,
    or a model reading the description is answering an easier question than one
    looking at the render. Each clause below names one form parameter.
    """
    a, f = float(form["peak_angle"]), float(form["flank"])
    summit = ("a sharp" if a < 0.95 else
              ("a broad flat-topped" if a > 1.6 else "a conical"))
    crest = float(form.get("crest", 0.0))
    body = ("peak" if crest < 0.15 else
            ("long ridge" if crest > 0.38 else "crested peak"))
    base = ("dropping in cliffs to its base" if f < 1.35 else
            ("spreading into a broad talus apron" if f > 2.0 else
             "with even flanks"))
    rough = float(form["roughness"])
    tex = ("smooth-sided" if rough < 0.25 else
           ("heavily broken and gullied" if rough > 0.42 else "gullied"))
    lean = "" if float(form["offset"]) < 0.09 else ", its summit set off-centre"
    return f"{summit} {tex} {body} {base}{lean}"

COMPASS = ["north", "north-northeast", "northeast", "east-northeast",
           "east", "east-southeast", "southeast", "south-southeast",
           "south", "south-southwest", "southwest", "west-southwest",
           "west", "west-northwest", "northwest", "north-northwest"]


def compass_name(bearing_deg):
    return COMPASS[int(round((bearing_deg % 360.0) / 22.5)) % 16]


def view_geometry(layout, camera):
    """
    Per-peak view quantities, ordered left-to-right in the image.

    Returns a list of dicts with the peak's landform, its egocentric bearing,
    distance and apparent width, and its allocentric position and compass
    bearing from the valley centre.
    """
    cam = geom.camera_position(camera["azimuth_deg"], camera["elevation_deg"],
                               camera["radius"], camera["target_z"])
    target = np.array([0.0, 0.0, camera["target_z"]])
    peaks = layout["peaks"]
    pts = np.array([[p["x"], p["y"], p["z"]] for p in peaks])
    radii = np.array([p["base_radius"] for p in peaks])

    bearings = geom.bearing_from_camera(pts, cam, target)
    dists = geom.distance_from_camera(pts, cam)
    widths = 2.0 * geom.angular_radius(pts, radii, cam)
    compass = geom.compass_bearing(pts)

    rows = [{
        "type": p["type"],
        "form": p.get("form"),
        "x": p["x"], "y": p["y"],
        "height": p["height"], "base_radius": p["base_radius"],
        "view_bearing_deg": float(bearings[i]),
        "distance_m": float(dists[i]),
        "apparent_width_deg": float(widths[i]),
        "compass_from_centre": float(compass[i]),
        "range_from_centre": float(math.hypot(p["x"], p["y"])),
    } for i, p in enumerate(peaks)]
    rows.sort(key=lambda r: r["view_bearing_deg"])
    return rows


# --------------------------------------------------------------------------- #
# L1 -- allocentric coordinates
# --------------------------------------------------------------------------- #

def _is_bare(layout):
    """
    True for the stimulus-ladder scenes: primitives on a featureless plane.

    The serialisers describe an alpine valley with a lake, which is a false
    statement about those scenes and would be handed to a model as part of the
    prompt. The layout already says which world it is: ladder landmarks carry an
    `obj` record and no form vector.
    """
    # `assign_objects` records the world explicitly, because it cannot be
    # inferred from the landmarks: c3 (bare plane) and c4 (alpine valley) use
    # identical peaks with no `obj` record, so the old heuristic called c3 a
    # valley and told the model a lake sat at the origin of an empty plane.
    world = layout.get("world")
    if world is not None:
        return world != "valley"
    return any(p.get("obj") for p in layout["peaks"])


def to_l1(layout, camera):
    rows = sorted(view_geometry(layout, camera), key=lambda r: r["compass_from_centre"])
    lines = (["Overhead map of a flat plain. The centre of the plain is the",
              "origin (0, 0)."] if _is_bare(layout) else
             ["Overhead map of the valley. The lake sits at the origin (0, 0)."])
    lines.append("+y is north and +x is east; all distances are in metres.")
    for r in rows:
        lines.append(
            f"- {r['type']}: position ({r['x']:.1f}, {r['y']:.1f}), "
            f"summit height {r['height']:.0f} m, base radius {r['base_radius']:.1f} m")
    lines.append(f"The viewer stands on the {compass_name(camera['azimuth_deg'])} side "
                 f"of the valley, {camera['radius']:.0f} m from the centre, looking inward.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# L2 -- egocentric bearings
# --------------------------------------------------------------------------- #

def to_l2(layout, camera):
    rows = view_geometry(layout, camera)
    centre = ("the centre of a flat plain" if _is_bare(layout)
              else "the lake at the centre of the valley")
    lines = [f"View from a single fixed viewpoint, looking towards {centre}.",
             "Objects are listed left to right across the view.",
             "Bearing is the angle from the centre of the view: negative is to the",
             "left, positive to the right. Apparent width is how wide the peak's",
             "base looks from here."]
    for r in rows:
        # Precision matters: at whole-metre distances and 0.1 deg bearings, two
        # different 4 m displacements can round to identical text, which makes
        # the item unanswerable in this channel even though the difference is
        # plainly visible in the image. L2 is meant to carry what the picture
        # affords, so it is quoted finely enough to preserve the smallest
        # manipulation on the ladder.
        lines.append(
            f"- {r['type']}: bearing {r['view_bearing_deg']:+.2f} deg, "
            f"distance {r['distance_m']:.1f} m, "
            f"apparent width {r['apparent_width_deg']:.2f} deg, "
            f"summit height {r['height']:.0f} m")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# L3 -- qualitative prose
# --------------------------------------------------------------------------- #

def _side_phrase(bearing):
    a = abs(bearing)
    side = "left" if bearing < 0 else "right"
    if a < 3.0:
        return "straight ahead"
    if a < 9.0:
        return f"a little to the {side}"
    if a < 18.0:
        return f"to the {side}"
    return f"far to the {side}"


def _rank_phrase(value, values, near_word, far_word):
    order = np.argsort(values)
    rank = int(np.where(order == list(values).index(value))[0][0])
    if rank == 0:
        return near_word
    if rank == len(values) - 1:
        return far_word
    return "midway"


def to_l3(layout, camera):
    rows = view_geometry(layout, camera)
    dists = [r["distance_m"] for r in rows]
    heights = [r["height"] for r in rows]
    nearest, farthest = min(dists), max(dists)

    # A comparative is only informative if the quantity actually varies. With
    # equal heights every landmark came out as "the tallest summit in view",
    # four times in a row -- a descriptor that is true, useless, and actively
    # misleading about the scene.
    heights_vary = max(heights) - min(heights) > 1e-6

    lines = (["Standing at one edge of a flat open plain, looking across its",
              "centre. Describing the objects from left to right:"]
             if _is_bare(layout) else
             ["Standing at one edge of an alpine valley, looking across the lake at",
              "its centre. Describing the peaks from left to right:"])
    for r in rows:
        if r.get("form"):
            phrase = form_phrase(r["form"])
        else:
            # No form vector: a ladder primitive, whose `type` is already a
            # readable name ("blue dome"). The twelve named landforms this
            # branch used to translate no longer exist.
            phrase = f"a {r['type']}"
        rel = r["distance_m"] / nearest
        if r["distance_m"] == nearest:
            dist_phrase = "the closest of them"
        elif r["distance_m"] == farthest:
            dist_phrase = f"the most distant, roughly {rel:.1f} times as far as the closest"
        else:
            dist_phrase = f"roughly {rel:.1f} times as far as the closest"
        if not heights_vary:
            h_phrase = None
        elif r["height"] == max(heights):
            h_phrase = "the tallest summit in view"
        elif r["height"] == min(heights):
            h_phrase = "the lowest summit in view"
        else:
            h_phrase = "of middling height"
        parts = [phrase, _side_phrase(r["view_bearing_deg"]), dist_phrase]
        if h_phrase:
            parts.append(h_phrase)
        lines.append("- " + ", ".join(parts) + ".")
    return "\n".join(lines)


LEVELS = {"L1": to_l1, "L2": to_l2, "L3": to_l3}


def scene_to_text(layout, camera, level="L2"):
    """Serialise one view. `level` is one of L1, L2, L3."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {sorted(LEVELS)}, got {level!r}")
    return LEVELS[level](layout, camera)


def parse_l1(text):
    """
    Recover {landform: (x, y, height)} from an L1 description.

    Used by the round-trip test that backs the information-equivalence claim:
    if L1 cannot be parsed back to the coordinates it was built from, the two
    channels are not carrying the same content.
    """
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        name, _, rest = line[2:].partition(":")
        pos = rest.split("position (")[1].split(")")[0]
        x, y = (float(v) for v in pos.split(","))
        h = float(rest.split("summit height")[1].split("m")[0])
        out[name.strip()] = (x, y, h)
    return out
