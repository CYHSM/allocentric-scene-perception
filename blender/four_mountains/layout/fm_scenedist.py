"""
How different are two scenes?

A scene is a set of landmarks, each an (identity, position) pair. Two scenes can
differ in three ways, and this module measures each of them separately:

    d_pos    metres      the furthest a landmark had to move
    d_bind   0..1        the same landmarks stand in different places
    d_id     0..1 / L2   they are not the same landmarks

`d_pos` is the axis the benchmark sweeps, because it is the one with units: it
lets the read-out be quoted in metres and compared against a viewpoint change in
degrees. `d_bind` and `d_id` are computed and recorded, but the cue ladder
(c0..c4) is what varies identity, so they are not swept.

**Why this replaces the old foil families.** `fm_foils.max_displacement` matched
landmarks *by slot name*, so a permutation -- which leaves the set of occupied
positions exactly as it found it -- scored 50-65 m, more than an explicit 20 m
translation. `fm_simple.layouts_collide` matched *by identity* and scored the
same permutation ~0 m. The two disagreed precisely on the case that separated
the families, so a binding-vs-metric contrast measured whichever convention the
caller happened to use. Here the matching is stated once, as an optimal
assignment between position sets, and the three quantities are orthogonal by
construction:

                   translation   permutation   substitution
    d_pos          = distance    0             0
    d_bind         0             > 0           > 0
    d_id           0             0             > 0

Optimal, not greedy: with n <= 4 landmarks the exact assignment is a search over
at most 24 permutations, so there is no reason to approximate it. A greedy
nearest-first match is not symmetric, and a distance that disagrees with itself
depending on argument order is not a distance.
"""

import copy
import itertools
import math

import fm_peak as peaklib

# Two scenes are the same place if every landmark is within this of its
# counterpart. Roughly half a landmark width -- close enough that no view
# distinguishes them.
SAME_PLACE_M = 8.0


# --------------------------------------------------------------------------- #
# The three distances
# --------------------------------------------------------------------------- #

def _positions(layout):
    return [(p["x"], p["y"]) for p in layout["peaks"]]


def _identities(layout):
    """A hashable identity per landmark, in landmark order.

    Discrete modes carry `obj`; peak modes carry a form vector, quantised so
    that two peaks built from the same form compare equal.
    """
    out = []
    for p in layout["peaks"]:
        obj = p.get("obj")
        if obj is not None:
            out.append(("obj", obj["shape"], obj["colour"]))
        elif p.get("form"):
            out.append(("form",) + tuple(round(float(p["form"][k]), 3)
                                         for k in peaklib.FORM_KEYS))
        else:
            out.append(("type", p.get("type")))
    return out


def _best_assignment(pa, pb):
    """
    (permutation, largest matched distance) under the *bottleneck* assignment.

    Minimising the largest displacement rather than the mean makes `d_pos` read
    the way the question is asked -- "how far did an object move?" -- so a probe
    built by displacing one landmark 20 m measures 20 m, not 20/n. It is still a
    metric: it is the L-infinity cost of an optimal matching, so symmetry and the
    triangle inequality both hold. Ties are broken by total distance so the
    permutation is well defined, which `d_bind` relies on.
    """
    n = len(pa)
    if n == 0:
        return (), 0.0
    best, best_cost = None, (float("inf"), float("inf"))
    for perm in itertools.permutations(range(n)):
        ds = [math.dist(pa[i], pb[perm[i]]) for i in range(n)]
        cost = (max(ds), sum(ds))
        if cost < best_cost:
            best, best_cost = perm, cost
    return best, best_cost[0]


def assignment(a, b):
    """The optimal landmark matching between two layouts, positions only.

    Returns `perm` with `perm[i] == j` meaning landmark `i` of `a` is matched
    to landmark `j` of `b`.
    """
    if len(a["peaks"]) != len(b["peaks"]):
        raise ValueError("scene distance is undefined between different "
                         f"landmark counts ({len(a['peaks'])} vs "
                         f"{len(b['peaks'])})")
    return _best_assignment(_positions(a), _positions(b))[0]


def d_pos(a, b):
    """Furthest a landmark has to move, in metres, under the best matching.

    Identity is ignored: this asks whether the *geometry* of the place changed.
    A permutation therefore scores 0 -- the same positions are occupied, just by
    different landmarks -- which is what makes it independent of `d_bind`.
    """
    if len(a["peaks"]) != len(b["peaks"]):
        raise ValueError("scene distance is undefined between different "
                         f"landmark counts ({len(a['peaks'])} vs "
                         f"{len(b['peaks'])})")
    return _best_assignment(_positions(a), _positions(b))[1]


def d_bind(a, b):
    """Fraction of places whose occupant changed identity, 0..1.

    Measured under the position matching, so it asks: standing where landmark
    `i` stood, do we now find the same thing? Both a permutation and a
    substitution move this; `d_id` is what separates them.
    """
    perm = assignment(a, b)
    ia, ib = _identities(a), _identities(b)
    n = len(ia)
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if ia[i] != ib[perm[i]]) / n


def d_id(a, b):
    """How different the *sets* of landmarks are, ignoring where they stand.

    For discrete modes: the fraction of identities that are not shared, 0..1.
    For peak modes: the mean form-space distance under the best identity
    matching, in `fm_peak.shape_distance` units (0 .. sqrt(6)).

    A permutation scores exactly 0 by construction -- it reuses every landmark.
    """
    ia, ib = _identities(a), _identities(b)
    if len(ia) != len(ib):
        raise ValueError("scene distance is undefined between different "
                         "landmark counts")
    n = len(ia)
    if n == 0:
        return 0.0

    forms_a = [p.get("form") for p in a["peaks"]]
    forms_b = [p.get("form") for p in b["peaks"]]
    if all(f for f in forms_a) and all(f for f in forms_b):
        best = float("inf")
        for perm in itertools.permutations(range(n)):
            cost = sum(peaklib.shape_distance(forms_a[i], forms_b[perm[i]])
                       for i in range(n))
            best = min(best, cost)
        return best / n

    # Discrete identities: multiset difference, normalised by landmark count.
    remaining = list(ib)
    shared = 0
    for k in ia:
        if k in remaining:
            remaining.remove(k)
            shared += 1
    return (n - shared) / n


def distance(a, b):
    """All three coordinates at once."""
    return {"d_pos": d_pos(a, b), "d_bind": d_bind(a, b), "d_id": d_id(a, b)}


# --------------------------------------------------------------------------- #
# Same place or not
# --------------------------------------------------------------------------- #

def same_place(a, b, tol_m=SAME_PLACE_M):
    """
    True if these are the same place: same landmarks, none of them meaningfully
    moved.

    Used to keep a retrieval gallery honest -- if two anchors are the same place,
    a retrieval miss between them is the pool's fault, not the model's.

    This supersedes `fm_simple.layouts_collide`, which compared an identity
    multiset built by `_identity_multiset`; that helper returned `None` for the
    identity of every peak-mode landmark, so on c3/c4 it compared sizes only and
    two scenes with identical positions but completely different landforms were
    called the same place.
    """
    if len(a["peaks"]) != len(b["peaks"]):
        return False
    if d_id(a, b) > 1e-9:
        return False
    return d_pos(a, b) < tol_m


def pool_is_separable(accepted, candidate, tol_m=SAME_PLACE_M):
    """True if `candidate` is a distinct place from everything accepted so far."""
    return not any(same_place(a, candidate, tol_m) for a in accepted)


def layout_key(layout):
    """Hashable identity of a layout, for rejecting exact duplicates."""
    return tuple(zip(_identities(layout),
                     ((round(p["x"], 2), round(p["y"], 2))
                      for p in layout["peaks"])))


# --------------------------------------------------------------------------- #
# Building a scene at a requested distance
# --------------------------------------------------------------------------- #

def translate(layout, name, dx, dy):
    """A new layout with one landmark moved. The input is untouched."""
    out = copy.deepcopy(layout)
    for p in out["peaks"]:
        if p["name"] == name:
            p["x"] += float(dx)
            p["y"] += float(dy)
            return out
    raise KeyError(f"no landmark named {name!r}")


def sample_probe(layout, rng, target_m, cams, name=None, tol=0.05,
                 max_tries=400):
    """
    A scene at `target_m` metres from `layout`, or a loud failure.

    One landmark moves by exactly `target_m`; the rest stay. `d_pos` is the
    largest matched displacement, so the requested and achieved distances agree
    directly and the probe ladder reads as "one object moved d metres".

    `name` fixes *which* landmark moves. The caller passes one per anchor so
    that an anchor's whole ladder is a single trajectory: letting it be redrawn
    at every rung made the cube move at 1-4 m and the cylinder at 8-24 m, which
    adds variance to D_scene(d) for nothing. Averaging over which landmark moves
    still happens, across anchors.

    Returns `(probe_layout, achieved)` where `achieved` is the measured
    `{d_pos, d_bind, d_id}`. **The achieved distance is what gets recorded**;
    a requested value that the geometry could not deliver would be a fabricated
    label, which is exactly the bug `make_permutation_foil` used to have when it
    silently clamped a 4-cycle down to a 3-cycle and still reported 4.

    Raises `ValueError` when no valid layout exists at that distance. The ring
    is 36-45 m and `FOOTPRINT_SLACK` forces ~38 m between landmark centres, so
    large requests genuinely cannot be built and must not be quietly rounded
    down to something that can.
    """
    import fm_layout as layoutlib

    n = len(layout["peaks"])
    if n == 0:
        raise ValueError("cannot probe an empty scene")
    step = float(target_m)              # d_pos is the largest single move
    names = [p["name"] for p in layout["peaks"]]

    for _ in range(max_tries):
        who = str(rng.choice(names)) if name is None else str(name)
        th = float(rng.uniform(0, 2 * math.pi))
        cand = translate(layout, who, step * math.cos(th), step * math.sin(th))
        if not layoutlib.layout_is_valid(cand["peaks"], cams=cams):
            continue
        got = distance(layout, cand)
        if abs(got["d_pos"] - target_m) > tol * max(target_m, 1.0):
            continue                    # a different landmark got matched
        cand["probe_of"] = layout.get("layout_id", layout.get("seed"))
        cand["probe_distance_m"] = got["d_pos"]
        cand["probe_moved"] = who
        return cand, got

    raise ValueError(
        f"no valid scene at d_pos={target_m} m"
        + (f" moving {name}" if name else "")
        + f" from layout "
        f"{layout.get('layout_id', layout.get('seed'))} after {max_tries} "
        f"tries; the ring geometry cannot place a landmark that far without "
        f"breaking framing or footprint separation")


def translate_all(layout, dx, dy):
    """A new layout with *every* landmark moved by the same vector."""
    out = copy.deepcopy(layout)
    for p in out["peaks"]:
        p["x"] += float(dx)
        p["y"] += float(dy)
    return out


def sample_rigid(layout, rng, target_m, cams, tol=0.05, max_tries=400):
    """
    The whole configuration translated `target_m` metres, shape intact.

    Every inter-landmark distance and angle is preserved, so the *configuration*
    is untouched while every pixel moves. Paired with `sample_scramble` at the
    same `target_m` this is the contrast that separates a representation of
    relative position from one of appearance: the two probes move the image by
    the same amount, and only a model that encodes the configuration can rank
    them differently.

    `d_pos` is the bottleneck matched displacement, and under a rigid
    translation every landmark moves the same distance, so `d_pos == target_m`
    exactly rather than approximately.
    """
    import fm_layout as layoutlib

    if not layout["peaks"]:
        raise ValueError("cannot probe an empty scene")
    for _ in range(max_tries):
        th = float(rng.uniform(0, 2 * math.pi))
        cand = translate_all(layout, target_m * math.cos(th),
                             target_m * math.sin(th))
        if not layoutlib.layout_is_valid(cand["peaks"], cams=cams):
            continue
        got = distance(layout, cand)
        if abs(got["d_pos"] - target_m) > tol * max(target_m, 1.0):
            continue           # a landmark got matched to a different one
        cand["probe_of"] = layout.get("layout_id", layout.get("seed"))
        cand["probe_distance_m"] = got["d_pos"]
        cand["probe_kind"] = "rigid"
        return cand, got
    raise ValueError(
        f"no valid rigid translation of {target_m} m from layout "
        f"{layout.get('layout_id', layout.get('seed'))} after {max_tries} "
        f"tries; the ring geometry cannot hold the whole configuration that "
        f"far off centre without breaking framing")


def sample_scramble(layout, rng, target_m, cams, tol=0.05, max_tries=600):
    """
    Every landmark displaced independently, bottleneck exactly `target_m`.

    The configuration is destroyed while the landmark *set* is untouched, so
    `d_id` stays 0 and the change is purely positional -- the same currency the
    ladder is measured in. This is the counterpart of `sample_rigid`: matched
    `d_pos`, opposite configural consequence.

    One landmark is given the full `target_m` and the rest a uniform share of
    it, so the bottleneck is the requested distance rather than an emergent
    property of the draw. Without that the achieved `d_pos` would drift above
    the request and the two families would no longer be matched, which is the
    single thing the comparison depends on.
    """
    import fm_layout as layoutlib

    peaks = layout["peaks"]
    if len(peaks) < 2:
        raise ValueError("scrambling needs at least two landmarks")
    names = [p["name"] for p in peaks]

    for _ in range(max_tries):
        lead = int(rng.integers(len(names)))
        cand = copy.deepcopy(layout)
        for i, p in enumerate(cand["peaks"]):
            step = target_m if i == lead else target_m * float(rng.uniform(0.5, 1.0))
            th = float(rng.uniform(0, 2 * math.pi))
            p["x"] += step * math.cos(th)
            p["y"] += step * math.sin(th)
        if not layoutlib.layout_is_valid(cand["peaks"], cams=cams):
            continue
        got = distance(layout, cand)
        if abs(got["d_pos"] - target_m) > tol * max(target_m, 1.0):
            continue
        if got["d_id"] > 1e-9:
            continue                    # identities must be untouched
        cand["probe_of"] = layout.get("layout_id", layout.get("seed"))
        cand["probe_distance_m"] = got["d_pos"]
        cand["probe_kind"] = "scramble"
        return cand, got
    raise ValueError(
        f"no valid scramble at d_pos={target_m} m from layout "
        f"{layout.get('layout_id', layout.get('seed'))} after {max_tries} "
        f"tries")
