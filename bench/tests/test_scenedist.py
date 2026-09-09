"""
Tests for the scene-distance metric.

The headline test is `test_the_three_axes_are_orthogonal`. It is the regression
test for the defect that made the old foil families unreportable: because
`fm_foils.max_displacement` matched landmarks by slot name, a permutation --
which leaves the set of occupied positions exactly as it found it -- recorded
50-65 m of "displacement", more than an explicit 20 m translation. Any
binding-vs-metric contrast was therefore comparing a manipulation against a
mislabelled version of itself.
"""

import copy
import math

import numpy as np
import pytest

import fm_layout as layoutlib
import fm_scenedist as D


@pytest.fixture(scope="module")
def cams():
    return layoutlib._cameras(layoutlib.benchmark_azimuths(45))


@pytest.fixture(scope="module")
def layout():
    return layoutlib.sample_layout(7, n_peaks=4, azimuth_step=45)


@pytest.fixture
def rng():
    return np.random.default_rng(20260908)


def _permute_positions(layout, order):
    """Rotate positions among landmarks: identities and places both preserved."""
    out = copy.deepcopy(layout)
    coords = [(out["peaks"][i]["x"], out["peaks"][i]["y"]) for i in order]
    for k, i in enumerate(order):
        out["peaks"][i]["x"], out["peaks"][i]["y"] = coords[(k - 1) % len(order)]
    return out


# --------------------------------------------------------------------------- #
# The point of the module
# --------------------------------------------------------------------------- #

def test_the_three_axes_are_orthogonal(layout, rng):
    """
    Each manipulation moves exactly one coordinate.

    translation  -> d_pos only
    permutation  -> d_bind only   (the old metric scored this at 50-65 m)
    substitution -> d_bind and d_id, never d_pos
    """
    # translation: move one landmark 20 m
    moved = D.translate(layout, layout["peaks"][0]["name"], 20.0, 0.0)
    got = D.distance(layout, moved)
    assert got["d_pos"] == pytest.approx(20.0, abs=1e-9)
    assert got["d_bind"] == 0.0
    assert got["d_id"] == pytest.approx(0.0, abs=1e-9)

    # permutation: same places, different occupants
    perm = _permute_positions(layout, [0, 1, 2, 3])
    got = D.distance(layout, perm)
    assert got["d_pos"] == pytest.approx(0.0, abs=1e-9), (
        "a permutation occupies exactly the same positions; any non-zero d_pos "
        "means landmarks are being matched by slot rather than by place")
    assert got["d_bind"] > 0.0
    assert got["d_id"] == pytest.approx(0.0, abs=1e-9)

    # substitution: a different landmark in one place
    sub = copy.deepcopy(layout)
    other = layoutlib.sample_layout(99, n_peaks=4, azimuth_step=45)
    sub["peaks"][0]["form"] = dict(other["peaks"][0]["form"])
    got = D.distance(layout, sub)
    assert got["d_pos"] == pytest.approx(0.0, abs=1e-9)
    assert got["d_id"] > 0.0


def test_permutation_is_not_scored_as_displacement(layout):
    """Explicitly: the old convention gave ~50-65 m here."""
    perm = _permute_positions(layout, [0, 1])
    assert D.d_pos(layout, perm) < 1e-9


# --------------------------------------------------------------------------- #
# It has to actually be a distance
# --------------------------------------------------------------------------- #

def test_identity_of_indiscernibles(layout):
    assert D.d_pos(layout, copy.deepcopy(layout)) == pytest.approx(0.0)
    assert D.d_bind(layout, copy.deepcopy(layout)) == 0.0
    assert D.d_id(layout, copy.deepcopy(layout)) == pytest.approx(0.0)


def test_symmetry(layout, rng):
    for _ in range(20):
        a = D.translate(layout, f"M{rng.integers(1, 5)}",
                        rng.uniform(-15, 15), rng.uniform(-15, 15))
        b = D.translate(layout, f"M{rng.integers(1, 5)}",
                        rng.uniform(-15, 15), rng.uniform(-15, 15))
        assert D.d_pos(a, b) == pytest.approx(D.d_pos(b, a), abs=1e-9)


def test_triangle_inequality(layout, rng):
    for _ in range(30):
        a = layout
        b = D.translate(a, "M1", rng.uniform(-12, 12), rng.uniform(-12, 12))
        c = D.translate(b, "M2", rng.uniform(-12, 12), rng.uniform(-12, 12))
        assert D.d_pos(a, c) <= D.d_pos(a, b) + D.d_pos(b, c) + 1e-9


def test_order_of_the_landmark_list_does_not_matter(layout):
    """A scene is a set. Shuffling the list must not change any distance."""
    shuffled = copy.deepcopy(layout)
    shuffled["peaks"] = list(reversed(shuffled["peaks"]))
    assert D.d_pos(layout, shuffled) == pytest.approx(0.0, abs=1e-9)
    assert D.d_bind(layout, shuffled) == 0.0
    assert D.d_id(layout, shuffled) == pytest.approx(0.0, abs=1e-9)


def test_different_landmark_counts_are_refused(layout):
    fewer = copy.deepcopy(layout)
    fewer["peaks"] = fewer["peaks"][:3]
    with pytest.raises(ValueError):
        D.d_pos(layout, fewer)


# --------------------------------------------------------------------------- #
# Sampling a scene at a requested distance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("target", [1.0, 2.0, 4.0, 8.0])
def test_probe_lands_at_the_requested_distance(layout, cams, rng, target):
    probe, got = D.sample_probe(layout, rng, target, cams)
    assert got["d_pos"] == pytest.approx(target, rel=0.05)
    assert probe["probe_distance_m"] == pytest.approx(got["d_pos"])
    assert layoutlib.layout_is_valid(probe["peaks"], cams=cams)


def test_probe_records_the_achieved_not_the_requested_distance(layout, cams, rng):
    """
    The recorded number must be measured from the layout that was built.

    `make_permutation_foil` used to clamp a 4-cycle to a 3-cycle and still
    record `foil_param: 4.0`; that fabricated label is the failure mode here.
    """
    probe, got = D.sample_probe(layout, rng, 4.0, cams)
    assert probe["probe_distance_m"] == pytest.approx(D.d_pos(layout, probe))


def test_an_unbuildable_distance_raises_rather_than_clamping(layout, cams, rng):
    """The ring cannot deliver an arbitrarily large displacement."""
    with pytest.raises(ValueError, match="no valid scene"):
        D.sample_probe(layout, rng, 500.0, cams, max_tries=40)


def test_probes_are_not_the_same_place_as_their_anchor(layout, cams, rng):
    """A probe far enough out must read as a different place."""
    probe, _ = D.sample_probe(layout, rng, 16.0, cams)
    assert not D.same_place(layout, probe)


# --------------------------------------------------------------------------- #
# same_place, which keeps the retrieval gallery honest
# --------------------------------------------------------------------------- #

def test_same_place_sees_identity_on_peak_scenes(layout):
    """
    The bug this replaces: `fm_simple._identity_multiset` returned None for the
    identity of every peak-mode landmark, so two c3/c4 scenes with identical
    positions and completely different landforms compared as the same place.
    """
    other = layoutlib.sample_layout(123, n_peaks=4, azimuth_step=45)
    swapped = copy.deepcopy(layout)
    for p, q in zip(swapped["peaks"], other["peaks"]):
        p["form"] = dict(q["form"])
    assert D.d_id(layout, swapped) > 0.0
    assert not D.same_place(layout, swapped)


def test_a_nudged_scene_is_still_the_same_place(layout):
    nudged = D.translate(layout, "M1", 1.0, 0.0)
    assert D.same_place(layout, nudged)


def test_pool_separation_rejects_a_duplicate_accepts_a_move(layout, cams, rng):
    far, _ = D.sample_probe(layout, rng, 24.0, cams)
    assert not D.pool_is_separable([layout], copy.deepcopy(layout))
    assert D.pool_is_separable([layout], far)


# --------------------------------------------------------------------------- #
# rigid vs scramble -- matched displacement, opposite configural consequence
# --------------------------------------------------------------------------- #

def _pairwise(lay):
    ps = [(p["x"], p["y"]) for p in lay["peaks"]]
    return sorted(round(math.dist(a, b), 6)
                  for i, a in enumerate(ps) for b in ps[i + 1:])


def test_a_rigid_translation_preserves_every_inter_landmark_distance(layout):
    """
    The whole point of the rigid probe. If the configuration were even slightly
    deformed, the contrast against scramble would no longer isolate configural
    change from displacement magnitude.
    """
    assert _pairwise(D.translate_all(layout, 7.0, -3.0)) == _pairwise(layout)


def test_a_rigid_translation_moves_every_landmark_the_same_distance(layout):
    moved = D.translate_all(layout, 3.0, 4.0)
    for a, b in zip(layout["peaks"], moved["peaks"]):
        assert math.dist((a["x"], a["y"]), (b["x"], b["y"])) == pytest.approx(5.0)
    assert D.d_pos(layout, moved) == pytest.approx(5.0)


def test_scrambling_changes_the_configuration_but_not_the_landmark_set(
        layout, cams, rng):
    """
    d_id must stay 0: the change has to be purely positional, or D_scene would
    be measuring identity while reported in metres.
    """
    cand, got = D.sample_scramble(layout, rng, 8.0, cams)
    assert got["d_id"] == pytest.approx(0.0)
    assert got["d_pos"] == pytest.approx(8.0, rel=0.05)
    assert _pairwise(cand) != _pairwise(layout), "the configuration survived"


def test_rigid_and_scramble_are_matched_on_displacement(layout, cams, rng):
    """
    The comparison rests entirely on this. If the two families differed in
    d_pos, any gap in embedding distance could just be the larger move and the
    configural index would measure nothing.
    """
    for target in (4.0, 8.0):
        _, r = D.sample_rigid(layout, rng, target, cams)
        _, s = D.sample_scramble(layout, rng, target, cams)
        assert r["d_pos"] == pytest.approx(target, rel=0.05)
        assert s["d_pos"] == pytest.approx(target, rel=0.05)
        assert r["d_pos"] == pytest.approx(s["d_pos"], rel=0.05)


def test_a_rigid_translation_leaves_binding_and_identity_alone(layout, cams, rng):
    _, got = D.sample_rigid(layout, rng, 8.0, cams)
    assert got["d_bind"] == pytest.approx(0.0)
    assert got["d_id"] == pytest.approx(0.0)


def test_the_two_families_are_labelled_so_they_cannot_be_confused(
        layout, cams, rng):
    r, _ = D.sample_rigid(layout, rng, 8.0, cams)
    s, _ = D.sample_scramble(layout, rng, 8.0, cams)
    assert r["probe_kind"] == "rigid" and s["probe_kind"] == "scramble"
    assert r["probe_distance_m"] == pytest.approx(s["probe_distance_m"], rel=0.05)
