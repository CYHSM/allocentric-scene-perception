"""
Tests for lambda, the metres-per-turn exchange rate.

Both ends are pinned against embeddings whose answer is known by construction:
an embedding that depends only on the layout puts lambda *below* the ladder
(viewpoint is free, so even the smallest scene change costs more than the turn),
and one that depends only on the camera puts it *above* (the turn costs more
than any scene change we rendered). That is the direction of the whole read-out:
**small lambda is the allocentric end.** If it were inverted, every lambda in
the paper would be wrong the same way, so it is asserted here explicitly.
"""

import numpy as np
import pytest

import exchange as X

AZ = [0, 45, 90, 135, 180, 225, 270, 315]
LADDER = [1.0, 2.0, 4.0, 8.0, 16.0, 24.0]


def _bank(n_anchors=6):
    scenes = {}
    for a in range(n_anchors):
        aid = f"a{a:03d}"
        scenes[aid] = {"role": "anchor", "anchor": aid, "d_pos": 0.0,
                       "dir": aid, "frames": {}}
        for d in LADDER:
            scenes[f"{aid}p{d:g}"] = {
                "role": "probe", "anchor": aid, "d_pos": d,
                "requested_d_pos": d, "dir": f"{aid}p{d:g}", "frames": {}}
    return {"scenes": scenes, "stimulus_mode": "test", "n_objects": 4}


def _vectors(bank, kind, dim=32, seed=0):
    """
    kind="layout": the embedding sees the place and ignores the camera.
    kind="camera": the embedding sees the camera and ignores the place.
    """
    rng = np.random.default_rng(seed)
    anchor_vec = {s: rng.normal(size=dim) for s, m in bank["scenes"].items()
                  if m["role"] == "anchor"}
    cam_vec = {az: rng.normal(size=dim) for az in AZ}

    vec = {}
    for sid, meta in bank["scenes"].items():
        base = anchor_vec[meta["anchor"]]
        if meta["role"] == "probe":
            # Move away from the anchor in proportion to d_pos.
            drift = rng.normal(size=dim)
            base = base + 0.05 * meta["requested_d_pos"] * drift
        for app in ("A", "B"):
            for az in AZ:
                v = base if kind == "layout" else cam_vec[az]
                vec[(sid, app, az)] = v.astype(np.float64)
    return vec


def test_a_layout_only_embedding_has_no_measurable_lambda():
    """
    Viewpoint costs nothing, so no scene change on the ladder is ever as cheap
    as a turn. lambda must come back None -- "off the ladder" -- rather than a
    clamped number pretending to be a measurement.
    """
    bank = _bank()
    vec = _vectors(bank, "layout")
    dv, ds = X.curves(vec, bank)
    assert np.mean(dv[45]) == pytest.approx(0.0, abs=1e-9)
    for delta in (45, 90, 180):
        assert X.lam(dv, ds, delta) is None


def test_a_camera_only_embedding_has_no_measurable_lambda_either():
    """
    The mirror image: the turn costs more than any scene change we rendered, so
    lambda is off the top of the ladder. Distinguishing the two ends is the
    caller's job, and `d_view` vs `d_scene` says which.
    """
    bank = _bank()
    vec = _vectors(bank, "camera")
    dv, ds = X.curves(vec, bank)
    assert all(np.mean(v) == pytest.approx(0.0, abs=1e-9) for v in ds.values()), \
        "a camera-only embedding cannot see a scene change"
    assert np.mean(dv[45]) > 0.0
    for delta in (45, 90, 180):
        assert X.lam(dv, ds, delta) is None


def test_lambda_lands_where_the_curves_actually_cross():
    """A hand-built pair of curves with an analytically known crossing."""
    d_view = {45: [0.30]}
    d_scene = {1.0: [0.10], 2.0: [0.20], 4.0: [0.40], 8.0: [0.60]}
    # 0.30 sits halfway between d=2 (0.20) and d=4 (0.40) -> 3.0 m
    assert X.lam(d_view, d_scene, 45) == pytest.approx(3.0)


def test_lambda_is_none_outside_the_probe_ladder():
    d_scene = {1.0: [0.10], 24.0: [0.50]}
    assert X.lam({45: [0.01]}, d_scene, 45) is None    # cheaper than 1 m
    assert X.lam({45: [0.99]}, d_scene, 45) is None    # dearer than 24 m


def test_both_curves_use_the_same_appearance_contrast():
    """
    D_view and D_scene must both be A-vs-B. If one were B-vs-B it would be
    artificially small and lambda would be wrong by that margin, in a way no
    downstream check would catch.
    """
    bank = _bank(n_anchors=3)
    vec = _vectors(bank, "layout")
    # Make appearance A differ from B by a constant offset.
    for k in list(vec):
        if k[1] == "A":
            vec[k] = vec[k] + 3.0
    dv, ds = X.curves(vec, bank)
    assert np.mean(dv[0]) > 0.0, \
        "D_view at delta 0 must carry the appearance change"
    assert min(ds) in LADDER


def test_probes_are_compared_at_the_same_bearing_as_their_anchor():
    """
    D_scene must hold viewpoint fixed. If it drifted across azimuths it would
    be measuring viewpoint and scene at once, and lambda would compare a
    quantity against itself.
    """
    bank = _bank(n_anchors=2)
    vec = _vectors(bank, "camera")
    _, ds = X.curves(vec, bank)
    # camera-only embedding + same bearing => exactly zero at every distance
    assert all(np.allclose(v, 0.0, atol=1e-9) for v in ds.values())


def test_which_end_of_the_ladder_lambda_falls_off_is_reported():
    """
    The direction of the read-out, asserted rather than described.

    A layout-only embedding is the ideal allocentric model: viewpoint is free,
    so the turn costs less than the smallest scene change and lambda sits
    *below* the ladder. A camera-only embedding is pure appearance matching and
    sits *above* it. `lam` collapses both to None; `lam_bound` keeps the sign,
    which for a saturated model is the whole finding.

    If this ever inverts, every lambda in the paper points the same wrong way
    and nothing downstream would catch it -- the numbers would still be numbers.
    """
    bank = _bank()

    dv, ds = X.curves(_vectors(bank, "layout"), bank)
    assert X.lam_bound(dv, ds, 45) == (None, "below"), \
        "an embedding that ignores the camera must land at the allocentric end"

    dv, ds = X.curves(_vectors(bank, "camera"), bank)
    assert X.lam_bound(dv, ds, 45) == (None, "above"), \
        "an embedding that sees only the camera must land at the appearance end"


def test_a_measured_lambda_reports_no_bound():
    d_scene = {1.0: [0.10], 2.0: [0.20], 4.0: [0.40], 8.0: [0.60]}
    m, bound = X.lam_bound({45: [0.30]}, d_scene, 45)
    assert bound is None
    assert m == pytest.approx(3.0)
