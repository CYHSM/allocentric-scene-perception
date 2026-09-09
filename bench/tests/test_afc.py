"""
Tests for the forced-choice arm -- the bridge from frozen encoders to models
that only emit text.

The arm is only worth having if it measures the *same* lambda as
`exchange.py`. So the load-bearing test here is not that the code runs, but
that a frozen encoder answering these items lands on the same metres as the
curve-crossing read-out does on the same embeddings.
"""

import numpy as np
import pytest

import afc
import exchange as X

AZ = [0, 45, 90, 135, 180, 225, 270, 315]
LADDER = [1.0, 2.0, 4.0, 8.0, 16.0, 24.0]


def _bank(n_anchors=8):
    scenes = {}
    for a in range(n_anchors):
        aid = f"a{a:03d}"
        frames = {f"{app}_{az}": {"rgb": f"{aid}/{aid}_{app}_az{az:03d}.png"}
                  for app in ("A", "B") for az in AZ}
        scenes[aid] = {"role": "anchor", "anchor": aid, "d_pos": 0.0,
                       "frames": frames}
        for d in LADDER:
            sid = f"{aid}p{d:g}"
            scenes[sid] = {
                "role": "probe", "anchor": aid, "d_pos": d,
                "requested_d_pos": d,
                "frames": {f"{app}_{az}": {"rgb": f"{sid}/{sid}_{app}_az{az:03d}.png"}
                           for app in ("A", "B") for az in AZ}}
    return {"scenes": scenes}


def _vectors(bank, kind, view_cost=0.0, scene_gain=0.0, dim=48, seed=0):
    """
    An embedding built so the answer is known in advance.

    `view_cost` scales how much a turn moves the vector; `scene_gain` how much a
    metre of displacement does. Their ratio fixes where lambda must land.
    """
    rng = np.random.default_rng(seed)
    anchor = {s: rng.normal(size=dim) for s, m in bank["scenes"].items()
              if m["role"] == "anchor"}
    cam = {az: rng.normal(size=dim) for az in AZ}
    drift = {s: rng.normal(size=dim) for s in bank["scenes"]}
    vec = {}
    for sid, meta in bank["scenes"].items():
        base = anchor[meta["anchor"]].copy()
        if meta["role"] == "probe":
            base = base + scene_gain * meta["requested_d_pos"] * drift[meta["anchor"]]
        for app in ("A", "B"):
            for az in AZ:
                v = base + view_cost * cam[az]
                if kind == "camera":
                    v = cam[az]
                vec[(sid, app, az)] = v
    return vec


def test_the_moved_scene_is_always_the_correct_answer():
    items = afc.build_items(_bank(), per_cell=6, seed=1)
    assert items
    for it in items:
        assert it["options"][it["answer"] - 1]["kind"] == "scene"


def test_option_order_carries_no_signal():
    """If the answer sat in the same slot the task would be solvable blind."""
    items = afc.build_items(_bank(), per_cell=20, seed=3)
    first = sum(1 for it in items if it["answer"] == 1)
    assert 0.35 < first / len(items) < 0.65, \
        f"the correct option sits first in {first}/{len(items)} items"


def test_both_options_carry_the_same_appearance():
    """
    The reference is appearance A and both candidates B. If only one candidate
    carried the change it would be gratuitously easy to reject, and lambda
    would shift by an amount no downstream check could see.
    """
    for it in afc.build_items(_bank(), per_cell=5, seed=2):
        assert it["study"]["app"] == "A"
        assert {o["app"] for o in it["options"]} == {"B"}


def test_the_two_candidates_differ_only_as_intended():
    """
    view = the anchor turned, at the same place; scene = the alternative at the
    reference bearing. Holding the scene option at the reference bearing is what
    makes one comparison pure viewpoint and the other pure place.

    On the ladder the alternative is the anchor's own probe; in the ceiling cell
    it is a different anchor. Both must sit at the reference bearing.
    """
    items = afc.build_items(_bank(), per_cell=5, seed=4)
    assert {it["cell"] for it in items} == {"ladder", "ceiling"}
    for it in items:
        view = next(o for o in it["options"] if o["kind"] == "view")
        scene = next(o for o in it["options"] if o["kind"] == "scene")
        assert view["scene"] == it["anchor"]
        assert view["az"] != it["azimuth"]
        assert scene["az"] == it["azimuth"]
        assert (view["az"] - it["azimuth"]) % 360 == it["delta"]
        if it["cell"] == "ladder":
            assert scene["scene"].startswith(it["anchor"] + "p")
        else:
            assert scene["scene"] != it["anchor"] and "p" not in scene["scene"]


def test_a_layout_only_encoder_is_right_everywhere():
    """Viewpoint free: the moved scene is the only thing that ever differs."""
    bank = _bank()
    vec = _vectors(bank, "layout", view_cost=0.0, scene_gain=0.05)
    items = afc.build_items(bank, per_cell=8, seed=5)
    cells = afc.accuracy(items, afc.encoder_answers(items, vec))
    assert all(c / n == 1.0 for c, n in cells.values())
    assert afc.lam_from_accuracy(cells, 45) == (None, "below")


def test_a_camera_only_encoder_is_wrong_everywhere():
    bank = _bank()
    vec = _vectors(bank, "camera")
    items = afc.build_items(bank, per_cell=8, seed=6)
    cells = afc.accuracy(items, afc.encoder_answers(items, vec))
    assert all(c / n == 0.0 for c, n in cells.values())
    assert afc.lam_from_accuracy(cells, 45) == (None, "above")


def test_accuracy_rises_with_displacement():
    bank = _bank()
    vec = _vectors(bank, "mix", view_cost=0.45, scene_gain=0.06, seed=7)
    items = afc.build_items(bank, per_cell=24, seed=7)
    cells = afc.accuracy(items, afc.encoder_answers(items, vec))
    acc = [cells[(45, d)][0] / cells[(45, d)][1] for d in LADDER]
    assert acc[0] < acc[-1], f"accuracy did not rise with d: {acc}"
    assert all(b >= a - 0.05 for a, b in zip(acc, acc[1:])), \
        f"accuracy should be monotone in d up to noise: {acc}"


def test_the_forced_choice_lambda_matches_the_curve_crossing_lambda():
    """
    The point of the whole module. Both read-outs are run on the *same*
    embeddings; if they disagreed, the frontier-model numbers would not be
    comparable with the encoder numbers and the metric would not span the range
    it claims to.
    """
    bank = _bank(n_anchors=12)
    vec = _vectors(bank, "mix", view_cost=0.45, scene_gain=0.06, seed=11)

    items = afc.build_items(bank, per_cell=40, seed=11)
    cells = afc.accuracy(items, afc.encoder_answers(items, vec))
    lam_afc, bound_afc = afc.lam_from_accuracy(cells, 45)

    dv, ds = X.curves(vec, bank)
    lam_curve, bound_curve = X.lam_bound(dv, ds, 45)

    assert bound_afc == bound_curve is None, \
        f"the two arms disagree on measurability: {bound_afc} vs {bound_curve}"
    ratio = lam_afc / lam_curve
    assert 0.5 < ratio < 2.0, \
        (f"forced-choice lambda {lam_afc:.2f} m and curve-crossing lambda "
         f"{lam_curve:.2f} m are more than a ladder step apart")


def test_lambda_is_not_clamped_when_it_leaves_the_ladder():
    below = {(45, d): (10, 10) for d in LADDER}
    above = {(45, d): (0, 10) for d in LADDER}
    assert afc.lam_from_accuracy(below, 45) == (None, "below")
    assert afc.lam_from_accuracy(above, 45) == (None, "above")


def test_a_cell_with_no_answers_is_skipped_not_scored_as_wrong():
    """An unanswered item is missing data; scoring it 0 would read as failure."""
    bank = _bank()
    items = afc.build_items(bank, per_cell=4, seed=8)
    cells = afc.accuracy(items, {})
    assert cells == {}


def test_the_ceiling_control_offers_a_different_place_not_a_nudged_one():
    """
    The upper counterpart of the identity gate. Its alternative is another
    anchor outright, so no model that can do the task at all should miss it --
    and a model that does has failed the task, not the viewpoint question.
    """
    bank = _bank()
    items = afc.build_items(bank, per_cell=6, seed=9)
    ceil = [it for it in items if it["cell"] == "ceiling"]
    assert ceil, "no ceiling items were built"
    for it in ceil:
        assert it["d"] is None
        scene = next(o for o in it["options"] if o["kind"] == "scene")
        assert scene["scene"] != it["anchor"]
        assert "p" not in scene["scene"], "the alternative must not be a probe"
        assert scene["scene"] in bank["scenes"]


def test_the_ceiling_cell_is_never_interpolated_into_lambda():
    """
    It is off the ladder by construction. If it leaked into the fit it would
    drag every lambda toward the largest distance in the set.
    """
    cells = {(45, d): (0, 10) for d in LADDER}
    cells[(45, None)] = (10, 10)                 # perfect on the control
    assert afc.lam_from_accuracy(cells, 45) == (None, "above")
    assert afc.ceiling(cells, 45) == 1.0


def test_a_layout_only_encoder_passes_the_ceiling():
    bank = _bank()
    vec = _vectors(bank, "layout", view_cost=0.0, scene_gain=0.05)
    items = afc.build_items(bank, per_cell=6, seed=10)
    cells = afc.accuracy(items, afc.encoder_answers(items, vec))
    assert afc.ceiling(cells) == 1.0
