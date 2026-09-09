"""
Gates for the text channel.

The text arm exists to remove perception from the task, so its own integrity has
to be checked more strictly than the image arm's: a picture cannot accidentally
print the camera's heading, and a coordinate list can.
"""

import json
import math
import os

import numpy as np
import pytest

from bench import text_views as tv
from bench.build_text_benchmark import GEOMETRY_MODE, add_text, load_layouts
from bench.build_vlm_benchmark import DELTAS, build_trials

ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data", "scenes_100")
FILES = {t: os.path.join(os.path.dirname(ROOT), f"text_benchmark_2afc_{t}.json")
         for t in ("id", "noid")}

needs_bank = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, GEOMETRY_MODE, "bank.json")),
    reason="data/scenes_100 not present")


def _layout(peaks):
    return {"world": "bare",
            "peaks": [{"x": x, "y": y, "z": 1.25, "height": 19.0,
                       "base_radius": 17.0, "type": t}
                      for (x, y, t) in peaks]}


LAYOUT = _layout([(37.0, 8.0, "purple cone"), (-7.0, 43.0, "cyan pyramid"),
                  (-40.0, 19.0, "yellow dome"), (11.0, -37.0, "green cylinder")])


# --------------------------------------------------------------------------- #
# the serialiser
# --------------------------------------------------------------------------- #

def test_rows_are_ordered_left_to_right():
    rows = tv.view_rows(LAYOUT, 0.0)
    assert [r["bearing_deg"] for r in rows] == sorted(r["bearing_deg"] for r in rows)


def test_polar_and_cartesian_agree():
    for r in tv.view_rows(LAYOUT, 135.0):
        assert math.hypot(r["right_m"], r["forward_m"]) == pytest.approx(r["distance_m"], abs=1e-6)
        assert math.degrees(math.atan2(r["right_m"], r["forward_m"])) == \
            pytest.approx(r["bearing_deg"], abs=1e-6)


def test_bearing_sign_matches_the_render():
    """
    Left in the text must be left in the picture. The flattened viewer frame is
    derived independently of `fm_geom.camera_basis`, so the two are checked
    against each other: same sign everywhere, and within a degree in magnitude
    (they differ only by the camera's 17 deg downward tilt).
    """
    import sys
    sys.path.insert(0, tv._BLENDER)
    import fm_geom as geom

    pts = np.array([[p["x"], p["y"], p["z"]] for p in LAYOUT["peaks"]])
    for az in (0.0, 45.0, 90.0, 135.0, 225.0, 315.0):
        cam = geom.camera_position(az, 17.0, 122.0, 6.0)
        ref = dict(zip([p["type"] for p in LAYOUT["peaks"]],
                       geom.bearing_from_camera(pts, cam, np.array([0.0, 0.0, 6.0]))))
        for r in tv.view_rows(LAYOUT, az):
            assert np.sign(r["bearing_deg"]) == np.sign(ref[r["type"]])
            assert abs(r["bearing_deg"] - ref[r["type"]]) < 1.0


def test_no_identity_leak_in_anonymous_variant():
    """
    A landmark name in the no-identity text would hand back the correspondence
    the variant exists to withhold.
    """
    text = tv.render_view(LAYOUT, 45.0, identity=False)
    for p in LAYOUT["peaks"]:
        assert p["type"] not in text


def test_no_allocentric_quantity_is_ever_printed():
    """
    World coordinates, compass bearings or the camera azimuth would make the
    rotation free, which is the one thing the task is measuring.
    """
    for identity in (True, False):
        for az in (0.0, 45.0, 90.0, 225.0):
            text = tv.render_view(LAYOUT, az, identity=identity).lower()
            assert "M1" not in text.upper().replace("M1LE", "")
            for word in ("north", "south", "east", "west", "azimuth", "heading",
                         "compass", "world"):
                assert word not in text


def test_round_trip_through_the_printed_text():
    """What the solver reads must be what the model reads: the file, not the layout."""
    rows = tv.view_rows(LAYOUT, 90.0)
    back = tv.view_rows_from_text(tv.render_view(LAYOUT, 90.0, identity=True))
    assert len(back) == len(rows)
    for a, b in zip(rows, back):
        assert a["type"] == b["type"]
        assert b["right_m"] == pytest.approx(a["right_m"], abs=0.05)
        assert b["forward_m"] == pytest.approx(a["forward_m"], abs=0.05)


def test_identity_and_anonymous_carry_the_same_numbers():
    """The id/noid contrast must be naming alone, not a different measurement."""
    a = tv.view_rows_from_text(tv.render_view(LAYOUT, 45.0, identity=True))
    b = tv.view_rows_from_text(tv.render_view(LAYOUT, 45.0, identity=False))
    assert [(r["right_m"], r["forward_m"]) for r in a] == \
           [(r["right_m"], r["forward_m"]) for r in b]


def test_left_to_right_order_does_not_survive_a_turn():
    """
    The premise of the anonymous variant: the within-view index is not a stable
    identity. If the order happened to be preserved at every turn, the numbering
    would be a correspondence cue after all.
    """
    base = [r["type"] for r in tv.view_rows(LAYOUT, 0.0)]
    turned = [[r["type"] for r in tv.view_rows(LAYOUT, d)] for d in (45, 90, 135, 180)]
    assert any(t != base for t in turned)


# --------------------------------------------------------------------------- #
# the solver, which is the solvability gate
# --------------------------------------------------------------------------- #

def test_distance_matrix_is_rotation_invariant():
    ref = tv.distance_matrix(tv.view_rows(LAYOUT, 0.0))
    for d in (45, 90, 135, 180, 270):
        assert tv.distance_matrix(tv.view_rows(LAYOUT, d)) == pytest.approx(ref, abs=1e-6)


def test_solver_finds_the_same_layout_after_any_turn():
    other = _layout([(20.0, 30.0, "purple cone"), (-35.0, -5.0, "cyan pyramid"),
                     (5.0, -40.0, "yellow dome"), (40.0, 12.0, "green cylinder")])
    for d in DELTAS:
        study = tv.view_rows(LAYOUT, 0.0)
        opts = [tv.view_rows(other, d), tv.view_rows(LAYOUT, d)]
        assert tv.solve(study, opts)[0] == 2


# --------------------------------------------------------------------------- #
# the built benchmark
# --------------------------------------------------------------------------- #

@needs_bank
@pytest.mark.parametrize("identity", [True, False])
def test_built_trials_are_all_answerable(identity):
    layouts, camera = load_layouts(ROOT)
    trials = build_trials(root=ROOT, modes=[GEOMETRY_MODE], deltas=DELTAS,
                          trials_per_delta=20, n_options=2, seed=42)
    tt = add_text(trials, layouts, camera, identity)
    for t in tt:
        study = tv.view_rows_from_text(t["study_text"])
        opts = [tv.view_rows_from_text(o["text"]) for o in t["options"]]
        assert tv.solve(study, opts)[0] == t["correct_choice"], t["id"]


@needs_bank
def test_delta_zero_is_flagged_degenerate_without_jitter():
    """
    With exact coordinates the delta = 0 candidate is a byte-for-byte copy of the
    study block. That cell is not an appearance gate and must not be read as one,
    so the builder labels it.
    """
    layouts, camera = load_layouts(ROOT)
    trials = build_trials(root=ROOT, modes=[GEOMETRY_MODE], deltas=DELTAS,
                          trials_per_delta=20, n_options=2, seed=42)
    tt = add_text(trials, layouts, camera, True)
    for t in tt:
        assert t["degenerate"] == (t["delta"] == 0)
    jittered = add_text(trials, layouts, camera, True, jitter=0.25, seed=1)
    assert not any(t["degenerate"] for t in jittered)


@needs_bank
def test_text_trials_are_paired_with_the_image_benchmark():
    """
    The claim that text and image arms run the same trials is checked, not
    assumed: identical scenes, turns and answer positions make the channel
    comparison within-trial.
    """
    img_path = os.path.join(os.path.dirname(ROOT), "vlm_benchmark_2afc.json")
    if not os.path.exists(img_path):
        pytest.skip("image 2AFC benchmark not built")
    img = {t["id"]: t for t in json.load(open(img_path))["trials"]
           if t["mode"] == GEOMETRY_MODE}
    layouts, camera = load_layouts(ROOT)
    trials = build_trials(root=ROOT, modes=[GEOMETRY_MODE], deltas=DELTAS,
                          trials_per_delta=20, n_options=2, seed=42)
    tt = add_text(trials, layouts, camera, True)
    assert len(tt) == len(img)
    for t in tt:
        i = img[t["paired_id"]]
        assert (i["delta"], i["study_scene"], i["study_azimuth"],
                i["correct_choice"]) == (t["delta"], t["study_scene"],
                                         t["study_azimuth"], t["correct_choice"])
        assert [o["scene_id"] for o in i["options"]] == \
               [o["scene_id"] for o in t["options"]]


@pytest.mark.parametrize("tag", ["id", "noid"])
def test_shipped_file_passes_its_own_gate(tag):
    if not os.path.exists(FILES[tag]):
        pytest.skip(f"{FILES[tag]} not built")
    bench = json.load(open(FILES[tag]))
    assert bench["channel"] == "text"
    for t in bench["trials"]:
        study = tv.view_rows_from_text(t["study_text"])
        opts = [tv.view_rows_from_text(o["text"]) for o in t["options"]]
        assert tv.solve(study, opts)[0] == t["correct_choice"], t["id"]
        if not bench["identity_given"]:
            assert "cone" not in t["study_text"]


# --------------------------------------------------------------------------- #
# the prompt
# --------------------------------------------------------------------------- #

def test_prompt_states_the_heading_is_unknown_and_never_states_the_turn():
    from bench.evaluate_vlm import get_text_prompt
    for identity in (True, False):
        p = get_text_prompt(2, identity, style="neutral").lower()
        assert "unknown direction" in p
        assert "45" not in p and "degree" not in p
    assert "not named" in get_text_prompt(2, False).lower()
    assert "same name" in get_text_prompt(2, True).lower()


@pytest.mark.parametrize("tag", ["id", "noid"])
def test_message_contains_every_view_exactly_once(tag):
    if not os.path.exists(FILES[tag]):
        pytest.skip(f"{FILES[tag]} not built")
    from bench.evaluate_vlm import build_text_message
    t = json.load(open(FILES[tag]))["trials"][7]
    msg = build_text_message(t, "neutral")
    assert msg.count("=== STUDY VIEW ===") == 1
    for i in range(1, t["n_options"] + 1):
        assert f"OPTION {i}:" in msg
    assert "Final Answer: Option X" in msg
