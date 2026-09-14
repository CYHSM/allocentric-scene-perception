"""
Text serialisation.

The benchmark claims its text and image channels carry the same information.
That claim is only as good as these tests: L1 must round-trip to the
coordinates it was built from, and no level may leak the M1..M4 indices, which
are stable across a target and its foils and would give the text channel a
correspondence the image channel has to earn.
"""

import numpy as np
import pytest

import fm_scenedist as D
import fm_layout as L
import fm_text as T

CAMERA = {"azimuth_deg": 45.0, "elevation_deg": 17.0, "radius": 94.0,
          "target_z": 6.0, "lens_mm": 28.0}


@pytest.fixture
def layout():
    return L.sample_layout(0, n_peaks=4)


@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_serialisation_is_deterministic(layout, level):
    assert T.scene_to_text(layout, CAMERA, level) == T.scene_to_text(layout, CAMERA, level)


@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_no_level_leaks_peak_indices(layout, level):
    text = T.scene_to_text(layout, CAMERA, level)
    for name in ("M1", "M2", "M3", "M4"):
        assert name not in text


def test_l1_round_trips_to_the_source_coordinates(layout):
    """The information-equivalence claim, asserted rather than assumed."""
    parsed = T.parse_l1(T.to_l1(layout, CAMERA))
    assert len(parsed) == len(layout["peaks"])
    for p in layout["peaks"]:
        x, y, h = parsed[p["type"]]
        assert abs(x - p["x"]) < 0.06
        assert abs(y - p["y"]) < 0.06
        assert abs(h - p["height"]) < 0.6


def test_l2_lists_peaks_left_to_right_as_the_image_shows_them(layout):
    rows = T.view_geometry(layout, CAMERA)
    bearings = [r["view_bearing_deg"] for r in rows]
    assert bearings == sorted(bearings)


def test_l2_changes_with_viewpoint_but_l1_does_not(layout):
    """
    L1 is a world-frame map, so it is viewpoint-invariant apart from the line
    naming where the viewer stands; L2 is egocentric and must change.
    """
    a = {**CAMERA, "azimuth_deg": 45.0}
    b = {**CAMERA, "azimuth_deg": 180.0}
    assert T.to_l2(layout, a) != T.to_l2(layout, b)
    body_a = "\n".join(l for l in T.to_l1(layout, a).splitlines() if l.startswith("- "))
    body_b = "\n".join(l for l in T.to_l1(layout, b).splitlines() if l.startswith("- "))
    assert body_a == body_b


def test_a_translation_foil_is_visible_in_every_level(layout):
    rng = np.random.default_rng(0)
    foil = D.translate(layout, layout["peaks"][0]["name"], 12.0, 0.0)
    for level in ("L1", "L2", "L3"):
        assert T.scene_to_text(layout, CAMERA, level) != T.scene_to_text(foil, CAMERA, level)


def test_identical_condition_gives_no_shape_cue(layout):
    """Every peak reads the same, so only the arrangement can distinguish them."""
    lay = L.sample_layout(3, n_peaks=4, distinctiveness="identical")
    rows = T.view_geometry(lay, CAMERA)
    assert len({r["type"] for r in rows}) == 1
    assert len({round(r["height"], 6) for r in rows}) == 1
