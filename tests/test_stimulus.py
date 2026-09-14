"""
The stimulus modes c0..c4.

One set of layouts is rendered in all five modes, so every test here pins one of
the ways that pairing could quietly fail: a size that varies between modes, an
identity assignment that drifts because the random stream diverged, a mode that
does not declare its world.
"""

import numpy as np
import pytest

import fm_layout as layoutlib
import fm_scenedist as D
import fm_stimulus as S

DISCRETE = ("c0_shape_colour", "c1_shape", "c2_colour")
PEAKS = ("c3_peaks_bare", "c4_valley")


@pytest.fixture(scope="module")
def base():
    return layoutlib.sample_layout(7, n_peaks=4, azimuth_step=45)


@pytest.fixture(scope="module")
def cams():
    return layoutlib._cameras(layoutlib.benchmark_azimuths(45))


@pytest.fixture
def rng():
    return np.random.default_rng(20260908)


# --------------------------------------------------------------------------- #
# The pairing: the same place in five worlds
# --------------------------------------------------------------------------- #

def test_positions_are_identical_in_every_mode(base):
    """
    The whole design rests on this. `assign_objects` may change what a landmark
    *is*, never where it stands -- otherwise c0 and c4 are different places and
    no cross-mode comparison means anything.
    """
    ref = None
    for mode in S.MODES:
        lay = S.assign_objects(base, np.random.default_rng(11), mode)
        xy = [(round(p["x"], 9), round(p["y"], 9)) for p in lay["peaks"]]
        if ref is None:
            ref = xy
        assert xy == ref, f"{mode} moved the landmarks"


def test_size_is_uniform_and_identical_in_every_mode(base):
    """
    Apparent size is the image's distance cue. If it varied by mode, a landmark
    would subtend a different angle in c0 than in c3 and the two would not be
    the same landmark seen two ways.
    """
    for mode in S.MODES:
        lay = S.assign_objects(base, np.random.default_rng(11), mode)
        assert {p["height"] for p in lay["peaks"]} == {S.UNIFORM_HEIGHT}, mode
        assert {p["base_radius"] for p in lay["peaks"]} == {S.UNIFORM_RADIUS}, mode


def test_c1_is_c0_with_the_colour_removed(base):
    """
    The cue ladder is only interpretable if the modes are nested: c1 must be the
    same shapes as c0 with colour taken away, and c2 the same colours with shape
    taken away.

    This failed before: the shape and colour permutations were drawn *only when
    the mode used them*, so c2 -- which skips the shape draw -- consumed c0's
    shape permutation as its colours. The modes silently disagreed about which
    landmark was which.
    """
    seed = 11
    c0 = S.assign_objects(base, np.random.default_rng(seed), "c0_shape_colour")
    c1 = S.assign_objects(base, np.random.default_rng(seed), "c1_shape")
    c2 = S.assign_objects(base, np.random.default_rng(seed), "c2_colour")

    assert [p["obj"]["shape"] for p in c0["peaks"]] == \
           [p["obj"]["shape"] for p in c1["peaks"]]
    assert all(p["obj"]["colour"] is None for p in c1["peaks"])

    assert [p["obj"]["colour"] for p in c0["peaks"]] == \
           [p["obj"]["colour"] for p in c2["peaks"]]
    assert len({p["obj"]["shape"] for p in c2["peaks"]}) == 1


def test_c4_is_c3_in_a_different_world(base):
    """c4 must differ from c3 in the world and in nothing else -- otherwise the
    comparison confounds "harder world" with "different landmarks"."""
    c3, c4 = S.MODES["c3_peaks_bare"], S.MODES["c4_valley"]
    for key in ("shape_varies", "colour_varies", "geometry"):
        assert c3[key] == c4[key], key
    assert c3["world"] == "bare" and c4["world"] == "valley"

    a = S.assign_objects(base, np.random.default_rng(11), "c3_peaks_bare")
    b = S.assign_objects(base, np.random.default_rng(11), "c4_valley")
    assert D.d_pos(a, b) == pytest.approx(0.0, abs=1e-9)
    assert D.d_id(a, b) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# The object space
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", DISCRETE)
def test_landmarks_in_a_scene_are_all_distinguishable(base, rng, mode):
    """Two identical landmarks make "which one stands where" unanswerable."""
    for _ in range(30):
        lay = S.assign_objects(base, rng, mode)
        ids = [(p["obj"]["shape"], p["obj"]["colour"]) for p in lay["peaks"]]
        assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("mode", DISCRETE)
def test_only_the_named_cue_varies(base, rng, mode):
    spec = S.MODES[mode]
    lay = S.assign_objects(base, rng, mode)
    shapes = {p["obj"]["shape"] for p in lay["peaks"]}
    colours = {p["obj"]["colour"] for p in lay["peaks"]}
    assert (len(shapes) > 1) == spec["shape_varies"]
    assert (len(colours) > 1) == spec["colour_varies"]


@pytest.mark.parametrize("mode", DISCRETE)
def test_the_form_vector_is_dropped(base, rng, mode):
    """A primitive has no landform; leaving `form` behind would make `fm_text`
    describe a peak that is not in the picture."""
    lay = S.assign_objects(base, rng, mode)
    assert all("form" not in p for p in lay["peaks"])


@pytest.mark.parametrize("mode", PEAKS)
def test_peak_modes_keep_their_form(base, rng, mode):
    lay = S.assign_objects(base, rng, mode)
    assert all(p.get("form") for p in lay["peaks"])
    assert all(p["obj"] is None for p in lay["peaks"])


# --------------------------------------------------------------------------- #
# Worlds
# --------------------------------------------------------------------------- #

def test_every_mode_declares_its_world():
    for name, spec in S.MODES.items():
        assert spec.get("world") in ("bare", "valley"), name


def test_assign_objects_records_the_world(rng):
    layout = layoutlib.sample_layout(7, n_peaks=4, azimuth_step=45)
    for mode in S.MODES:
        out = S.assign_objects(layout, rng, mode)
        assert out["world"] == S.MODES[mode]["world"]


def test_text_does_not_put_a_lake_on_a_bare_plane(rng):
    """
    The serialisers describe an alpine valley with a lake at the origin. On c3
    that is a false statement handed to a model inside the prompt, and it cannot
    be caught by inspecting the landmarks -- c3's are identical to c4's.
    """
    import fm_text as text
    layout = layoutlib.sample_layout(7, n_peaks=4, azimuth_step=45)
    bare = S.assign_objects(layout, rng, "c3_peaks_bare")
    valley = S.assign_objects(layout, rng, "c4_valley")
    assert text._is_bare(bare) is True
    assert text._is_bare(valley) is False
    cam = {"azimuth_deg": 0.0, "elevation_deg": layoutlib.CAM_ELEVATION,
           "radius": layoutlib.CAM_RADIUS, "target_z": layoutlib.CAM_TARGET_Z}
    assert "lake" not in text.to_l1(bare, cam).lower()
    assert "lake" in text.to_l1(valley, cam).lower()


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def test_primitives_sit_on_the_ground_and_reach_their_height():
    for shape in S.SHAPES:
        verts, faces = S._primitive_mesh(shape, 17.0, 19.0)
        zs = [v[2] for v in verts]
        assert min(zs) == pytest.approx(0.0, abs=1e-9)
        assert max(zs) == pytest.approx(19.0, abs=1e-9)
        assert faces and all(len(f) in (3, 4) for f in faces)
        assert max(max(f) for f in faces) < len(verts)


def test_primitive_footprints_stay_inside_the_base_radius():
    """Framing validity is computed from `base_radius`; a wider primitive would
    be clipped in frames the layout sampler passed as valid."""
    for shape in S.SHAPES:
        verts, _ = S._primitive_mesh(shape, 17.0, 19.0)
        r = max((v[0] ** 2 + v[1] ** 2) ** 0.5 for v in verts)
        assert r <= 17.0 + 1e-9


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_every_object_count_yields_a_visible_scene(n, cams):
    """`radius_bounds` derives its packing limit from 2*RING_MIN*sin(pi/n),
    which is zero at n=1 -- a single landmark was once sampled with base radius
    0.0 and rendered as nothing at all."""
    lo, hi = layoutlib.radius_bounds(n)
    assert lo > 0 and hi >= lo
    lay = layoutlib.sample_layout(7, n_peaks=n, azimuth_step=45)
    assert len(lay["peaks"]) == n
    assert all(p["base_radius"] > 1.0 for p in lay["peaks"])
    assert all(p["height"] > 1.0 for p in lay["peaks"])
    assert layoutlib.layout_is_valid(lay["peaks"], cams)
