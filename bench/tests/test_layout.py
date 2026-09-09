"""Layout sampler constraints. Runs in plain Python -- no Blender needed."""

import math

import numpy as np
import pytest

import fm_geom as geom
import fm_layout as L


@pytest.mark.parametrize("n_peaks", [2, 3, 4, 5, 6, 8])
def test_sampler_yields_valid_layouts(n_peaks):
    for seed in range(10):
        lay = L.sample_layout(1000 * n_peaks + seed, n_peaks=n_peaks)
        assert len(lay["peaks"]) == n_peaks
        assert L.layout_is_valid(lay["peaks"])


def test_distinct_morphologies_are_all_different():
    lay = L.sample_layout(0, n_peaks=4, distinctiveness="distinct")
    assert len({p["type"] for p in lay["peaks"]}) == 4


def test_identical_removes_every_shape_cue():
    """Under 'identical' only the arrangement can distinguish target from foil."""
    lay = L.sample_layout(3, n_peaks=4, distinctiveness="identical")
    assert len({p["type"] for p in lay["peaks"]}) == 1
    assert len({round(p["height"], 6) for p in lay["peaks"]}) == 1
    assert len({round(p["base_radius"], 6) for p in lay["peaks"]}) == 1


def test_sampling_is_deterministic():
    a = L.sample_layout(7, n_peaks=4)
    b = L.sample_layout(7, n_peaks=4)
    assert a == b


def test_every_peak_is_in_frame_at_every_benchmark_azimuth():
    lay = L.sample_layout(11, n_peaks=4)
    tgt = np.array([0.0, 0.0, L.CAM_TARGET_Z])
    for az in L.benchmark_azimuths():
        cam = geom.camera_position(az, L.CAM_ELEVATION, L.CAM_RADIUS, L.CAM_TARGET_Z)
        assert geom.layout_in_frame(lay["peaks"], L.BASE_Z, cam, tgt,
                                    L.CAM_LENS_MM, L.RES_X, L.RES_Y,
                                    margin=L.FRAME_MARGIN)


def test_footprints_never_intersect():
    lay = L.sample_layout(5, n_peaks=5)
    for i, a in enumerate(lay["peaks"]):
        for b in lay["peaks"][i + 1:]:
            d = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            assert d >= a["base_radius"] + b["base_radius"]


def test_shipped_canonical_layout_clips_the_frame():
    """
    The hand-tuned valley in generate_scene.py is *not* benchmark-valid: M2
    runs off the right edge at three azimuths. This is why layouts are sampled
    against an explicit framing constraint rather than reused.
    """
    peaks = [
        {"x": -26.05, "y": 21.85, "height": 34.0, "base_radius": 17.0},
        {"x": 26.05, "y": 21.85, "height": 27.0, "base_radius": 19.0},
        {"x": -22.31, "y": -25.66, "height": 24.0, "base_radius": 15.0},
        {"x": 25.27, "y": -22.75, "height": 30.0, "base_radius": 18.0},
    ]
    tgt = np.array([0.0, 0.0, 6.0])
    clipped = [az for az in L.benchmark_azimuths()
               if not geom.layout_in_frame(
                   peaks, L.BASE_Z, geom.camera_position(az, 17.0, 94.0, 6.0),
                   tgt, 28.0, 640, 440, margin=0.0)]
    assert clipped == [90.0, 105.0, 345.0]
