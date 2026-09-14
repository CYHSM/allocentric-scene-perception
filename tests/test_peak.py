"""
Tests for the parametric peak shape space.

The invariants here are what let the foil families stay separable: a peak's
summit must be exactly its stated height (so `identity` cannot leak into the
size channel), its base must close at Z == 0 on a circle (so it seats on the
valley floor), and distance in form space must actually control how different
two peaks look.
"""

import numpy as np
import pytest

import fm_peak as P


@pytest.fixture
def rng():
    return np.random.default_rng(20260907)


def test_summit_is_exactly_the_requested_height(rng):
    """
    `height` is the summit altitude by definition.

    An earlier version let the surface texture lift the ring just below the apex
    a hair above it, and a leaning needle built on a base-centred grid came out
    *short* because the grid never sampled the apex at all.
    """
    for i in range(120):
        form = P.sample_form(rng)
        size = P.sample_size(rng)
        _, _, Z = P.build_peak(form, size["height"], size["width"],
                               rot_z=rng.uniform(0, 2 * np.pi), seed=i)
        assert Z.max() == pytest.approx(size["height"], abs=1e-6)


def test_base_closes_at_zero_on_a_circle(rng):
    """The outermost ring must be flat at Z == 0 and circular, or the peak
    either floats above the valley floor or Z-fights with it."""
    for i in range(60):
        form = P.sample_form(rng)
        size = P.sample_size(rng)
        X, Y, Z = P.build_peak(form, size["height"], size["width"],
                               rot_z=rng.uniform(0, 2 * np.pi), seed=i)
        assert np.abs(Z[-1, :]).max() < 1e-12
        assert np.hypot(X[-1], Y[-1]).max() == pytest.approx(size["width"], abs=1e-6)
        assert (Z >= 0.0).all()


def test_peaks_are_never_spires(rng):
    """
    Height / base radius stays near 1.

    Sampling height and width independently allowed a 35 m summit on an 11 m
    base -- a 2.8:1 spire. That single ratio, not any surface detail, is what
    made the rendered peaks read as tipis.
    """
    for _ in range(400):
        s = P.sample_size(rng)
        assert 0.7 <= s["height"] / s["width"] <= 1.4


def test_shape_distance_is_a_metric(rng):
    a, b, c = (P.sample_form(rng) for _ in range(3))
    assert P.shape_distance(a, a) == pytest.approx(0.0)
    assert P.shape_distance(a, b) == pytest.approx(P.shape_distance(b, a))
    assert (P.shape_distance(a, c)
            <= P.shape_distance(a, b) + P.shape_distance(b, c) + 1e-9)


def test_substitution_lands_near_the_requested_distance(rng):
    """Identity's difficulty dial has to actually be a dial."""
    for target in (0.5, 0.8, 1.1, 1.4):
        got = []
        for _ in range(40):
            base = P.sample_form(rng)
            sub = P.substitute_form(rng, base, target)
            assert sub is not None, f"no substitution found at distance {target}"
            got.append(P.shape_distance(base, sub))
        assert np.mean(got) == pytest.approx(target, abs=0.15)


def test_scene_peaks_are_mutually_separated(rng):
    """
    Peaks in one scene must be individually recognisable.

    If two are near-identical the `binding` family -- which asks which landmark
    stands where -- has no answer, and the item is unanswerable rather than hard.
    """
    for _ in range(80):
        forms = P.sample_distinct_forms(rng, 4)
        for i, a in enumerate(forms):
            for b in forms[i + 1:]:
                assert P.shape_distance(a, b) >= P.MIN_SEPARATION - 1e-9


def test_substitution_avoids_the_other_peaks(rng):
    """A substituted peak must not become a copy of one already in the scene."""
    for _ in range(60):
        forms = P.sample_distinct_forms(rng, 4)
        sub = P.substitute_form(rng, forms[0], 1.0, avoid=forms[1:])
        if sub is None:
            continue
        for other in forms[1:]:
            assert P.shape_distance(sub, other) >= P.MIN_SEPARATION - 1e-9


def test_form_distance_controls_how_different_a_peak_actually_is(rng):
    """
    Distance in form space must translate into visible difference, measured
    against the noise floor of the peak's own rock texture.

    A silhouette-only metric barely registers this (the outline is dominated by
    height and width, which identity foils deliberately hold fixed), so the
    comparison is made on the surface itself -- which is what shading reveals
    and what a vision model actually sees.
    """
    def field(form, h, w, seed=7):
        _, _, Z = P.build_peak(form, h, w, seed=seed, n_r=90, n_theta=180)
        return Z / h

    # Baseline: the same landform with different rock detail.
    floor = []
    for _ in range(30):
        f = P.sample_form(rng)
        s = P.sample_size(rng)
        floor.append(np.abs(field(f, s["height"], s["width"], seed=7)
                            - field(f, s["height"], s["width"], seed=99)).mean())
    floor = float(np.mean(floor))

    got = {}
    for d in (0.8, 1.4):
        vals = []
        for _ in range(30):
            f = P.sample_form(rng)
            s = P.sample_size(rng)
            g = P.substitute_form(rng, f, d)
            if g is not None:
                vals.append(np.abs(field(f, s["height"], s["width"])
                                   - field(g, s["height"], s["width"])).mean())
        got[d] = float(np.mean(vals))

    # Both ladder rungs must clear the texture noise floor, and the harder rung
    # must be measurably smaller than the easier one or the dial does nothing.
    assert got[0.8] > floor, "identity(0.8) is lost in the rock texture"
    assert got[1.4] > 1.6 * floor
    assert got[1.4] > got[0.8]


def test_describe_is_stable_and_readable(rng):
    for _ in range(50):
        form = P.sample_form(rng)
        text = P.describe(form)
        assert text and text == P.describe(form)
        assert "/" in text
