"""
The metric that lets a 1%-chance retrieval score and a 50%-chance 2AFC score sit
on one axis.

The headline test is `test_chance_is_zero_on_every_scale`: if d' did not return
exactly 0 at chance for every m, VII would be a ratio of two small biased
numbers and the whole cross-arm comparison would be an artefact of the arm.
"""

import math
from statistics import NormalDist

import pytest

import vii as V


# --------------------------------------------------------------------------- #
# d' has to be right before anything built on it means anything
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("m", [2, 3, 4, 8, 100, 1000])
def test_chance_is_zero_on_every_scale(m):
    """d' = 0 must give exactly 1/m correct, or the arms are not commensurable."""
    assert V.pc_from_dprime(0.0, m) == pytest.approx(1.0 / m, abs=2e-5)
    assert V.dprime(1.0 / m, m) == 0.0


@pytest.mark.parametrize("p", [0.55, 0.6, 0.75, 0.9, 0.99])
def test_two_afc_matches_the_closed_form(p):
    """For m = 2 the answer is sqrt(2) * z(p) and there is no excuse to differ."""
    assert V.dprime(p, 2) == pytest.approx(
        math.sqrt(2) * NormalDist().inv_cdf(p), abs=1e-6)


@pytest.mark.parametrize("m", [2, 4, 16])
@pytest.mark.parametrize("d", [0.25, 1.0, 2.5, 5.0])
def test_dprime_inverts_pc_from_dprime(m, d):
    assert V.dprime(V.pc_from_dprime(d, m), m) == pytest.approx(d, abs=1e-6)


@pytest.mark.parametrize("m", [2, 4, 100])
def test_below_chance_clamps_to_zero_rather_than_going_negative(m):
    """
    A below-chance cell in this dataset means the model is riding a positional
    prior, not that it discriminates in reverse. A negative d' would enter VII
    as a sign flip and read as "anti-invariant", which is not a thing.
    """
    assert V.dprime(0.5 / m, m) == 0.0
    assert V.dprime(0.0, m) == 0.0


def test_a_perfect_cell_is_capped_by_how_many_trials_were_run():
    """100/100 is not infinite sensitivity, it is the ceiling of 100 trials."""
    assert V.dprime(1.0, 4) == float("inf")
    assert math.isfinite(V.dprime(1.0, 4, n_trials=100))
    assert V.dprime(1.0, 4, n_trials=100) > V.dprime(1.0, 4, n_trials=20)


def test_pc_is_monotone_in_d_and_decreasing_in_m():
    assert V.pc_from_dprime(1.0, 4) < V.pc_from_dprime(2.0, 4)
    assert V.pc_from_dprime(1.5, 8) < V.pc_from_dprime(1.5, 2)


# --------------------------------------------------------------------------- #
# VII
# --------------------------------------------------------------------------- #

def test_perfect_invariance_is_one_and_chance_is_zero():
    assert V.vii(0.6, 0.6, 4) == pytest.approx(1.0)
    assert V.vii(0.25, 0.6, 4) == pytest.approx(0.0)


def test_the_same_model_gets_the_same_vii_whichever_arm_it_is_run_on():
    """
    The point of the whole module. A model with sensitivity d0 at delta=0 and
    d45 at delta=45 must yield the same VII whether it was measured by 2AFC,
    4AFC or 100-way retrieval -- otherwise the axis compares task formats and
    not models, and the frontier-VLM column could not be plotted beside the
    frozen-encoder column.
    """
    d0, d45 = 2.2, 0.8
    got = [V.vii(V.pc_from_dprime(d45, m), V.pc_from_dprime(d0, m), m)
           for m in (2, 4, 100)]
    for g in got:
        assert g == pytest.approx(d45 / d0, rel=1e-4)


def test_raw_accuracy_would_have_ranked_them_the_other_way():
    """
    Why the normalisation is not cosmetic. A strong model with a poor viewpoint
    ratio outscores a weak model with a good one on raw accuracy at every delta;
    only VII separates competence from invariance.
    """
    strong = {0: V.pc_from_dprime(3.0, 4), 45: V.pc_from_dprime(0.9, 4)}
    weak = {0: V.pc_from_dprime(1.0, 4), 45: V.pc_from_dprime(0.7, 4)}
    assert strong[45] > weak[45]                       # raw accuracy prefers strong
    assert V.vii_curve(strong, 4)[45] < V.vii_curve(weak, 4)[45]


def test_a_model_with_no_appearance_gate_has_no_vii():
    """Noise over noise is not 1.0, it is undefined, and must not be plotted."""
    assert V.vii(0.25, 0.25, 4) is None
    assert V.vii(0.30, 0.20, 4) is None


def test_vii_curve_refuses_to_guess_a_missing_baseline():
    with pytest.raises(KeyError, match="delta=0"):
        V.vii_curve({45: 0.4, 90: 0.3}, 4)


# --------------------------------------------------------------------------- #
# The positional-prior control
# --------------------------------------------------------------------------- #

def test_a_model_that_always_says_four_is_scored_at_chance_not_credited():
    correct = [1, 2, 3, 4] * 25
    got = V.constant_answer_rate([4] * 100, correct)
    assert got["best_fixed"] == pytest.approx(0.25)
    assert got["distribution"] == pytest.approx(0.25)


def test_it_catches_a_benchmark_whose_answers_are_skewed():
    """
    The failure this exists for: the 2AFC calibration slice had all ten
    delta=180 answers at position 1, so "always answer 1" scored 100% there.
    The control has to say so.
    """
    correct = [1] * 10
    got = V.constant_answer_rate([1] * 10, correct)
    assert got["best_fixed"] == pytest.approx(1.0)
    assert got["distribution"] == pytest.approx(1.0)


def test_a_skewed_model_on_a_balanced_benchmark_still_only_gets_chance():
    """Qwen2.5-VL-7B's actual prior: option 4 on 59% of trials."""
    correct = [1, 2, 3, 4] * 125
    choices = [4] * 295 + [3] * 129 + [2] * 56 + [1] * 20
    got = V.constant_answer_rate(choices, correct)
    assert got["distribution"] == pytest.approx(0.25, abs=1e-9)


def test_unanswered_trials_do_not_count_toward_the_prior():
    got = V.constant_answer_rate([1, None, None, 2], [1, 2, 3, 4])
    assert got["model_prior"][1] == pytest.approx(0.5)


def test_no_trials_fails_loudly():
    with pytest.raises(ValueError):
        V.constant_answer_rate([1], [])
