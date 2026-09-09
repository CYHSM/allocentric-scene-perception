"""
The graded metrics.

4AFC bottoms out at 25% and every interesting cell on this dataset is already
there, so these three carry the trend instead. Each is tested against inputs
whose answer is known analytically, at both ends: a representation that is
perfectly viewpoint-invariant, and one that encodes only the viewpoint.

The scale-invariance tests are not decoration. These numbers are compared
across architectures with different embedding widths and norms, so a metric
that moves when the embedding is rescaled would rank models by their weight
initialisation.
"""

import numpy as np
import pytest

import metrics as M

S, D = 40, 16


@pytest.fixture
def rng():
    return np.random.default_rng(20260908)


@pytest.fixture
def invariant(rng):
    """Scene identity fully preserved across the viewpoint change."""
    base = rng.normal(size=(S, D))
    jitter = 0.02 * rng.normal(size=(S, D))
    return base, base + jitter


@pytest.fixture
def viewpoint_only(rng):
    """
    Every scene collapses onto a per-viewpoint point.

    This is the failure the dataset is built to detect: the embedding encodes
    where the camera was, not what it was looking at.
    """
    vx = rng.normal(size=D)
    vy = rng.normal(size=D)
    return (vx + 0.01 * rng.normal(size=(S, D)),
            vy + 0.01 * rng.normal(size=(S, D)))


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def test_perfect_invariance_retrieves_every_scene(invariant):
    q, g = invariant
    r = M.retrieval(q, g)
    assert r["recall@1"] == 1.0
    assert r["map"] == pytest.approx(1.0)
    assert r["median_rank"] == 1.0
    assert r["n"] == S


def test_identical_inputs_are_the_ceiling(rng):
    """The Delta-0/same-appearance cell is a byte-identical match, and is used
    as a gate on every bank -- so it has to read exactly 1.0."""
    x = rng.normal(size=(S, D))
    assert M.retrieval(x, x)["recall@1"] == 1.0
    assert M.rsa(x, x) == pytest.approx(1.0)


def test_viewpoint_dominance_lands_at_chance(viewpoint_only):
    q, g = viewpoint_only
    r = M.retrieval(q, g)
    assert r["recall@1"] < 4 * r["chance_recall@1"] + 0.02
    assert r["chance_recall@1"] == pytest.approx(1.0 / S)


def test_recall_at_5_is_never_below_recall_at_1(rng):
    q = rng.normal(size=(S, D))
    g = q + 0.7 * rng.normal(size=(S, D))
    r = M.retrieval(q, g)
    assert r["recall@5"] >= r["recall@1"]
    assert 0.0 <= r["map"] <= 1.0


def test_ties_are_scored_against_the_model():
    """A degenerate embedding maps everything to one point. Counting a tie as a
    hit would report that as perfect retrieval."""
    x = np.ones((S, D))
    r = M.retrieval(x, x)
    assert r["recall@1"] == 0.0
    assert r["median_rank"] == float(S)
    assert r["map"] == pytest.approx(1.0 / S)


# --------------------------------------------------------------------------- #
# NVM
# --------------------------------------------------------------------------- #

def test_nvm_is_positive_under_invariance_and_zero_under_collapse(
        invariant, viewpoint_only):
    """
    Collapse gives ~0, not a negative value: if every scene maps to one point
    per viewpoint then the same place and a different place are equidistant.
    Below zero means something stronger -- the same place is reliably *further*
    -- which is what a below-chance 4AFC looks like from the inside.
    """
    assert M.nvm(*invariant) > 0.9
    assert abs(M.nvm(*viewpoint_only)) < 0.05


def test_nvm_goes_negative_only_when_the_signal_is_anti_correlated(rng):
    base = rng.normal(size=(S, D))
    # Pair each scene with a *different* scene's viewpoint-Y embedding, so the
    # correct match is systematically the wrong one.
    shifted = np.roll(base, 1, axis=0) + 0.02 * rng.normal(size=(S, D))
    assert M.nvm(base, shifted) < 0.0


def test_nvm_stays_in_range(rng):
    for scale in (0.05, 0.5, 5.0):
        q = rng.normal(size=(S, D))
        g = q + scale * rng.normal(size=(S, D))
        v = M.nvm(q, g)
        assert -1.0 <= v <= 1.0


def test_the_hardest_distractor_is_never_kinder_than_the_average(invariant):
    q, g = invariant
    assert M.nvm(q, g, hardest=True) <= M.nvm(q, g) + 1e-9


# --------------------------------------------------------------------------- #
# RSA
# --------------------------------------------------------------------------- #

def test_rsa_is_one_when_the_manifold_is_preserved(invariant):
    assert M.rsa(*invariant) > 0.95


def test_rsa_collapses_when_only_the_viewpoint_is_encoded(viewpoint_only):
    assert abs(M.rsa(*viewpoint_only)) < 0.3


def test_rsa_survives_a_rotation_of_the_embedding_space(rng):
    """
    RSA asks about *relative* geometry, so an orthogonal transform of the whole
    space must not change it -- which is exactly why it can answer "is the
    structure still there" where nearest-neighbour retrieval says no.
    """
    x = rng.normal(size=(S, D))
    q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    assert M.rsa(x, x @ q) == pytest.approx(1.0, abs=1e-9)


def test_spearman_matches_a_known_value():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert M._spearman(a, a) == pytest.approx(1.0)
    assert M._spearman(a, list(reversed(a))) == pytest.approx(-1.0)
    # Ties take the average rank, or a constant vector would produce nan noise.
    assert np.isnan(M._spearman(a, [1.0] * 5))


# --------------------------------------------------------------------------- #
# Comparability across models
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_every_metric_is_invariant_to_embedding_scale(rng, scale):
    """These numbers are compared across architectures with different embedding
    widths and norms; a metric that moved with the scale would be ranking
    weight initialisations."""
    q = rng.normal(size=(S, D))
    g = q + 0.4 * rng.normal(size=(S, D))
    ref = M.all_metrics(q, g, view_x=q, view_y=g)
    got = M.all_metrics(q * scale, g * scale, view_x=q * scale, view_y=g * scale)
    for k in ("recall@1", "recall@5", "map", "nvm", "nvm_hardest", "rsa"):
        assert got[k] == pytest.approx(ref[k], abs=1e-9), k


def test_all_metrics_reports_every_field(rng):
    q = rng.normal(size=(S, D))
    g = rng.normal(size=(S, D))
    got = M.all_metrics(q, g, view_x=q, view_y=g)
    assert set(got) >= {"recall@1", "recall@5", "map", "median_rank",
                        "n", "chance_recall@1", "nvm", "nvm_hardest", "rsa"}


def test_angular_separation_wraps():
    assert M.angular_separation(0, 315) == 45
    assert M.angular_separation(0, 180) == 180
    assert M.angular_separation(90, 90) == 0


def test_metrics_json_merges_across_models(tmp_path, monkeypatch):
    """
    A cell is scored by several models in sequence and `grid_report.py` reads
    them all out of one `metrics.json`. A plain write left only the last model,
    silently discarding the rest -- which looks exactly like a cell that was
    only ever scored once.
    """
    import json
    import sys
    import metrics as M

    out = tmp_path / "metrics.json"

    def fake_score_bank(bank, model, appearance="changed", batch=64):
        return {"bank": bank, "model": model, "appearance": appearance,
                "n_scenes": 4, "stimulus_mode": "c0", "n_objects": 2,
                "by_delta": {}}

    monkeypatch.setattr(M, "score_bank", fake_score_bank)

    for model in ("alpha", "beta"):
        monkeypatch.setattr(sys, "argv",
                            ["metrics.py", "--bank", str(tmp_path),
                             "--model", model, "--appearance", "both",
                             "--out", str(out)])
        M.main()

    got = json.load(open(out))
    assert {(r["model"], r["appearance"]) for r in got} == {
        ("alpha", "changed"), ("alpha", "same"),
        ("beta", "changed"), ("beta", "same")}

    # Re-scoring one model replaces only its own rows.
    monkeypatch.setattr(sys, "argv",
                        ["metrics.py", "--bank", str(tmp_path),
                         "--model", "alpha", "--appearance", "changed",
                         "--out", str(out)])
    M.main()
    got = json.load(open(out))
    assert len(got) == 4
    assert sum(r["model"] == "alpha" for r in got) == 2
