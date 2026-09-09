"""
The bank plan, and the pairing it exists to guarantee.

The dataset's whole design is that one set of layouts is rendered in all five
stimulus modes, so `a007` in c0 and `a007` in c4 are the same place. Every
cross-mode comparison in the paper depends on that, and none of it is visible in
the rendered images -- two banks with different layouts look perfectly fine side
by side. So it is checked here rather than assumed.
"""

import itertools
import numpy as np
import pytest

import fm_bank
import fm_layout as layoutlib
import fm_scenedist as D
import fm_stimulus as S

MODES = tuple(S.MODES)


@pytest.fixture(scope="module")
def plan():
    return fm_bank.plan_bank(seed0=0, n_anchors=8, probes_per_anchor=3)


# --------------------------------------------------------------------------- #
# The pairing
# --------------------------------------------------------------------------- #

def test_the_plan_does_not_depend_on_the_stimulus_mode():
    """
    `plan_bank` takes no mode argument, and must not acquire one. If the layouts
    ever became mode-dependent the five renders would silently stop being the
    same places -- and the images would give no sign of it.
    """
    import inspect
    params = set(inspect.signature(fm_bank.plan_bank).parameters)
    assert "mode" not in params and "stimulus_mode" not in params


def test_the_same_seed_gives_the_same_layouts_every_time(plan):
    again = fm_bank.plan_bank(seed0=0, n_anchors=8, probes_per_anchor=3)
    assert [e["layout"]["layout_id"] for e in plan] == \
           [e["layout"]["layout_id"] for e in again]
    for a, b in zip(plan, again):
        assert D.d_pos(a["layout"], b["layout"]) == pytest.approx(0.0, abs=1e-12)


def test_every_mode_renders_the_same_places(plan):
    """
    Assigning identities must never move a landmark. This is the invariant that
    makes `a007` in c0 and `a007` in c4 comparable.
    """
    for entry in plan[:6]:
        base = entry["layout"]
        laid = [S.assign_objects(base, np.random.default_rng(3), m)
                for m in MODES]
        for lay in laid:
            assert D.d_pos(base, lay) == pytest.approx(0.0, abs=1e-9)
        for a, b in zip(laid, laid[1:]):
            assert D.d_pos(a, b) == pytest.approx(0.0, abs=1e-9)


def test_scene_ids_are_identical_across_modes(plan):
    """The id is what joins the five banks; it comes from the plan, which is
    mode-free, so it cannot drift."""
    ids = [e["layout"]["layout_id"] for e in plan]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("a") for i in ids)


# --------------------------------------------------------------------------- #
# Anchors are distinct places
# --------------------------------------------------------------------------- #

def test_no_two_anchors_are_the_same_place(plan):
    """A retrieval miss between two anchors that are really one place would be
    the pool's fault, and would read as a model failure."""
    anchors = [e["layout"] for e in plan if e["role"] == "anchor"]
    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            assert not D.same_place(a, b)


def test_anchor_sampling_fails_loudly_rather_than_returning_a_short_pool():
    """
    Ask for a separation the ring geometry cannot deliver and the sampler must
    raise, not hand back fewer anchors than asked for. A silently smaller
    gallery would change the retrieval chance level without saying so.

    This asks for it honestly rather than by monkeypatching the rejection
    predicate: the previous version patched `pool_is_separable`, which
    `sample_anchors` no longer calls -- it now tests `d_pos` directly, because
    `same_place` short-circuits on identity and its floor never bound.
    """
    with pytest.raises(RuntimeError, match="apart"):
        fm_bank.sample_anchors(0, 8, 4, 45, separation_m=200.0)


def test_every_probe_lands_at_its_requested_distance(plan):
    for e in plan:
        if e["role"] != "probe":
            continue
        anchor = next(x["layout"] for x in plan
                      if x["layout"]["layout_id"] == e["anchor"])
        assert D.d_pos(anchor, e["layout"]) == pytest.approx(
            e["requested_d_pos"], rel=0.05)
        assert e["d_pos"] == pytest.approx(D.d_pos(anchor, e["layout"]))


def test_probes_change_only_position(plan):
    """A probe must differ from its anchor in geometry alone. If it also changed
    identity, `D_scene(d)` would be measuring two things and lambda would not be
    in metres of anything."""
    for e in plan:
        if e["role"] != "probe":
            continue
        anchor = next(x["layout"] for x in plan
                      if x["layout"]["layout_id"] == e["anchor"])
        assert D.d_id(anchor, e["layout"]) == pytest.approx(0.0, abs=1e-9)


def test_the_probe_ladder_is_covered_for_every_probed_anchor(plan):
    probed = {e["anchor"] for e in plan if e["role"] == "probe"}
    for a in probed:
        got = sorted(e["requested_d_pos"] for e in plan
                     if e["role"] == "probe" and e["anchor"] == a)
        assert got == sorted(fm_bank.PROBE_LADDER)


def test_only_the_first_k_anchors_are_probed():
    p = fm_bank.plan_bank(seed0=0, n_anchors=6, probes_per_anchor=2)
    probed = {e["anchor"] for e in p if e["role"] == "probe"}
    assert probed == {fm_bank.anchor_id(0), fm_bank.anchor_id(1)}
    assert sum(1 for e in p if e["role"] == "anchor") == 6


def test_a_probe_inherits_its_anchors_identities(plan):
    """
    The bug this guards: the identity draw was seeded on the *scene* id, so
    every probe got its own objects and `a000p4` was a different scene rather
    than `a000` with one landmark moved. `D_scene(d)` would then have measured
    identity change while being reported in metres, and lambda would have been
    meaningless -- with nothing in the images to show it, since each probe
    looked like a perfectly good scene on its own.
    """
    by_id = {e["layout"]["layout_id"]: e for e in plan}
    for e in plan:
        if e["role"] != "probe":
            continue
        anchor = by_id[e["anchor"]]["layout"]
        for mode in MODES:
            a = S.assign_objects(
                anchor, np.random.default_rng(fm_bank.identity_seed(e["anchor"])),
                mode)
            p = S.assign_objects(
                e["layout"],
                np.random.default_rng(fm_bank.identity_seed(e["anchor"])), mode)
            assert D.d_id(a, p) == pytest.approx(0.0, abs=1e-9), mode
            assert D.d_bind(a, p) == 0.0, mode
            assert D.d_pos(a, p) == pytest.approx(e["requested_d_pos"], rel=0.05)


def test_identity_seed_is_stable_across_processes():
    """`hash()` is randomised per process; a run-dependent seed would give the
    five modes different objects and break the pairing invisibly."""
    assert fm_bank.identity_seed("a007") == fm_bank.identity_seed("a007")
    assert fm_bank.identity_seed("a007") != fm_bank.identity_seed("a008")


def test_one_anchors_ladder_moves_a_single_landmark(plan):
    """
    An anchor's probe ladder must be one trajectory. Redrawing the landmark at
    every rung made the cube move at 1-4 m and the cylinder at 8-24 m, so
    D_scene(d) mixed two different manipulations and carried variance that had
    nothing to do with distance. Averaging over which landmark moves still
    happens -- across anchors, where it belongs.
    """
    by_anchor = {}
    for e in plan:
        if e["role"] == "probe":
            by_anchor.setdefault(e["anchor"], set()).add(e["moved"])
    assert by_anchor, "the plan has no probes"
    for anchor, moved in by_anchor.items():
        assert len(moved) == 1, f"{anchor} moved {sorted(moved)}"


def test_different_anchors_do_not_all_move_the_same_landmark(plan):
    """...but it must not be the same landmark every time, or the ladder would
    only ever probe one position on the ring."""
    moved = {e["moved"] for e in plan if e["role"] == "probe"}
    assert len(moved) > 1


# --------------------------------------------------------------------------- #
# The canonical landmark set: what makes this a benchmark about place
# --------------------------------------------------------------------------- #

def _laid_out(plan, mode, k=24):
    import fm_stimulus as stim
    return [stim.assign_objects(
                e["layout"],
                np.random.default_rng(fm_bank.identity_seed(e["anchor"])), mode)
            for e in plan[:k]]


@pytest.mark.parametrize("mode", ["c0_shape_colour", "c1_shape", "c2_colour",
                                  "c3_peaks_bare", "c4_valley"])
def test_every_scene_contains_the_same_landmarks(mode):
    """
    The load-bearing property. If scenes differed in *which* landmarks they
    hold, a model could recognise a place by spotting the purple sphere from any
    angle, and cross-view retrieval would measure viewpoint-invariant object
    recognition while being reported as spatial memory.
    """
    plan = fm_bank.plan_bank(seed0=0, n_anchors=12)
    laid = _laid_out(plan, mode, k=12)
    for a, b in itertools.combinations(laid, 2):
        assert D.d_id(a, b) == pytest.approx(0.0), \
            f"{mode}: two scenes hold different landmark sets"


def test_which_landmark_stands_where_still_varies():
    """The set is fixed; the binding is not. Otherwise every scene would be the
    same arrangement of the same objects and only the positions would move as a
    rigid whole."""
    plan = fm_bank.plan_bank(seed0=0, n_anchors=24)
    laid = _laid_out(plan, "c0_shape_colour")
    bindings = {tuple(str(i) for i in D._identities(l)) for l in laid}
    assert len(bindings) > 1, "every scene used the same landmark ordering"


def test_the_binding_is_the_same_in_every_mode():
    """
    A landmark standing at a given place must be *the same landmark* in all five
    renders, or the modes are not paired and no cross-mode comparison is valid.

    The identities themselves are not comparable across modes -- c0 holds a
    shape/colour pair and c3 a form vector -- so what has to match is the
    permutation: the canonical set is indexed the same way everywhere, so slot
    k must receive canonical entry `order[k]` in every mode. Comparing the peak
    *coordinates* instead, as an earlier version of this test did, checks only
    that the layouts were not reordered, which `test_every_mode_renders_the_same_places`
    already covers -- it would pass even if c0 and c3 disagreed completely about
    which landmark stands where.
    """
    import fm_stimulus as stim
    plan = fm_bank.plan_bank(seed0=0, n_anchors=8)
    n = len(plan[0]["layout"]["peaks"])
    objs, forms = stim.canonical_objects(n), stim.canonical_forms(n)

    for e in plan[:8]:
        seed = fm_bank.identity_seed(e["anchor"])
        c0 = stim.assign_objects(e["layout"], np.random.default_rng(seed),
                                 "c0_shape_colour")
        c3 = stim.assign_objects(e["layout"], np.random.default_rng(seed),
                                 "c3_peaks_bare")
        order_c0 = [objs.index({"shape": p["obj"]["shape"],
                                "colour": p["obj"]["colour"]})
                    for p in c0["peaks"]]
        order_c3 = [forms.index(p["form"]) for p in c3["peaks"]]
        assert order_c0 == order_c3, (
            f"{e['anchor']}: c0 places landmarks in order {order_c0} but c3 "
            f"uses {order_c3} -- the same place holds different landmarks")
        assert sorted(order_c0) == list(range(n)), \
            "a landmark was used twice or omitted"


def test_anchors_are_actually_separated_in_space():
    """
    The floor must bind. It previously did not: `same_place` returns False the
    moment identities differ, so with per-seed forms the positional test was
    never reached and the bank held anchor pairs 4.8 m apart.
    """
    plan = fm_bank.plan_bank(seed0=0, n_anchors=30, separation_m=12.0)
    lays = [e["layout"] for e in plan]
    worst = min(D.d_pos(a, b)
                for a, b in itertools.combinations(lays, 2))
    assert worst >= 12.0 - 1e-6, f"closest anchor pair is only {worst:.1f} m apart"


def test_the_plan_is_anchors_only_by_default():
    """The one-object ladder is off unless asked for: it measures whether a
    single object moved, which needs no map."""
    plan = fm_bank.plan_bank(seed0=0, n_anchors=10)
    assert {e["role"] for e in plan} == {"anchor"}
    assert len(plan) == 10
