"""
What is in the bank, decided before anything is rendered.

Kept separate from `render_bank.py` for one reason: that module imports Blender,
so nothing in it can be unit-tested. The plan is the part that has to be right --
if the layouts differ between modes, the five renders are not the same places and
every cross-mode number in the paper is meaningless -- so it lives here, in plain
Python, and `bench/tests/test_bank.py` checks it.

The plan is a function of `(seed0, n_anchors, n_peaks, azimuth_step,
probe_ladder, probes_per_anchor)` and **nothing else**. In particular it does not
take a stimulus mode: that is what makes `a007` in c0 and `a007` in c4 the same
place.
"""

import binascii

import numpy as np

import fm_layout as layoutlib
import fm_scenedist as scenedist

# One landmark moved this far, in metres. Roughly logarithmic so the curve is
# sampled where it bends: below ~4 m the change is expected to be invisible,
# above ~16 m obviously a different place, and the interesting region is between.
PROBE_LADDER = (1.0, 2.0, 4.0, 8.0, 16.0, 24.0)

# Minimum bottleneck displacement between any two scenes in the gallery.
# Four landmarks on a 36-45 m ring is a packing problem: 100 scenes fit at 12 m,
# 50 at 16 m, 26 at 20 m. 12 m keeps the gallery large enough for a meaningful
# chance level (1%) while the median pair sits around 32 m apart.
SEPARATION_M = 12.0


def anchor_id(i):
    return f"a{i:03d}"


def identity_seed(anchor):
    """
    The seed that decides which landmark is which, keyed on the **anchor**.

    A probe must inherit its anchor's identities. Seeding on the scene id
    instead gave every probe its own draw, so `a000p4` was a different set of
    objects rather than `a000` with one of them moved -- and `D_scene(d)` would
    have measured identity change while being reported in metres.
    """
    return binascii.crc32(anchor.encode())


def probe_id(anchor, d):
    return f"{anchor}p{d:g}"


def sample_anchors(seed0, n_anchors, n_peaks, azimuth_step,
                   separation_m=SEPARATION_M):
    """`n_anchors` layouts no two of which are the same place.

    Rejection on `d_pos` **directly**, not through `scenedist.same_place`. That
    helper returns False the moment two layouts differ in identity, and at plan
    time a layout still carries the per-seed forms `sample_layout` gave it -- so
    the test short-circuited and the separation floor never bound. The bank it
    produced had anchor pairs 4.8 m apart, which the canonical landmark set now
    makes indistinguishable by anything but position.

    Identity is assigned later and is the *same* in every scene, so position is
    the only thing that separates two anchors and the only thing worth testing.
    """
    out, seed = [], seed0
    budget = 400 * n_anchors
    while len(out) < n_anchors:
        if seed - seed0 > budget:
            raise RuntimeError(
                f"only found {len(out)}/{n_anchors} anchors at least "
                f"{separation_m} m apart after {seed - seed0} seeds; the ring "
                f"geometry cannot pack that many that far apart")
        try:
            lay = layoutlib.sample_layout(seed, n_peaks=n_peaks,
                                          azimuth_step=azimuth_step)
        except Exception:
            seed += 1
            continue
        seed += 1
        if any(scenedist.d_pos(a, lay) < separation_m for a in out):
            continue
        lay["layout_id"] = anchor_id(len(out))
        out.append(lay)
    return out


def plan_bank(seed0=0, n_anchors=100, n_peaks=4, azimuth_step=45,
              probe_ladder=PROBE_LADDER, probes_per_anchor=0,
              separation_m=SEPARATION_M):
    """
    Every scene in the bank, as `[{layout, role, anchor, d_pos, ...}]`.

    Mode-independent by construction; identity is assigned later, per mode, and
    is the same set in every scene -- so two scenes differ only in where the
    landmarks stand and in which one stands where.

    `probes_per_anchor` defaults to 0: the one-landmark displacement ladder is
    off by default. It measures how far a *single* object must move to be
    noticed, which a model can do by watching one object's relation to its
    neighbours -- no map required. The benchmark proper is the gallery of
    mutually distinct places.
    """
    anchors = sample_anchors(seed0, n_anchors, n_peaks, azimuth_step,
                             separation_m=separation_m)
    cams = layoutlib._cameras(layoutlib.benchmark_azimuths(azimuth_step))

    out = [{"layout": lay, "role": "anchor", "anchor": lay["layout_id"],
            "d_pos": 0.0, "requested_d_pos": 0.0, "d_bind": 0.0, "d_id": 0.0}
           for lay in anchors]

    for i, lay in enumerate(anchors[:probes_per_anchor]):
        rng = np.random.default_rng(1_000_003 * (seed0 + 1) + i)
        # One landmark per anchor, so the anchor's ladder is a single
        # trajectory rather than a different object at every rung.
        moved = str(rng.choice([p["name"] for p in lay["peaks"]]))
        for d in probe_ladder:
            probe, got = scenedist.sample_probe(lay, rng, float(d), cams,
                                                name=moved)
            probe["layout_id"] = probe_id(lay["layout_id"], d)
            out.append({"layout": probe, "role": "probe",
                        "anchor": lay["layout_id"], "d_pos": got["d_pos"],
                        "requested_d_pos": float(d),
                        "d_bind": got["d_bind"], "d_id": got["d_id"],
                        "moved": moved})
    return out
