"""
How many metres is a degree?

    python bench/exchange.py --bank data/scenes/c0_shape_colour \
        --model vit_base_patch14_dinov2.lvd142m

Two images of a place differ for exactly two reasons: the place changed, or you
moved. Both are measured here in the same currency -- cosine distance in a
frozen encoder's embedding -- which lets them be traded against each other:

    D_view(delta)   the same place, seen from delta degrees away
    D_scene(d)      the same viewpoint, one landmark moved d metres

    lambda(delta) = the d at which D_scene(d) == D_view(delta)

**lambda is in metres.** It answers: *how far must the landmarks physically move
to confuse the model as much as walking `delta` degrees around the scene does?*

**Small lambda means viewpoint is cheap.** A model with a genuine allocentric
representation barely notices the walk, so only a small displacement is needed
to cost it the same: a lambda of a metre or two. A model matching appearance is
thrown by the walk, and the landmarks would have to be rebuilt somewhere else
entirely before a scene change cost as much: lambda of tens of metres, or off
the top of the ladder. Both ends are pinned by construction in
`tests/test_exchange.py` -- a layout-only embedding falls below the ladder, a
camera-only embedding above it -- because the sign of this claim is the one
error that would point every number in the paper the same wrong way.

Because the units are metres, the number is comparable across architectures,
across the cue ladder, and against a human threshold measured the same way.

Both curves are computed on the *same* anchors and the *same* appearance
contrast, so the comparison is within-scene throughout. A probe shares its
anchor's identities and all but one of its positions, which is what makes
D_scene a scene-distance curve rather than a scene-identity one.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import angular_separation, bank_frames, embed_bank


def _cos(a, b):
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    return float(1.0 - a @ b)


def curves(vec, bank, q_app="A", g_app="B"):
    """
    (D_view, D_scene) as {delta: [distances]} and {requested_d: [distances]}.

    Every distance is anchor-under-A against something-under-B, so the
    appearance change is present in both curves and cancels when they are
    compared. Leaving it out of one would make that curve artificially small and
    lambda correspondingly wrong.
    """
    scenes = bank["scenes"]
    anchors = [s for s, m in scenes.items() if m.get("role") == "anchor"]
    azimuths = sorted({k[2] for k in vec})

    d_view = defaultdict(list)
    for a in anchors:
        for x in azimuths:
            q = vec.get((a, q_app, x))
            if q is None:
                continue
            for y in azimuths:
                g = vec.get((a, g_app, y))
                if g is None:
                    continue
                d_view[angular_separation(x, y)].append(_cos(q, g))

    d_scene = defaultdict(list)
    for sid, meta in scenes.items():
        if meta.get("role") != "probe":
            continue
        a = meta["anchor"]
        req = meta.get("requested_d_pos", meta.get("d_pos"))
        for x in azimuths:
            q = vec.get((a, q_app, x))
            g = vec.get((sid, g_app, x))     # same bearing: viewpoint held fixed
            if q is None or g is None:
                continue
            d_scene[float(req)].append(_cos(q, g))
    return d_view, d_scene


def _mean_curve(d):
    xs = sorted(d)
    return np.array(xs, dtype=float), np.array([np.mean(d[x]) for x in xs])


def lam_bound(d_view, d_scene, delta):
    """
    (lambda in metres, bound) -- bound is None when measured, else "below" or
    "above".

    A crossing off the end of the ladder is not one outcome but two opposite
    ones, and for a saturated model it *is* the result: "below" means viewpoint
    costs less than the smallest scene change we rendered (the allocentric end),
    "above" means it costs more than any of them (the appearance end). Collapsing
    both to a bare None throws away the direction, and clamping either to an
    endpoint would turn "we did not measure this" into a number.
    """
    if delta not in d_view:
        return None, None
    target = float(np.mean(d_view[delta]))
    xs, ys = _mean_curve(d_scene)
    if len(xs) < 2:
        return None, None
    if target <= ys[0]:
        return None, "below"
    if target >= ys[-1]:
        return None, "above"
    i = int(np.searchsorted(ys, target))
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if y1 == y0:
        return float(x1), None
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0)), None


def lam(d_view, d_scene, delta):
    """lambda(delta) in metres, or None when it falls off the probe ladder."""
    return lam_bound(d_view, d_scene, delta)[0]


def bootstrap_lambda(vec, bank, delta, n_boot=400, seed=0):
    """lambda with a percentile CI, resampling anchors (the unit of replication)."""
    rng = np.random.default_rng(seed)
    scenes = bank["scenes"]
    anchors = sorted(s for s, m in scenes.items() if m.get("role") == "anchor")
    out = []
    for _ in range(n_boot):
        pick = set(rng.choice(anchors, size=len(anchors), replace=True))
        sub = {"scenes": {s: m for s, m in scenes.items()
                          if (m.get("role") == "anchor" and s in pick)
                          or (m.get("role") == "probe" and m["anchor"] in pick)}}
        dv, ds = curves(vec, sub)
        v = lam(dv, ds, delta)
        if v is not None:
            out.append(v)
    if len(out) < 0.5 * n_boot:
        return None, None                # too often unmeasurable to quote a CI
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def score(bank_dir, model_id, batch=64, workers=8, n_boot=400):
    with open(os.path.join(bank_dir, "bank.json")) as f:
        bank = json.load(f)
    vec = embed_bank(bank_dir, bank, model_id, batch=batch, workers=workers)
    d_view, d_scene = curves(vec, bank)

    res = {"bank": os.path.basename(os.path.abspath(bank_dir)),
           "model": model_id,
           "stimulus_mode": bank.get("stimulus_mode"),
           "n_objects": bank.get("n_objects"),
           "n_anchors": sum(1 for m in bank["scenes"].values()
                            if m.get("role") == "anchor"),
           "d_view": {str(k): float(np.mean(v)) for k, v in sorted(d_view.items())},
           "d_scene": {str(k): float(np.mean(v)) for k, v in sorted(d_scene.items())},
           "lambda": {}}
    for delta in sorted(d_view):
        if delta == 0:
            continue
        point, bound = lam_bound(d_view, d_scene, delta)
        lo, hi = (None, None) if point is None else \
            bootstrap_lambda(vec, bank, delta, n_boot=n_boot)
        res["lambda"][str(delta)] = {"m": point, "lo": lo, "hi": hi,
                                     "bound": bound}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = score(args.bank, args.model, batch=args.batch, n_boot=args.boot)
    print(f"\n{res['bank']}  |  {res['model']}  |  {res['n_anchors']} anchors")
    print("  viewpoint cost      scene cost")
    print(f"  {'delta':>6} {'D_view':>8}    {'d (m)':>6} {'D_scene':>8}")
    dv = sorted(res["d_view"].items(), key=lambda kv: float(kv[0]))
    ds = sorted(res["d_scene"].items(), key=lambda kv: float(kv[0]))
    for i in range(max(len(dv), len(ds))):
        left = f"  {dv[i][0]:>6} {dv[i][1]:>8.4f}" if i < len(dv) else " " * 17
        right = f"    {ds[i][0]:>6} {ds[i][1]:>8.4f}" if i < len(ds) else ""
        print(left + right)
    print("\n  lambda -- metres of landmark movement that cost as much as the turn")
    ladder = sorted(float(k) for k in res["d_scene"])
    for delta, v in sorted(res["lambda"].items(), key=lambda kv: float(kv[0])):
        if v["m"] is None:
            side = {"below": f"< {ladder[0]:g} m  (the turn is cheaper than the "
                             f"smallest scene change: allocentric end)",
                    "above": f"> {ladder[-1]:g} m  (the turn costs more than any "
                             f"scene change we rendered: appearance end)"}
            print(f"    {delta:>4}deg : {side.get(v.get('bound'), 'unmeasured')}")
        else:
            ci = "" if v["lo"] is None else f"  [95% CI {v['lo']:.1f}-{v['hi']:.1f}]"
            print(f"    {delta:>4}deg : {v['m']:5.1f} m{ci}")

    out = args.out or os.path.join(args.bank, "exchange.json")
    prior = []
    if os.path.exists(out):
        try:
            with open(out) as f:
                prior = [r for r in json.load(f) if model_key(r) != model_key(res)]
        except (ValueError, KeyError, TypeError):
            prior = []
    with open(out, "w") as f:
        json.dump(prior + [res], f, indent=2)
    print(f"\n[exchange] -> {out}  (1 new, {len(prior)} kept)")


def model_key(res):
    """Identity of a result row: a model scored on a bank. Keying on the model
    alone is enough for the default per-bank file, but drops the other four
    modes the moment --out points at a shared one."""
    return (res.get("model"), res.get("bank"))


if __name__ == "__main__":
    main()
