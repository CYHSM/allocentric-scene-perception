"""
The same lambda, from a model that only emits text.

    python bench/afc.py --bank data/scenes/c0_shape_colour --build items.json
    python bench/afc.py --bank data/scenes/c0_shape_colour --model <timm id>

`exchange.py` reads lambda off two embedding-distance curves. A closed model
behind an API has no embeddings, so that route stops at the models we can
download -- exactly the models least worth the headline. This module gets the
same number, in the same metres, from a forced choice that any model can make.

**The item is the trade-off itself.** Show a reference view of a place, then two
candidates:

    view    the same place, seen `delta` degrees around
    scene   one landmark moved `d` metres, seen from the reference bearing

and ask which one is a *different place*. The answer is `scene`, always. But a
model that leans on appearance sees the rotated view as the bigger change and
picks it instead, and it keeps doing so until `d` grows large enough to
outweigh the turn.

    accuracy(delta, d) = P(the model calls the moved scene the different one)
    lambda(delta)      = the d at which accuracy = 0.5

At that d the model is indifferent between "the place changed" and "I moved":
the point of subjective equality, in metres. It is the same quantity
`exchange.py` interpolates, read per item instead of off two averaged curves.

**Why this bridges the model range.** A frozen encoder answers the identical
item with no free parameter and no threshold to tune -- it picks whichever
candidate is farther away in its own embedding, which is `D_scene(d)` against
`D_view(delta)`, the very comparison lambda is defined by. A generative model
answers by replying "1" or "2". A person answers by pointing. Three
read-outs, one item set, one number in metres. Nothing is calibrated to make
them comparable, so nothing can quietly decalibrate.

Chance is 0.5, not 0.25: a below-chance cell means the model is systematically
choosing the rotated view, which is a direction, not noise.

**The ceiling control.** One extra cell replaces the moved scene with a
*different anchor entirely*. Nothing on the ladder is as different as that, so
any model that can do the task at all should be near 1.0 there. If it is not,
the failure is the task -- prompt, resolution, image ordering -- and not the
model's viewpoint tolerance, and every lambda below it is uninterpretable. It is
the upper counterpart of the Delta-0 identity gate, and it costs one cell.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import angular_separation


def _frame(scene, app, az):
    return (scene.get("frames") or {}).get(f"{app}_{az}")


def build_items(bank, deltas=(45, 90, 135, 180), per_cell=20, seed=0,
                q_app="A", g_app="B", ceiling=True):
    """
    2AFC trade-off items, balanced over (delta, d) cells.

    Both options carry appearance `g_app` against a reference in `q_app`, so the
    appearance change is present in *both* arms and cancels. Putting it in only
    one would make that option gratuitously easier to reject and would shift
    lambda by an amount nothing downstream could detect.
    """
    rng = np.random.default_rng(seed)
    scenes = bank["scenes"]
    azimuths = sorted({int(k.split("_")[1]) for s in scenes.values()
                       for k in (s.get("frames") or {})})
    probes = [(sid, s) for sid, s in scenes.items() if s.get("role") == "probe"]

    pool = defaultdict(list)
    for sid, p in probes:
        anchor = scenes.get(p.get("anchor"))
        if not anchor:
            continue
        d = float(p.get("requested_d_pos", p.get("d_pos", 0.0)))
        for x in azimuths:
            for delta in deltas:
                y = (x + delta) % 360
                if y not in azimuths:
                    continue
                if not (_frame(anchor, q_app, x) and _frame(anchor, g_app, y)
                        and _frame(p, g_app, x)):
                    continue
                pool[(delta, d)].append((p["anchor"], sid, x, y))

    # The ceiling cell: the alternative is a different place outright, not the
    # same place nudged. `d` is None because it is off the ladder by
    # construction -- it is a control, and lambda must never interpolate it.
    if ceiling:
        anchors = sorted(s for s, m in scenes.items() if m.get("role") == "anchor")
        for delta in deltas:
            for x in azimuths:
                y = (x + delta) % 360
                if y not in azimuths or len(anchors) < 2:
                    continue
                for a in anchors:
                    other = anchors[(anchors.index(a) + 1 + int(
                        rng.integers(len(anchors) - 1))) % len(anchors)]
                    if other == a:
                        continue
                    if not (_frame(scenes[a], q_app, x)
                            and _frame(scenes[a], g_app, y)
                            and _frame(scenes[other], g_app, x)):
                        continue
                    pool[(delta, None)].append((a, other, x, y))

    items = []
    for (delta, d) in sorted(pool, key=lambda k: (k[0], -1.0 if k[1] is None
                                                  else k[1])):
        cands = pool[(delta, d)]
        idx = rng.permutation(len(cands))[:per_cell]
        for j in idx:
            anchor, sid, x, y = cands[j]
            view = {"kind": "view", "scene": anchor, "app": g_app, "az": y}
            scene = {"kind": "scene", "scene": sid, "app": g_app, "az": x}
            opts = [view, scene]
            if rng.random() < 0.5:                 # order carries no signal
                opts = opts[::-1]
            items.append({
                "id": f"{anchor}_{sid}_x{x}_d{delta}",
                "delta": int(delta),
                "d": None if d is None else float(d),
                "cell": "ceiling" if d is None else "ladder",
                "anchor": anchor,
                "azimuth": int(x),
                "study": {"scene": anchor, "app": q_app, "az": x},
                "options": opts,
                "answer": 1 + next(i for i, o in enumerate(opts)
                                   if o["kind"] == "scene"),
            })
    return items


def image_path(bank, ref):
    """Bank-relative path of one (scene, appearance, azimuth) frame."""
    fr = _frame(bank["scenes"][ref["scene"]], ref["app"], ref["az"])
    if fr is None:
        raise KeyError(f"no frame {ref['app']}_{ref['az']} for {ref['scene']}")
    return fr["rgb"] if isinstance(fr, dict) else fr


def encoder_answers(items, vec):
    """
    How a frozen encoder answers: the farther candidate is the different place.

    No threshold and no fitting -- the comparison *is* D_scene(d) vs
    D_view(delta), so the encoder arm and the generative arm are answering the
    same question rather than two questions declared equivalent.
    """
    def cos(a, b):
        a = a / max(np.linalg.norm(a), 1e-12)
        b = b / max(np.linalg.norm(b), 1e-12)
        return float(1.0 - a @ b)

    out = {}
    for it in items:
        q = vec.get((it["study"]["scene"], it["study"]["app"], it["study"]["az"]))
        if q is None:
            continue
        ds = []
        for o in it["options"]:
            g = vec.get((o["scene"], o["app"], o["az"]))
            ds.append(None if g is None else cos(q, g))
        if any(v is None for v in ds):
            continue
        out[it["id"]] = 1 + int(np.argmax(ds))
    return out


def accuracy(items, answers):
    """{(delta, d): (correct, n)} over the items that were actually answered."""
    cells = defaultdict(lambda: [0, 0])
    for it in items:
        got = answers.get(it["id"])
        if got is None:
            continue
        cells[(it["delta"], it["d"])][1] += 1
        cells[(it["delta"], it["d"])][0] += int(got == it["answer"])
    return {k: tuple(v) for k, v in cells.items()}


def ceiling(cells, delta=None):
    """Accuracy on the different-place-entirely control, or None if not built."""
    hits = [(c, n) for (dl, d), (c, n) in cells.items()
            if d is None and (delta is None or dl == delta) and n]
    if not hits:
        return None
    return sum(c for c, _ in hits) / sum(n for _, n in hits)


def lam_from_accuracy(cells, delta, target=0.5):
    """
    (lambda, bound) -- the d where accuracy crosses `target`.

    Same contract as `exchange.lam_bound`: None with "below" or "above" when the
    crossing is off the ladder, never clamped to an endpoint. "below" is the
    allocentric end (even a 1 m move already outweighs the turn); "above" is the
    appearance end.
    """
    pts = sorted((d, c / n) for (dl, d), (c, n) in cells.items()
                 if dl == delta and n and d is not None)
    if len(pts) < 2:
        return None, None
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    if ys[0] >= target:
        return None, "below"
    if ys[-1] <= target:
        return None, "above"
    i = int(np.argmax(ys >= target))
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if y1 == y0:
        return float(x1), None
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--build", default=None, help="write items here and stop")
    ap.add_argument("--model", default=None, help="timm id; scores the items")
    ap.add_argument("--per_cell", type=int, default=20)
    ap.add_argument("--deltas", type=int, nargs="+", default=[45, 90, 135, 180])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.bank, "bank.json")) as f:
        bank = json.load(f)
    items = build_items(bank, deltas=tuple(args.deltas),
                        per_cell=args.per_cell, seed=args.seed)
    cells = defaultdict(int)
    for it in items:
        cells[(it["delta"], it["d"])] += 1
    print(f"{len(items)} items over {len(cells)} (delta, d) cells "
          f"({min(cells.values())}-{max(cells.values())} per cell)")

    if args.build:
        for it in items:                       # resolve paths only when writing
            it["study"]["image"] = image_path(bank, it["study"])
            for o in it["options"]:
                o["image"] = image_path(bank, o)
        with open(args.build, "w") as f:
            json.dump({"bank": os.path.basename(os.path.abspath(args.bank)),
                       "items": items}, f, indent=1)
        print(f"[afc] -> {args.build}")
        return

    if not args.model:
        raise SystemExit("give --model to score, or --build to write items")

    from metrics import embed_bank
    vec = embed_bank(args.bank, bank, args.model)
    cells = accuracy(items, encoder_answers(items, vec))

    print(f"\n{args.model} on {os.path.basename(os.path.abspath(args.bank))}")
    ds = sorted({d for _, d in cells if d is not None})
    print("  accuracy (chance 0.5; below chance = the turn wins)")
    print("   delta " + "".join(f"{d:>8g}m" for d in ds))
    res = {"bank": os.path.basename(os.path.abspath(args.bank)),
           "model": args.model, "read_out": "2afc", "accuracy": {}, "lambda": {}}
    for delta in sorted({dl for dl, _ in cells}):
        row = []
        for d in ds:
            c, n = cells.get((delta, d), (0, 0))
            row.append(f"{c / n:>8.2f}" if n else f"{'--':>8s}")
            if n:
                res["accuracy"][f"{delta}_{d:g}"] = c / n
        print(f"  {delta:>5}°" + "".join(row))
    ceil = ceiling(cells)
    if ceil is not None:
        flag = "OK" if ceil > 0.9 else "FAILED -- lambda below is uninterpretable"
        print(f"\n  ceiling control (alternative is a different place "
              f"outright): {ceil:.3f}  [{flag}]")
        res["ceiling"] = ceil

    print("\n  lambda -- metres at which the model is indifferent")
    for delta in sorted({dl for dl, _ in cells}):
        m, bound = lam_from_accuracy(cells, delta)
        res["lambda"][str(delta)] = {"m": m, "bound": bound}
        print(f"    {delta:>4}° : " + (f"{m:5.1f} m" if m is not None else
              {"below": "< ladder (allocentric end)",
               "above": "> ladder (appearance end)"}.get(bound, "unmeasured")))

    out = args.out or os.path.join(args.bank, "afc.json")
    prior = []
    if os.path.exists(out):
        try:
            with open(out) as f:
                prior = [r for r in json.load(f) if r.get("model") != args.model]
        except (ValueError, TypeError):
            prior = []
    with open(out, "w") as f:
        json.dump(prior + [res], f, indent=2)
    print(f"\n[afc] -> {out}  (1 new, {len(prior)} kept)")


if __name__ == "__main__":
    main()
