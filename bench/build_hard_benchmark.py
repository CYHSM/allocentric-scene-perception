"""
A forced-choice benchmark whose foils sit in a controlled band of layout distance.

`build_vlm_benchmark.py` draws distractors with `rng.sample(other_scenes, ...)`,
so a trial's difficulty is whatever the lottery gave it: the median random foil
is 32.4 m of layout displacement away, off the top of the configured probe
ladder, while the occasional near-duplicate lands at 3 m. Gemini 3.8 Flash's
errors are almost exactly those near-duplicates. Difficulty therefore has to be
selected, not sampled.

Foils here are drawn from a band [lo, hi) of `bench/layout_distance.py`'s
rotation-optimal distance, in metres. Band (0, 8] is roughly the hard end of the
probe ladder `[1, 2, 4, 8, 16, 24]` and needs no new renders; the wide band
reproduces the existing benchmark's difficulty for a matched control.

    python3 bench/build_hard_benchmark.py --lo 0 --hi 8 \
        --out data/vlm_benchmark_4afc_hard.json
"""

import argparse
import collections
import json
import os
import random

import numpy as np

AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
DELTAS = [0, 45, 90, 135, 180]
MODES = ["c0_shape_colour", "c1_shape", "c2_colour", "c3_peaks_bare", "c4_valley"]


def relpath(sid, appearance, az):
    return f"{sid}/{sid}_{appearance}_az{az:03d}.png"



def _band(store, mode, args):
    """(lo, hi) in metres for this mode, from absolute args or a percentile."""
    if not args.pct:
        return args.lo, args.hi
    D = store[f"{mode}__D"]
    off = D[~np.eye(D.shape[0], dtype=bool)]
    return (float(np.percentile(off, args.pct[0])),
            float(np.percentile(off, args.pct[1])))


def _rebuild_matched(args, store, rng, n_foils):
    """
    Take a reference benchmark's trials and swap only their foils.

    Everything that defines the *question* is copied: the trial id, the mode,
    the viewpoint change, the study scene, both azimuths, both appearance draws,
    and therefore the study image the observer looks at. Only the three foils
    and the answer slot are redrawn, from this band.

    A target with too few neighbours in the band keeps its original foils and is
    flagged; that is rare at the wide bands and would otherwise silently drop
    trials and break the pairing the flag exists to create.
    """
    with open(args.match) as f:
        ref = json.load(f)

    out, short = [], collections.Counter()
    bands, pools = {}, {}
    for mode in args.modes:
        ids = list(store[f"{mode}__ids"])
        D = store[f"{mode}__D"]
        idx = {s: i for i, s in enumerate(ids)}
        lo, hi = _band(store, mode, args)
        bands[mode] = (lo, hi, ids, D, idx)
        pools[mode] = {s: [ids[j] for j in np.argsort(D[idx[s]])
                           if ids[j] != s and lo <= D[idx[s], j] < hi]
                       for s in ids}
        print(f"  {mode}: band [{lo:.1f}, {hi:.1f}) m")

    for t in ref["trials"]:
        mode = t["mode"]
        if mode not in bands:
            continue
        lo, hi, ids, D, idx = bands[mode]
        target, target_az = t["study_scene"], t["target_azimuth"]
        cand = pools[mode].get(target, [])
        if len(cand) < n_foils:
            short[mode] += 1
            out.append(dict(t, foil_distance_band_m=[round(lo, 2), round(hi, 2)],
                            rebanded=False))
            continue
        foils = rng.sample(cand, n_foils)

        opts = [{"scene_id": f, "is_target": False,
                 "image_path": os.path.join(args.root, mode,
                                            relpath(f, "B", target_az)),
                 "rel_path": relpath(f, "B", target_az),
                 "appearance": "B", "azimuth": target_az,
                 "layout_distance_m": round(float(D[idx[target], idx[f]]), 2)}
                for f in foils]
        head = {"scene_id": target, "is_target": True,
                "image_path": os.path.join(args.root, mode,
                                           relpath(target, "B", target_az)),
                "rel_path": relpath(target, "B", target_az),
                "appearance": "B", "azimuth": target_az,
                "layout_distance_m": 0.0}
        # Keep the reference trial's answer slot. The answer key is then
        # identical across bands, so a positional prior scores the same on all
        # of them and cannot masquerade as a difficulty effect.
        slot = t["correct_choice"]
        opts.insert(slot - 1, head)
        assert opts[slot - 1]["is_target"]

        out.append({**{k: v for k, v in t.items() if k != "options"},
                    "options": opts, "correct_choice": slot,
                    "foil_distance_band_m": [round(lo, 2), round(hi, 2)],
                    "min_foil_distance_m": round(
                        min(o["layout_distance_m"] for o in opts
                            if not o["is_target"]), 2),
                    "rebanded": True})
    for m, k in short.items():
        print(f"  WARNING {m}: {k} trials kept their original foils "
              f"(target had fewer than {n_foils} neighbours in band)")
    return out


def _write(args, trials, matched_to=None):
    blob = {"n_options": args.n_options, "trials_per_delta": args.trials_per_cell,
            "deltas": DELTAS, "modes": args.modes, "seed": args.seed,
            "foil_selection": {"metric": "rotation-optimal layout distance, metres",
                               "band_percentile": args.pct,
                               "band": [args.lo, args.hi],
                               "source": args.distances},
            "total_trials": len(trials), "trials": trials}
    if matched_to:
        blob["matched_to"] = matched_to
        blob["matching"] = ("trial id, mode, delta, study scene, both azimuths and "
                            "both appearance draws are copied from matched_to; only "
                            "the foils differ, so difficulty is within-items")
    json.dump(blob, open(args.out, "w"))

    dists = [t["min_foil_distance_m"] for t in trials]
    ans = collections.Counter(t["correct_choice"] for t in trials)
    print(f"{len(trials)} trials -> {args.out}")
    print(f"  band {'pct ' + str(args.pct) if args.pct else f'[{args.lo}, {args.hi}) m'}"
          f"   nearest foil: "
          f"median {np.median(dists):.1f} m, range {min(dists):.1f}-{max(dists):.1f}")
    print(f"  answer key {dict(sorted(ans.items()))}  "
          f"best-fixed {max(ans.values())/len(trials):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", default="data/layout_distance.npz")
    ap.add_argument("--root", default="data/scenes_100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.0, help="metres, inclusive")
    ap.add_argument("--hi", type=float, default=8.0, help="metres, exclusive")
    ap.add_argument("--pct", type=float, nargs=2, metavar=("LO", "HI"),
                    help="select the band by percentile of THIS mode's own pairwise "
                         "distance distribution instead of absolute metres. The scales "
                         "are not comparable across modes -- c0-c2 have a median "
                         "pairwise distance of 32.4 m and c3/c4 of 7.3 m, because "
                         "c3/c4 minimise over landmark permutations too -- so an "
                         "absolute band means 'hard' in one mode and 'everything' in "
                         "another. Percentiles make a rung mean the same thing "
                         "everywhere, which is what a ladder needs.")
    ap.add_argument("--n_options", type=int, default=4, choices=[2, 4])
    ap.add_argument("--trials_per_cell", type=int, default=20)
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match", metavar="BENCHMARK.json",
                    help="Keep this benchmark's trials -- the same ids, the same "
                         "study scene, the same azimuths, the same appearance "
                         "draw -- and re-draw ONLY the foils from the new band. "
                         "Without it each band picks its own target scenes (a "
                         "scene is usable only if it has enough neighbours in "
                         "the band), so two bands share trial ids while probing "
                         "different places: 483 of 500 study scenes differ "
                         "between the random and hard banks. Difficulty is then "
                         "a between-items manipulation and any comparison across "
                         "banks is confounded by which places were asked about. "
                         "With --match it is within-items: identical question, "
                         "different foils, and one human session covers every "
                         "band.")
    args = ap.parse_args()

    store = np.load(args.distances, allow_pickle=True)
    rng = random.Random(args.seed)
    n_foils = args.n_options - 1

    if args.match:
        trials = _rebuild_matched(args, store, rng, n_foils)
        _write(args, trials, matched_to=args.match)
        return

    trials, skipped = [], collections.Counter()
    for mode in args.modes:
        ids = list(store[f"{mode}__ids"])
        D = store[f"{mode}__D"]
        index = {s: i for i, s in enumerate(ids)}

        lo, hi = args.lo, args.hi
        if args.pct:
            off = D[~np.eye(len(ids), dtype=bool)]
            lo, hi = (float(np.percentile(off, args.pct[0])),
                      float(np.percentile(off, args.pct[1])))
            print(f"  {mode}: percentile {args.pct} -> [{lo:.1f}, {hi:.1f}) m")

        # Which scenes can even host a trial in this band?
        pool = {}
        for s in ids:
            row = D[index[s]]
            near = [ids[j] for j in np.argsort(row)
                    if ids[j] != s and lo <= row[j] < hi]
            if len(near) >= n_foils:
                pool[s] = near
        usable = sorted(pool)
        if len(usable) < args.trials_per_cell:
            skipped[mode] = len(usable)

        for delta in DELTAS:
            order = usable[:]
            rng.shuffle(order)
            # Balance the answer slot within the cell, as the original builder does.
            slots = ([i for i in range(1, args.n_options + 1)]
                     * (args.trials_per_cell // args.n_options + 1))[:args.trials_per_cell]
            rng.shuffle(slots)
            for t in range(args.trials_per_cell):
                if not order:
                    break
                target = order[t % len(order)]
                study_az = rng.choice(AZIMUTHS)
                target_az = (study_az + delta) % 360
                foils = rng.sample(pool[target], n_foils)

                opts = [{"scene_id": target, "is_target": True,
                         "image_path": os.path.join(args.root, mode,
                                                    relpath(target, "B", target_az)),
                         "rel_path": relpath(target, "B", target_az),
                         "appearance": "B", "azimuth": target_az,
                         "layout_distance_m": 0.0}]
                for f in foils:
                    opts.append({"scene_id": f, "is_target": False,
                                 "image_path": os.path.join(args.root, mode,
                                                            relpath(f, "B", target_az)),
                                 "rel_path": relpath(f, "B", target_az),
                                 "appearance": "B", "azimuth": target_az,
                                 "layout_distance_m": round(
                                     float(D[index[target], index[f]]), 2)})

                slot = slots[t]
                head, rest = opts[0], opts[1:]
                rng.shuffle(rest)
                rest.insert(slot - 1, head)
                opts = rest
                assert opts[slot - 1]["is_target"]

                trials.append({
                    "id": f"{mode}_d{delta:03d}_t{t:02d}",
                    "mode": mode, "delta": delta,
                    "study_scene": target, "study_azimuth": study_az,
                    "target_azimuth": target_az,
                    "study_appearance": "A", "target_appearance": "B",
                    "study_image": os.path.join(args.root, mode,
                                                relpath(target, "A", study_az)),
                    "study_relpath": relpath(target, "A", study_az),
                    "n_options": args.n_options,
                    "options": opts,
                    "correct_choice": slot,
                    "foil_distance_band_m": [round(lo, 2), round(hi, 2)],
                    "min_foil_distance_m": round(
                        min(o["layout_distance_m"] for o in opts if not o["is_target"]), 2),
                })

    _write(args, trials)
    for m, k in skipped.items():
        print(f"  WARNING {m}: only {k} scenes have {n_foils} foils in band")


if __name__ == "__main__":
    main()
