"""
Merge a sharded render into one bank, and check it is complete.

    python bench/merge_bank.py --root data/scenes

Each shard writes `bank.shard<k>.json` next to its scene directories; the scene
directories themselves are already in the right place, because every shard was
given the same `--out`. So merging is a JSON join, not a file move.

It also verifies the thing the dataset's design rests on and that no image shows:
**the five modes must contain the same scenes.** A mode that silently lost a
shard, or was rendered from a different plan, looks entirely normal on its own.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

EXPECTED_FRAMES = 16          # 2 appearances x 8 azimuths
HIDDEN_PX = 500               # below this a landmark is not readably present


def merge_mode(mode_dir):
    """Join the shard banks of one mode into `bank.json`; return the bank."""
    shards = sorted(glob.glob(os.path.join(mode_dir, "bank.shard*.json")))
    if not shards:
        raise SystemExit(f"no bank.shard*.json under {mode_dir}")

    bank, seen = None, {}
    for path in shards:
        with open(path) as f:
            part = json.load(f)
        if bank is None:
            bank = {k: v for k, v in part.items() if k != "scenes"}
            bank["scenes"] = {}
        for sid, scene in part["scenes"].items():
            if sid in seen:
                raise SystemExit(
                    f"{mode_dir}: scene {sid} rendered by two shards "
                    f"({seen[sid]} and {os.path.basename(path)})")
            seen[sid] = os.path.basename(path)
            bank["scenes"][sid] = scene

    bank["n_shards"] = len(shards)
    out = os.path.join(mode_dir, "bank.json")
    with open(out, "w") as f:
        json.dump(bank, f, indent=1)
    return bank


def occlusion_census(banks):
    """
    How often a probe hides a landmark its anchor showed.

    Displacing a landmark can move it in front of another one. The *scene*
    distance is still pure position -- d_id and d_bind stay 0 -- but the *image*
    has lost an object, and D_scene would then be partly measuring
    "a landmark vanished" rather than "a landmark moved d metres". That is a
    property of the stimulus, not of the model, so it has to be a number printed
    beside the results rather than something a reader infers from a steep curve.

    Not a gate: some occlusion is an honest consequence of rearranging a scene,
    and where the threshold should sit is a judgement about the experiment. It
    is reported per rung so the judgement is made on the data.
    """
    rows = defaultdict(lambda: [0, 0])          # d -> [occluded frames, frames]
    for b in banks.values():
        scenes = b["scenes"]
        for sid, s in scenes.items():
            if s.get("role") != "probe":
                continue
            anch = scenes.get(s.get("anchor"))
            if not anch:
                continue
            d = float(s.get("requested_d_pos", s.get("d_pos", 0.0)))
            for key, fr in s.get("frames", {}).items():
                av = (anch.get("frames", {}).get(key) or {}).get("visible_px")
                pv = fr.get("visible_px")
                if not av or not pv:
                    continue
                rows[d][1] += 1
                if any(pv.get(k, 0) < HIDDEN_PX <= av.get(k, 0) for k in av):
                    rows[d][0] += 1
    return rows


def report_occlusion(rows):
    if not rows:
        print("\nocclusion census: no visible_px recorded -- masks were skipped")
        return
    print(f"\nocclusion census (a landmark the anchor showed drops below "
          f"{HIDDEN_PX} px in the probe):")
    for d in sorted(rows):
        occ, n = rows[d]
        bar = "#" * int(round(30 * occ / max(n, 1)))
        print(f"  d = {d:>5g} m  {occ:6d}/{n:<6d} frames  "
              f"{100 * occ / max(n, 1):5.1f}%  {bar}")
    worst = max(rows, key=lambda d: rows[d][0] / max(rows[d][1], 1))
    frac = rows[worst][0] / max(rows[worst][1], 1)
    if frac > 0.10:
        print(f"  NOTE: at d = {worst:g} m, {100 * frac:.0f}% of frames hide a "
              f"landmark. D_scene there is not purely a displacement.")


def check(banks):
    """Everything that must hold before a number is read off this dataset."""
    problems = []

    # 1. Same scenes in every mode. The pairing is the design.
    ids = {m: set(b["scenes"]) for m, b in banks.items()}
    common = set.intersection(*ids.values()) if ids else set()
    for m, s in ids.items():
        missing = common ^ s
        if missing:
            problems.append(f"{m}: {len(missing)} scenes not shared with the "
                            f"other modes, e.g. {sorted(missing)[:4]}")

    # 2. Same layouts, scene for scene. Two banks with different layouts look
    #    perfectly fine side by side, so this cannot be left to the eye.
    modes = sorted(banks)
    if len(modes) > 1:
        ref = banks[modes[0]]
        for m in modes[1:]:
            for sid in sorted(common):
                a = ref["scenes"][sid]["layout"]["peaks"]
                b = banks[m]["scenes"][sid]["layout"]["peaks"]
                pa = sorted((round(p["x"], 6), round(p["y"], 6)) for p in a)
                pb = sorted((round(p["x"], 6), round(p["y"], 6)) for p in b)
                if pa != pb:
                    problems.append(
                        f"{m}/{sid}: landmark positions differ from "
                        f"{modes[0]} -- these are not the same place")
                    break

    # 3. Every scene fully rendered.
    for m, b in banks.items():
        short = [sid for sid, s in b["scenes"].items()
                 if len(s.get("frames", {})) != EXPECTED_FRAMES]
        if short:
            problems.append(f"{m}: {len(short)} scenes with != "
                            f"{EXPECTED_FRAMES} frames, e.g. {short[:4]}")

    # 4. Probes point at anchors that exist, and moved only in position.
    for m, b in banks.items():
        for sid, s in b["scenes"].items():
            if s.get("role") != "probe":
                continue
            if s["anchor"] not in b["scenes"]:
                problems.append(f"{m}/{sid}: anchor {s['anchor']} missing")
            if abs(s.get("d_id", 0.0)) > 1e-9 or abs(s.get("d_bind", 0.0)) > 1e-9:
                problems.append(
                    f"{m}/{sid}: probe changed identity (d_id={s.get('d_id')}, "
                    f"d_bind={s.get('d_bind')}); D_scene would not be in metres")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/scenes")
    args = ap.parse_args()

    modes = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                   if os.path.isdir(d))
    if not modes:
        raise SystemExit(f"no mode directories under {args.root}")

    banks = {}
    for d in modes:
        m = os.path.basename(d)
        banks[m] = merge_mode(d)
        roles = defaultdict(int)
        for s in banks[m]["scenes"].values():
            roles[s.get("role")] += 1
        print(f"{m:20s} {len(banks[m]['scenes']):4d} scenes  "
              f"({roles['anchor']} anchors, {roles['probe']} probes)  "
              f"from {banks[m]['n_shards']} shards")

    report_occlusion(occlusion_census(banks))

    problems = check(banks)
    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S) -- do not score this bank:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("checks passed: the five modes hold the same scenes, at the same "
          "places, fully rendered.")


if __name__ == "__main__":
    main()
