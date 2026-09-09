"""
Build standardized 4AFC and 2AFC benchmark datasets from data/scenes_100 for VLMs.

Usage:
    python bench/build_vlm_benchmark.py --root data/scenes_100 --out data/vlm_benchmark_4afc.json --n_options 4
    python bench/build_vlm_benchmark.py --root data/scenes_100 --out data/vlm_benchmark_2afc.json --n_options 2
"""

import argparse
import json
import os
import random
import sys

MODES = [
    "c0_shape_colour",
    "c1_shape",
    "c2_colour",
    "c3_peaks_bare",
    "c4_valley",
]

AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
DELTAS = [0, 45, 90, 135, 180]


def balanced_positions(rng, n_trials, n_options):
    """
    Where the correct answer sits, one entry per trial in a (mode x delta) cell.

    A bare `rng.shuffle(options)` per trial is unbiased in expectation but says
    nothing about any particular cell, and the cells here hold 10-20 trials. At
    seed 42 that produced a 2AFC set skewed 273/227 overall (p = 0.044) and a
    calibration slice whose entire delta=180 cell had the answer at position 1 --
    a cell on which a model that always answers "1" scores 100%. Since every
    per-delta claim in the paper is read off one such cell, the position has to
    be balanced *by construction* rather than in expectation.
    """
    pool = [p for p in range(1, n_options + 1)
            for _ in range(n_trials // n_options)]
    short = n_trials - len(pool)
    if short:  # cell size not divisible by n_options; spread the remainder
        pool += rng.sample(range(1, n_options + 1), short)
    rng.shuffle(pool)
    return pool


def get_image_relpath(bank, scene_id, app, az):
    scene_meta = bank["scenes"][scene_id]
    frame_key = f"{app}_{az}"
    frame = scene_meta["frames"].get(frame_key)
    if frame is None:
        raise KeyError(f"Frame {frame_key} not found for scene {scene_id}")
    img_name = frame["image"]
    return os.path.join(scene_meta["dir"], img_name)


def build_trials(root, modes=MODES, deltas=DELTAS, trials_per_delta=20,
                 n_options=4, seed=42):
    rng = random.Random(seed)
    all_trials = []

    for mode in modes:
        mode_dir = os.path.join(root, mode)
        bank_path = os.path.join(mode_dir, "bank.json")
        if not os.path.exists(bank_path):
            print(f"[WARN] {bank_path} not found; skipping {mode}")
            continue

        with open(bank_path) as f:
            bank = json.load(f)

        scene_ids = sorted(bank["scenes"].keys())
        if len(scene_ids) < n_options:
            raise ValueError(f"Mode {mode} has only {len(scene_ids)} scenes; need at least {n_options}")

        trial_idx = 0
        for delta in deltas:
            # Sample scenes without replacement for target across trials when possible
            shuffled_targets = list(scene_ids)
            rng.shuffle(shuffled_targets)
            answer_slots = balanced_positions(rng, trials_per_delta, n_options)

            for t in range(trials_per_delta):
                trial_idx += 1
                target_id = shuffled_targets[t % len(shuffled_targets)]

                # Pick a random reference azimuth
                study_az = rng.choice(AZIMUTHS)
                target_az = (study_az + delta) % 360

                study_rel = get_image_relpath(bank, target_id, "A", study_az)
                study_path = os.path.join(root, mode, study_rel)

                target_rel = get_image_relpath(bank, target_id, "B", target_az)
                target_path = os.path.join(root, mode, target_rel)

                # Sample distractors from other scenes
                other_scenes = [s for s in scene_ids if s != target_id]
                distractor_ids = rng.sample(other_scenes, n_options - 1)

                options = [
                    {
                        "scene_id": target_id,
                        "is_target": True,
                        "image_path": target_path,
                        "rel_path": target_rel,
                        "appearance": "B",
                        "azimuth": target_az,
                    }
                ]

                for d_id in distractor_ids:
                    d_rel = get_image_relpath(bank, d_id, "B", target_az)
                    d_path = os.path.join(root, mode, d_rel)
                    options.append({
                        "scene_id": d_id,
                        "is_target": False,
                        "image_path": d_path,
                        "rel_path": d_rel,
                        "appearance": "B",
                        "azimuth": target_az,
                    })

                # Place the target at its scheduled slot and shuffle only the
                # distractors around it, so the answer position is balanced
                # within this (mode, delta) cell rather than merely in
                # expectation over the whole benchmark.
                target_opt, rest = options[0], options[1:]
                rng.shuffle(rest)
                correct_idx = answer_slots[t]
                options = rest[:correct_idx - 1] + [target_opt] + rest[correct_idx - 1:]

                assert options[correct_idx - 1]["is_target"]

                trial = {
                    "id": f"{mode}_d{delta:03d}_t{t:02d}",
                    "mode": mode,
                    "delta": int(delta),
                    "study_scene": target_id,
                    "study_azimuth": int(study_az),
                    "target_azimuth": int(target_az),
                    "study_appearance": "A",
                    "target_appearance": "B",
                    "study_image": study_path,
                    "study_relpath": study_rel,
                    "n_options": n_options,
                    "options": options,
                    "correct_choice": correct_idx,  # 1-indexed (1, 2, 3, 4)
                }
                all_trials.append(trial)

    return all_trials


def verify_trials(trials, check_files=True):
    total = len(trials)
    missing = 0
    for it in trials:
        if check_files:
            if not os.path.exists(it["study_image"]):
                missing += 1
            for opt in it["options"]:
                if not os.path.exists(opt["image_path"]):
                    missing += 1
    print(f"Verified {total} trials. Missing image files: {missing}")
    return missing == 0


def main():
    parser = argparse.ArgumentParser(description="Build VLM benchmark dataset from scenes_100")
    parser.add_argument("--root", default="data/scenes_100", help="Path to data/scenes_100")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    parser.add_argument("--trials_per_delta", type=int, default=20, help="Number of trials per delta per mode")
    parser.add_argument("--n_options", type=int, default=4, choices=[2, 4], help="2 for 2AFC, 4 for 4AFC")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verify", action="store_true", help="Verify image paths exist locally")
    args = parser.parse_args()

    trials = build_trials(
        root=args.root,
        trials_per_delta=args.trials_per_delta,
        n_options=args.n_options,
        seed=args.seed
    )

    print(f"Generated {len(trials)} {args.n_options}AFC trials across {len(MODES)} modes and {len(DELTAS)} deltas.")

    if args.verify:
        verify_trials(trials)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_options": args.n_options,
            "trials_per_delta": args.trials_per_delta,
            "deltas": DELTAS,
            "modes": MODES,
            "seed": args.seed,
            "total_trials": len(trials),
            "trials": trials
        }, f, indent=2)

    print(f"Saved benchmark dataset to {args.out}")


if __name__ == "__main__":
    main()
