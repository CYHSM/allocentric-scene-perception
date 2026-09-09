"""
Build the text-channel 2AFC benchmark: the same trials, written as coordinates.

    python bench/build_text_benchmark.py --root data/scenes_100 --out_dir data

Writes two files, `text_benchmark_2afc_id.json` (landmarks named) and
`text_benchmark_2afc_noid.json` (landmarks anonymous). They hold the *same*
trials -- same scenes, same turn angles, same answer positions -- so the
identity effect is a within-trial contrast rather than two separate samples.

Trials come from `build_vlm_benchmark.build_trials`, not from a second copy of
the sampling code. With the default seed and 20 trials per delta, the draws are
bit-identical to the `c0_shape_colour` block of `data/vlm_benchmark_2afc.json`,
because that mode is drawn first and the RNG is consumed mode by mode. Each
trial therefore keeps `paired_id`, the image trial it corresponds to, and the
image paths themselves: the text and image channels can be compared trial by
trial rather than only in aggregate. `bench/tests/test_text_views.py` asserts
that pairing rather than trusting it.

Only c0 geometry is used, and one mode is not an oversight. The five rendered
modes share their layouts exactly -- c0 through c4 place the same landmarks at
the same coordinates and differ only in how they are drawn -- so serialising
all five would produce five near-copies of one geometry set and quintuple the
apparent n without adding a single independent scene.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import text_views  # noqa: E402
from build_vlm_benchmark import DELTAS, build_trials  # noqa: E402

GEOMETRY_MODE = "c0_shape_colour"


def load_layouts(root, mode=GEOMETRY_MODE):
    with open(os.path.join(root, mode, "bank.json")) as f:
        bank = json.load(f)
    layouts = {sid: s["layout"] for sid, s in bank["scenes"].items()}
    return layouts, bank.get("camera")


def add_text(trials, layouts, camera, identity, jitter=0.0, seed=0):
    """Return a copy of `trials` carrying the text serialisation of every view."""
    import numpy as np

    tag = "id" if identity else "noid"
    # One stream, drawn in a fixed order, so a rebuild reproduces the file and
    # the two identity variants get the *same* measurement error on the same
    # view -- otherwise the id/noid contrast would confound naming with noise.
    rng = np.random.default_rng(seed)
    out = []
    for t in trials:
        t = json.loads(json.dumps(t))  # deep copy; the two variants share nothing
        t["paired_id"] = t["id"]
        t["image_mode"] = t["mode"]
        t["id"] = t["id"].replace(GEOMETRY_MODE, f"text_{tag}")
        t["mode"] = f"text_{tag}"
        t["identity_given"] = identity
        t["study_text"] = text_views.render_view(
            layouts[t["study_scene"]], t["study_azimuth"], camera, identity,
            jitter=jitter, rng=rng)
        for opt in t["options"]:
            opt["text"] = text_views.render_view(
                layouts[opt["scene_id"]], opt["azimuth"], camera, identity,
                jitter=jitter, rng=rng)
        # Without jitter, delta = 0 makes the correct candidate a byte-for-byte
        # copy of the study block. That is not a viewpoint-invariance trial and
        # must never be the denominator of a VII: it is an attention check, and
        # it is labelled as one here rather than in a caption nobody reads.
        t["degenerate"] = bool(jitter == 0.0 and t["delta"] == 0
                               and t["options"][t["correct_choice"] - 1]["text"]
                               == t["study_text"])
        out.append(t)
    return out


def check_solvable(trials):
    """
    The gate: an analytic matcher must score 100%, or the item is unanswerable.

    Text items can fail in a way image items cannot -- if two layouts happen to
    have near-identical inter-landmark distances, no amount of reasoning
    separates them, and a model would be scored wrong for being right. This
    runs on the shipped file, not on a fixture.
    """
    layouts_ok, wrong = 0, []
    for t in trials:
        study = text_views.view_rows_from_text(t["study_text"])
        opts = [text_views.view_rows_from_text(o["text"]) for o in t["options"]]
        pick, _ = text_views.solve(study, opts)
        if pick == t["correct_choice"]:
            layouts_ok += 1
        else:
            wrong.append(t["id"])
    return layouts_ok, wrong


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/scenes_100")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--trials_per_delta", type=int, default=20)
    ap.add_argument("--n_options", type=int, default=2, choices=[2, 4])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="Gaussian measurement error on each egocentric position, "
                         "in metres. 0 (default) gives exact coordinates and a "
                         "degenerate delta=0 cell; see text_views.render_view.")
    args = ap.parse_args()

    layouts, camera = load_layouts(args.root)
    trials = build_trials(root=args.root, modes=[GEOMETRY_MODE], deltas=DELTAS,
                          trials_per_delta=args.trials_per_delta,
                          n_options=args.n_options, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    for identity in (True, False):
        tag = "id" if identity else "noid"
        tt = add_text(trials, layouts, camera, identity, jitter=args.jitter,
                      seed=args.seed)
        ok, wrong = check_solvable(tt)
        if wrong:
            raise SystemExit(f"{len(wrong)} unanswerable trials in the {tag} variant "
                             f"(analytic solver failed): {wrong[:5]}")
        n_deg = sum(t["degenerate"] for t in tt)
        out = os.path.join(args.out_dir,
                           f"text_benchmark_{args.n_options}afc_{tag}.json")
        with open(out, "w") as f:
            json.dump({"channel": "text", "identity_given": identity,
                       "n_options": args.n_options,
                       "trials_per_delta": args.trials_per_delta,
                       "deltas": DELTAS, "modes": [f"text_{tag}"],
                       "geometry_mode": GEOMETRY_MODE, "seed": args.seed,
                       "jitter_m": args.jitter,
                       "degenerate_trials": n_deg,
                       "total_trials": len(tt), "trials": tt}, f, indent=2)
        print(f"{out}: {len(tt)} trials, analytic solver {ok}/{len(tt)}")
        if n_deg:
            print(f"  WARNING: {n_deg} trials (delta=0) have a candidate identical "
                  f"to the study text. Report them as an attention check, not as "
                  f"an appearance gate; --jitter makes the cell non-degenerate.")


if __name__ == "__main__":
    main()
