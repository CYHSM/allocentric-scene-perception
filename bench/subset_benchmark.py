"""
A benchmark file restricted to some of the stimulus modes.

c3_peaks_bare and c4_valley are not landmark-inventory controlled: each of the
100 scenes in those modes has its own unique set of terrain morphologies, so
"which option contains the same landmarks" answers the trial with no spatial
reasoning. c0/c1/c2 share one inventory across all 100 scenes, so there the only
thing separating target from foil is where the landmarks stand.

    python3 bench/subset_benchmark.py data/vlm_benchmark_4afc.json \
        --modes c0_shape_colour c1_shape c2_colour \
        --out data/vlm_benchmark_4afc_ctrl.json
"""

import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("benchmark")
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blob = json.load(open(args.benchmark))
    keep = [t for t in blob["trials"] if t["mode"] in args.modes]
    blob["trials"] = keep
    blob["total_trials"] = len(keep)
    blob["modes"] = list(args.modes)
    blob["derived_from"] = args.benchmark
    json.dump(blob, open(args.out, "w"))

    cells = collections.Counter((t["mode"], t["delta"]) for t in keep)
    ans = collections.Counter(t["correct_choice"] for t in keep)
    print(f"{len(keep)} trials -> {args.out}")
    print(f"  {len(cells)} cells, {sorted(set(cells.values()))} trials each")
    print(f"  answer key {dict(sorted(ans.items()))}  "
          f"best-fixed baseline {max(ans.values()) / len(keep):.3f}")


if __name__ == "__main__":
    main()
