"""
Merge multiple VLM evaluation result files into a single unified result JSON.

Usage:
    python bench/merge_vlm_results.py results/part1.json results/part2.json --out results/merged.json
"""

import argparse
import json
import os
from collections import defaultdict


def merge_results(file_paths, out_path):
    all_results = {}
    model_id = None
    prompt_style = None
    n_options = 4
    benchmark_path = None

    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"Warning: {fp} does not exist, skipping.")
            continue
        with open(fp) as f:
            data = json.load(f)

        summary = data.get("summary", {})
        if model_id is None:
            model_id = summary.get("model")
            prompt_style = summary.get("prompt_style")
            n_options = summary.get("n_options", 4)
            benchmark_path = summary.get("benchmark")

        for r in data.get("results", []):
            all_results[r["trial_id"]] = r

    chance_level = 1.0 / n_options
    by_mode_delta = defaultdict(lambda: {"correct": 0, "total": 0})
    choice_dist = defaultdict(int)

    for r in all_results.values():
        key = (r["mode"], r["delta"])
        by_mode_delta[key]["total"] += 1
        if r["is_correct"]:
            by_mode_delta[key]["correct"] += 1
        if r["model_choice"]:
            choice_dist[r["model_choice"]] += 1

    summary = {
        "model": model_id,
        "benchmark": benchmark_path,
        "prompt_style": prompt_style,
        "n_options": n_options,
        "chance_level": chance_level,
        "total_trials": len(all_results),
        "overall_accuracy": float(sum(r["is_correct"] for r in all_results.values()) / max(len(all_results), 1)),
        "choice_distribution": dict(choice_dist),
        "by_mode_delta": {},
    }

    print("\n" + "=" * 65)
    print(f"MERGED RESULTS SUMMARY: {model_id} ({prompt_style})")
    print("=" * 65)
    print(f"Total Trials: {len(all_results)}")
    print(f"Overall Accuracy: {100 * summary['overall_accuracy']:.1f}% (Chance: {100 * chance_level:.1f}%)")
    print(f"Choice Distribution: {dict(choice_dist)}\n")
    print(f"{'Mode':20s} | {'Delta':5s} | {'Acc (%)':8s} | {'N':4s}")
    print("-" * 45)

    for (mode, delta), stats in sorted(by_mode_delta.items()):
        acc = 100.0 * stats["correct"] / stats["total"] if stats["total"] else 0.0
        summary["by_mode_delta"][f"{mode}_d{delta}"] = {
            "mode": mode,
            "delta": delta,
            "accuracy": float(acc / 100.0),
            "correct": stats["correct"],
            "total": stats["total"],
        }
        print(f"{mode:20s} | {delta:5d} | {acc:7.1f}% | {stats['total']:4d}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "results": list(all_results.values())
        }, f, indent=2)

    print(f"\nSaved merged results to {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Merge VLM evaluation results")
    parser.add_argument("inputs", nargs="+", help="Input result JSON files")
    parser.add_argument("--out", required=True, help="Output merged JSON file")
    args = parser.parse_args()

    merge_results(args.inputs, args.out)


if __name__ == "__main__":
    main()
