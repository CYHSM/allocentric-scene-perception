#!/usr/bin/env python3
"""Collate and analyze all VLM evaluation results on Four Mountains Benchmark."""
import os
import json
from collections import defaultdict

def analyze_model(path, title):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    trials = data.get("results", [])
    if not trials:
        print(f"No trials in {path}")
        return None

    by_mode_delta = defaultdict(lambda: {"correct": 0, "total": 0})
    choice_dist = defaultdict(int)
    for t in trials:
        m, d = t["mode"], t["delta"]
        by_mode_delta[(m, d)]["total"] += 1
        if t.get("is_correct"):
            by_mode_delta[(m, d)]["correct"] += 1
        if t.get("model_choice"):
            choice_dist[t["model_choice"]] += 1

    deltas = [0, 45, 90, 135, 180]
    modes = sorted(list(set(m for m, d in by_mode_delta.keys())))

    print(f"\n{'='*78}")
    print(f"{title}: {len(trials)} trials completed")
    print(f"{'='*78}")
    header = " | ".join(f"d{d:03d} (N)" for d in deltas)
    print(f"{'Mode':20s} | {header} | Total")
    print("-" * 78)

    for m in modes:
        row = []
        tot_c, tot_n = 0, 0
        for d in deltas:
            st = by_mode_delta.get((m, d), {"correct": 0, "total": 0})
            tot_c += st["correct"]
            tot_n += st["total"]
            if st["total"] > 0:
                row.append(f"{100*st['correct']/st['total']:4.0f}% ({st['total']:2d})")
            else:
                row.append("  --  ( 0)")
        tot_str = f"{100*tot_c/tot_n:4.1f}% ({tot_n:3d})" if tot_n > 0 else "0.0%"
        print(f"{m:20s} | {' | '.join(row)} | {tot_str}")

    print("-" * 78)
    d_tot_c = [sum(by_mode_delta[(m, d)]["correct"] for m in modes) for d in deltas]
    d_tot_n = [sum(by_mode_delta[(m, d)]["total"] for m in modes) for d in deltas]
    d_acc_str = " | ".join(f"{100*c/n:4.1f}% ({n:2d})" if n > 0 else "  --  " for c, n in zip(d_tot_c, d_tot_n))
    overall_c = sum(d_tot_c)
    overall_n = sum(d_tot_n)
    print(f"{'OVERALL BY DELTA':20s} | {d_acc_str} | {100*overall_c/overall_n:4.1f}% ({overall_n:3d})")
    print(f"Choice Distribution: {dict(choice_dist)}")
    return {
        "trials": len(trials),
        "overall": 100 * overall_c / overall_n if overall_n else 0.0,
        "by_delta": {d: (100*c/n if n else 0.0) for d, c, n in zip(deltas, d_tot_c, d_tot_n)},
        "by_mode": {m: (100*sum(by_mode_delta[(m, d)]["correct"] for d in deltas)/sum(by_mode_delta[(m, d)]["total"] for d in deltas)) for m in modes}
    }

if __name__ == "__main__":
    analyze_model("results/qwen2_5_vl_3b_4afc_full.json", "Qwen2.5-VL-3B-Instruct (4AFC CoT)")
    analyze_model("results/qwen2_5_vl_7b_4afc_full.json", "Qwen2.5-VL-7B-Instruct (4AFC CoT)")
    analyze_model("results/qwen2_5_vl_32b_4afc_full.json", "Qwen2.5-VL-32B-Instruct (4AFC CoT)")
    analyze_model("results/calib_7b_mental_rotation.json", "Qwen2.5-VL-7B-Instruct (Mental Rotation Prompt)")
    analyze_model("results/calib_7b_2afc_mr.json", "Qwen2.5-VL-7B-Instruct (2AFC Pairwise Mental Rotation)")
