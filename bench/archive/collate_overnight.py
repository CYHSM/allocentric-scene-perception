#!/usr/bin/env python3
"""
Collate and analyze all VLM overnight evaluation results.
"""
import glob
import json
import os
import numpy as np

def analyze_file(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    results = d.get("results", [])
    if not results:
        return None
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    acc = correct / total
    by_delta = {}
    by_mode = {}
    by_mode_delta = {}
    choice_dist = {}
    latencies = []
    
    for r in results:
        delta = r["delta"]
        mode = r["mode"]
        corr = r["is_correct"]
        c = r.get("model_choice")
        lat = r.get("latency", 0)
        if lat:
            latencies.append(lat)
        if c is not None:
            choice_dist[c] = choice_dist.get(c, 0) + 1
        
        by_delta.setdefault(delta, []).append(corr)
        by_mode.setdefault(mode, []).append(corr)
        by_mode_delta.setdefault((mode, delta), []).append(corr)
        
    delta_acc = {k: float(np.mean(v)) for k, v in sorted(by_delta.items())}
    mode_acc = {k: float(np.mean(v)) for k, v in sorted(by_mode.items())}
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    
    return {
        "file": os.path.basename(path),
        "total": total,
        "correct": correct,
        "acc": acc,
        "mean_latency": mean_lat,
        "delta_acc": delta_acc,
        "mode_acc": mode_acc,
        "choice_dist": choice_dist,
        "by_mode_delta": {f"{m}_d{d}": float(np.mean(v)) for (m, d), v in sorted(by_mode_delta.items())}
    }

def main():
    print("=" * 75)
    print("FULL 500-TRIAL BENCHMARK BATTERY")
    print("=" * 75)
    full_files = [
        "results/qwen2_5_vl_3b_4afc_full.json",
        "results/qwen2_5_vl_7b_4afc_full.json",
        "results/qwen2_5_vl_32b_4afc_full.json",
        "results/qwen2_5_vl_7b_mental_rot_full.json",
        "results/qwen2_5_vl_7b_anchor_full.json",
        "results/qwen2_5_vl_7b_2afc_full.json",
        "results/qwen2_5_vl_72b_4afc_cot.json",
    ]
    for f in full_files:
        res = analyze_file(f)
        if res:
            name = res["file"]
            n = res["total"]
            acc = res["acc"] * 100
            lat = res["mean_latency"]
            print(f"\n>> {name} (N={n}, Acc={acc:.1f}%, Mean Latency={lat:.2f}s)")
            delta_str = " | ".join([f"Δ={k}°: {v*100:4.1f}%" for k, v in res["delta_acc"].items()])
            print(f"   Deltas:  {delta_str}")
            mode_str = " | ".join([f"{k[:8]}: {v*100:4.1f}%" for k, v in res["mode_acc"].items()])
            print(f"   Modes:   {mode_str}")
            print(f"   Choices: {res['choice_dist']}")

    print("\n" + "=" * 75)
    print("100-TRIAL PROMPT ABLATION CALIBRATION BATTERY (7B)")
    print("=" * 75)
    calib_files = sorted(glob.glob("results/calib_7b_*.json"))
    for f in calib_files:
        res = analyze_file(f)
        if res:
            delta_str = " ".join([f"d{k}:{v*100:4.1f}%" for k, v in res["delta_acc"].items()])
            print(f"{res['file']:30s} | N={res['total']:3d} | Acc={res['acc']*100:5.1f}% | Deltas: {delta_str}")

if __name__ == "__main__":
    main()
