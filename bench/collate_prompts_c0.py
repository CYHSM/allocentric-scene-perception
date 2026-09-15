#!/usr/bin/env python3
"""
Collate and analyze prompt calibration results on the c0_shape_colour benchmark for Qwen2.5-VL-7B.
"""

import os
import glob
import json
import argparse
from collections import Counter, defaultdict

def analyze_result_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # Handle either raw trials list or wrapped dict
    trials = data.get("results", []) or data.get("trials", [])
    if not trials and isinstance(data, list):
        trials = data
        
    summary = data.get("summary", {})
    prompt_style = summary.get("prompt_style") or os.path.basename(filepath).split("_c0_")[1].replace(".json", "")
    
    total = len(trials)
    if total == 0:
        return None
        
    correct_total = 0
    gate_total = 0
    gate_correct = 0
    rot_total = 0
    rot_correct = 0
    
    by_delta = defaultdict(lambda: {"correct": 0, "total": 0})
    choice_dist = Counter()
    
    for t in trials:
        pred = t.get("model_choice")
        gt = t.get("correct_choice")
        delta = t.get("delta", 0)
        
        if pred is not None:
            choice_dist[str(pred)] += 1
            is_corr = (pred == gt)
        else:
            is_corr = False
            
        if is_corr:
            correct_total += 1
            
        by_delta[delta]["total"] += 1
        if is_corr:
            by_delta[delta]["correct"] += 1
            
        if delta == 0:
            gate_total += 1
            if is_corr:
                gate_correct += 1
        else:
            rot_total += 1
            if is_corr:
                rot_correct += 1
                
    acc_overall = correct_total / total if total > 0 else 0.0
    acc_gate = gate_correct / gate_total if gate_total > 0 else 0.0
    acc_rot = rot_correct / rot_total if rot_total > 0 else 0.0
    
    delta_accs = {}
    for d in [0, 45, 90, 135, 180]:
        tot = by_delta[d]["total"]
        corr = by_delta[d]["correct"]
        delta_accs[d] = (corr / tot) if tot > 0 else 0.0
        
    return {
        "prompt": prompt_style,
        "file": filepath,
        "total": total,
        "acc_overall": acc_overall,
        "acc_gate": acc_gate,
        "acc_rot": acc_rot,
        "delta_accs": delta_accs,
        "choice_dist": dict(choice_dist),
        "correct_counts": {
            "total": f"{correct_total}/{total}",
            "gate": f"{gate_correct}/{gate_total}",
            "rot": f"{rot_correct}/{rot_total}"
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Collate c0 prompt sweep results")
    parser.add_argument("--results_dir", default="results/prompts_c0", help="Directory containing prompt result JSONs")
    args = parser.parse_args()
    
    pattern = os.path.join(args.results_dir, "*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No result JSON files found in {args.results_dir}")
        return
        
    results = []
    for f in files:
        res = analyze_result_file(f)
        if res:
            results.append(res)
            
    # Sort by Rotated Accuracy descending, then Overall Accuracy descending
    results.sort(key=lambda r: (r["acc_rot"], r["acc_overall"]), reverse=True)
    
    print("\n# Prompt Calibration Results on c0_shape_colour (Qwen2.5-VL-7B-Instruct, N=100)")
    print("=" * 110)
    header = f"| {'Prompt Style':<24} | {'Overall':<7} | {'Gate (0°)':<9} | {'Rot (≥45°)':<10} | {'45°':<6} | {'90°':<6} | {'135°':<6} | {'180°':<6} | {'Choices (1/2/3/4)':<17} |"
    print(header)
    print("|" + "-" * 26 + "|" + "-" * 9 + "|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 19 + "|")
    
    baseline_rot = None
    for r in results:
        if r["prompt"] == "cot_anyview":
            baseline_rot = r["acc_rot"]
            break
            
    for r in results:
        p = r["prompt"]
        ovr = f"{r['acc_overall']*100:5.1f}%"
        gate = f"{r['acc_gate']*100:5.1f}%"
        rot = f"{r['acc_rot']*100:5.1f}%"
        d45 = f"{r['delta_accs'].get(45, 0)*100:4.1f}%"
        d90 = f"{r['delta_accs'].get(90, 0)*100:4.1f}%"
        d135 = f"{r['delta_accs'].get(135, 0)*100:4.1f}%"
        d180 = f"{r['delta_accs'].get(180, 0)*100:4.1f}%"
        
        c = r["choice_dist"]
        cdist = f"{c.get('1',0):2d}/{c.get('2',0):2d}/{c.get('3',0):2d}/{c.get('4',0):2d}"
        
        # Mark baseline and top performer
        marker = ""
        if p == "cot_anyview":
            marker = " (baseline)"
        line = f"| {p:<24} | {ovr:<7} | {gate:<9} | {rot:<10} | {d45:<6} | {d90:<6} | {d135:<6} | {d180:<6} | {cdist:<17} |"
        print(line)
        
    print("=" * 110)
    print(f"Chance level: 25.0% across all metrics. Total trials per prompt: 100 (20 gate, 80 rotated).\n")

if __name__ == "__main__":
    main()
