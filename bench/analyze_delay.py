"""
Delay Analysis and Visualization for Allocentric Scene Perception (COSYNE 2027).

Analyzes performance across simulated working-memory delay durations (or noise scales sigma),
stratified by:
  - Appearance gate (Delta = 0 deg): viewpoint-identical probe testing visual memory retention.
  - Rotated trials (Delta in {45, 90, 135, 180} deg): allocentric perspective transformation under memory decay.

Performs biophysical curve fitting:
  - Exponential decay: Acc(t) = 0.25 + (Acc_0 - 0.25) * exp(-lambda * t)
  - Half-life estimation: t_{1/2} = ln(2) / lambda

Outputs:
  - Publication-quality two-panel figures (PDF / PNG).
  - LaTeX table snippet for the COSYNE abstract.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def load_delay_results(directory: str, pattern: str = "delayed_*.json") -> List[Dict[str, Any]]:
    """Load all delay result JSON files matching pattern in directory."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    records = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        data["_filepath"] = fp
        records.append(data)
    return records


def exp_decay(t, acc_0, lam):
    """Exponential decay towards chance (0.25)."""
    return 0.25 + (acc_0 - 0.25) * np.exp(-lam * t)


def analyze_delay_series(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract stratified performance across delay / noise levels."""
    series = []
    for r in records:
        summary = r.get("summary", {})
        results = r.get("results", [])
        
        # Determine delay key: either delay_seconds or noise_scale
        delay_s = summary.get("delay_seconds")
        sigma = summary.get("effective_sigma", summary.get("noise_scale", 0.0))
        x_val = delay_s if delay_s is not None else sigma
        x_type = "delay_seconds" if delay_s is not None else "noise_scale"

        # Stratified calculations
        total = len(results)
        if total == 0:
            continue
        correct_total = sum(1 for t in results if t.get("is_correct"))
        acc_total = correct_total / total

        # Gate (delta == 0)
        gate_trials = [t for t in results if t.get("delta") == 0]
        gate_n = len(gate_trials)
        gate_acc = (sum(1 for t in gate_trials if t.get("is_correct")) / gate_n) if gate_n > 0 else np.nan

        # Rotated (delta > 0)
        rot_trials = [t for t in results if t.get("delta", 0) > 0]
        rot_n = len(rot_trials)
        rot_acc = (sum(1 for t in rot_trials if t.get("is_correct")) / rot_n) if rot_n > 0 else np.nan

        # Breakdown by angle
        angles = {}
        for d in (45, 90, 135, 180):
            d_trials = [t for t in results if t.get("delta") == d]
            if d_trials:
                angles[d] = sum(1 for t in d_trials if t.get("is_correct")) / len(d_trials)
            else:
                angles[d] = np.nan

        series.append({
            "x": x_val,
            "x_type": x_type,
            "delay_seconds": delay_s,
            "sigma": sigma,
            "overall_acc": acc_total,
            "gate_acc": gate_acc,
            "rot_acc": rot_acc,
            "angles": angles,
            "n_trials": total,
            "summary": summary,
        })

    # Sort series by x
    series.sort(key=lambda item: item["x"])
    return {
        "x_type": series[0]["x_type"] if series else "noise_scale",
        "series": series,
    }


def plot_delay_decay(analysis: Dict[str, Any], out_path_prefix: str, title_suffix: str = ""):
    """Generate publication-quality two-panel figure."""
    series = analysis["series"]
    if not series:
        print("No series data to plot.")
        return

    x_vals = np.array([s["x"] for s in series])
    gate_accs = np.array([s["gate_acc"] for s in series])
    rot_accs = np.array([s["rot_acc"] for s in series])
    all_accs = np.array([s["overall_acc"] for s in series])
    x_type = analysis["x_type"]

    x_label = "Working Memory Delay $t$ (seconds)" if x_type == "delay_seconds" else "Latent Perturbation Scale $\\sigma$"

    # Style configuration
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4), dpi=300)

    # Panel A: Stratified Decay (Gate vs Rotated)
    ax1.axhline(0.25, color="gray", linestyle="--", alpha=0.7, label="Chance (25%)")
    
    # Plot Gate
    valid_gate = ~np.isnan(gate_accs)
    if np.any(valid_gate):
        ax1.plot(x_vals[valid_gate], gate_accs[valid_gate] * 100, "o-", color="#1f77b4", linewidth=2, markersize=5, label="Gate ($\\Delta = 0^\\circ$)")
        # Try exponential fit
        if len(x_vals[valid_gate]) >= 3:
            try:
                popt, _ = curve_fit(exp_decay, x_vals[valid_gate], gate_accs[valid_gate], p0=[gate_accs[valid_gate][0], 0.5], maxfev=2000)
                x_dense = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax1.plot(x_dense, exp_decay(x_dense, *popt) * 100, "--", color="#1f77b4", alpha=0.5)
            except Exception:
                pass

    # Plot Rotated
    valid_rot = ~np.isnan(rot_accs)
    if np.any(valid_rot):
        ax1.plot(x_vals[valid_rot], rot_accs[valid_rot] * 100, "s-", color="#d62728", linewidth=2, markersize=5, label="Rotated ($\\Delta \\ge 45^\\circ$)")
        if len(x_vals[valid_rot]) >= 3:
            try:
                popt, _ = curve_fit(exp_decay, x_vals[valid_rot], rot_accs[valid_rot], p0=[rot_accs[valid_rot][0], 0.5], maxfev=2000)
                x_dense = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax1.plot(x_dense, exp_decay(x_dense, *popt) * 100, "--", color="#d62728", alpha=0.5)
            except Exception:
                pass

    # Plot Overall
    ax1.plot(x_vals, all_accs * 100, "^:", color="#2ca02c", linewidth=1.5, markersize=4, label="Overall", alpha=0.8)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("A. Working Memory Decay Trajectory")
    ax1.set_ylim(15, 105)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel B: Breakdown by Rotation Angle
    colors = {45: "#2ca02c", 90: "#ff7f0e", 135: "#9467bd", 180: "#8c564b"}
    for d, c in colors.items():
        ang_vals = np.array([s["angles"].get(d, np.nan) for s in series])
        valid = ~np.isnan(ang_vals)
        if np.any(valid):
            ax2.plot(x_vals[valid], ang_vals[valid] * 100, "o-", color=c, linewidth=1.5, markersize=4, label=f"$\\Delta = {d}^\\circ$")

    ax2.axhline(0.25, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("B. Angular Invariance Under Delay")
    ax2.set_ylim(15, 105)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, title="Rotation Angle")

    plt.tight_layout()
    pdf_path = f"{out_path_prefix}.pdf"
    png_path = f"{out_path_prefix}.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated figures: {pdf_path} and {png_path}")


def generate_latex_table(analysis: Dict[str, Any]) -> str:
    """Generate LaTeX table code for the COSYNE abstract."""
    series = analysis["series"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Working Memory Decay under Targeted Latent Perturbations.}",
        r"\label{tab:delay_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Delay / $\sigma$ & Gate ($\Delta = 0^\circ$) & Rotated ($\Delta \ge 45^\circ$) & Overall & $\Delta = 180^\circ$ \\",
        r"\midrule",
    ]
    for s in series:
        x_str = f"{s[x]:.2f}"
        gate = f"{s[gate_acc]*100:.1f}\\%" if not np.isnan(s[gate_acc]) else "--"
        rot = f"{s[rot_acc]*100:.1f}\\%" if not np.isnan(s[rot_acc]) else "--"
        all_ = f"{s[overall_acc]*100:.1f}\\%"
        a180 = f"{s[angles].get(180, np.nan)*100:.1f}\\%" if not np.isnan(s[angles].get(180, np.nan)) else "--"
        lines.append(f"{x_str} & {gate} & {rot} & {all_} & {a180} \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze delayed VLM evaluation results")
    parser.add_argument("--dir", type=str, default="results")
    parser.add_argument("--pattern", type=str, default="delayed_*.json")
    parser.add_argument("--out_prefix", type=str, default="paper/cosyne/figures/working_memory_decay")
    args = parser.parse_args()

    records = load_delay_results(args.dir, args.pattern)
    print(f"Found {len(records)} matching result files in {args.dir}")
    if not records:
        return

    analysis = analyze_delay_series(records)
    plot_delay_decay(analysis, args.out_prefix)
    tex = generate_latex_table(analysis)
    print("\n--- Generated LaTeX Table ---")
    print(tex)


if __name__ == "__main__":
    main()
