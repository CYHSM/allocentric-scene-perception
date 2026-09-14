#!/usr/bin/env python3
"""Plot comparison between 4AFC and 2AFC task formats on Four Mountains Benchmark."""
import os
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("paper", exist_ok=True)

with open("results/calib_7b_mental_rotation.json") as f:
    d_4afc = json.load(f)["results"]
with open("results/calib_7b_2afc_mr.json") as f:
    d_2afc = json.load(f)["results"]

deltas = [0, 45, 90, 135, 180]
modes = ["c0_shape_colour", "c1_shape", "c2_colour", "c3_peaks_bare", "c4_valley"]
mode_names = ["c0: Shape+Col", "c1: Shape Only", "c2: Colour Only", "c3: Bare Peaks", "c4: Valley"]

def get_delta_accs(trials):
    by_d = defaultdict(list)
    for t in trials:
        by_d[t["delta"]].append(t["is_correct"])
    return [100.0 * sum(by_d[d]) / len(by_d[d]) if by_d[d] else 0.0 for d in deltas]

def get_mode_accs(trials):
    by_m = defaultdict(list)
    for t in trials:
        by_m[t["mode"]].append(t["is_correct"])
    return [100.0 * sum(by_m[m]) / len(by_m[m]) if by_m[m] else 0.0 for m in modes]

accs_4afc_d = get_delta_accs(d_4afc)
accs_2afc_d = get_delta_accs(d_2afc)

accs_4afc_m = get_mode_accs(d_4afc)
accs_2afc_m = get_mode_accs(d_2afc)

overall_4afc = 100.0 * sum(t["is_correct"] for t in d_4afc) / len(d_4afc)
overall_2afc = 100.0 * sum(t["is_correct"] for t in d_2afc) / len(d_2afc)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15
})

fig, axes = plt.subplots(1, 3, figsize=(17, 5.0), dpi=300)

# Panel A: Accuracy vs Rotation Delta
ax = axes[0]
ax.plot(deltas, accs_2afc_d, marker="s", markersize=7, linewidth=2.4, color="#2980b9", label="Pairwise 2AFC (Chance: 50%)")
ax.axhline(50.0, color="#2980b9", linestyle=":", linewidth=1.5, alpha=0.7)

ax.plot(deltas, accs_4afc_d, marker="o", markersize=7, linewidth=2.4, color="#27ae60", label="Standard 4AFC (Chance: 25%)")
ax.axhline(25.0, color="#27ae60", linestyle=":", linewidth=1.5, alpha=0.7)

ax.set_xlabel("Camera Rotation Angle $\\Delta$ (degrees)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("A. Viewpoint Invariance (4AFC vs. 2AFC)", weight="bold")
ax.set_xticks(deltas)
ax.set_xticklabels([f"{d}°" for d in deltas])
ax.set_ylim(0, 105)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(frameon=True, loc="upper right")

# Panel B: Stimulus Condition Breakdown
ax = axes[1]
x = np.arange(len(modes))
bar_width = 0.35

ax.bar(x - bar_width/2, accs_4afc_m, width=bar_width, color="#27ae60", label=f"4AFC (Mean: {overall_4afc:.1f}%)", alpha=0.9)
ax.bar(x + bar_width/2, accs_2afc_m, width=bar_width, color="#2980b9", label=f"2AFC (Mean: {overall_2afc:.1f}%)", alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(mode_names, rotation=25, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.set_title("B. Performance by Stimulus Mode", weight="bold")
ax.set_ylim(0, 105)
ax.grid(True, linestyle=":", alpha=0.5, axis="y")
ax.legend(frameon=True, loc="upper right")

# Panel C: Excess Information Above Chance Level (Net Lift)
ax = axes[2]
excess_4afc = [acc - 25.0 for acc in accs_4afc_d]
excess_2afc = [acc - 50.0 for acc in accs_2afc_d]

ax.plot(deltas, excess_2afc, marker="s", markersize=7, linewidth=2.4, color="#2980b9", label="2AFC Excess (Acc - 50%)")
ax.plot(deltas, excess_4afc, marker="o", markersize=7, linewidth=2.4, color="#27ae60", label="4AFC Excess (Acc - 25%)")
ax.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.5)

ax.set_xlabel("Camera Rotation Angle $\\Delta$ (degrees)")
ax.set_ylabel("Excess Accuracy Over Chance (%)")
ax.set_title("C. Signal Above Random Chance", weight="bold")
ax.set_xticks(deltas)
ax.set_xticklabels([f"{d}°" for d in deltas])
ax.set_ylim(-30, 40)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(frameon=True, loc="upper right")

plt.tight_layout()
fig.savefig("paper/fig6_format_2afc.png", bbox_inches="tight")
fig.savefig("paper/fig6_format_2afc.pdf", bbox_inches="tight")
print("Saved paper/fig6_format_2afc.png and paper/fig6_format_2afc.pdf")
