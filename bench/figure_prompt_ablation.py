#!/usr/bin/env bash
# figure_prompt_ablation.py: Plot comparison of prompt strategies on Four Mountains Benchmark
import os
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("paper", exist_ok=True)

prompts = ["cot", "mental_rotation", "anchor", "birdseye", "elimination", "elevation"]
labels = {
    "cot": "Standard CoT",
    "mental_rotation": "Mental Rotation (Cognitive)",
    "anchor": "Landmark Anchor",
    "birdseye": "Top-Down Cognitive Map",
    "elimination": "Distractor Elimination",
    "elevation": "Topographic Elevation"
}
colors = {
    "cot": "#7f8c8d",            # gray
    "mental_rotation": "#27ae60",  # vibrant green
    "anchor": "#e67e22",          # orange
    "birdseye": "#9b59b6",        # purple
    "elimination": "#2980b9",     # blue
    "elevation": "#c0392b"        # red
}

deltas = [0, 45, 90, 135, 180]
modes = ["c0_shape_colour", "c1_shape", "c2_colour", "c3_peaks_bare", "c4_valley"]
mode_names = ["c0: Shape+Col", "c1: Shape Only", "c2: Colour Only", "c3: Bare Peaks", "c4: Valley"]

# Load results
data = {}
for p in prompts:
    with open(f"results/calib_7b_{p}.json") as f:
        data[p] = json.load(f)["results"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 15
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

# -------------------------------------------------------------
# Panel A: Accuracy vs Rotation Delta (Mental Rotation Curve)
# -------------------------------------------------------------
ax = axes[0]
for p in prompts:
    trials = data[p]
    by_d = defaultdict(list)
    for t in trials:
        by_d[t["delta"]].append(t["is_correct"])
    accs = [100.0 * sum(by_d[d]) / len(by_d[d]) if by_d[d] else 0.0 for d in deltas]
    lw = 2.8 if p == "mental_rotation" else 1.8
    marker = "s" if p == "mental_rotation" else "o"
    ax.plot(deltas, accs, marker=marker, markersize=7, linewidth=lw, label=labels[p], color=colors[p])

ax.axhline(25.0, color="#7f8c8d", linestyle="--", linewidth=1.5, label="Chance (25%)")
ax.set_xlabel("Camera Rotation Angle $\Delta$ (degrees)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("A. Viewpoint Invariance by Prompt", weight="bold")
ax.set_xticks(deltas)
ax.set_xticklabels([f"{d}°" for d in deltas])
ax.set_ylim(0, 80)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(frameon=True, loc="upper right")

# -------------------------------------------------------------
# Panel B: Accuracy by Topographic Condition
# -------------------------------------------------------------
ax = axes[1]
x = np.arange(len(modes))
bar_width = 0.13

for i, p in enumerate(prompts):
    trials = data[p]
    by_m = defaultdict(list)
    for t in trials:
        by_m[t["mode"]].append(t["is_correct"])
    accs = [100.0 * sum(by_m[m]) / len(by_m[m]) if by_m[m] else 0.0 for m in modes]
    offset = (i - 2.5) * bar_width
    ax.bar(x + offset, accs, width=bar_width, label=labels[p], color=colors[p], alpha=0.9)

ax.axhline(25.0, color="#7f8c8d", linestyle="--", linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(mode_names, rotation=25, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.set_title("B. Performance Across Stimulus Conditions", weight="bold")
ax.set_ylim(0, 70)
ax.grid(True, linestyle=":", alpha=0.5, axis="y")

# -------------------------------------------------------------
# Panel C: Overall Benchmark Accuracy Ranking
# -------------------------------------------------------------
ax = axes[2]
overalls = []
for p in prompts:
    trials = data[p]
    acc = 100.0 * sum(t["is_correct"] for t in trials) / len(trials)
    overalls.append((acc, p))

overalls.sort(key=lambda x: x[0])
y_pos = np.arange(len(overalls))
bar_colors = [colors[p] for _, p in overalls]
bar_labels = [labels[p] for _, p in overalls]
bar_vals = [acc for acc, _ in overalls]

bars = ax.barh(y_pos, bar_vals, color=bar_colors, height=0.6, alpha=0.9)
ax.axvline(25.0, color="#7f8c8d", linestyle="--", linewidth=1.5, label="Chance")

for bar, val in zip(bars, bar_vals):
    ax.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
            va="center", ha="left", weight="bold", fontsize=10)

ax.set_yticks(y_pos)
ax.set_yticklabels(bar_labels)
ax.set_xlabel("Overall Accuracy (%)")
ax.set_title("C. Prompt Strategy Ranking (7B)", weight="bold")
ax.set_xlim(0, 50)
ax.grid(True, linestyle=":", alpha=0.5, axis="x")

plt.tight_layout()
fig.savefig("paper/fig_prompt_ablation.png", bbox_inches="tight")
fig.savefig("paper/fig_prompt_ablation.pdf", bbox_inches="tight")
print("Saved paper/fig_prompt_ablation.png and paper/fig_prompt_ablation.pdf")
