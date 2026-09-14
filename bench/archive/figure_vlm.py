"""
Generate Figure 4 and Table for Vision-Language Models (VLMs)
on the Four Mountains Allocentric Benchmark.

Usage:
    python bench/figure_vlm.py --results results/qwen2_5_vl_3b_cot.json results/qwen2_5_vl_7b_cot.json --out paper/fig4_vlm.png
"""

import argparse
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DELTAS = [0, 45, 90, 135, 180]
MODES = [
    ("c0_shape_colour", "c0: Shape + Colour"),
    ("c1_shape", "c1: Shape Only"),
    ("c2_colour", "c2: Colour Only"),
    ("c3_peaks_bare", "c3: Landforms Only"),
    ("c4_valley", "c4: Valley (Depression)"),
]

MODEL_STYLES = {
    "Qwen/Qwen2-VL-2B-Instruct": {"label": "Qwen2-VL (2B)", "color": "#888888", "ls": "--", "marker": "^"},
    "Qwen/Qwen2.5-VL-3B-Instruct": {"label": "Qwen2.5-VL (3B)", "color": "#1f77b4", "ls": "-", "marker": "o"},
    "Qwen/Qwen2.5-VL-7B-Instruct": {"label": "Qwen2.5-VL (7B)", "color": "#2ca02c", "ls": "-", "marker": "s"},
    "Qwen/Qwen2.5-VL-7B-Instruct (mental_rotation)": {"label": "Qwen2.5-VL (7B) + Mental Rot", "color": "#17becf", "ls": "--", "marker": "D"},
    "Qwen/Qwen2.5-VL-7B-Instruct (anchor)": {"label": "Qwen2.5-VL (7B) + Anchor", "color": "#bcbd22", "ls": ":", "marker": "v"},
    "Qwen/Qwen2.5-VL-32B-Instruct": {"label": "Qwen2.5-VL (32B)", "color": "#ff7f0e", "ls": "-", "marker": "p"},
    "Qwen/Qwen2.5-VL-72B-Instruct": {"label": "Qwen2.5-VL (72B)", "color": "#9467bd", "ls": "-", "marker": "h"},
    "gpt-4o": {"label": "GPT-4o (Closed)", "color": "#d62728", "ls": "-", "marker": "D"},
    "claude-3-5-sonnet-20241022": {"label": "Claude 3.5 Sonnet", "color": "#8c564b", "ls": "-", "marker": "*"},
}


def load_result(path):
    with open(path) as f:
        data = json.load(f)
    return data


def plot_vlm_scaling(results_dict, out_fig="paper/fig4_vlm.png", out_table="paper/table_vlm.tex"):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)

    # Left Panel: Overall Accuracy vs Rotation Delta
    ax = axes[0]
    ax.axhline(25.0, color="#aaaaaa", linestyle=":", linewidth=1.5, label="Chance Level (4AFC = 25%)")

    for model_id, data in results_dict.items():
        summary = data.get("summary", {})
        results = data.get("results", [])
        style = MODEL_STYLES.get(model_id, {"label": model_id.split("/")[-1], "color": "#333333", "ls": "-", "marker": "o"})

        # Compute accuracy per delta
        delta_accs = []
        for d in DELTAS:
            matching = [r for r in results if r.get("delta") == d]
            if matching:
                acc = 100.0 * sum(r["is_correct"] for r in matching) / len(matching)
            else:
                acc = np.nan
            delta_accs.append(acc)

        ax.plot(DELTAS, delta_accs, label=style["label"], color=style["color"],
                linestyle=style["ls"], marker=style["marker"], linewidth=2, markersize=6)

    ax.set_title("(A) Overall Allocentric Invariance vs. Viewpoint $\\Delta$", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Viewpoint Rotation $\\Delta$ (degrees)", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(DELTAS)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    # Right Panel: Performance across Cue Conditions at Delta = 45 degrees
    ax2 = axes[1]
    ax2.axhline(25.0, color="#aaaaaa", linestyle=":", linewidth=1.5)

    mode_keys = [m[0] for m in MODES]
    mode_labels = [m[1].split(":")[1].strip() for m in MODES]
    x_indices = np.arange(len(mode_keys))
    n_models = len(results_dict)
    width = 0.75 / max(n_models, 1)

    for idx, (model_id, data) in enumerate(results_dict.items()):
        results = data.get("results", [])
        style = MODEL_STYLES.get(model_id, {"label": model_id.split("/")[-1], "color": "#333333", "ls": "-", "marker": "o"})

        mode_accs = []
        for mk in mode_keys:
            # Look at Delta >= 45 deg or Delta = 45 deg
            matching = [r for r in results if r.get("mode") == mk and r.get("delta") == 45]
            if not matching:
                matching = [r for r in results if r.get("mode") == mk]
            if matching:
                acc = 100.0 * sum(r["is_correct"] for r in matching) / len(matching)
            else:
                acc = 0.0
            mode_accs.append(acc)

        offsets = x_indices - (n_models - 1) * width / 2.0 + idx * width
        ax2.bar(offsets, mode_accs, width=width, label=style["label"], color=style["color"], alpha=0.85, edgecolor="black", linewidth=0.5)

    ax2.set_title("(B) Condition Sensitivity at $\\Delta = 45^\\circ$", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Scene Cue Condition", fontsize=10)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(mode_labels, rotation=20, ha="right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    pdf_path = out_fig.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to {out_fig} and {pdf_path}")

    # Generate LaTeX table
    generate_latex_table(results_dict, out_table)


def generate_latex_table(results_dict, out_table):
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & $\\Delta=0^\\circ$ & $\\Delta=45^\\circ$ & $\\Delta=90^\\circ$ & $\\Delta=135^\\circ$ & $\\Delta=180^\\circ$ & Overall \\\\",
        "\\midrule",
    ]

    for model_id, data in results_dict.items():
        results = data.get("results", [])
        label = MODEL_STYLES.get(model_id, {}).get("label", model_id.split("/")[-1])
        accs = []
        for d in DELTAS:
            matching = [r for r in results if r.get("delta") == d]
            if matching:
                acc = 100.0 * sum(r["is_correct"] for r in matching) / len(matching)
                accs.append(f"{acc:.1f}\\%")
            else:
                accs.append("--")
        total_acc = 100.0 * sum(r["is_correct"] for r in results) / max(len(results), 1)
        lines.append(f"{label} & {' & '.join(accs)} & \\textbf{{{total_acc:.1f}\\%}} \\\\")

    lines.extend([
        "\\midrule",
        "Chance (4AFC) & 25.0\\% & 25.0\\% & 25.0\\% & 25.0\\% & 25.0\\% & 25.0\\% \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ])

    table_content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(out_table)), exist_ok=True)
    with open(out_table, "w") as f:
        f.write(table_content)
    print(f"Saved LaTeX table to {out_table}")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 4 and Table for VLMs")
    parser.add_argument("--results", nargs="+", required=True, help="Result JSON files")
    parser.add_argument("--out", default="paper/fig4_vlm.png", help="Output PNG path")
    parser.add_argument("--table", default="paper/table_vlm.tex", help="Output LaTeX table path")
    args = parser.parse_args()

    results_dict = {}
    for fp in args.results:
        d = load_result(fp)
        model = d.get("summary", {}).get("model")
        if not model:
            if "72b" in fp.lower():
                model = "Qwen/Qwen2.5-VL-72B-Instruct"
            elif "32b" in fp.lower():
                model = "Qwen/Qwen2.5-VL-32B-Instruct"
            elif "7b" in fp.lower():
                model = "Qwen/Qwen2.5-VL-7B-Instruct"
            elif "3b" in fp.lower():
                model = "Qwen/Qwen2.5-VL-3B-Instruct"
            elif "2b" in fp.lower():
                model = "Qwen/Qwen2-VL-2B-Instruct"
            else:
                model = fp

        prompt_style = d.get("summary", {}).get("prompt_style", "cot")
        if "mental_rot" in fp.lower():
            prompt_style = "mental_rotation"
        elif "anchor" in fp.lower():
            prompt_style = "anchor"

        key = f"{model} ({prompt_style})" if prompt_style not in ("cot", "direct") else model
        results_dict[key] = d

    plot_vlm_scaling(results_dict, out_fig=args.out, out_table=args.table)


if __name__ == "__main__":
    main()
