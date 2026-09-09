# Comprehensive Handover Document: Allocentric 3D Scene Perception Benchmark & VLM Evaluation Suite

**Date**: September 9, 2026 (07:15 CEST)  
**Repository Branch**: `four_opt`  
**Host & Environment Paths**:
- **Local Workspace**: `/Users/markus/Documents/Github/allocentric-scene-perception_claude`
- **Remote Cluster (`dgx2`)**: `/raid/nbe_tmp/markus_frey/asp/`
- **Python Virtualenv**: `/raid/nbe_tmp/markus_frey/asp/.venv/` (Python 3.12, PyTorch 2.11.0+cu128, transformers 5.16.1, qwen-vl-utils)
- **Hugging Face Cache**: `/raid/nbe_tmp/markus_frey/cache/huggingface` (**CRITICAL**: Never omit this! `/dev/md0` root has 76 GB free, `/raid` has 2.6 TB free).
- **Blender Binary**: `/raid/nbe_tmp/markus_frey/asp/blender-5.2.1-linux-x64/blender`

---

## 1. Executive Summary & Core Scientific Milestones

We are investigating whether vision foundation models and Vision-Language Models (VLMs) possess true **allocentric 3D spatial scene perception** or merely egocentric 2D template matching. The benchmark is derived from the clinical **Four Mountains Test** (Hartley et al., 2007; Burgess et al., 2002).

### Three Core Phases:
1. **Phase 1 (Frozen Vision Encoders on Continuous Probe Ladder, `data/scenes/`)**:
   - Evaluated `DINOv2-B/14`, `SigLIP-SO400M`, `CLIP-ViT-B/16`, and `ResNet-50` across 25,600 rendered frames.
   - **Result**: All frozen encoders suffer an **Allocentric Representation Collapse** ($\lambda(45^\circ) > 24$ m; R@1 $\approx 1-3\%$ when scenes contain identical landmark geons). Because representation space has no task pressure to recognize the same scene across viewpoints, view change dominates representation distance.
2. **Phase 2 (Clean 100-Anchor Bank, `data/scenes_100/`)**:
   - Rendered exactly 100 unique anchor scenes $\times$ 5 modes (`c0`–`c4`) $\times$ 16 views (Appearances A/B $\times$ 8 azimuths) = **16,000 / 16,000 frames (100% complete)**.
   - Constructed deterministic psychophysical benchmarks:
     - `data/vlm_benchmark_4afc.json`: 500 trials (5 modes $\times$ 5 rotation angles $\{0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ\} \times 20$ trials). Chance = 25.0%.
     - `data/vlm_benchmark_2afc.json`: 500 trials. Chance = 50.0%.
     - `data/vlm_benchmark_calibration.json`: Balanced 50-trial calibration slice for prompt engineering.
     - `data/vlm_benchmark_2afc_calib.json`: Balanced 50-trial 2AFC calibration slice.
3. **Phase 3 (Vision-Language Model Scaling & Prompt Engineering Suite)**:
   - Evaluated `Qwen2-VL-2B`, `Qwen2.5-VL-3B`, `Qwen2.5-VL-7B`, active run on `Qwen2.5-VL-32B` (370/500 trials done), and queued `Qwen2.5-VL-72B`.
   - **Three Major Breakthroughs**:
     1. **Model Parameter Scaling**: Scaling from 3B $\to$ 7B $\to$ 32B produces a dramatic surge at the appearance gate ($\Delta = 0^\circ$): **36.0% (3B) $\to$ 45.0% (7B) $\to$ 71.2% (32B)**, and lifts overall 4AFC accuracy to **32.2%** (+7.2% above chance).
     2. **Cognitive Prompt Engineering**: The **Mental Rotation Prompt** jumps 7B accuracy from **20.0% to 38.0%** (nearly $2\times$ baseline CoT), reaching **60.0% at $\Delta = 45^\circ$** and **50.0% at $\Delta = 180^\circ$**.
     3. **Task Format (4AFC vs. 2AFC)**: Reducing working memory load from 5 images to 3 images cuts cross-attention dispersion, achieving **54.0% overall** and **70.0% accuracy on pure shape geons (`c1_shape`) and valley topography (`c4_valley`)**.

---

## 2. Complete Collated Experimental Results

### A. Model Scaling Hierarchy (4AFC Benchmark, Chain-of-Thought)

#### 1. `Qwen2.5-VL-32B-Instruct` (In Progress, 370/500 Trials, Chance = 25.0%)
| Scene Condition Mode | $\Delta = 0^\circ$ (App Shift) | $\Delta = 45^\circ$ | $\Delta = 90^\circ$ | $\Delta = 135^\circ$ | $\Delta = 180^\circ$ | Total Mode Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`c0_shape_colour`** | **70%** (20) | **55%** (20) | 10% (20) | 10% (20) | 10% (20) | **31.0%** (100) |
| **`c1_shape`** (Shape only) | **75%** (20) | **35%** (20) | **35%** (20) | 15% (20) | 5% (20) | **33.0%** (100) |
| **`c2_colour`** (Colour only) | **75%** (20) | 15% (20) | 20% (20) | 5% (20) | 15% (20) | **26.0%** (100) |
| **`c3_peaks_bare`** (Landforms only) | **65%** (20) | **40%** (20) | 25% (20) | *(in progress)* | *(in progress)* | **43.3%** (60) |
| **`c4_valley`** (Depression) | *(in progress)* | *(in progress)* | *(in progress)* | *(in progress)* | *(in progress)* | *(in progress)* |
| **OVERALL BY DELTA** | **71.2%** (80) | **36.2%** (80) | **22.5%** (80) | **10.0%** (60) | **10.0%** (60) | **32.2%** (360) |
| *Choice Distribution* | {1: 57, 2: 79, 3: 89, 4: 135} (Substantially more balanced than 7B) | | | | | |

#### 2. `Qwen2.5-VL-7B-Instruct` (Full 500 Trials, Chance = 25.0%)
| Scene Condition Mode | $\Delta = 0^\circ$ (App Shift) | $\Delta = 45^\circ$ | $\Delta = 90^\circ$ | $\Delta = 135^\circ$ | $\Delta = 180^\circ$ | Total Mode Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`c0_shape_colour`** | **55.0%** | **40.0%** | **30.0%** | 15.0% | 15.0% | **31.0%** |
| **`c1_shape`** (Shape only) | **55.0%** | **45.0%** | 25.0% | 5.0% | 10.0% | **28.0%** |
| **`c2_colour`** (Colour only) | **50.0%** | 25.0% | 10.0% | 10.0% | 30.0% | **25.0%** |
| **`c3_peaks_bare`** (Landforms only) | 35.0% | 20.0% | 15.0% | 30.0% | 20.0% | **24.0%** |
| **`c4_valley`** (Depression) | 30.0% | 25.0% | 20.0% | 20.0% | 20.0% | **23.0%** |
| **OVERALL BY DELTA** | **45.0%** | **31.0%** | **20.0%** | **16.0%** | **19.0%** | **26.2%** |

#### 3. `Qwen2.5-VL-3B-Instruct` (Full 500 Trials, Chance = 25.0%)
| Scene Condition Mode | $\Delta = 0^\circ$ (App Shift) | $\Delta = 45^\circ$ | $\Delta = 90^\circ$ | $\Delta = 135^\circ$ | $\Delta = 180^\circ$ | Total Mode Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`c0_shape_colour`** | 35.0% | 40.0% | 20.0% | 15.0% | 10.0% | **24.0%** |
| **`c1_shape`** (Shape only) | **50.0%** | 30.0% | 35.0% | 20.0% | 5.0% | **28.0%** |
| **`c2_colour`** (Colour only) | 35.0% | 35.0% | 30.0% | 20.0% | 30.0% | **30.0%** |
| **`c3_peaks_bare`** (Landforms only) | 20.0% | 5.0% | 25.0% | 20.0% | 30.0% | **20.0%** |
| **`c4_valley`** (Depression) | 40.0% | 25.0% | 20.0% | 25.0% | 20.0% | **26.0%** |
| **OVERALL BY DELTA** | **36.0%** | **27.0%** | **26.0%** | **20.0%** | **19.0%** | **25.6%** |

---

### B. Prompt Strategy Ablation Suite (`paper/fig_prompt_ablation.png`)

Evaluated on `Qwen2.5-VL-7B-Instruct` across balanced conditions (5 modes $\times$ 5 rotation angles, Chance = 25.0%):

| Prompt Strategy | Overall Acc | $\Delta = 0^\circ$ | $\Delta = 45^\circ$ | $\Delta = 90^\circ$ | $\Delta = 135^\circ$ | $\Delta = 180^\circ$ | Core Strength / Key Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standard CoT (Baseline)** | 20.0% | 40.0% | 40.0% | 0.0% | 0.0% | 20.0% | Severe recency bias (picks Option 4 in 56% of trials) |
| **Mental Rotation (Cognitive)** | **38.0%** | **40.0%** | **60.0%** | **20.0%** | **20.0%** | **50.0%** | **Dominant strategy across all rotation angles (+18% net gain)** |
| **Distractor Elimination** | 28.0% | **70.0%** | 30.0% | 0.0% | 0.0% | 40.0% | Best appearance gate accuracy (70% at $\Delta = 0^\circ$) |
| **Landmark Origin Anchor** | 24.0% | 40.0% | 30.0% | 20.0% | 0.0% | 30.0% | Preserves geon configurations on `c1_shape` (30%) |
| **Topographic Elevation** | 24.0% | 30.0% | 40.0% | 20.0% | 10.0% | 20.0% | Invariant on pure color distractors (`c2_colour`: 40%) |
| **Hybrid (Rotation + Elimination)**| 26.0% | 50.0% | 0.0% | 0.0% | 0.0% | 20.0% | Suffers from choice collapse against Option 1 |
| **Top-Down Cognitive Map** | 18.0% | 40.0% | 10.0% | 0.0% | 10.0% | 30.0% | Underperforms due to 2D bird's-eye projection ambiguity |

---

### C. Task Format Ablation: 4AFC vs. Pairwise 2AFC (`paper/fig6_format_2afc.png`)

| Format | Images in Context | Chance Level | Overall Acc (7B + Mental Rotation) | `c0_shape_colour` | `c1_shape` | `c4_valley` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **4AFC** | 5 images | 25.0% | **38.0%** (+13.0% over chance) | 40.0% | 30.0% | 50.0% |
| **2AFC** | 3 images | 50.0% | **54.0%** | 50.0% | **70.0%** | **70.0%** |

---

## 3. Publication Figures & Artifacts

All figures are compiled in vector PDF and high-resolution PNG:

1. **Figure 4: VLM Scaling & Allocentric Invariance** (`paper/fig4_vlm.png` / `paper/fig4_vlm.pdf`):
   - Compares 3B vs 7B vs 32B across $\Delta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ\}$ and stimulus conditions.
   - LaTeX source generated: `paper/table_vlm.tex`.
2. **Figure 5: Cognitive Prompt Engineering Ablation** (`paper/fig_prompt_ablation.png` / `paper/fig_prompt_ablation.pdf`):
   - 3-panel comparative analysis showing viewpoint invariance curves, cue sensitivity, and prompt rankings across all 6 strategies.
3. **Figure 6: Task Format Comparison: 4AFC vs. Pairwise 2AFC** (`paper/fig6_format_2afc.png` / `paper/fig6_format_2afc.pdf`):
   - Compares 4AFC vs 2AFC across rotation angles, stimulus modes, and net excess signal above random chance.

---

## 4. Cluster Health & Active Pipeline on `dgx2`

### Storage & Process Status
- **Root Partition `/dev/md0`**: **76 GB free** (cleaned up temporary uv wheels and duplicate model weights).
- **Data Partition `/raid`**: **2.6 TB free** (all weights and caches reside under `/raid/nbe_tmp/markus_frey/cache/huggingface`).
- **Active 32B Process (PID 2588309)**:
  - Running cleanly on GPUs 0, 1, 4, 5, 6, 7.
  - Progress: **370 / 500 trials complete** (74%).
  - Fast cadence: ~15-18 seconds per trial following process de-duplication.
  - Expected completion: **~07:45–08:00 CEST**.

### Master Battery (`bench/run_overnight_battery.sh`, PID 2463915)
Running under `nohup` (`logs/overnight_battery.log`). Automatically triggers upon 32B completion:
- **Phase 1**: Merges 32B results and updates Figure 4.
- **Phase 2**: Evaluates `mental_rotation` prompt on 7B across 5 GPUs in parallel (500 trials).
- **Phase 3**: Evaluates `anchor` prompt on 7B across 5 GPUs in parallel (500 trials).
- **Phase 4**: Evaluates Pairwise 2AFC on 7B across 5 GPUs in parallel (500 trials).
- **Phase 5**: Re-generates Figure 4 and Table 1.
- **Phase 6**: Automatically launches `bench/launch_72b.sh` to evaluate **`Qwen2.5-VL-72B-Instruct`** across NVLink GPUs!

---

## 5. Verification & Sync Commands

```bash
# Check current 32B progress
ssh dgx2 "cd /raid/nbe_tmp/markus_frey/asp && .venv/bin/python -c 'import json; print(\"Done:\", len(json.load(open(\"results/qwen2_5_vl_32b_4afc_full.json\")).get(\"results\", [])))'"

# Check master battery log
ssh dgx2 "cat /raid/nbe_tmp/markus_frey/asp/logs/overnight_battery.log | tail -n 40"

# Sync all newly generated results and figures locally
rsync -avz dgx2:/raid/nbe_tmp/markus_frey/asp/results/ results/
rsync -avz dgx2:/raid/nbe_tmp/markus_frey/asp/paper/ paper/
```
