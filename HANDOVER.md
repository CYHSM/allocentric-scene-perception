# Handover: Four Mountains Allocentric Scene Perception Benchmark

**Branch**: `vlm-paper-clean` (Parent: `four-mountains-blender`, fully synced to `origin/four-mountains-blender`)  
**Date**: September 9, 2026  
**Objective**: Clean, publication-grade benchmark battery evaluating **Allocentric 3D Scene Perception** in Vision-Language Models (VLMs) against human performance.

---

## 1. Clean Branch Architecture: What Was Stripped & Kept

All legacy exploratory work from earlier phases is safely preserved in git history under commit `7466363` on `origin/four-mountains-blender`. 

In this branch (`vlm-paper-clean`), the codebase is stripped to focus purely on the VLM publication:

### What Was Stripped:
1. **Phase 1 Frozen Encoder Continuous Probe Ladder**:
   - `data/scenes/` (25,600 legacy continuous probe frames).
   - Embedding distance scripts (`bench/exchange.py`, `bench/metrics.py`, `bench/merge_bank.py`).
   - Retrieval and lambda figures (`bench/figure_results.py`, `figures/archive_lambda/`, `figures/fig_rsa.png`).
2. **Text Coordinate Channel**:
   - Symbolic serialiser & tests (`bench/text_views.py`, `bench/build_text_benchmark.py`, `bench/tests/test_text_views.py`, `bench/run_text_task.sh`).
   - Text datasets (`data/text_benchmark_*.json`).
   - *Rationale*: Providing numeric coordinates transforms the task from spatial vision perception into symbolic algebra/Pythagorean distance matrix matching. The scientific core of this paper is visual perception under perspective shifts.
3. **Temporary Logs & Intermediate Outputs**:
   - `results/corrupt/`, `results/qwen2_vl_2b_*`, `results/qwen2_5_vl_7b_smoke.json`, `results/qwen2_5_vl_3b_cot_test.json`.

### What Is Kept (The Publication Pipeline):
* **`data/scenes_100/`**: 100 anchor scenes $\times$ 5 modes (`c0`–`c4`) $\times$ 16 views (Appearances A/B $\times$ 8 azimuths) = 16,000 pristine rendered frames.
* **`data/vlm_benchmark_4afc.json`**: 500 trials (5 modes $\times$ 5 deltas $\times$ 20 trials, Chance = 25.0%).
* **`data/vlm_benchmark_2afc.json`**: 500 trials (5 modes $\times$ 5 deltas $\times$ 20 trials, Chance = 50.0%).
* **`data/vlm_benchmark_calibration.json` & `vlm_benchmark_2afc_calib.json`**: Balanced 50-trial slices for rapid calibration and prompt ablation.
* **`results/`**: All completed 500-trial benchmark runs (3B, 7B, 32B, 7B Mental Rotation, 7B Anchor, 7B 2AFC) + calibration runs for prompt ablations.
* **`human_task/`**: Zero-dependency browser evaluation app (`index.html`), task slice (`task.json`), and human baseline (`human_p01_2afc.json`).
* **`paper/`**: LaTeX draft (`main.tex`), tables (`table_vlm.tex`), figures (`fig4_vlm.*`, `fig6_format_2afc.*`, `fig_prompt_ablation.*`).

---

## 2. Benchmark Design & Visual Cue Ladder

The benchmark adapts the clinical **Four Mountains Test** (Burgess et al., 2002; Hartley et al., 2007) to probe whether VLMs form an internal allocentric 3D cognitive map or rely on 2D retinotopic template matching.

### Experimental Controls:
* **Rotation Angle ($\Delta$)**: Parametric sweep at $0^\circ$ (appearance gate), $45^\circ$, $90^\circ$, $135^\circ$, and $180^\circ$ (opposite viewpoint).
* **Appearance Resampling**: All candidate views have weather, sun elevation, atmospheric haze, and ground texture resampled (Appearance A $\to$ Appearance B). A model cannot rely on pixel/color matching.
* **The 5-Condition Cue Ladder**:
  * **`c0_shape_colour`**: Distinctive 3D geon shapes + distinctive colors (full cues).
  * **`c1_shape`**: Distinctive 3D geon shapes, neutral gray textures (color removed).
  * **`c2_colour`**: Identical mountain shapes, distinctive colors (shape removed).
  * **`c3_peaks_bare`**: Continuous naturalistic mountain terrain (texture/geons removed).
  * **`c4_valley`**: Central topographic depression surrounded by natural peaks.

---

## 3. Key Findings & Metric Framework (Adopting SpinBench Insights)

SpinBench (arXiv:2509.25390, 2025) recently evaluated 43 VLMs on general spatial rotation. Our benchmark extends and deepens these findings with a clinical neuropsychological paradigm, parametric degradation curves, and a visual cue ladder.

### Core Metrics:

#### 1. Raw Accuracy & Cohen's Kappa ($\kappa$)
Because we evaluate both **4AFC** ($p_{\text{chance}} = 0.25$) and **2AFC** ($p_{\text{chance}} = 0.50$), raw accuracy is not directly comparable across formats. We use **Cohen's Kappa**:
$$\kappa = \frac{\text{Acc} - p_{\text{chance}}}{1 - p_{\text{chance}}}$$
* **4AFC**: $\kappa = \frac{\text{Acc} - 0.25}{0.75}$
* **2AFC**: $\kappa = \frac{\text{Acc} - 0.50}{0.50}$

#### Current Collated Performance (N=500 each):
| Agent / Model | Format | Chance | Raw Acc | Cohen's $\kappa$ | $\Delta=0^\circ$ | $\Delta=45^\circ$ | $\Delta=90^\circ$ | $\Delta=135^\circ$ | $\Delta=180^\circ$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Human (p01, N=50)** | 2AFC | 50% | **98.0%** | **0.960** | 100% | 90% | 100% | 100% | 100% |
| **Qwen2.5-VL-7B (2AFC)** | 2AFC | 50% | **56.4%** | **0.128** | 70.0% | 70.0% | 57.0% | 47.0% | 38.0% |
| **Qwen2.5-VL-32B (4AFC)**| 4AFC | 25% | **31.4%** | **0.085** | 67.0% | 34.0% | 24.0% | 14.0% | 18.0% |
| **Qwen2.5-VL-7B (Mental Rot)**| 4AFC | 25% | **29.4%** | **0.059** | 49.0% | 36.0% | 25.0% | 17.0% | 20.0% |
| **Qwen2.5-VL-7B (Baseline CoT)**| 4AFC | 25% | **26.2%** | **0.016** | 45.0% | 31.0% | 20.0% | 16.0% | 19.0% |
| **Qwen2.5-VL-3B (Baseline CoT)**| 4AFC | 25% | **25.6%** | **0.008** | 36.0% | 27.0% | 26.0% | 20.0% | 19.0% |

#### 2. Viewpoint Invariance Index (VII)
$$\text{VII} = \frac{\text{Mean Accuracy}(\Delta \ge 45^\circ)}{\text{Accuracy}(\Delta = 0^\circ)}$$
* Measures how much appearance-recognition survives spatial transformation.
* Humans: $\text{VII} \approx \frac{97.5\%}{100\%} = \mathbf{0.975}$ (near-perfect invariance).
* 32B VLM: $\text{VII} = \frac{22.5\%}{67.0\%} = \mathbf{0.336}$ (dramatic collapse under rotation).

#### 3. The Retinotopic Trap (Below-Chance Performance at $180^\circ$)
* At $\Delta = 135^\circ$ and $\Delta = 180^\circ$, 32B drops to **14% - 18%** in 4AFC (chance = 25%) and 7B drops to **38%** in 2AFC (chance = 50%).
* **Scientific Significance**: Models are not answering randomly; they are actively biased by 2D retinotopic template matching. When rotated $180^\circ$, left becomes right; models naively pick foils that preserve 2D left-right screen positions.

---

## 4. How to Run Frontier Models via OpenRouter

Frontier vision models (GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Pro) can be evaluated directly via `bench/run_openrouter.sh`.

### Setup:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### 1. Free Smoke Test (Validates prompt formatting & image base64 encoding):
```bash
bash bench/run_openrouter.sh meta-llama/llama-3.2-11b-vision-instruct:free 20 0
```

### 2. Evaluated on Human-Matched Slice (50 Trials, Paired with Human p01):
```bash
# GPT-4o (budget: $1.50)
BENCHMARK=human_task/task.json bash bench/run_openrouter.sh openai/gpt-4o 50 1.50

# Claude 3.5 Sonnet (budget: $1.50)
BENCHMARK=human_task/task.json bash bench/run_openrouter.sh anthropic/claude-3.5-sonnet 50 1.50

# Gemini 2.0 Flash (budget: $0.50)
BENCHMARK=human_task/task.json bash bench/run_openrouter.sh google/gemini-2.0-flash-001 50 0.50
```

### 3. Evaluated on Full 500-Trial 2AFC / 4AFC Benchmarks:
```bash
BENCHMARK=data/vlm_benchmark_2afc.json bash bench/run_openrouter.sh openai/gpt-4o 500 12.00
```

---

## 5. Human Baseline: Adding 50 More Trials

Human subject `p01` completed 50 trials (`human_task/human_p01_2afc.json`) scoring **98%**.

To reach 100 human trials:
1. Open `human_task/index.html` in any web browser.
2. Enter Participant ID: `p02`.
3. Complete the 50 trials (takes ~5–7 minutes).
4. Save the resulting JSON as `human_task/human_p02_2afc.json`.
5. The analysis scripts (`bench/agents.py`, `bench/analyze_results.py`) automatically pool `human:p01` and `human:p02` into `Human (N=100)`.

---

## 6. Verifying & Compiling Paper Figures

All paper figures can be regenerated with a single command:
```bash
# Figure 4 & LaTeX Table (Scaling & Invariance curves across deltas):
python3 bench/figure_vlm.py \
  --results results/qwen2_5_vl_3b_4afc_full.json \
            results/qwen2_5_vl_7b_4afc_full.json \
            results/qwen2_5_vl_32b_4afc_full.json \
  --out paper/fig4_vlm.png

# Figure 5 (Prompt Strategy Ablation: Mental Rotation, Anchor, etc.):
python3 bench/figure_prompt_ablation.py

# Figure 6 (Task Format Comparison: 4AFC vs 2AFC):
python3 bench/figure_format_2afc.py

# Summarize all model scores in terminal:
python3 bench/analyze_results.py
```
