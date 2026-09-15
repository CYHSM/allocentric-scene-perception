# COSYNE 2027: Experiments & Working-Memory Delay Investigation

This document tracks all computational neuroscience experiments, hypotheses, GPU allocations on `dgx2`, benchmark runs, and findings for the COSYNE 2027 submission:
**"Viewpoint-invariant scene memory dissociates cognitive maps from visual gates"**

---

## 1. Core Neuroscience Research Questions

1. **Working-Memory Delay Dynamics (2s Delay Analog)**:
   - *Biological context*: In the clinical Four Mountains Task, human observers are tested immediately (perception) or after a delay (e.g. 2s working memory maintenance). Patients with selective hippocampal damage show catastrophic viewpoint-dependent deficits across delays.
   - *Computational implementation*: In biological attractor networks, delay induces stochastic Brownian drift in neural activity ($\Delta \mathbf{z} \sim \mathcal{N}(0, \sigma^2 t \mathbf{I})$). We simulate this by injecting calibrated Gaussian noise into the **study scene latent representation** while keeping probe alternatives clean.
   - *Hypothesis*: Appearance memory (the $\Delta=0^\circ$ gate) and allocentric relational memory ($\Delta \ge 45^\circ$) will exhibit distinct decay rates under latent diffusion.

2. **Benchmark Headroom & Foil Difficulty**:
   - On the `hard` benchmark (7 m foil separation), model rotated accuracy is near chance floor (~25%), causing a severe floor effect.
   - To observe a continuous, graded memory decay curve across delay/noise scales, we must evaluate open models across simpler foil difficulty tiers (`mid_m`: 31 m, `easy_m`: 43 m) and stimulus modes (`c0_shape_colour`).

---

## 2. Cluster Resources & GPU Allocation (`dgx2`)

- **Host**: `dgx2` (`/raid/nbe_tmp/markus_frey/asp`)
- **Hardware**: 8 $\times$ NVIDIA A100-SXM4-80GB
- **Live Status** (Mon Sep 14 14:38:45 2026):
  - GPUs 0–3: Occupied by other user (VLLM processes, 100% util, 75 GB used)
  - **GPU 4: FREE (1 MiB used, 0% util, 80 GB available)** $\leftarrow$ **PRIMARY PINNED DEVICE**
  - GPU 5: 1.2 GB used, 0% util (collab-wm)
  - GPU 6: 1.2 GB used, 54% util (collab-wm)
  - **GPU 7: FREE (1 MiB used, 0% util, 80 GB available)** $\leftarrow$ **SECONDARY PINNED DEVICE**
- **Pinned Execution**: All jobs must use `GPUS=4` or `GPUS=7` to avoid colliding with other users.

---

## 3. Open Models Baseline Matrix across Difficulty Tiers

Evaluated across 100 stratified trials per benchmark (chance level: 25.0%):

| Model | Parameters | Benchmark | Overall Acc | Gate ($\Delta=0^\circ$) | Rotated ($\Delta \ge 45^\circ$) | Status |
|---|---|---|---|---|---|---|
| **Qwen2.5-VL-7B-Instruct** | 7B | `4afc_hard` | **36.0%** | 55.0% | **31.2%** | Complete |
| Qwen2.5-VL-7B-Instruct | 7B | `4afc_mid_m` | 33.0% | 65.0% | 25.0% | Complete |
| Qwen2.5-VL-7B-Instruct | 7B | `4afc_easy_m`| 35.0% | 60.0% | 28.7% | Complete |
| **Qwen2.5-VL-72B-Instruct** | 72B | `4afc_hard` | 31.0% | 75.0% | 20.0% | Complete |
| Qwen2.5-VL-72B-Instruct | 72B | `4afc_mid_m` | 37.0% | 90.0% | 23.8% | Complete |
| Qwen2.5-VL-72B-Instruct | 72B | `4afc_easy_m`| **42.0%** | 85.0% | **31.2%** | Complete |
| **InternVL3-38B-hf** | 38B | `4afc_hard` | 34.0% | **85.0%** | 21.2% | Complete |
| InternVL3-38B-hf | 38B | `4afc_mid_m` | 32.0% | **90.0%** | 17.5% | Complete |
| InternVL3-38B-hf | 38B | `4afc_easy_m`| 39.0% | **100.0%** | 23.8% | Complete |
| **Qwen2.5-VL-32B-Instruct** | 32B | `4afc_hard` | 29.0% | 75.0% | 17.5% | Complete |
| Qwen2.5-VL-32B-Instruct | 32B | `4afc_mid_m` | 39.0% | 85.0% | 27.5% | Complete |
| Qwen2.5-VL-32B-Instruct | 32B | `4afc_easy_m`| 35.0% | 80.0% | 23.8% | Complete |
| **Qwen2.5-VL-3B-Instruct** | 3B | `4afc_hard` | 29.0% | 30.0% | 28.7% | Complete |
| Qwen2.5-VL-3B-Instruct | 3B | `4afc_mid_m` | 32.0% | 60.0% | 25.0% | Complete |
| Qwen2.5-VL-3B-Instruct | 3B | `4afc_easy_m`| 29.0% | 50.0% | 23.8% | Complete |
| **InternVL3.5-38B-HF** | 38B | `4afc_hard` | 28.0% | 65.0% | 18.8% | Complete |
| InternVL3.5-38B-HF | 38B | `4afc_mid_m` | *TBD* | *TBD* | *TBD* | Queued |
| InternVL3.5-38B-HF | 38B | `4afc_easy_m`| *TBD* | *TBD* | *TBD* | Queued |
| **InternVL3.5-14B-HF** | 14B | `4afc_hard` | 31.0% | 50.0% | 26.2% | Complete |
| InternVL3.5-14B-HF | 14B | `4afc_mid_m` | *TBD* | *TBD* | *TBD* | Queued |
| **InternVL3.5-14B-HF** | 14B | `4afc_easy_m`| **27.0%** | **65.0%** | **17.5%** | **Complete** |
| **InternVL3.5-8B-HF** | 8B | `4afc_hard` | 28.0% | 55.0% | 21.2% | Complete |
| InternVL3.5-8B-HF | 8B | `4afc_mid_m` | *TBD* | *TBD* | *TBD* | Queued |
| **InternVL3.5-8B-HF** | 8B | `4afc_easy_m`| **39.0%** | **75.0%** | **30.0%** | **Complete** |

*(For comparison: Human baseline on hard is 86% overall, 100% gate, 82% rotated; Gemini 3.8 Flash on easy_m is 84% overall, 90% gate, 82.5% rotated).*

---

## 4. Work in Progress & Active Queues

### A. Queue 1: InternVL3.5 Family on `easy_m` and `mid_m`
- **Objective**: Complete the open-weights matrix on simpler foils to identify whether any InternVL3.5 model exhibits higher rotated headroom than Qwen2.5-VL-7B (31.2%).
- **Runner**: `scripts/launch_local_queue.sh` pinned to `GPUS=4`.
- **Target Models**: `OpenGVLab/InternVL3_5-38B-HF`, `InternVL3_5-14B-HF`, `InternVL3_5-8B-HF`.

### B. Queue 2: Latent Noise Sweep (2s Working Memory Delay)
- **Objective**: Measure psychometric decay curves $\text{Accuracy}(\sigma)$ under simulated working-memory drift.
- **Model**: `Qwen/Qwen2.5-VL-7B-Instruct` (primary workhorse).
- **Intervention**: Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ injected solely into the study scene visual tokens.
- **Noise Levels**: $\sigma \in \{0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0\}$.
- **Conditions**:
  1. Gate maintenance vs Rotated maintenance on `4afc_hard`.
  2. Memory decay on `4afc_easy_m` and `4afc_mid_m` (simpler foils).
  3. Memory decay on `c0_shape_colour` (high baseline headroom: 50% at $\sigma=0$).

---

## 5. Completed Runs: Open Models on `easy_m` and `mid_m`

Updated matrix incorporating the completed InternVL3.5 runs on `dgx2`:

| Model | Parameters | Benchmark | Overall Acc | Gate ($\Delta=0^\circ$) | Rotated ($\Delta \ge 45^\circ$) | Status |
|---|---|---|---|---|---|---|
| **InternVL3.5-8B-HF** | 8B | `4afc_easy_m` | **39.0%** | **75.0%** | **30.0%** | Complete |
| InternVL3.5-8B-HF | 8B | `4afc_mid_m` | 33.0% | 65.0% | 25.0% | Complete |
| InternVL3.5-14B-HF | 14B | `4afc_easy_m` | 27.0% | 65.0% | 17.5% | Complete |
| InternVL3.5-14B-HF | 14B | `4afc_mid_m` | 32.0% | 60.0% | 25.0% | Complete |
| InternVL3.5-38B-HF | 38B | `4afc_easy_m` | 33.0% | 55.0% | 27.5% | Complete |
| InternVL3.5-38B-HF | 38B | `4afc_mid_m` | 34.0% | 70.0% | 25.0% | Complete |

---

## 6. Prompt Calibration Study on `c0_shape_colour` (Qwen2.5-VL-7B)

*Full report and methodology: [EXPERIMENT_PROMPTS_C0.md](file:///Users/markus/Documents/Github/allocentric-scene-perception_claude/EXPERIMENT_PROMPTS_C0.md)*

We conducted a 14-condition prompt calibration experiment on `c0_shape_colour` (100 trials, 1,400 trials evaluated on `dgx2` GPU 7) to investigate if cognitive scaffolding could elevate open-weights allocentric performance:

| Strategy | Prompt Style | Overall Acc | Gate ($\Delta=0^\circ$) | Rotated ($\Delta \ge 45^\circ$) | Key Takeaway |
|:---|:---|:---:|:---:|:---:|:---|
| **Novel Falsification** | `c0_falsification` | **31.0%** | **65.0%** | 22.5% | Highest gate headroom (+20% over baseline); balanced choices |
| **Baseline CoT** | `cot_anyview` | **31.0%** | 45.0% | **27.5%** | Standard CoT; strong Option 4 recency bias (51%) |
| **Novel Vector Triangulation** | `c0_anchor_triangulation` | 30.0% | 50.0% | 25.0% | Peak performance at $135^\circ$ inversion (40%); balanced choices |
| **Novel BBB Translation** | `c0_ego_to_allo` | 29.0% | 50.0% | 23.8% | Egocentric depth to allocentric bearing translation |
| **Legacy Falsification** | `elimination` | 28.0% | 60.0% | 20.0% | Mountain prompt semantics; good gate, weak rotated |
| **Zero-Shot Direct** | `direct` | 27.0% | 55.0% | 20.0% | Direct choice without CoT |
| **Novel Mental Rotation** | `c0_mental_rotation` | 26.0% | 60.0% | 17.5% | Stimulus-aligned 3D rotation; gate 60%, rotated near chance |
| **Legacy Mental Rotation** | `mental_rotation` | 24.0% | 50.0% | 17.5% | Mountain prompt semantic mismatch |
| **Neutral Human Text** | `neutral_anyview` | 23.0% | 25.0% | 22.5% | Total positional collapse: 90% Option 1 choices |

### Takeaways:
1. **Gate Optimization**: `c0_falsification` pushed the appearance gate from 45.0% to **65.0%** (13/20), establishing strong above-chance preservation.
2. **Positional Bias Mitigation**: Structured relational prompts (`c0_anchor_triangulation`, `c0_falsification`) successfully broke the ~54% Option 4 recency prior.
3. **Capacity Limit**: Across 80 rotated trials per prompt, rotated accuracy tightly bounded between 17.5% and 27.5% (chance = 25.0%). The apparent 50% rotated finding in preliminary runs was a small-sample artifact ($N=16$).

---

## 7. Working-Memory Delay Latent Perturbation Sweep (Qwen2.5-VL-7B)

Evaluated on `data/vlm_benchmark_4afc_easy_m.json` across 7 temporal delay points ($t \in \{0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0\}\text{s}$) with diffusion coefficient $D = 0.05$:

| Simulated Delay ($t$) | Latent Noise ($\sigma = \sqrt{2Dt}$) | Overall Acc | Gate ($\Delta=0^\circ$) | Rotated ($\Delta \ge 45^\circ$) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$0.0\,\text{s}$ (Baseline)** | $0.000$ | 26.0% | 45.0% | 21.25% | Baseline floor on rotated foils |
| **$0.5\,\text{s}$** | $0.224$ | 31.0% | 50.0% | 26.25% | Position prior noise interaction |
| **$1.0\,\text{s}$** | $0.316$ | 29.0% | 50.0% | 23.75% | Stable gate |
| **$2.0\,\text{s}$ (Clinical 4MT)** | $0.447$ | 28.0% | 50.0% | 22.50% | Clinical working memory analog |
| **$4.0\,\text{s}$** | $0.632$ | 32.0% | 60.0% | 25.00% | Fluctuations within $\pm 8.6\%$ Wilson CI |
| **$8.0\,\text{s}$** | $0.894$ | **25.0%** | 45.0% | 20.00% | **Exact chance floor (25.0% overall)** |
| **$16.0\,\text{s}$** | $1.265$ | 30.0% | 50.0% | 25.00% | Degraded representation dominated by priors |

*Figures generated*: `paper/cosyne/figures/working_memory_decay_qwen7b_easym.pdf` and `.png`.
