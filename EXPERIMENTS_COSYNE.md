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
