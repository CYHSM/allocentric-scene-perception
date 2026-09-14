# Archived and Superseded Benchmark Scripts

Scripts here are preserved for historical provenance and exploration records,
mirroring `results/archive/` and `human_task/hard/superseded/`. None of these
are part of the active publication pipeline.

The live pipeline is `bash bench/build_paper.sh` (which orchestrates
`collate.py`, `make_tables.py`, `make_figures.py`, and `make_prompts.py`).

---

## 1. Superseded Figure Scripts

These one-off exploratory plotting scripts each read result JSONs directly.
They are superseded by `bench/make_figures.py` and `bench/build_paper.sh`,
which regenerate all four figures and table strips for the paper from collated data.

| Script | Purpose / History |
|---|---|
| `figure_vlm.py` | Earliest raw accuracy vs scale plot on pilot 2AFC/4AFC results. |
| `figure_dissociation.py` | Pilot dissociation plot between gate (delta=0) and rotated accuracy. |
| `figure_double_dissociation.py` | Early human vs Gemini double dissociation figure; superseded by Figure 1/2 in `make_figures.py`. |
| `figure_format_2afc.py` | Pilot comparison between 2AFC and 4AFC task framing. |
| `figure_modes_full.py` | Exploration of accuracy across the 5 visual stimulus modes. |
| `figure_mode_detail.py` | Breakdown of foil difficulty vs delta across modes. |
| `figure_model_modes.py` | Radar/bar chart breakdown across stimulus modes. |
| `figure_prompt_ablation.py` | Old prompt strategy ablation plot on superseded benchmark slice. |

---

## 2. Superseded Analysis & Collation Scripts

Superseded by `bench/collate.py` (the single source of truth for processing run outputs).

| Script | Purpose / History |
|---|---|
| `analyze_results.py` | Early exploratory script to print model accuracy breakdowns. |
| `analyze_vii.py` | Early standalone VII / d-prime calculation script; unified into `bench/collate.py` and `bench/vii.py`. |
| `collate_overnight.py` | Ad-hoc collation script from the 8-9 September overnight run. |
| `merge_vlm_results.py` | Ad-hoc utility to concatenate partitioned result JSON files. |

---

## 3. Legacy Launch Scripts

Superseded by `bench/launch_local_queue.sh`, `bench/launch_prompt_sweep.sh`, and `bench/run_openrouter.sh`.

| Script | Purpose / History |
|---|---|
| `run_overnight_battery.sh` | Pilot overnight test battery for early model runs. |
| `run_prompt_ablation.sh` | Early prompt ablation run script for GPU 3. |
| `launch_vlm_parallel.sh` | Legacy 5-GPU parallel evaluation launcher across stimulus modes. |
