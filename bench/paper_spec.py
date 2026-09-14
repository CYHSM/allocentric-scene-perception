"""
The locked configuration of the paper's headline arm, and the roster of
observers that ran under it.

This module exists because the same model on the same benchmark can produce two
different numbers, and nothing in a result filename says which one you are
looking at. Three separate comparisons in this project were invalidated before
the config was pinned down:

* A random-foil arm ran at `max_tokens=4000` and the hard-foil arm at 8000.
  Half the c4 replies in the short arm were truncated mid-reasoning, which read
  as "c4 is harder" until the two configs were diffed by hand.
* Two foil banks wrote to one output filename and overwrote each other.
* Six of the nine prompt styles told the model the scene *was* rotated, on
  trials where it was not. Fixing one clause took a model from 5/10 to 10/10.
  The style name did not change, so only a hash of the prompt text catches it.

So: one benchmark, one prompt, one token ceiling, one trial slice. A run that
deviates on any of them is not dropped -- it is reported with the deviation
named, and it stays out of the headline table. `collate.py` enforces this; no
figure or table script re-derives it.

Adding an observer means adding a row to `OBSERVERS`. Result files that match
the arm but are in no row are reported as unclaimed rather than silently
included, because a stale `_n500` or `_partial` file sitting in `results/` is
exactly how a superseded run gets back into a plot.
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- the arm ---

ARM = {
    "name": "4afc_hard",
    "benchmark": "data/vlm_benchmark_4afc_hard.json",
    # sha256[:12] of the benchmark file. Recomputed and checked by collate.py.
    "benchmark_sha256_12": "5e64a7575cb7",
    "n_options": 4,
    "chance": 0.25,
    "n_trials": 100,
    "sampling": "stratified_mode_x_delta",
    # Models get the chain-of-thought prompt; people get the same task without
    # the "think step by step" scaffold, which is not a thing a person needs to
    # be told. Both carry the fixed any-view clause.
    "prompt_style_model": "cot_anyview",
    "prompt_style_human": "neutral_anyview",
    # High enough that a reasoning model's hidden thinking does not eat the
    # whole budget and return empty content. 8000 was where truncation stopped.
    "max_tokens": 8000,
    "reasoning_effort": "medium",
}

# The order the modes were built in, and the order they appear in every data
# file. Nothing reads this for display.
# --------------------------------------------------------------- the ladder ---
# The same 500 questions at three foil difficulties. `--match` in
# build_hard_benchmark.py copies the trial spine -- id, mode, delta, study
# scene, both azimuths, both appearance draws, and the answer slot -- and
# redraws only the foils, so difficulty is a within-items manipulation. Without
# that the bands pick their own target scenes (483 of 500 study scenes differed
# between the random and hard banks) and a cross-band comparison is confounded
# by which places were asked about.
#
# `tag` is the substring that appears in every result filename for that bank,
# under both naming conventions, which is how a run is matched to its rung.
BANKS = [
    dict(key="hard",   tag="4afc_hard",   pct=(0, 10),
         benchmark="data/vlm_benchmark_4afc_hard.json",   median_foil_m=6.8),
    dict(key="mid",    tag="4afc_mid_m",  pct=(40, 60),
         benchmark="data/vlm_benchmark_4afc_mid_m.json",  median_foil_m=31.4),
    dict(key="easy",   tag="4afc_easy_m", pct=(90, 100),
         benchmark="data/vlm_benchmark_4afc_easy_m.json", median_foil_m=42.8),
]
PRIMARY_BANK = "hard"

# ------------------------------------------------------- the set-size axis ---
# How many landmarks a scene contains. Four separately rendered scene banks,
# identical in every render setting (8 azimuths, 640x440, 24 samples, seed 0)
# and differing only in landmark count. c0 only: every landmark must be unique
# in both shape and colour and the inventory holds six of each, so six is the
# ceiling.
#
# **Foils are uniform random draws here, not a percentile band.** Layout
# distance is a mean over landmarks, so its distribution shifts with landmark
# count and no band is comparable across N; there is no absolute band that even
# exists at all four sizes (at N=6, 95 of 100 scenes have fewer than three
# candidates within 12 m). Holding the *sampling rule* fixed is what makes the
# banks comparable instead.
#
# `median_foil_m` is therefore a measured property of each bank, not a setting.
# It falls steeply with landmark count because fewer landmarks compress layout
# space, so set size and layout distinctiveness co-vary by construction. Report
# them together; "N=1 is hard" and "the N=1 foils are 0.9 m away" are the same
# fact stated twice.
SETSIZE_BANKS = [
    dict(key="n01", n_landmarks=1, tag="setsize_n01", median_foil_m=0.9,
         anchor_separation_m=4.0),
    dict(key="n02", n_landmarks=2, tag="setsize_n02", median_foil_m=7.6,
         anchor_separation_m=10.0),
    dict(key="n04", n_landmarks=4, tag="setsize_n04", median_foil_m=23.2,
         anchor_separation_m=12.0),
    dict(key="n06", n_landmarks=6, tag="setsize_n06", median_foil_m=28.6,
         anchor_separation_m=12.0),
]

# ------------------------------------------------------------ prompt sweep ---
# One model across several instruction styles, on the locked benchmark. The
# question is whether the paper's conclusion is a property of the models or of
# how we asked.
#
# **Read the rotated trials only.** Six of the nine styles contain a clause
# asserting that the test views are shown from a different viewpoint. That is
# true at every delta >= 45 and false at delta = 0, where it is worth several
# points. Restricting to rotated trials is the subset on which all styles ask
# the same honest question, and it is the paper's dependent variable anyway.
SWEEP_MODEL = "Qwen2.5-VL-32B"
SWEEP_PATH = "results/qwen_qwen2.5-vl-32b-instruct_4afc_hard_{style}_n100.json"
SWEEP_STYLES = [
    ("cot_anyview", "chain of thought, viewpoint unstated"),
    ("cot", "chain of thought, viewpoint asserted"),
    ("mental_rotation", "instructed to mentally rotate"),
    ("anchor", "instructed to pick an anchor landmark"),
    ("birdseye", "instructed to imagine a plan view"),
    ("elimination", "instructed to eliminate alternatives"),
]

MODES = ["c0_shape_colour", "c1_shape", "c2_colour", "c3_peaks_bare", "c4_valley"]

# The order they are *plotted* in, which is the order that makes the axis mean
# something: one cue, the other cue, both cues, then the two conditions that
# remove the objects entirely. The build order c0..c4 puts the richest
# condition first and reads as an arbitrary sequence; this reads as a ladder.
PLOT_MODES = ["c2_colour", "c1_shape", "c0_shape_colour",
              "c3_peaks_bare", "c4_valley"]

MODE_LABELS = {
    "c0_shape_colour": "c0  shape + colour",
    "c1_shape":        "c1  shape only",
    "c2_colour":       "c2  colour only",
    "c3_peaks_bare":   "c3  bare peaks",
    "c4_valley":       "c4  valley",
}
# Short form for a crowded axis: the cue, not the build index.
MODE_SHORT = {
    "c2_colour":       "colour\nonly",
    "c1_shape":        "shape\nonly",
    "c0_shape_colour": "colour\n+ shape",
    "c3_peaks_bare":   "bare\npeaks",
    "c4_valley":       "valley",
}
DELTAS = [0, 45, 90, 135, 180]

# ---------------------------------------------------------- the observers ---
# access: "human" | "api" (frontier, via OpenRouter) | "local" (open weights)
# params_b: total parameters in billions; None where undisclosed. For the
# mixture models this is the total, not the active count -- the active count is
# in `active_b` because a scaling axis wants both and they differ by 10x.

OBSERVERS = [
    dict(key="human_p01", label="Human", family="human", access="human",
         params_b=None, active_b=None,
         path="human_task/hard/human_p01_4afc.json",
         prompt="neutral_anyview"),

    dict(key="luna", label="GPT-5.6 Luna", family="openai", access="api",
         params_b=None, active_b=None,
         path="results/or_openai_gpt-5.6-luna_cot_anyview_4afc_hard_n100.json",
         prompt="cot_anyview"),

    dict(key="gemini38", label="Gemini 3.8 Flash", family="gemini", access="api",
         params_b=None, active_b=None,
         path="results/or_google_gemini-3.8-flash_cot_anyview_4afc_hard_n100.json",
         prompt="cot_anyview"),

    dict(key="qwen3vl_235b_it", label="Qwen3-VL-235B (i)", family="qwen",
         access="api", params_b=235.0, active_b=22.0,
         path="results/or_qwen_qwen3-vl-235b-a22b-instruct_cot_anyview_4afc_hard_n100.json",
         prompt="cot_anyview"),

    dict(key="qwen3vl_235b_th", label="Qwen3-VL-235B (t)", family="qwen",
         access="api", params_b=235.0, active_b=22.0,
         path="results/or_qwen_qwen3-vl-235b-a22b-thinking_cot_anyview_4afc_hard_n100.json",
         prompt="cot_anyview", role="reasoning_control"),

    dict(key="qwen25vl_3b", label="Qwen2.5-VL-3B", family="qwen", access="local",
         params_b=3.0, active_b=3.0,
         path="results/qwen_qwen2.5-vl-3b-instruct_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    dict(key="qwen25vl_7b", label="Qwen2.5-VL-7B", family="qwen", access="local",
         params_b=7.0, active_b=7.0,
         path="results/qwen_qwen2.5-vl-7b-instruct_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    dict(key="qwen25vl_32b", label="Qwen2.5-VL-32B", family="qwen", access="local",
         params_b=32.0, active_b=32.0,
         path="results/qwen_qwen2.5-vl-32b-instruct_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    dict(key="qwen25vl_72b", label="Qwen2.5-VL-72B", family="qwen", access="local",
         params_b=72.0, active_b=72.0,
         path="results/qwen_qwen2.5-vl-72b-instruct_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    dict(key="internvl3_38b", label="InternVL3-38B", family="internvl",
         access="local", params_b=38.0, active_b=38.0,
         path="results/opengvlab_internvl3-38b-hf_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    # InternVL3.5, the second scaling series. Qwen2.5-VL has no more public
    # sizes, so the axis could only be replicated, not extended: six dense
    # points across a 38x parameter range in an unrelated architecture, run on
    # the identical trials at the identical prompt and token ceiling. The MoE
    # member is deliberately not marked `scaling` -- 3B active against 30B total
    # is not the same axis, and joining it to the dense line would draw a curve
    # through two different quantities.
    dict(key="internvl35_1b", label="InternVL3.5-1B", family="internvl35",
         access="local", params_b=1.0, active_b=1.0,
         path="results/opengvlab_internvl3_5-1b-hf_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    dict(key="internvl35_2b", label="InternVL3.5-2B", family="internvl35",
         access="local", params_b=2.0, active_b=2.0,
         path="results/opengvlab_internvl3_5-2b-hf_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    dict(key="internvl35_4b", label="InternVL3.5-4B", family="internvl35",
         access="local", params_b=4.0, active_b=4.0,
         path="results/opengvlab_internvl3_5-4b-hf_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    dict(key="internvl35_8b", label="InternVL3.5-8B", family="internvl35",
         access="local", params_b=8.0, active_b=8.0,
         path="results/opengvlab_internvl3_5-8b-hf_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    dict(key="internvl35_14b", label="InternVL3.5-14B", family="internvl35",
         access="local", params_b=14.0, active_b=14.0,
         path="results/opengvlab_internvl3_5-14b-hf_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    dict(key="internvl35_38b", label="InternVL3.5-38B", family="internvl35",
         access="local", params_b=38.0, active_b=38.0,
         path="results/opengvlab_internvl3_5-38b-hf_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", scaling=True),

    dict(key="internvl35_30b_a3b", label="InternVL3.5-30B-A3B", family="internvl35",
         access="local", params_b=30.0, active_b=3.0,
         path="results/opengvlab_internvl3_5-30b-a3b-hf_4afc_hard_cot_anyview_n100.json",
         prompt="cot_anyview", role="moe_control"),
]

# Which observers carry a text label in the gate-vs-rotation figure. Ten labels
# on ten points that cluster in one corner is unreadable, and the identity of
# each model is Table 1's job. These are the ones the argument names: the human
# ceiling, the best gate, the best rotated score, and the two ends of the
# Qwen2.5-VL scaling series. Everything else is still plotted, in grey, so the
# spread is honest.
FIG_HIGHLIGHT = ["human_p01", "luna", "gemini38", "qwen25vl_3b", "qwen25vl_72b"]

# --------------------------------------------------------------- the tables ---
# Every table orders its rows the same way: the human first, then the models by
# overall accuracy, worst first. Worst first rather than best first because the
# claim is about the bottom of the column, and it puts the two frontier models
# last, next to the human they are being compared with.
# Where a run's own colour comes from, so every figure agrees.
FAMILY_COLOUR = {
    "human":    "#111111",
    "openai":   "#10a37f",
    "gemini":   "#4285f4",
    "qwen":     "#8b5cf6",
    "internvl": "#f59e0b",
    "internvl35": "#0891b2",
    "gemma":    "#ef4444",
}


def deviations(summary, observer, provenance=None):
    """
    Diff one run against `ARM`. Returns (blocking, gaps).

    **blocking** -- the run measured something other than the locked arm: a
    different foil bank, a different prompt, the wrong number of alternatives,
    fewer trials than the slice, or numbers inherited from a previous output
    file. A run with any of these is not in the headline table.

    **gaps** -- the run is almost certainly the locked arm but cannot prove it,
    because it predates the `run_config` block that `evaluate_vlm.py` now
    stamps into every result. These are provenance debts, not confounds. They
    close on the next re-run of that observer; `paper/provenance.json` records
    what the launch script used in the meantime, with its source named, so the
    value is never silently invented.

    The split matters because the two demand different responses. A blocking
    deviation means the number is wrong. A gap means the number is probably
    right and nobody can yet prove it, which is a thing to fix before
    submission and not a reason to throw the run away today.
    """
    blocking, gaps = [], []
    cfg = dict(summary.get("run_config") or {})
    if not cfg and provenance:
        cfg = dict(provenance.get(observer["key"], {}).get("run_config", {}))
        recovered = bool(cfg)
    else:
        recovered = False

    bench = os.path.basename(summary.get("benchmark") or "")
    if bench != os.path.basename(ARM["benchmark"]):
        blocking.append(f"benchmark={bench or 'unrecorded'}")
    elif cfg.get("benchmark_sha256_12") not in (None, ARM["benchmark_sha256_12"]):
        blocking.append(f"benchmark content differs (sha {cfg['benchmark_sha256_12']})")

    want_prompt = observer.get("prompt") or (
        ARM["prompt_style_human"] if observer.get("access") == "human"
        else ARM["prompt_style_model"])
    got_prompt = summary.get("prompt_style")
    if got_prompt != want_prompt:
        blocking.append(f"prompt={got_prompt} (expected {want_prompt})")

    if summary.get("n_options") != ARM["n_options"]:
        blocking.append(f"n_options={summary.get('n_options')}")

    n = summary.get("total_trials") or 0
    if n < ARM["n_trials"]:
        blocking.append(f"partial {n}/{ARM['n_trials']} trials")
    elif n > ARM["n_trials"]:
        blocking.append(f"{n} trials (expected {ARM['n_trials']})")

    if observer.get("access") != "human":
        mt = cfg.get("max_tokens", "unrecorded")
        if mt == "unrecorded":
            gaps.append("max_tokens not recorded in the result file")
        elif mt != ARM["max_tokens"]:
            blocking.append(f"max_tokens={mt} (expected {ARM['max_tokens']})")
        elif recovered:
            gaps.append("config recovered from the launch script, "
                        "not stamped by the run")

    if cfg.get("resumed_from_existing_output"):
        blocking.append(f"resumed {cfg['resumed_from_existing_output']} trials "
                        f"from a pre-existing output file")
    if not cfg:
        gaps.append("no run_config block (predates provenance stamping)")
    return blocking, gaps
