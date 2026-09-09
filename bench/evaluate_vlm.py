"""
Evaluate Vision-Language Models (open Qwen-VL or closed API models)
on the Four Mountains Allocentric Benchmark.

Usage:
    # Open local model on GPU:
    python bench/evaluate_vlm.py --benchmark data/vlm_benchmark_4afc.json \
        --model Qwen/Qwen2-VL-2B-Instruct --out results/qwen2_vl_2b_4afc.json

    # With Chain-of-Thought reasoning:
    python bench/evaluate_vlm.py --benchmark data/vlm_benchmark_4afc.json \
        --model Qwen/Qwen2.5-VL-7B-Instruct --prompt_style cot --out results/qwen2_5_vl_7b_cot.json

    # API model (OpenAI / OpenRouter):
    python bench/evaluate_vlm.py --benchmark data/vlm_benchmark_4afc.json \
        --model gpt-4o --api_base https://api.openai.com/v1 --out results/gpt4o_4afc.json
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np


def parse_answer(text, n_options=4, prefer="last"):
    """
    Extract chosen option index (1-based, e.g. 1..4).
    Checks for patterns like 'Final Answer: Option 2', 'The final answer is Option 3',
    'Option 2', or trailing digits. Filters out range patterns like '[1-4]' or '1-4'.
    """
    if not text:
        return None

    # Clean out template artifacts like [1-4], [1-2], 1-4, 1-2
    cleaned = re.sub(r"\[[1-%d]-[1-%d]\]" % (n_options, n_options), "", text)
    cleaned = re.sub(r"\b1-[1-%d]\b" % n_options, "", cleaned)

    # Pattern 1: Explicit concluding statements
    final_patterns = [
        r"(?:final\s+answer|correct\s+(?:option|choice|answer)|the\s+answer|i\s+choose|therefore|hence)\s*(?:is)?\s*:?\s*(?:option|choice)?\s*([1-%d])\b" % n_options,
        r"\b(?:option|choice)\s*([1-%d])\b(?:\s+is\s+(?:the\s+)?(?:correct|same|matching))" % n_options,
    ]
    for pat in final_patterns:
        matches = re.findall(pat, cleaned, re.IGNORECASE)
        if matches:
            return int(matches[-1])

    # Pattern 2: Standalone "Final Answer: X" without word Option
    m = re.findall(r"final\s+answer\s*:?\s*([1-%d])\b" % n_options, cleaned, re.IGNORECASE)
    if m:
        return int(m[-1])

    # Pattern 3: If prefer == "first" (e.g. direct answering where model just outputs "Option 1" or "1")
    if prefer == "first":
        m = re.search(r"(?:Option|Choice)\s*([1-%d])\b" % n_options, cleaned, re.IGNORECASE)
        if m:
            return int(m.group(1))
        digits = re.findall(r"\b([1-%d])\b" % n_options, cleaned)
        if digits:
            return int(digits[0])
    else:
        # If prefer == "last" (e.g. CoT reasoning), take the last mentioned Option or digit
        matches = re.findall(r"(?:Option|Choice)\s*([1-%d])\b" % n_options, cleaned, re.IGNORECASE)
        if matches:
            return int(matches[-1])
        digits = re.findall(r"\b([1-%d])\b" % n_options, cleaned)
        if digits:
            return int(digits[-1])

    return None


def get_prompt_text(n_options=4, style="cot"):
    if style == "mental_rotation":
        return (
            f"You are taking the Four Mountains Test of allocentric spatial perception.\n\n"
            f"Image 1 is the STUDY view of a landscape with four distinct mountain peaks.\n"
            f"The subsequent {n_options} images are OPTION 1 to OPTION {n_options}.\n"
            f"Exactly ONE option shows the EXACT SAME mountain landscape (the same four peaks in the same relative geometric layout), "
            f"viewed from a shifted camera viewpoint and under different weather/lighting.\n"
            f"The other options are distractors where the relative 3D spatial arrangement of the peaks has been altered.\n\n"
            f"To determine the matching scene, perform mental rotation:\n"
            f"1. Estimate the camera perspective shift (rotation angle) between the study image and candidate options.\n"
            f"2. Mentally rotate the four peaks to verify if topological handedness (e.g. clockwise/counterclockwise order of peaks, "
            f"which peak is opposite or between others) matches the study scene.\n"
            f"3. Ignore superficial differences in sunlight, shadows, fog, and seasonal color.\n\n"
            f"In 2-4 sentences, explain your mental rotation reasoning, then conclude on the last line:\n"
            f"Final Answer: Option X"
        )
    elif style == "anchor":
        return (
            f"You are taking the Four Mountains Test of allocentric spatial perception.\n\n"
            f"Image 1 is the STUDY view of a landscape with four distinct mountain peaks.\n"
            f"The subsequent {n_options} images are OPTION 1 to OPTION {n_options}.\n"
            f"Exactly ONE option shows the EXACT SAME mountain landscape under a camera viewpoint rotation and weather change.\n"
            f"The remaining options are distractors.\n\n"
            f"Spatial Strategy:\n"
            f"1. Select the single most prominent or unique landmark peak as an anchor (origin).\n"
            f"2. Trace the relative bearings and distances of the other 3 peaks surrounding this anchor.\n"
            f"3. Identify which option preserves this exact 3D spatial configuration around the anchor under the new camera angle.\n\n"
            f"In 2-4 concise sentences, explain your reasoning and conclude on the last line:\n"
            f"Final Answer: Option X"
        )
    elif style == "birdseye":
        return (
            f"You are taking the Four Mountains Test of allocentric spatial perception.\n\n"
            f"Image 1 is the STUDY view of a landscape with four mountain peaks.\n"
            f"The subsequent {n_options} images are OPTION 1 to OPTION {n_options}.\n"
            f"Exactly ONE option shows the EXACT SAME mountain landscape viewed from a different camera viewpoint and lighting.\n"
            f"The other options are distractors with altered 3D mountain configurations.\n\n"
            f"Top-Down Cognitive Mapping Strategy:\n"
            f"1. Imagine looking down at the four peaks from directly above (a 2D bird's-eye map). Note their relative positions (which forms a triangle, which is isolated, which is tallest).\n"
            f"2. For each candidate option, determine where the camera would be standing on that same bird's-eye map.\n"
            f"3. Verify which option is geometrically consistent with the study scene's top-down layout under the new camera angle.\n\n"
            f"In 2-4 sentences, describe the bird's-eye spatial layout and conclude on the last line:\n"
            f"Final Answer: Option X"
        )
    elif style == "elimination":
        return (
            f"You are taking the Four Mountains Test of allocentric spatial perception.\n\n"
            f"Image 1 is the STUDY view of a landscape with four mountain peaks.\n"
            f"The subsequent {n_options} images are OPTION 1 to OPTION {n_options}.\n"
            f"Exactly ONE option shows the EXACT SAME mountain landscape viewed from a different camera angle and lighting.\n"
            f"The other options are geometric distractors.\n\n"
            f"Falsification Strategy:\n"
            f"1. Inspect each candidate option one by one to find geometric contradictions with the study scene (e.g. impossible relative peak heights, wrong peak ordering, or missing ridges).\n"
            f"2. Eliminate the distractor options that cannot possibly match the study landscape under any viewpoint rotation.\n"
            f"3. Select the remaining single candidate that has no geometric contradictions.\n\n"
            f"Briefly eliminate the distractors and conclude on the last line:\n"
            f"Final Answer: Option X"
        )
    elif style == "elevation":
        return (
            f"You are taking the Four Mountains Test of allocentric spatial perception.\n\n"
            f"Image 1 is the STUDY view of a landscape with four distinct mountain peaks.\n"
            f"The subsequent {n_options} images are OPTION 1 to OPTION {n_options}.\n"
            f"Exactly ONE option shows the EXACT SAME mountain landscape.\n\n"
            f"Elevation & Topography Strategy:\n"
            f"1. Rank the four peaks by physical elevation/height (tallest to shortest).\n"
            f"2. Note the shape profile of the highest peak and the ridges connecting the peaks.\n"
            f"3. Camera rotation changes left/right ordering but PRESERVES physical peak heights and ridge connections.\n"
            f"4. Identify the option whose peak elevations and ridge topography match the study scene.\n\n"
            f"Explain your reasoning in 2-4 sentences and conclude on the last line:\n"
            f"Final Answer: Option X"
        )
    elif style == "hybrid":
        return (
            f"You are taking the Four Mountains Test of allocentric spatial perception.\n\n"
            f"Image 1 is the STUDY view of a landscape with four distinct mountain peaks.\n"
            f"The subsequent {n_options} images are OPTION 1 to OPTION {n_options}.\n"
            f"Exactly ONE option shows the EXACT SAME mountain landscape viewed from a shifted camera viewpoint and under different weather/lighting.\n"
            f"The other options are geometric distractors.\n\n"
            f"Allocentric Reasoning & Elimination Strategy:\n"
            f"1. Perform mental rotation: trace the 3D topological handedness (clockwise/counterclockwise order and relative bearings) of the 4 peaks.\n"
            f"2. Systematically inspect each option and ELIMINATE distractors that show geometric or topological contradictions under rotation (wrong peak order or impossible angles).\n"
            f"3. Verify that the remaining candidate preserves the exact 3D peak configuration.\n\n"
            f"Briefly eliminate the distractors using mental rotation, then conclude on the last line:\n"
            f"Final Answer: Option X"
        )
    elif style == "cot":
        return (
            f"You are taking the Four Mountains Test of spatial allocentric perception.\n\n"
            f"Image 1 is the STUDY view of a landscape with four mountain peaks.\n"
            f"The subsequent {n_options} images are OPTION 1 to OPTION {n_options}.\n"
            f"Exactly ONE option shows the EXACT SAME mountain landscape (the same four peaks in the same relative spatial arrangement), "
            f"simply viewed from a different viewpoint and under different lighting/weather.\n"
            f"The other options show different mountain landscapes.\n\n"
            f"In 2-4 sentences, compare the 3D spatial layout of the peaks (e.g. relative positions such as in front, behind, left, right) "
            f"between the study view and the options, accounting for camera rotation. Avoid lengthy itemized lists.\n\n"
            f"State your final decision on the last line as:\n"
            f"Final Answer: Option X"
        )
    else:  # direct
        return (
            f"You are taking the Four Mountains Test of spatial perception.\n\n"
            f"Image 1 is the STUDY view of a landscape with four mountain peaks.\n"
            f"The subsequent {n_options} images are OPTION 1 to OPTION {n_options}.\n"
            f"Exactly ONE option shows the EXACT SAME mountain landscape viewed from a different angle "
            f"and under different weather/lighting conditions.\n"
            f"The other options show different mountain landscapes.\n\n"
            f"Which option shows the same place as the study image?\n"
            f"State your choice clearly:\n"
            f"Final Answer: Option X"
        )


REASONING_STYLES = {"cot", "mental_rotation", "anchor", "birdseye", "elimination", "elevation", "hybrid"}


class OpenVLMBackend:
    def __init__(self, model_id, device="cuda", max_pixels=640 * 480):
        import torch
        from transformers import AutoProcessor

        self.device = device
        self.torch = torch
        os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/raid/nbe_tmp/markus_frey/cache/huggingface")

        print(f"Loading {model_id} on {device}...", flush=True)

        # `max_pixels` is a Qwen processor kwarg. Passing it to a processor that
        # does not take it is a TypeError after the weights are already on the
        # GPU, so the family is decided before anything is loaded.
        self.family = ("qwen" if "qwen" in model_id.lower()
                       else "internvl" if "internvl" in model_id.lower()
                       else "generic")
        kwargs = {"max_pixels": max_pixels} if self.family == "qwen" else {}
        self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)

        # Check model family
        if "qwen2.5-vl" in model_id.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
        elif "qwen2-vl" in model_id.lower():
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
        else:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )

        self.model.eval()
        print(f"Model {model_id} loaded successfully.", flush=True)

    def _vision(self, messages):
        """
        Resolve the image paths in `messages` to whatever the processor wants.

        `qwen_vl_utils.process_vision_info` also applies Qwen's smart-resize and
        returns the (images, videos) pair its processor expects. Nothing outside
        the Qwen family takes that pair, and importing it for an InternVL run
        would fail on a machine that never installed it, so non-Qwen models get
        plain PIL images and no video argument.
        """
        if self.family == "qwen":
            from qwen_vl_utils import process_vision_info
            return process_vision_info(messages)

        from PIL import Image
        images = [Image.open(part["image"]).convert("RGB")
                  for m in messages for part in m["content"]
                  if part.get("type") == "image"]
        return images, None

    def predict(self, trial, prompt_style="direct", max_new_tokens=64):
        n_opts = trial["n_options"]
        instructions = get_prompt_text(n_opts, style=prompt_style)

        content = [
            {"type": "text", "text": "=== STUDY IMAGE ==="},
            {"type": "image", "image": trial["study_image"]},
            {"type": "text", "text": "\n=== CANDIDATE OPTIONS ==="},
        ]
        for i, opt in enumerate(trial["options"], 1):
            content.append({"type": "text", "text": f"\nOPTION {i}:"})
            content.append({"type": "image", "image": opt["image_path"]})

        content.append({"type": "text", "text": f"\n{instructions}"})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._vision(messages)

        proc_kwargs = {"text": [text], "images": image_inputs,
                       "padding": True, "return_tensors": "pt"}
        if video_inputs is not None:
            proc_kwargs["videos"] = video_inputs
        inputs = self.processor(**proc_kwargs)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with self.torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)]
        reply = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        choice = parse_answer(reply, n_options=n_opts, prefer="last" if prompt_style in REASONING_STYLES else "first")
        return choice, reply.strip()


class APIBackend:
    def __init__(self, model_id, api_base=None, api_key=None):
        import urllib.request
        self.model_id = model_id
        self.base = (api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")).rstrip("/")
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.key:
            raise ValueError("API key must be provided via --api_key or OPENAI_API_KEY env var")

    def predict(self, trial, prompt_style="cot", max_new_tokens=64):
        import json as _json
        import urllib.request

        n_opts = trial["n_options"]
        instructions = get_prompt_text(n_opts, style=prompt_style)

        content = [{"type": "text", "text": "=== STUDY IMAGE ==="}]
        with open(trial["study_image"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        content.append({"type": "text", "text": "\n=== CANDIDATE OPTIONS ==="})
        for i, opt in enumerate(trial["options"], 1):
            content.append({"type": "text", "text": f"\nOPTION {i}:"})
            with open(opt["image_path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        content.append({"type": "text", "text": f"\n{instructions}"})

        body = _json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_new_tokens,
            "temperature": 0,
        }).encode()

        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        )

        with urllib.request.urlopen(req, timeout=120) as r:
            res = _json.loads(r.read())

        reply = res["choices"][0]["message"]["content"]
        choice = parse_answer(reply, n_options=n_opts, prefer="last" if prompt_style in REASONING_STYLES else "first")
        return choice, reply.strip()


def run_evaluation(benchmark_path, model_id, out_path, prompt_style="direct",
                   api_base=None, api_key=None, max_trials=None, max_tokens=None,
                   modes=None, resume=True):
    with open(benchmark_path) as f:
        bench = json.load(f)

    trials = bench["trials"]
    n_options = bench.get("n_options", 4)
    chance_level = 1.0 / n_options

    # Filter by mode if specified
    if modes is not None:
        selected_modes = [m.strip() for m in modes.split(",")]
        trials = [t for t in trials if t["mode"] in selected_modes]
        print(f"Filtered trials to modes: {selected_modes} ({len(trials)} trials remaining)", flush=True)

    if max_trials is not None:
        trials = trials[:max_trials]

    results = {}
    if resume and os.path.exists(out_path):
        try:
            with open(out_path) as f:
                old = json.load(f)
                results = {r["trial_id"]: r for r in old.get("results", [])}
                print(f"Resuming: found {len(results)} previously evaluated trials.", flush=True)
        except Exception as e:
            print(f"Could not load previous results for resume: {e}", flush=True)

    # Initialize backend
    if api_base or "gpt" in model_id.lower() or "claude" in model_id.lower():
        backend = APIBackend(model_id, api_base=api_base, api_key=api_key)
    else:
        backend = OpenVLMBackend(model_id)

    if max_tokens is None:
        max_tokens = 512 if prompt_style in REASONING_STYLES else 32

    print(f"\nEvaluating {model_id} on {len(trials)} trials ({benchmark_path})...", flush=True)
    t_start = time.time()

    for idx, trial in enumerate(trials, 1):
        tid = trial["id"]
        if tid in results:
            continue

        t0 = time.time()
        try:
            choice, reply = backend.predict(trial, prompt_style=prompt_style, max_new_tokens=max_tokens)
            err = None
        except Exception as exc:
            choice, reply, err = None, "", str(exc)
            print(f"[{idx}/{len(trials)}] ERROR on {tid}: {exc}")

        dur = time.time() - t0
        correct = (choice == trial["correct_choice"]) if choice is not None else False

        results[tid] = {
            "trial_id": tid,
            "mode": trial["mode"],
            "delta": trial["delta"],
            "correct_choice": trial["correct_choice"],
            "model_choice": choice,
            "is_correct": correct,
            "latency": dur,
            "reply": reply,
            "error": err,
        }

        if idx % 5 == 0 or idx == len(trials):
            n_done = len(results)
            n_corr = sum(1 for r in results.values() if r["is_correct"])
            acc = 100.0 * n_corr / max(n_done, 1)
            print(f"[{idx:3d}/{len(trials):3d}] {tid:20s} | pred={choice} (gt={trial['correct_choice']}) "
                  f"| Acc={acc:5.1f}% (chance={100 * chance_level:.1f}%) | {dur:.2f}s", flush=True)

        # Periodic intermediate saving
        if idx % 10 == 0 or idx == len(trials):
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({
                    "summary": {"in_progress": True, "done": len(results), "total": len(trials)},
                    "results": list(results.values())
                }, f, indent=2)

    # Aggregate metrics
    by_mode_delta = defaultdict(lambda: {"correct": 0, "total": 0})
    choice_dist = defaultdict(int)

    for r in results.values():
        key = (r["mode"], r["delta"])
        by_mode_delta[key]["total"] += 1
        if r["is_correct"]:
            by_mode_delta[key]["correct"] += 1
        if r["model_choice"]:
            choice_dist[r["model_choice"]] += 1

    summary = {
        "model": model_id,
        "benchmark": benchmark_path,
        "prompt_style": prompt_style,
        "n_options": n_options,
        "chance_level": chance_level,
        "total_trials": len(results),
        "overall_accuracy": float(sum(r["is_correct"] for r in results.values()) / max(len(results), 1)),
        "choice_distribution": dict(choice_dist),
        "by_mode_delta": {},
    }

    print("\n" + "=" * 65, flush=True)
    print(f"RESULTS SUMMARY: {model_id} ({prompt_style})", flush=True)
    print("=" * 65, flush=True)
    print(f"Overall Accuracy: {100 * summary['overall_accuracy']:.1f}% (Chance: {100 * chance_level:.1f}%)", flush=True)
    print(f"Choice Distribution: {dict(choice_dist)}\n", flush=True)
    print(f"{'Mode':20s} | {'Delta':5s} | {'Acc (%)':8s} | {'N':4s}", flush=True)
    print("-" * 45, flush=True)

    for (mode, delta), stats in sorted(by_mode_delta.items()):
        acc = 100.0 * stats["correct"] / stats["total"] if stats["total"] else 0.0
        summary["by_mode_delta"][f"{mode}_d{delta}"] = {
            "mode": mode,
            "delta": delta,
            "accuracy": float(acc / 100.0),
            "correct": stats["correct"],
            "total": stats["total"],
        }
        print(f"{mode:20s} | {delta:5d} | {acc:7.1f}% | {stats['total']:4d}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "results": list(results.values())
        }, f, indent=2)

    print(f"\nSaved full results to {out_path} in {time.time() - t_start:.1f}s", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on Four Mountains Benchmark")
    parser.add_argument("--benchmark", default="data/vlm_benchmark_4afc.json", help="Path to benchmark JSON")
    parser.add_argument("--model", required=True, help="Model ID (e.g. Qwen/Qwen2-VL-2B-Instruct)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--prompt_style", default="cot", choices=["direct", "cot", "mental_rotation", "anchor", "birdseye", "elimination", "elevation", "hybrid"], help="Prompting style")
    parser.add_argument("--api_base", default=None, help="API Base URL for OpenAI/OpenRouter")
    parser.add_argument("--api_key", default=None, help="API Key for API backend")
    parser.add_argument("--max_trials", type=int, default=None, help="Limit number of trials for testing")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max generated tokens per trial")
    parser.add_argument("--modes", default=None, help="Comma-separated list of modes to evaluate (e.g. c0_shape_colour,c1_shape)")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume previous run")
    args = parser.parse_args()

    run_evaluation(
        benchmark_path=args.benchmark,
        model_id=args.model,
        out_path=args.out,
        prompt_style=args.prompt_style,
        api_base=args.api_base,
        api_key=args.api_key,
        max_trials=args.max_trials,
        max_tokens=args.max_tokens,
        modes=args.modes,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
