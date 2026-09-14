"""
Delayed VLM Evaluation Backend: Simulating Working-Memory Delay via Latent Noise Injection.

In the clinical Four Mountains Task, human observers maintain an internal allocentric
representation of a study scene across a delay period (e.g. 2s) before identifying it
from a 4-choice probe array viewed from a novel azimuth.

In computational neuroscience, working-memory maintenance across delay time t is
modeled as stochastic drift in neural attractor space:
    z_study(t) = z_study(0) + epsilon,   epsilon ~ N(0, sigma^2 * t * I)

This module implements targeted latent perturbations on the study scene representation
in open-weights VLMs (e.g. Qwen2.5-VL and InternVL families), keeping test options
and prompt instructions completely uncorrupted.
"""

import argparse
import collections
import dataclasses
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure bench and root are on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

def load_benchmark(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("trials", [])

from bench.evaluate_vlm import (
    OpenVLMBackend,
    parse_answer,
    _stratified,
    REASONING_STYLES,
    get_prompt_text,
)


@dataclasses.dataclass
class LatentPerturbationConfig:
    noise_scale: float = 0.0          # sigma: 0.0 = no noise (immediate / baseline)
    noise_type: str = "gaussian"      # "gaussian", "spherical", "dropout"
    noise_target: str = "study"       # "study" (image 0), "options" (images 1..4), "all"
    noise_seed: Optional[int] = 42    # Random seed for perturbation reproducibility
    relative_std: bool = True         # Scale noise by empirical std(z)
    dropout_prob: float = 0.1         # Used if noise_type == "dropout"


def apply_perturbation(
    tensor,
    config: LatentPerturbationConfig,
    generator=None
):
    """Apply configured perturbation to a latent tensor."""
    if config.noise_scale == 0.0 and config.noise_type != "dropout":
        return tensor

    import torch
    dtype = tensor.dtype
    device = tensor.device
    flat = tensor.to(torch.float32)

    if config.noise_type == "gaussian":
        eps = torch.randn(flat.shape, generator=generator, device=device, dtype=torch.float32)
        if config.relative_std:
            scale = config.noise_scale * (flat.std() + 1e-7)
        else:
            scale = config.noise_scale
        perturbed = flat + scale * eps

    elif config.noise_type == "spherical":
        eps = torch.randn(flat.shape, generator=generator, device=device, dtype=torch.float32)
        eps = eps / (torch.norm(eps, dim=-1, keepdim=True) + 1e-8)
        norm_orig = torch.norm(flat, dim=-1, keepdim=True)
        perturbed = flat + config.noise_scale * norm_orig * eps
        # Rescale to preserve original norm
        perturbed = perturbed / (torch.norm(perturbed, dim=-1, keepdim=True) + 1e-8) * norm_orig

    elif config.noise_type == "dropout":
        mask = (torch.rand(flat.shape, generator=generator, device=device) > config.dropout_prob).float()
        perturbed = flat * mask / (1.0 - config.dropout_prob)

    else:
        raise ValueError(f"Unknown noise_type: {config.noise_type}")

    return perturbed.to(dtype)


class DelayedQwenHook:
    """
    Targets the image features of Qwen2.5-VL / Qwen2-VL.
    In Qwen2.5-VL, get_image_features() splits vision tokens across images:
      image_embeds[0] is strictly the STUDY IMAGE.
      image_embeds[1..4] are the CANDIDATE OPTIONS.
    """
    def __init__(self, model, config: LatentPerturbationConfig):
        self.model = model
        self.config = config
        self.orig_get_image_features = None
        self.target_obj = None
        self.generator = None

    def __enter__(self):
        if self.config.noise_scale == 0.0 and self.config.noise_type != "dropout":
            return self

        import torch
        if self.config.noise_seed is not None:
            self.generator = torch.Generator(device=self.model.device)
            self.generator.manual_seed(self.config.noise_seed)

        target_obj = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "get_image_features"):
            target_obj = self.model.model
        elif hasattr(self.model, "get_image_features"):
            target_obj = self.model

        if target_obj is not None:
            self.target_obj = target_obj
            self.orig_get_image_features = target_obj.get_image_features
            config = self.config
            generator = self.generator
            orig_fn = self.orig_get_image_features

            def hooked_get_image_features(*args, **kwargs):
                vision_outputs = orig_fn(*args, **kwargs)
                image_embeds = list(vision_outputs.pooler_output)
                if len(image_embeds) >= 1:
                    new_embeds = []
                    for idx, emb in enumerate(image_embeds):
                        should_perturb = False
                        if config.noise_target == "all":
                            should_perturb = True
                        elif config.noise_target == "study" and idx == 0:
                            should_perturb = True
                        elif config.noise_target == "options" and idx > 0:
                            should_perturb = True

                        if should_perturb:
                            emb = apply_perturbation(emb, config, generator=generator)
                        new_embeds.append(emb)
                    vision_outputs.pooler_output = tuple(new_embeds)
                return vision_outputs

            target_obj.get_image_features = hooked_get_image_features
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.orig_get_image_features is not None and self.target_obj is not None:
            self.target_obj.get_image_features = self.orig_get_image_features


class DelayedInternVLHook:
    """
    Targets the image features of InternVL / InternVL3 / InternVL3.5.
    Hooks into extract_feature or vision_model output.
    """
    def __init__(self, model, config: LatentPerturbationConfig):
        self.model = model
        self.config = config
        self.orig_extract_feature = None
        self.generator = None

    def __enter__(self):
        if self.config.noise_scale == 0.0 and self.config.noise_type != "dropout":
            return self

        import torch
        if self.config.noise_seed is not None:
            self.generator = torch.Generator(device=self.model.device)
            self.generator.manual_seed(self.config.noise_seed)

        if hasattr(self.model, "extract_feature"):
            self.orig_extract_feature = self.model.extract_feature
            config = self.config
            generator = self.generator
            orig_fn = self.orig_extract_feature

            def hooked_extract_feature(pixel_values, *args, **kwargs):
                vit_embeds = orig_fn(pixel_values, *args, **kwargs)
                # In InternVL, multi-image inputs have shape (num_images * num_tiles, len, dim)
                # or (total_tiles, len, dim). The first image occupies the leading tiles.
                if vit_embeds.shape[0] >= 1:
                    # Perturb leading tile(s) for study scene
                    if config.noise_target in ("study", "all"):
                        vit_embeds[0:1] = apply_perturbation(vit_embeds[0:1], config, generator=generator)
                    if config.noise_target in ("options", "all") and vit_embeds.shape[0] > 1:
                        vit_embeds[1:] = apply_perturbation(vit_embeds[1:], config, generator=generator)
                return vit_embeds

            self.model.extract_feature = hooked_extract_feature
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.orig_extract_feature is not None:
            self.model.extract_feature = self.orig_extract_feature


class DelayedVLMBackend(OpenVLMBackend):
    """VLM evaluator with targeted working-memory latent perturbation support."""

    def predict_delayed(
        self,
        trial: Dict[str, Any],
        prompt_style: str = "cot_anyview",
        max_new_tokens: int = 8000,
        perturbation_config: Optional[LatentPerturbationConfig] = None,
    ) -> Tuple[Optional[int], str]:
        if perturbation_config is None or (perturbation_config.noise_scale == 0.0 and perturbation_config.noise_type != "dropout"):
            return self.predict(trial, prompt_style=prompt_style, max_new_tokens=max_new_tokens)

        # Apply hook appropriate for model family
        if self.family == "qwen":
            with DelayedQwenHook(self.model, perturbation_config):
                return self.predict(trial, prompt_style=prompt_style, max_new_tokens=max_new_tokens)
        elif self.family == "internvl":
            with DelayedInternVLHook(self.model, perturbation_config):
                return self.predict(trial, prompt_style=prompt_style, max_new_tokens=max_new_tokens)
        else:
            return self.predict(trial, prompt_style=prompt_style, max_new_tokens=max_new_tokens)


def evaluate_with_delay(
    model_id: str,
    benchmark_path: str,
    noise_scales: List[float],
    noise_type: str = "gaussian",
    noise_target: str = "study",
    prompt_style: str = "cot_anyview",
    max_trials: int = 100,
    seed: int = 0,
    out_dir: str = "results",
) -> Dict[float, Dict[str, Any]]:
    """Run an evaluation sweep across multiple noise scales (simulating delay duration)."""
    trials = load_benchmark(benchmark_path)
    if max_trials and len(trials) > max_trials:
        trials = _stratified(trials, max_trials, seed=seed)

    print(f"Loaded {len(trials)} trials from {benchmark_path}")
    print(f"Initializing model {model_id}...")
    backend = DelayedVLMBackend(model_id=model_id)

    results_by_scale = {}
    bench_tag = os.path.basename(benchmark_path).replace(".json", "").replace("vlm_benchmark_", "")
    slug = model_id.replace("/", "_").lower()
    os.makedirs(out_dir, exist_ok=True)

    for scale in noise_scales:
        print(f"\n=======================================================")
        print(f"EVALUATING NOISE SCALE sigma = {scale:.2f} (target: {noise_target}, type: {noise_type})")
        print(f"=======================================================")
        config = LatentPerturbationConfig(
            noise_scale=scale,
            noise_type=noise_type,
            noise_target=noise_target,
            noise_seed=seed,
        )

        trial_results = []
        for i, trial in enumerate(trials, 1):
            t0 = time.time()
            choice, reply = backend.predict_delayed(
                trial,
                prompt_style=prompt_style,
                perturbation_config=config,
            )
            elapsed = time.time() - t0
            gt = trial.get("correct_choice", trial.get("answer"))
            correct = (choice == gt) if choice is not None else False
            trial_results.append({
                "trial_id": trial.get("id"),
                "mode": trial.get("mode"),
                "delta": trial.get("delta"),
                "correct_choice": gt,
                "model_choice": choice,
                "is_correct": correct,
                "latency": round(elapsed, 2),
                "reply": reply,
            })
            if i % 10 == 0 or i == len(trials):
                acc_so_far = sum(r["is_correct"] for r in trial_results) / len(trial_results)
                print(f"[{i:3d}/{len(trials)}] Acc: {acc_so_far*100:5.1f}% | latest latency: {elapsed:.2f}s")

        acc_all = sum(r["is_correct"] for r in trial_results) / len(trial_results)
        d0_trials = [r for r in trial_results if r.get("delta") == 0]
        acc_d0 = sum(r["is_correct"] for r in d0_trials) / len(d0_trials) if d0_trials else 0.0
        rot_trials = [r for r in trial_results if r.get("delta", 0) > 0]
        acc_rot = sum(r["is_correct"] for r in rot_trials) / len(rot_trials) if rot_trials else 0.0

        summary = {
            "model": model_id,
            "benchmark": benchmark_path,
            "noise_scale": scale,
            "noise_type": noise_type,
            "noise_target": noise_target,
            "prompt_style": prompt_style,
            "n_trials": len(trial_results),
            "overall_accuracy": round(acc_all, 4),
            "gate_accuracy_delta0": round(acc_d0, 4),
            "rotated_accuracy": round(acc_rot, 4),
        }
        results_by_scale[scale] = summary

        out_path = os.path.join(out_dir, f"delayed_{slug}_{bench_tag}_sigma_{scale:.2f}.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "results": trial_results}, f, indent=2)
        print(f"Saved: {out_path} -> Overall: {acc_all*100:.1f}%, Gate: {acc_d0*100:.1f}%, Rot: {acc_rot*100:.1f}%")

    return results_by_scale


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM under simulated working memory delay")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--benchmark", type=str, default="data/benchmarks/vlm_benchmark_4afc_hard.json")
    parser.add_argument("--noise_scales", type=str, default="0.0,0.1,0.25,0.5,1.0")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "spherical", "dropout"])
    parser.add_argument("--noise_target", type=str, default="study", choices=["study", "options", "all"])
    parser.add_argument("--prompt_style", type=str, default="cot_anyview")
    parser.add_argument("--max_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    scales = [float(s.strip()) for s in args.noise_scales.split(",") if s.strip()]
    evaluate_with_delay(
        model_id=args.model,
        benchmark_path=args.benchmark,
        noise_scales=scales,
        noise_type=args.noise_type,
        noise_target=args.noise_target,
        prompt_style=args.prompt_style,
        max_trials=args.max_trials,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
