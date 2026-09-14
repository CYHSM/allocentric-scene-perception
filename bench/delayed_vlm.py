"""
Delayed VLM Evaluation Backend: Simulating Working-Memory Delay via Latent Perturbations.

In the clinical Four Mountains Task, human observers maintain an internal allocentric
representation of a study scene across a delay period (e.g. 2s) before identifying it
from a 4-choice probe array viewed from a novel azimuth.

In computational neuroscience, working-memory maintenance across delay duration t is
modeled as stochastic drift in neural attractor space:
    z_study(t) = z_study(0) + epsilon,   epsilon ~ N(0, sigma^2 * t * I)
or as mean-reverting attractor drift (Ornstein-Uhlenbeck):
    dz = -theta * (z - mu) * dt + sigma * dW_t

This module implements targeted latent perturbations on the study scene representation
in open-weights VLMs (e.g. Qwen2.5-VL and InternVL families), keeping test options
and prompt instructions completely uncorrupted.

Extensible Architecture:
- Noise Types: Gaussian / Brownian diffusion, Spherical / Angular, Ornstein-Uhlenbeck attractor drift,
  Feature dropout, and Spatial patch masking.
- Temporal Delay Parameterization: direct duration t (seconds) and diffusion rate D.
- Hook Registry: @register_delay_hook decorator supporting arbitrary new model architectures.
- Locus Selection: vision encoder, multimodal projector, or decoder input embeddings.
"""

import abc
import argparse
import collections
import dataclasses
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

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
    """Configuration for delay-period latent perturbations."""
    noise_scale: float = 0.0               # Explicit sigma (multiplier if relative_std=True, else absolute)
    delay_seconds: Optional[float] = None  # Working memory delay duration t (seconds)
    diffusion_rate: float = 0.05           # Diffusion constant D such that sigma(t) = sqrt(D * t)
    noise_type: str = "gaussian"           # "gaussian"/"brownian", "spherical"/"angular", "ornstein_uhlenbeck", "dropout", "spatial_patch"
    noise_target: str = "study"            # "study" (image 0), "options" (images 1..4), "all"
    noise_seed: Optional[int] = 42         # Seed for reproducible perturbation
    relative_std: bool = True              # Scale noise by empirical std(z)
    dropout_prob: float = 0.1              # Feature dropout probability
    patch_fraction: float = 0.25           # Fraction of patches to corrupt for spatial_patch
    ou_theta: float = 0.5                  # Reversion rate for Ornstein-Uhlenbeck drift
    ou_prior_mode: str = "zero"            # Prior mu for OU: "zero" or "mean"
    locus: str = "vision_embed"            # "vision_embed", "projector", "inputs_embeds"

    @property
    def effective_sigma(self) -> float:
        """Returns effective perturbation scale sigma, computing sqrt(D * t) if delay_seconds is set."""
        if self.delay_seconds is not None:
            return math.sqrt(max(0.0, self.diffusion_rate * self.delay_seconds))
        return self.noise_scale


def apply_perturbation(
    tensor,
    config: LatentPerturbationConfig,
    generator=None,
):
    """
    Apply configured perturbation to a latent tensor.

    Args:
        tensor: PyTorch tensor (e.g. vision token embeddings or pooled features).
        config: LatentPerturbationConfig specifying noise dynamics.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Perturbed PyTorch tensor matching input dtype and device.
    """
    sigma = config.effective_sigma
    if sigma == 0.0 and config.noise_type not in ("dropout", "spatial_patch"):
        return tensor

    import torch
    dtype = tensor.dtype
    device = tensor.device
    flat = tensor.to(torch.float32)
    noise_type = config.noise_type.lower()

    if noise_type in ("gaussian", "brownian"):
        eps = torch.randn(flat.shape, generator=generator, device=device, dtype=torch.float32)
        if config.relative_std:
            scale = sigma * (flat.std() + 1e-7)
        else:
            scale = sigma
        perturbed = flat + scale * eps

    elif noise_type in ("spherical", "angular"):
        # Angular perturbation preserving token norm
        eps = torch.randn(flat.shape, generator=generator, device=device, dtype=torch.float32)
        eps = eps / (torch.norm(eps, dim=-1, keepdim=True) + 1e-8)
        norm_orig = torch.norm(flat, dim=-1, keepdim=True)
        perturbed = flat + sigma * norm_orig * eps
        perturbed = perturbed / (torch.norm(perturbed, dim=-1, keepdim=True) + 1e-8) * norm_orig

    elif noise_type in ("ornstein_uhlenbeck", "attractor_decay"):
        # Attractor drift towards prior mu with diffusive noise:
        # z(t) = exp(-theta*t)*z(0) + (1 - exp(-theta*t))*mu + sigma_diff * eps
        t = config.delay_seconds if config.delay_seconds is not None else 1.0
        theta = max(1e-4, config.ou_theta)
        alpha = math.exp(-theta * t)
        
        if config.ou_prior_mode == "mean":
            mu = flat.mean(dim=-2, keepdim=True) if flat.ndim >= 2 else flat.mean()
        else:
            mu = torch.zeros_like(flat)

        std_scale = (flat.std() + 1e-7) if config.relative_std else 1.0
        var_diff = (1.0 - math.exp(-2.0 * theta * t)) / (2.0 * theta)
        scale = sigma * std_scale * math.sqrt(max(0.0, var_diff))
        
        eps = torch.randn(flat.shape, generator=generator, device=device, dtype=torch.float32)
        perturbed = alpha * flat + (1.0 - alpha) * mu + scale * eps

    elif noise_type == "dropout":
        mask = (torch.rand(flat.shape, generator=generator, device=device) > config.dropout_prob).float()
        perturbed = flat * mask / (1.0 - config.dropout_prob)

    elif noise_type == "spatial_patch":
        # Randomly perturb a subset of spatial tokens
        perturbed = flat.clone()
        if flat.ndim >= 2 and flat.shape[-2] > 1:
            num_tokens = flat.shape[-2]
            n_corrupt = max(1, int(num_tokens * config.patch_fraction))
            perm = torch.randperm(num_tokens, generator=generator, device=device)[:n_corrupt]
            std_scale = (flat.std() + 1e-7) if config.relative_std else 1.0
            eps = torch.randn(n_corrupt, flat.shape[-1], generator=generator, device=device, dtype=torch.float32)
            scale = (sigma if sigma > 0.0 else 1.0) * std_scale
            perturbed[..., perm, :] += scale * eps
        else:
            eps = torch.randn(flat.shape, generator=generator, device=device, dtype=torch.float32)
            perturbed += sigma * (flat.std() + 1e-7) * eps

    else:
        raise ValueError(f"Unknown noise_type: {config.noise_type}. Supported: gaussian, spherical, ornstein_uhlenbeck, dropout, spatial_patch")

    return perturbed.to(dtype)


# ==============================================================================
# Base Hook & Extensible Hook Registry
# ==============================================================================

class BaseDelayedHook(abc.ABC):
    """Abstract base class for model-specific latent working memory delay hooks."""

    def __init__(self, model: Any, config: LatentPerturbationConfig):
        self.model = model
        self.config = config
        self.generator = None

    def is_active(self) -> bool:
        return (self.config.effective_sigma > 0.0 or self.config.noise_type in ("dropout", "spatial_patch"))

    def init_generator(self):
        import torch
        if self.config.noise_seed is not None:
            # Safely extract device
            device = getattr(self.model, "device", "cpu")
            try:
                self.generator = torch.Generator(device=device)
            except Exception:
                self.generator = torch.Generator(device="cpu")
            self.generator.manual_seed(self.config.noise_seed)

    def should_perturb_item(self, idx: int, total_items: int) -> bool:
        """Helper to determine if item at index  should be perturbed."""
        target = self.config.noise_target.lower()
        if target == "all":
            return True
        if target == "study" and idx == 0:
            return True
        if target == "options" and idx > 0:
            return True
        return False

    @abc.abstractmethod
    def __enter__(self):
        """Attach hook to model."""
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Detach hook and restore original model state."""
        pass


HOOK_REGISTRY: Dict[str, Type[BaseDelayedHook]] = {}


def register_delay_hook(*aliases: str):
    """Decorator to register a delayed hook class for one or more model family aliases."""
    def decorator(cls: Type[BaseDelayedHook]):
        for a in aliases:
            HOOK_REGISTRY[a.lower().strip()] = cls
        return cls
    return decorator


def get_hook_for_model(model: Any, family: str, config: LatentPerturbationConfig) -> BaseDelayedHook:
    """
    Resolve and instantiate the appropriate hook for a given model.
    Checks family name, model class name, and falls back to generic hook.
    """
    fam = family.lower().strip()
    if fam in HOOK_REGISTRY:
        return HOOK_REGISTRY[fam](model, config)

    # Check substring matches in model class name
    cls_name = model.__class__.__name__.lower()
    for key, hook_cls in HOOK_REGISTRY.items():
        if key in fam or key in cls_name:
            return hook_cls(model, config)

    # Fallback
    if "generic" in HOOK_REGISTRY:
        return HOOK_REGISTRY["generic"](model, config)
    return DelayedGenericHook(model, config)


# ==============================================================================
# Model-Specific Hooks
# ==============================================================================

@register_delay_hook("qwen", "qwen2", "qwen2.5", "qwen2_5_vl", "qwen2-vl", "qwen2.5-vl")
class DelayedQwenHook(BaseDelayedHook):
    """
    Targets the image features of Qwen2.5-VL / Qwen2-VL.
    In Qwen2.5-VL, get_image_features() splits vision tokens across images:
      image_embeds[0] is strictly the STUDY IMAGE.
      image_embeds[1..4] are the CANDIDATE OPTIONS.
    """

    def __init__(self, model: Any, config: LatentPerturbationConfig):
        super().__init__(model, config)
        self.orig_get_image_features = None
        self.target_obj = None

    def __enter__(self):
        if not self.is_active():
            return self

        self.init_generator()
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
            hook_self = self

            def hooked_get_image_features(*args, **kwargs):
                vision_outputs = orig_fn(*args, **kwargs)
                image_embeds = list(vision_outputs.pooler_output)
                if len(image_embeds) >= 1:
                    new_embeds = []
                    for idx, emb in enumerate(image_embeds):
                        if hook_self.should_perturb_item(idx, len(image_embeds)):
                            emb = apply_perturbation(emb, config, generator=generator)
                        new_embeds.append(emb)
                    vision_outputs.pooler_output = tuple(new_embeds)
                return vision_outputs

            target_obj.get_image_features = hooked_get_image_features
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.orig_get_image_features is not None and self.target_obj is not None:
            self.target_obj.get_image_features = self.orig_get_image_features


@register_delay_hook("internvl", "internvl2", "internvl2_5", "internvl3", "internvl3_5")
class DelayedInternVLHook(BaseDelayedHook):
    """
    Targets the image features of InternVL / InternVL3 / InternVL3.5.
    Hooks into extract_feature or vision_model output.
    """

    def __init__(self, model: Any, config: LatentPerturbationConfig):
        super().__init__(model, config)
        self.orig_extract_feature = None

    def __enter__(self):
        if not self.is_active():
            return self

        self.init_generator()
        if hasattr(self.model, "extract_feature"):
            self.orig_extract_feature = self.model.extract_feature
            config = self.config
            generator = self.generator
            orig_fn = self.orig_extract_feature

            def hooked_extract_feature(pixel_values, *args, **kwargs):
                vit_embeds = orig_fn(pixel_values, *args, **kwargs)
                # In InternVL, multi-image inputs have leading tiles corresponding to Image 0 (study scene).
                if vit_embeds.shape[0] >= 1:
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


@register_delay_hook("generic", "fallback", "default")
class DelayedGenericHook(BaseDelayedHook):
    """
    Generic PyTorch forward hook targeting vision encoder or multimodal projector modules.
    """

    def __init__(self, model: Any, config: LatentPerturbationConfig):
        super().__init__(model, config)
        self.hook_handle = None

    def __enter__(self):
        if not self.is_active():
            return self

        self.init_generator()
        # Find visual module candidate
        target_mod = None
        for attr in ("visual", "vision_model", "vision_tower", "multi_modal_projector", "mlp1"):
            if hasattr(self.model, attr):
                target_mod = getattr(self.model, attr)
                break
            if hasattr(self.model, "model") and hasattr(self.model.model, attr):
                target_mod = getattr(self.model.model, attr)
                break

        if target_mod is not None:
            config = self.config
            generator = self.generator
            hook_self = self

            def forward_hook(module, args, output):
                if isinstance(output, tuple):
                    first = output[0]
                    if hasattr(first, "ndim") and first.ndim >= 2:
                        perturbed = apply_perturbation(first, config, generator=generator)
                        return (perturbed,) + output[1:]
                elif hasattr(output, "ndim") and output.ndim >= 2:
                    return apply_perturbation(output, config, generator=generator)
                return output

            self.hook_handle = target_mod.register_forward_hook(forward_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


# ==============================================================================
# Delayed VLM Backend
# ==============================================================================

class DelayedVLMBackend(OpenVLMBackend):
    """VLM evaluator with targeted working-memory latent perturbation support."""

    def predict_delayed(
        self,
        trial: Dict[str, Any],
        prompt_style: str = "cot_anyview",
        max_new_tokens: int = 8000,
        perturbation_config: Optional[LatentPerturbationConfig] = None,
    ) -> Tuple[Optional[int], str]:
        if perturbation_config is None or not perturbation_config.effective_sigma and perturbation_config.noise_type not in ("dropout", "spatial_patch"):
            return self.predict(trial, prompt_style=prompt_style, max_new_tokens=max_new_tokens)

        # Dynamically resolve hook from registry
        hook = get_hook_for_model(self.model, self.family, perturbation_config)
        with hook:
            return self.predict(trial, prompt_style=prompt_style, max_new_tokens=max_new_tokens)


# ==============================================================================
# Evaluation Sweep Runner
# ==============================================================================

def evaluate_with_delay(
    model_id: str,
    benchmark_path: str,
    noise_scales: Optional[List[float]] = None,
    delay_seconds_list: Optional[List[float]] = None,
    diffusion_rate: float = 0.05,
    noise_type: str = "gaussian",
    noise_target: str = "study",
    locus: str = "vision_embed",
    prompt_style: str = "cot_anyview",
    max_trials: int = 100,
    seed: int = 0,
    out_dir: str = "results",
) -> Dict[Union[float, str], Dict[str, Any]]:
    """Run an evaluation sweep across multiple noise scales or delay durations."""
    trials = load_benchmark(benchmark_path)
    if max_trials and len(trials) > max_trials:
        trials = _stratified(trials, max_trials, seed=seed)

    print(f"Loaded {len(trials)} trials from {benchmark_path}")
    print(f"Initializing model {model_id}...")
    backend = DelayedVLMBackend(model_id=model_id)

    results_by_sweep = {}
    bench_tag = os.path.basename(benchmark_path).replace(".json", "").replace("vlm_benchmark_", "")
    slug = model_id.replace("/", "_").lower()
    os.makedirs(out_dir, exist_ok=True)

    # Determine sweep items: either delay_seconds or noise_scales
    is_temporal = delay_seconds_list is not None and len(delay_seconds_list) > 0
    sweep_items = delay_seconds_list if is_temporal else (noise_scales or [0.0])

    for item in sweep_items:
        if is_temporal:
            delay_s = float(item)
            scale = math.sqrt(max(0.0, diffusion_rate * delay_s))
            tag = f"delay_{delay_s:.1f}s"
            print("\n=======================================================")
            print(f"EVALUATING DELAY t = {delay_s:.1f}s (effective sigma = {scale:.3f}, D = {diffusion_rate})")
            print(f"Target: {noise_target} | Type: {noise_type} | Locus: {locus}")
            print("=======================================================")
            config = LatentPerturbationConfig(
                noise_scale=scale,
                delay_seconds=delay_s,
                diffusion_rate=diffusion_rate,
                noise_type=noise_type,
                noise_target=noise_target,
                noise_seed=seed,
                locus=locus,
            )
        else:
            scale = float(item)
            delay_s = None
            tag = f"sigma_{scale:.2f}"
            print("\n=======================================================")
            print(f"EVALUATING NOISE SCALE sigma = {scale:.2f} (target: {noise_target}, type: {noise_type})")
            print("=======================================================")
            config = LatentPerturbationConfig(
                noise_scale=scale,
                delay_seconds=None,
                noise_type=noise_type,
                noise_target=noise_target,
                noise_seed=seed,
                locus=locus,
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
            "delay_seconds": delay_s,
            "diffusion_rate": diffusion_rate,
            "noise_scale": scale,
            "effective_sigma": config.effective_sigma,
            "noise_type": noise_type,
            "noise_target": noise_target,
            "locus": locus,
            "prompt_style": prompt_style,
            "n_trials": len(trial_results),
            "overall_accuracy": round(acc_all, 4),
            "gate_accuracy_delta0": round(acc_d0, 4),
            "rotated_accuracy": round(acc_rot, 4),
        }
        results_by_sweep[item] = summary

        out_path = os.path.join(out_dir, f"delayed_{slug}_{bench_tag}_{tag}.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "results": trial_results}, f, indent=2)
        print(f"Saved: {out_path} -> Overall: {acc_all*100:.1f}%, Gate: {acc_d0*100:.1f}%, Rot: {acc_rot*100:.1f}%")

    return results_by_sweep


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM under simulated working memory delay")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--benchmark", type=str, default="data/benchmarks/vlm_benchmark_4afc_hard.json")
    parser.add_argument("--noise_scales", type=str, default=None, help="Comma-separated list of sigmas, e.g. 0.0,0.1,0.25,0.5,1.0")
    parser.add_argument("--delay_seconds", type=str, default=None, help="Comma-separated list of delay durations in seconds, e.g. 0.0,0.5,1.0,2.0,5.0")
    parser.add_argument("--diffusion_rate", type=float, default=0.05, help="Diffusion rate D for sigma = sqrt(D * t)")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "brownian", "spherical", "angular", "ornstein_uhlenbeck", "dropout", "spatial_patch"])
    parser.add_argument("--noise_target", type=str, default="study", choices=["study", "options", "all"])
    parser.add_argument("--locus", type=str, default="vision_embed", choices=["vision_embed", "projector", "inputs_embeds"])
    parser.add_argument("--prompt_style", type=str, default="cot_anyview")
    parser.add_argument("--max_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    noise_scales = None
    if args.noise_scales:
        noise_scales = [float(s.strip()) for s in args.noise_scales.split(",") if s.strip()]

    delay_seconds = None
    if args.delay_seconds:
        delay_seconds = [float(s.strip()) for s in args.delay_seconds.split(",") if s.strip()]

    if noise_scales is None and delay_seconds is None:
        noise_scales = [0.0, 0.1, 0.25, 0.5, 1.0]

    evaluate_with_delay(
        model_id=args.model,
        benchmark_path=args.benchmark,
        noise_scales=noise_scales,
        delay_seconds_list=delay_seconds,
        diffusion_rate=args.diffusion_rate,
        noise_type=args.noise_type,
        noise_target=args.noise_target,
        locus=args.locus,
        prompt_style=args.prompt_style,
        max_trials=args.max_trials,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
