import pytest
import torch
from bench.delayed_vlm import (
    LatentPerturbationConfig,
    apply_perturbation,
    DelayedQwenHook,
)


def test_perturbation_zero_noise():
    x = torch.randn(10, 64)
    cfg = LatentPerturbationConfig(noise_scale=0.0)
    out = apply_perturbation(x, cfg)
    assert torch.equal(x, out)


def test_perturbation_gaussian():
    x = torch.randn(10, 64)
    cfg = LatentPerturbationConfig(noise_scale=0.5, noise_type="gaussian", noise_seed=123)
    out1 = apply_perturbation(x, cfg, generator=torch.Generator().manual_seed(123))
    out2 = apply_perturbation(x, cfg, generator=torch.Generator().manual_seed(123))
    
    assert out1.shape == x.shape
    assert not torch.equal(x, out1)
    assert torch.equal(out1, out2), "Identical seeds must yield identical perturbations"


def test_perturbation_spherical():
    x = torch.randn(10, 64)
    orig_norm = torch.norm(x, dim=-1)
    cfg = LatentPerturbationConfig(noise_scale=0.5, noise_type="spherical")
    out = apply_perturbation(x, cfg)
    new_norm = torch.norm(out, dim=-1)
    
    assert torch.allclose(orig_norm, new_norm, atol=1e-5), "Spherical noise must preserve norm"


def test_perturbation_dropout():
    x = torch.ones(100, 100)
    cfg = LatentPerturbationConfig(noise_type="dropout", dropout_prob=0.3)
    out = apply_perturbation(x, cfg)
    zero_frac = (out == 0.0).float().mean().item()
    assert 0.2 < zero_frac < 0.4, f"Zero fraction should be ~0.3, got {zero_frac}"


def test_delayed_qwen_hook_lifecycle():
    class DummyVision:
        def __init__(self):
            self.pooler_output = [torch.randn(5, 32) for _ in range(5)]

    class DummyModelInner:
        def get_image_features(self, *args, **kwargs):
            return DummyVision()

    class DummyModel:
        def __init__(self):
            self.model = DummyModelInner()
            self.device = "cpu"

    model = DummyModel()
    orig_fn = model.model.get_image_features
    cfg = LatentPerturbationConfig(noise_scale=0.5, noise_target="study")

    with DelayedQwenHook(model, cfg):
        assert model.model.get_image_features != orig_fn
        out = model.model.get_image_features()
        assert len(out.pooler_output) == 5

    assert model.model.get_image_features == orig_fn, "Hook must restore original method on exit"
