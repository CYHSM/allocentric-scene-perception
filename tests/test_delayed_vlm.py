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

def test_temporal_delay_effective_sigma():
    # sigma = sqrt(D * t)
    cfg = LatentPerturbationConfig(delay_seconds=2.0, diffusion_rate=0.08)
    expected_sigma = (0.08 * 2.0) ** 0.5
    assert abs(cfg.effective_sigma - expected_sigma) < 1e-6

    # Explicit noise_scale takes precedence if delay_seconds is None
    cfg2 = LatentPerturbationConfig(noise_scale=0.35, delay_seconds=None)
    assert cfg2.effective_sigma == 0.35


def test_perturbation_ornstein_uhlenbeck():
    # High theta means strong reversion to prior
    x = torch.ones(10, 64) * 5.0
    cfg = LatentPerturbationConfig(
        noise_scale=0.1,
        noise_type="ornstein_uhlenbeck",
        delay_seconds=10.0,
        ou_theta=2.0,
        ou_prior_mode="zero",
        noise_seed=42,
    )
    out = apply_perturbation(x, cfg)
    # Output should decay significantly towards 0
    assert out.mean().item() < 1.0


def test_perturbation_spatial_patch():
    # 20 spatial patches of dim 32
    x = torch.zeros(20, 32)
    cfg = LatentPerturbationConfig(
        noise_scale=1.0,
        noise_type="spatial_patch",
        patch_fraction=0.3,
        noise_seed=123,
    )
    out = apply_perturbation(x, cfg)
    # Patches that were not perturbed should remain exactly zero
    zero_patches = (out.abs().sum(dim=-1) == 0.0).sum().item()
    assert 12 <= zero_patches <= 16, f"Expected ~14 unperturbed patches (70%), got {zero_patches}"


def test_hook_registry_and_custom_extension():
    from bench.delayed_vlm import register_delay_hook, get_hook_for_model, BaseDelayedHook

    @register_delay_hook("custom_neuro_model")
    class CustomNeuroHook(BaseDelayedHook):
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class DummyNeuroModel:
        pass

    cfg = LatentPerturbationConfig(noise_scale=0.5)
    hook = get_hook_for_model(DummyNeuroModel(), family="custom_neuro_model", config=cfg)
    assert isinstance(hook, CustomNeuroHook), "Registry must resolve custom registered hook"


def test_delayed_internvl_hook():
    from bench.delayed_vlm import DelayedInternVLHook

    class DummyInternVLModel:
        def __init__(self):
            self.device = "cpu"
        def extract_feature(self, pixel_values):
            # 5 tiles: 1 study scene + 4 options
            torch.manual_seed(42); return torch.randn(5, 10, 64)

    model = DummyInternVLModel()
    orig_fn = model.extract_feature
    cfg = LatentPerturbationConfig(noise_scale=0.5, noise_target="study", noise_seed=42)

    with DelayedInternVLHook(model, cfg):
        assert model.extract_feature != orig_fn
        out = model.extract_feature(None)
        # Tile 0 (study scene) must be perturbed
        torch.manual_seed(42); orig = torch.randn(5, 10, 64); assert not torch.allclose(out[0], orig[0])
        # Tiles 1..4 (options) must remain completely uncorrupted (all 1.0)
        torch.manual_seed(42); orig = torch.randn(5, 10, 64); assert torch.allclose(out[1:], orig[1:])

    assert model.extract_feature == orig_fn, "Hook must restore original extract_feature"
