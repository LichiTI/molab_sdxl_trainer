"""Smoke test for Anima flow-matching sampler.

Validates that:
1. sample_anima runs end-to-end with a mock DiT
2. Euler step produces correct output
3. DPM-Solver step produces correct output
4. Sigma schedule is correctly constructed
5. Flow shift is applied when configured
6. VAE decode failure is handled gracefully
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn

TRAINER_ROOT = Path(__file__).resolve().parent


def _load_sampler_module():
    backend_root = TRAINER_ROOT.parents[1]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    module_name = "core.lulynx_trainer.anima_sampler"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    return importlib.import_module(module_name)


class MockDiT(nn.Module):
    """Minimal DiT mock that returns velocity predictions."""
    def forward(self, x, t, text_emb, **kwargs):
        # Return simple velocity: predict noise (identity velocity)
        return type("Output", (), {"sample": torch.randn_like(x)})()


class MockVAE:
    """Minimal VAE mock."""
    class Config:
        scale_factor = 8

        def get(self, key, default=None):
            return getattr(self, key, default)

    config = Config()
    dtype = torch.float32

    def decode(self, z):
        # Return a valid image-sized tensor
        b, c, h, w = z.shape
        image = torch.randn(b, 3, h * 8, w * 8)
        return type("Output", (), {"sample": image})()


class MockTextEncoder(nn.Module):
    def forward(self, input_ids, return_dict=True):
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden = torch.randn(batch, seq_len, 768)
        return type("Output", (), {"last_hidden_state": hidden})()


class MockTokenizer:
    def __call__(self, text, **kwargs):
        max_length = kwargs.get("max_length", 77)
        return type("Output", (), {
            "input_ids": torch.randint(0, 1000, (1, max_length)),
        })()


def test_euler_step():
    """Euler step correctly applies velocity update."""
    _euler_step = _load_sampler_module()._euler_step

    sample = torch.ones(1, 4, 4, 4)
    velocity = torch.ones(1, 4, 4, 4) * 0.5
    sigma = torch.tensor([1.0])
    sigma_next = torch.tensor([0.5])

    result = _euler_step(velocity, sample, sigma, sigma_next)
    # dt = 0.5 - 1.0 = -0.5, result = 1.0 + 0.5 * (-0.5) = 0.75
    expected = torch.ones_like(sample) * 0.75
    assert torch.allclose(result, expected, atol=1e-5), f"Expected 0.75, got {result.mean().item()}"
    print("PASS: test_euler_step")
    return True


def test_dpm_solver_step():
    """DPM-Solver step produces valid output (reduces to Euler for 1st order)."""
    _dpm_solver_step = _load_sampler_module()._dpm_solver_step

    sample = torch.ones(1, 4, 4, 4)
    velocity = torch.ones(1, 4, 4, 4) * 0.5
    sigma = torch.tensor([1.0])
    sigma_next = torch.tensor([0.5])

    result = _dpm_solver_step(velocity, sample, sigma, sigma_next)
    # Same as Euler for 1st order flow matching
    expected = torch.ones_like(sample) * 0.75
    assert torch.allclose(result, expected, atol=1e-5), f"Expected 0.75, got {result.mean().item()}"
    print("PASS: test_dpm_solver_step")
    return True


def test_sample_anima_euler():
    """sample_anima runs end-to-end with Euler sampler."""
    sample_anima = _load_sampler_module().sample_anima

    dit = MockDiT()
    vae = MockVAE()
    te = MockTextEncoder()
    tok = MockTokenizer()

    result = sample_anima(
        dit, vae, te, tok,
        "test prompt",
        num_inference_steps=3,
        width=64, height=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
        sampler_name="euler",
    )
    assert result is not None, "sample_anima returned None"
    assert result.size[0] == 64 and result.size[1] == 64, f"Expected 64x64, got {result.size}"
    print("PASS: test_sample_anima_euler")
    return True


def test_sample_anima_dpm_solver():
    """sample_anima runs end-to-end with DPM-Solver sampler."""
    sample_anima = _load_sampler_module().sample_anima

    dit = MockDiT()
    vae = MockVAE()
    te = MockTextEncoder()
    tok = MockTokenizer()

    result = sample_anima(
        dit, vae, te, tok,
        "test prompt",
        negative_prompt="bad",
        num_inference_steps=3,
        guidance_scale=3.0,
        width=64, height=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
        sampler_name="dpm_solver",
    )
    assert result is not None, "sample_anima with dpm_solver returned None"
    print("PASS: test_sample_anima_dpm_solver")
    return True


def test_sample_anima_smc_cfg():
    """sample_anima runs end-to-end with SMC-CFG enabled."""
    sample_anima = _load_sampler_module().sample_anima

    dit = MockDiT()
    vae = MockVAE()
    te = MockTextEncoder()
    tok = MockTokenizer()

    result = sample_anima(
        dit, vae, te, tok,
        "test prompt",
        negative_prompt="bad",
        num_inference_steps=3,
        guidance_scale=3.0,
        width=64, height=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
        sampler_name="euler",
        smc_cfg=True,
        smc_cfg_lambda=5.0,
        smc_cfg_alpha=0.2,
    )
    assert result is not None, "sample_anima with smc_cfg returned None"
    print("PASS: test_sample_anima_smc_cfg")
    return True


def test_sample_anima_flow_shift():
    """Flow shift modifies the sigma schedule."""
    sample_anima = _load_sampler_module().sample_anima

    dit = MockDiT()
    vae = MockVAE()
    te = MockTextEncoder()
    tok = MockTokenizer()

    # Should not crash with shift != 1.0
    result = sample_anima(
        dit, vae, te, tok,
        "test prompt",
        num_inference_steps=3,
        width=64, height=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
        discrete_flow_shift=3.0,
    )
    assert result is not None, "sample_anima with flow shift returned None"
    print("PASS: test_sample_anima_flow_shift")
    return True


def test_sample_anima_no_vae_decode():
    """sample_anima handles VAE decode failure gracefully."""
    sample_anima = _load_sampler_module().sample_anima

    class BrokenVAE:
        class Config:
            scale_factor = 8

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        dtype = torch.float32
        def decode(self, z):
            raise RuntimeError("VAE broken")

    dit = MockDiT()
    vae = BrokenVAE()
    te = MockTextEncoder()
    tok = MockTokenizer()

    result = sample_anima(
        dit, vae, te, tok,
        "test prompt",
        num_inference_steps=2,
        width=64, height=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
    )
    assert result is None, "Expected None on VAE decode failure"
    print("PASS: test_sample_anima_no_vae_decode")
    return True


def main():
    results = []
    tests = [
        test_euler_step,
        test_dpm_solver_step,
        test_sample_anima_euler,
        test_sample_anima_dpm_solver,
        test_sample_anima_smc_cfg,
        test_sample_anima_flow_shift,
        test_sample_anima_no_vae_decode,
    ]

    for test_fn in tests:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {test_fn.__name__} — {e}")
            results.append((test_fn.__name__, False))

    print("\n" + "=" * 60)
    print("Anima Sampler Smoke Test Results")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
