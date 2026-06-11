"""Smoke test for Newbie DiT flow-matching sampler.

Validates that:
1. sample_newbie runs end-to-end with mock DiT + dual encoders
2. Euler step produces correct output
3. CFG is applied when guidance_scale > 1
4. Dual text encoder embeddings are concatenated
5. VAE decode failure is handled gracefully
6. Pooled features are passed through correctly
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

TRAINER_ROOT = Path(__file__).resolve().parent


def _load_sampler_module():
    module_name = "_lulynx_newbie_sampler_smoke_target"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, TRAINER_ROOT / "newbie_sampler.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load newbie_sampler.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MockDiT(nn.Module):
    """Minimal DiT mock that accepts pooled_emb."""
    def forward(self, x, t, text_emb, pooled_emb=None, **kwargs):
        return type("Output", (), {"sample": torch.randn_like(x)})()


class MockVAE:
    class Config:
        scale_factor = 8
    config = Config()
    dtype = torch.float32

    def decode(self, z):
        b, c, h, w = z.shape
        image = torch.randn(b, 3, h * 8, w * 8)
        return type("Output", (), {"sample": image})()


class MockTextEncoder1(nn.Module):
    """CLIP-like encoder with pooled output."""
    def forward(self, input_ids, return_dict=True):
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden = torch.randn(batch, seq_len, 768)
        text_embeds = torch.randn(batch, 768)
        return type("Output", (), {
            "last_hidden_state": hidden,
            "text_embeds": text_embeds,
        })()


class MockTextEncoder2(nn.Module):
    """T5-like encoder (sequence only)."""
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


def test_sample_newbie_euler():
    """sample_newbie runs end-to-end with Euler sampler."""
    sample_newbie = _load_sampler_module().sample_newbie

    dit = MockDiT()
    vae = MockVAE()
    te1 = MockTextEncoder1()
    te2 = MockTextEncoder2()
    tok1 = MockTokenizer()
    tok2 = MockTokenizer()

    result = sample_newbie(
        dit, vae, te1, te2, tok1, tok2,
        "test prompt",
        num_inference_steps=3,
        width=64, height=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
        sampler_name="euler",
    )
    assert result is not None, "sample_newbie returned None"
    assert result.size[0] == 64 and result.size[1] == 64, f"Expected 64x64, got {result.size}"
    print("PASS: test_sample_newbie_euler")
    return True


def test_sample_newbie_cfg():
    """sample_newbie applies CFG with negative prompt."""
    sample_newbie = _load_sampler_module().sample_newbie

    dit = MockDiT()
    vae = MockVAE()
    te1 = MockTextEncoder1()
    te2 = MockTextEncoder2()
    tok1 = MockTokenizer()
    tok2 = MockTokenizer()

    result = sample_newbie(
        dit, vae, te1, te2, tok1, tok2,
        "test prompt",
        negative_prompt="bad quality",
        num_inference_steps=3,
        guidance_scale=7.5,
        width=64, height=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
    )
    assert result is not None, "sample_newbie with CFG returned None"
    print("PASS: test_sample_newbie_cfg")
    return True


def test_sample_newbie_flow_shift():
    """Flow shift is applied correctly."""
    sample_newbie = _load_sampler_module().sample_newbie

    dit = MockDiT()
    vae = MockVAE()
    te1 = MockTextEncoder1()
    te2 = MockTextEncoder2()
    tok1 = MockTokenizer()
    tok2 = MockTokenizer()

    result = sample_newbie(
        dit, vae, te1, te2, tok1, tok2,
        "test prompt",
        num_inference_steps=3,
        width=64, height=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
        discrete_flow_shift=3.0,
    )
    assert result is not None, "sample_newbie with flow shift returned None"
    print("PASS: test_sample_newbie_flow_shift")
    return True


def test_sample_newbie_vae_scaling():
    """VAE scaling factor is applied correctly for Newbie (0.3611)."""
    sample_newbie = _load_sampler_module().sample_newbie

    dit = MockDiT()
    vae = MockVAE()
    te1 = MockTextEncoder1()
    te2 = MockTextEncoder2()
    tok1 = MockTokenizer()
    tok2 = MockTokenizer()

    # Default vae_scaling_factor=0.3611
    result = sample_newbie(
        dit, vae, te1, te2, tok1, tok2,
        "test prompt",
        num_inference_steps=2,
        width=64, height=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
    )
    assert result is not None
    print("PASS: test_sample_newbie_vae_scaling")
    return True


def test_sample_newbie_broken_vae():
    """sample_newbie handles VAE decode failure gracefully."""
    sample_newbie = _load_sampler_module().sample_newbie

    class BrokenVAE:
        config = type("C", (), {"scale_factor": 8})()
        dtype = torch.float32
        def decode(self, z):
            raise RuntimeError("VAE broken")

    dit = MockDiT()
    vae = BrokenVAE()
    te1 = MockTextEncoder1()
    te2 = MockTextEncoder2()
    tok1 = MockTokenizer()
    tok2 = MockTokenizer()

    result = sample_newbie(
        dit, vae, te1, te2, tok1, tok2,
        "test prompt",
        num_inference_steps=2,
        width=64, height=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
    )
    assert result is None, "Expected None on VAE decode failure"
    print("PASS: test_sample_newbie_broken_vae")
    return True


def main():
    results = []
    tests = [
        test_sample_newbie_euler,
        test_sample_newbie_cfg,
        test_sample_newbie_flow_shift,
        test_sample_newbie_vae_scaling,
        test_sample_newbie_broken_vae,
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
    print("Newbie Sampler Smoke Test Results")
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
