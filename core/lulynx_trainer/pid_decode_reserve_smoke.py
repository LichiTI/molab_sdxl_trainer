# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test: P5 PiD decode-backend opt-in reserve seam.

Proves the decode seam is a genuine default-off, weight-gated reserve:

* No request / ``allow_pid=False`` -> VAE backend, bitwise-identical to calling
  ``vae_decode_fn`` directly (parity).
* PiD requested but not allowed -> still VAE (parity).
* PiD requested + allowed but no user weights -> honest VAE fallback with
  ``pid_blocked`` reason (no error, no download, no bundle).
* PiD requested + allowed + a user-supplied local tiny checkpoint -> the real PiD
  tiny decode runs and produces finite, upscaled pixels.

Run:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/pid_decode_reserve_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch

from core.lulynx_trainer.pid_decoder_backend import SMOKE_CHECKPOINT_KIND
from core.lulynx_trainer.pid_decode_reserve_seam import (
    decode_with_pid_reserve,
    pid_decode_reserve_readiness,
)


def _vae_decode_fn(latents):
    # Deterministic stand-in VAE decode: 16ch latent -> 3ch pixels at same HW.
    return latents[:, :3].tanh()


def _latents():
    return torch.randn(1, 16, 2, 2)


def _write_tiny_checkpoint(path):
    torch.save(
        {
            "kind": SMOKE_CHECKPOINT_KIND,
            "weight": torch.ones(3, 16, 1, 1) * 0.01,
            "bias": torch.zeros(3),
            "pid_scale": 4,
            "vae_down_factor": 8,
        },
        path,
    )


def test_default_is_vae_parity():
    torch.manual_seed(0)
    latents = _latents()
    out = decode_with_pid_reserve(latents, vae_decode_fn=_vae_decode_fn)
    assert out["backend_used"] == "vae"
    assert out["pid_blocked"] is False
    assert torch.equal(out["pixels"], _vae_decode_fn(latents)), "default decode must be bitwise VAE parity"
    print("PASS: default decode is bitwise VAE parity")


def test_pid_requested_but_not_allowed_is_vae():
    torch.manual_seed(1)
    latents = _latents()
    out = decode_with_pid_reserve(
        latents, vae_decode_fn=_vae_decode_fn, request={"decode_backend": "pid"}, allow_pid=False
    )
    assert out["backend_used"] == "vae"
    assert out["pid_requested"] is True
    assert torch.equal(out["pixels"], _vae_decode_fn(latents)), "pid-not-allowed must stay VAE parity"
    print("PASS: PiD requested but allow_pid=False stays VAE (parity)")


def test_pid_allowed_but_weights_missing_falls_back():
    torch.manual_seed(2)
    latents = _latents()
    with tempfile.TemporaryDirectory() as empty_root:
        out = decode_with_pid_reserve(
            latents,
            vae_decode_fn=_vae_decode_fn,
            request={"decode_backend": "pid"},
            allow_pid=True,
            checkpoint_root=empty_root,
        )
    assert out["backend_used"] == "vae"
    assert out["pid_blocked"] is True
    assert "pid_checkpoint_missing" in out["blocked_reasons"]
    assert torch.equal(out["pixels"], _vae_decode_fn(latents)), "fallback must still produce VAE pixels"
    print("PASS: PiD allowed but no user weights -> honest VAE fallback with reason")


def test_pid_allowed_with_user_weights_runs_pid():
    torch.manual_seed(3)
    latents = _latents()
    with tempfile.TemporaryDirectory() as root:
        ckpt = os.path.join(root, "pid_tiny.ckpt")
        _write_tiny_checkpoint(ckpt)
        out = decode_with_pid_reserve(
            latents,
            vae_decode_fn=_vae_decode_fn,
            request={"decode_backend": "pid", "pid_checkpoint": ckpt},
            allow_pid=True,
        )
    assert out["backend_used"] == "pid", out["backend_used"]
    assert out["pid_blocked"] is False
    pixels = out["pixels"]
    assert torch.isfinite(pixels).all(), "PiD pixels must be finite"
    assert pixels.shape[1] == 3, pixels.shape
    assert pixels.shape[-1] == 2 * 4 * 8, pixels.shape  # H * pid_scale * vae_down
    print(f"PASS: PiD allowed + user weights runs real tiny decode (pixels {list(pixels.shape)})")


def test_readiness_flags():
    report = pid_decode_reserve_readiness()
    assert report["wired"] is True
    assert report["default_backend"] == "vae"
    for flag in (
        "vae_replacement_allowed",
        "runtime_activation_enabled",
        "request_fields_emitted",
        "auto_download_allowed",
        "bundle_weights_allowed",
        "default_behavior_changed",
        "promotion_ready",
    ):
        assert report[flag] is False, flag
    print("PASS: readiness reports wired reserve with all activation/weight gates False")


def main():
    test_default_is_vae_parity()
    test_pid_requested_but_not_allowed_is_vae()
    test_pid_allowed_but_weights_missing_falls_back()
    test_pid_allowed_with_user_weights_runs_pid()
    test_readiness_flags()
    print("\n[pid_decode_reserve_smoke] 5/5 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
