# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test: native-Anima colorize / EasyControl v2 INFERENCE path.

CPU-only, no real weights. Drives ``anima_colorize_inference`` against the same
synthetic-shape ``AnimaNativeExecutableSubset`` the patch smoke uses, plus a toy
VAE encode, to prove the inference mirror of the training two-stream consumption:

* **No control -> bitwise parity.** ``colorize_condition_context`` with a
  ``None`` adapter (or ``None`` condition) is a pure no-op: the DiT forward is
  byte-for-byte the unconditioned render.
* **Per-forward reset (the crux).** Inside the context the condition is reset to
  the original control tokens before *every* forward, so two successive forwards
  both start from the same base — even though the patched blocks evolve and
  republish the condition within a single forward.
* **Adapter load round-trip.** A saved ``easycontrol_v2.*`` state dict (+ optional
  metadata) reconstructs the adapter config from tensor shapes and reloads exactly.
* **Engage changes output.** A non-trivial adapter + condition really alters the
  render (the condition is used, not silently dropped).
* **Honest degrade.** A checkpoint with no ``easycontrol_v2.*`` keys loads to
  ``None`` (caller falls back to plain text-to-image).
* **Control-image encode.** A control image encodes to ``[1, 16, h, w]`` condition
  latents (as-is and grayscale-derive paths).

Run directly:
    backend/env/python-flashattention/python.exe \\
        backend/core/lulynx_trainer/anima_colorize_inference_smoke.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch
import torch.nn.functional as F
from PIL import Image

from core.lulynx_trainer.anima_colorize_inference import (
    build_cond_vae_encode_fn,  # noqa: F401  (kept for import-surface coverage)
    colorize_condition_context,
    encode_control_image_to_cond_latents,
    load_easycontrol_v2_adapter_for_inference,
)
from core.lulynx_trainer.anima_colorize_inference import ADAPTER_PREFIX
from core.lulynx_trainer.easycontrol_v2_anima_patch_smoke import (
    COND_CHANNELS,
    HIDDEN,
    _adapter,
    _build_subset,
    _inputs,
)


def _cond_latents(h: int = 4, w: int = 4) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(1, COND_CHANNELS, h, w)


def test_no_control_is_bitwise_parity() -> None:
    torch.manual_seed(0)
    model = _build_subset(2)
    sample, ts, text = _inputs()
    with torch.no_grad():
        baseline = model(sample, ts, text).sample

    # None adapter -> no-op context.
    with colorize_condition_context(model, None, _cond_latents()) as handle:
        assert handle is None
        with torch.no_grad():
            out_none_adapter = model(sample, ts, text).sample
    assert torch.equal(baseline, out_none_adapter), "None-adapter context broke parity"

    # None condition -> no-op context.
    adapter = _adapter(2)
    with colorize_condition_context(model, adapter, None) as handle:
        assert handle is None
        with torch.no_grad():
            out_none_cond = model(sample, ts, text).sample
    assert torch.equal(baseline, out_none_cond), "None-condition context broke parity"
    print("PASS: no control image -> context is a no-op (bitwise parity)")


def test_condition_resets_before_every_forward() -> None:
    torch.manual_seed(1)
    model = _build_subset(2)
    adapter = _adapter(2)  # zero-init, b_cond=-10; cond still evolves via cond self-attn
    sample, ts, text = _inputs()
    cond = _cond_latents()
    base = adapter.encode_cond_latents(cond).detach()

    observed = []
    with colorize_condition_context(model, adapter, cond):
        # Registered AFTER the context's reset hook -> fires second -> sees the reset value.
        obs = model.register_forward_pre_hook(
            lambda _m, _a: observed.append(adapter._cond_tokens.clone())
        )
        try:
            with torch.no_grad():
                model(sample, ts, text)              # forward 1
            evolved_1 = adapter._cond_tokens.clone()  # end of forward 1 (republished, evolved)
            with torch.no_grad():
                model(sample, ts, text)              # forward 2
        finally:
            obs.remove()

    assert len(observed) == 2, observed
    assert torch.allclose(observed[0], base), "start of forward 1 was not the base control condition"
    assert torch.equal(observed[0], observed[1]), "condition was NOT reset before forward 2 (stale evolved state)"
    assert not torch.allclose(evolved_1, base, atol=1e-6), "condition did not evolve within a forward (test is vacuous)"
    print("PASS: condition resets to the base control tokens before every forward (evolves within, resets across)")


def test_adapter_load_roundtrip() -> None:
    torch.manual_seed(2)
    adapter = _adapter(2, b_cond=-3.0, init_zero_out=False)
    # Make a couple of weights non-trivial so a faithful reload is observable.
    with torch.no_grad():
        adapter.cond_proj.weight.add_(0.1)
        adapter.blocks[0].b_cond.add_(1.0)
    from safetensors.torch import save_file

    state = {f"{ADAPTER_PREFIX}{k}": v.contiguous().cpu() for k, v in adapter.state_dict().items()}
    metadata = {"ss_easycontrol_v2_cond_lora_alpha": "16.0", "ss_easycontrol_v2_cond_scale": "1.0"}
    tmp = tempfile.mkdtemp(prefix="lulynx_colorize_load_")
    try:
        path = os.path.join(tmp, "colorize_lora.safetensors")
        save_file(state, path, metadata=metadata)
        loaded = load_easycontrol_v2_adapter_for_inference(path, device="cpu", dtype=torch.float32)
        assert loaded is not None, "loader returned None for a real easycontrol_v2 checkpoint"
        assert loaded.config.hidden_size == HIDDEN, loaded.config.hidden_size
        assert loaded.config.cond_channels == COND_CHANNELS, loaded.config.cond_channels
        assert loaded.config.num_blocks == 2, loaded.config.num_blocks
        assert loaded.config.cond_lora_rank == 2, loaded.config.cond_lora_rank
        assert loaded.config.apply_ffn_lora is True, loaded.config.apply_ffn_lora
        assert torch.allclose(loaded.cond_proj.weight, adapter.cond_proj.weight), "cond_proj did not round-trip"
        assert torch.allclose(loaded.blocks[0].b_cond, adapter.blocks[0].b_cond), "b_cond did not round-trip"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS: adapter load round-trips (config inferred from shapes, weights match)")


def test_engage_changes_output() -> None:
    torch.manual_seed(3)
    model = _build_subset(2)
    sample, ts, text = _inputs()
    with torch.no_grad():
        baseline = model(sample, ts, text).sample  # plain (control-off)

    adapter = _adapter(2, b_cond=15.0, init_zero_out=False)  # condition really used
    cond = _cond_latents()
    with colorize_condition_context(model, adapter, cond):
        with torch.no_grad():
            conditioned = model(sample, ts, text).sample
    assert torch.isfinite(conditioned).all(), "conditioned render produced non-finite values"
    drift = float((conditioned - baseline).norm() / (baseline.norm() + 1e-8))
    assert drift > 1e-2, f"condition had no visible effect on the render (drift={drift})"
    # And after the context exits, parity is restored.
    with torch.no_grad():
        restored = model(sample, ts, text).sample
    assert torch.equal(restored, baseline), "context exit did not restore the plain forward"
    print(f"PASS: condition engages the render (drift={drift:.3e}); context exit restores parity")


def test_honest_degrade_no_easycontrol_keys() -> None:
    from safetensors.torch import save_file

    tmp = tempfile.mkdtemp(prefix="lulynx_colorize_degrade_")
    try:
        path = os.path.join(tmp, "plain_lora.safetensors")
        save_file({"lora_unet_blocks_0.lora_down.weight": torch.zeros(4, 8)}, path)
        loaded = load_easycontrol_v2_adapter_for_inference(path, device="cpu", dtype=torch.float32)
        assert loaded is None, "loader should return None for a checkpoint without easycontrol_v2.* keys"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS: checkpoint without easycontrol_v2.* keys loads to None (plain t2i fallback)")


def _toy_vae_encode(image: torch.Tensor) -> torch.Tensor:
    """Toy [1,3,H,W] -> [1,16,H//8,W//8] (mean-lift to 16 ch + 8x avg-pool)."""
    gray = image.mean(dim=1, keepdim=True)            # [1,1,H,W]
    lifted = gray.repeat(1, COND_CHANNELS, 1, 1)      # [1,16,H,W]
    return F.avg_pool2d(lifted, kernel_size=8)        # [1,16,H//8,W//8]


def test_encode_control_image() -> None:
    tmp = tempfile.mkdtemp(prefix="lulynx_colorize_encode_")
    try:
        # Synthetic 64x64 control image.
        arr = (torch.rand(64, 64, 3) * 255).to(torch.uint8).numpy()
        img_path = os.path.join(tmp, "control.png")
        Image.fromarray(arr, "RGB").save(img_path)

        cond_asis = encode_control_image_to_cond_latents(_toy_vae_encode, img_path, derive_mode="asis")
        assert tuple(cond_asis.shape) == (1, COND_CHANNELS, 8, 8), cond_asis.shape

        cond_gray = encode_control_image_to_cond_latents(_toy_vae_encode, img_path, derive_mode="grayscale")
        assert tuple(cond_gray.shape) == (1, COND_CHANNELS, 8, 8), cond_gray.shape
        assert torch.isfinite(cond_asis).all() and torch.isfinite(cond_gray).all()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS: control image encodes to [1,16,h,w] cond latents (asis + grayscale-derive)")


def test_condition_warmup_skips_then_engages() -> None:
    from core.lulynx_trainer.anima_colorize_inference import _resolve_condition_warmup_forwards

    # Pure mapping (the #214 colorize default = ratio 0.6 -> w12 at 20 steps).
    assert _resolve_condition_warmup_forwards(20, 4.0, 0.6) == 24   # CFG -> 12 steps * 2 branches
    assert _resolve_condition_warmup_forwards(20, 1.0, 0.6) == 12   # no CFG -> 1 branch
    assert _resolve_condition_warmup_forwards(0, 4.0, 0.6) == 0     # no steps -> inject every forward
    assert _resolve_condition_warmup_forwards(20, 4.0, 0.0) == 0    # ratio 0 -> inject every forward

    # Behaviour: warmup=2 forwards run as plain t2i (no condition), then engage.
    torch.manual_seed(5)
    model = _build_subset(2)
    sample, ts, text = _inputs()
    with torch.no_grad():
        baseline = model(sample, ts, text).sample
    adapter = _adapter(2, b_cond=15.0, init_zero_out=False)  # condition really used once engaged
    cond = _cond_latents()
    seen = []
    with colorize_condition_context(model, adapter, cond, condition_warmup_forwards=2):
        # Fires AFTER the context's reset hook -> sees the post-reset condition.
        obs = model.register_forward_pre_hook(lambda _m, _a: seen.append(adapter._cond_tokens))
        try:
            with torch.no_grad():
                w1 = model(sample, ts, text).sample   # warm-up forward 1 (no condition)
                w2 = model(sample, ts, text).sample   # warm-up forward 2 (no condition)
                engaged = model(sample, ts, text).sample  # forward 3 (condition engages)
        finally:
            obs.remove()

    assert seen[0] is None and seen[1] is None, "warm-up forwards must carry no condition"
    assert seen[2] is not None, "condition must engage after the warm-up forwards"
    assert torch.equal(w1, baseline) and torch.equal(w2, baseline), "warm-up forwards must be plain-t2i parity"
    assert not torch.equal(engaged, baseline), "post-warm-up forward must engage the condition"
    print("PASS: condition warm-up skips early forwards (t2i parity) then engages (ratio 0.6 = w12 default)")


def main() -> int:
    test_no_control_is_bitwise_parity()
    test_condition_resets_before_every_forward()
    test_adapter_load_roundtrip()
    test_engage_changes_output()
    test_honest_degrade_no_easycontrol_keys()
    test_encode_control_image()
    test_condition_warmup_skips_then_engages()
    print("\n[anima_colorize_inference_smoke] 7/7 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
