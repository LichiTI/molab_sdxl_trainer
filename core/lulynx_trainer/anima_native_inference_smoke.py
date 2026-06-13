"""CPU-safe smoke for the native-Anima inference stack (#133).

Validates the new wiring units that do NOT need GPU or real weights:

* ``_normalize`` / ``_denormalize`` qwen-image latent round-trip (decode inverse).
* ``anima_sampler_native.resolve_text_embeds`` injected-vs-CLIP routing.
* ``anima_sampler_native.decode_anima_image`` qwen-image (denorm + 5D + squeeze)
  vs standard (``/vae_scaling_factor``) branches, via fakes.
* ``anima_native_inference.encode_qwen3_hidden`` shape ``[1, seq, hidden]``.
* ``NativeAnimaInferenceBundle.is_native_qwen3`` property.
* ``anima_inference_cli.resolve_native_anima_paths`` convention + override.

Faithful native forward (#139-142), all parameter-shape level (no real weights):
* ``AnimaRope3D`` axis split (42/42/44), angle table + apply shape, rotate_half.
* ``AnimaLlmAdapter`` forward shape + pad-position zeroing.
* ``_resolve_t5_tokenizer_dir`` discovery + ``is_faithful`` bundle property.

Run with the flashattention env (CPU is fine, no CUDA required):
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/anima_native_inference_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.lulynx_trainer.anima_cache_runtime import (  # noqa: E402
    _denormalize_qwen_image_latents,
    _normalize_qwen_image_latents,
)
from core.lulynx_trainer.anima_native_inference import (  # noqa: E402
    NativeAnimaInferenceBundle,
    _resolve_faithful_mode,
    _resolve_t5_tokenizer_dir,
    encode_qwen3_hidden,
    faithful_inference_assets_available,
)
from core.lulynx_trainer.anima_native_faithful import (  # noqa: E402
    AnimaLlmAdapter,
    AnimaRope3D,
    _rotate_half_noninterleaved,
)
from core.lulynx_trainer.anima_sampler_native import (  # noqa: E402
    decode_anima_image,
    resolve_text_embeds,
)
from core.lulynx_trainer.anima_native_inference import resolve_native_anima_paths  # noqa: E402

Z_DIM = 16


def _fake_qwen_image_vae():
    """Fake qwen-image VAE: config triggers _is_qwen_image_vae; decode → 5D."""
    config = SimpleNamespace(
        latents_mean=[0.1 * (i + 1) for i in range(Z_DIM)],
        latents_std=[0.5 + 0.01 * i for i in range(Z_DIM)],
        z_dim=Z_DIM,
    )

    class _VAE:
        dtype = torch.float32

        def __init__(self):
            self.config = config

        def to(self, *args, **kwargs):
            return self

        def decode(self, latents):
            # qwen-image decode expects 5D [B,C,1,H,W] → returns [B,3,1,8H,8W].
            assert latents.dim() == 5, latents.shape
            b, _c, t, h, w = latents.shape
            return SimpleNamespace(sample=torch.zeros(b, 3, t, h * 8, w * 8))

    return _VAE()


def _fake_standard_vae():
    """Fake standard VAE: no latents_mean → not qwen-image; 4D decode."""

    class _VAE:
        dtype = torch.float32
        config = SimpleNamespace(scaling_factor=0.18215)

        def to(self, *args, **kwargs):
            return self

        def decode(self, latents):
            assert latents.dim() == 4, latents.shape
            b, _c, h, w = latents.shape
            return SimpleNamespace(sample=torch.zeros(b, 3, h * 8, w * 8))

    return _VAE()


def check_latent_norm_roundtrip() -> bool:
    print("== qwen-image latent normalize/denormalize round-trip ==")
    vae = _fake_qwen_image_vae()
    latents = torch.randn(1, Z_DIM, 8, 8)
    normalized = _normalize_qwen_image_latents(vae, latents)
    restored = _denormalize_qwen_image_latents(vae, normalized)
    err = (restored - latents).abs().max().item()
    ok = err < 1e-4
    print(f"  max_abs_err={err:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def check_resolve_text_embeds() -> bool:
    print("== resolve_text_embeds (injected vs CLIP fallback) ==")
    ok = True
    pe = torch.randn(1, 12, 64)
    ne = torch.randn(1, 12, 64)

    t, n, handled = resolve_text_embeds(pe, ne, device="cpu", dtype=torch.float32, guidance_scale=5.0)
    cond = handled and t is not None and n is not None
    ok &= cond
    print(f"  injected+CFG: handled={handled} neg={'set' if n is not None else 'none'} {'OK' if cond else 'FAIL'}")

    t, n, handled = resolve_text_embeds(pe, ne, device="cpu", dtype=torch.float32, guidance_scale=1.0)
    cond = handled and t is not None and n is None  # CFG off -> no negative
    ok &= cond
    print(f"  injected+noCFG: handled={handled} neg={'set' if n is not None else 'none'} {'OK' if cond else 'FAIL'}")

    t, n, handled = resolve_text_embeds(None, None, device="cpu", dtype=torch.float32, guidance_scale=5.0)
    cond = (not handled) and t is None and n is None  # fall through to CLIP
    ok &= cond
    print(f"  none: handled={handled} {'OK' if cond else 'FAIL'}")
    return ok


def check_decode_anima_image() -> bool:
    print("== decode_anima_image (qwen-image 5D vs standard 4D) ==")
    ok = True

    qi = _fake_qwen_image_vae()
    latents = torch.randn(1, Z_DIM, 16, 16)
    img = decode_anima_image(qi, latents, vae_scaling_factor=1.0)
    cond = img.dim() == 4 and tuple(img.shape) == (1, 3, 128, 128)  # 16*8, temporal dropped
    ok &= cond
    print(f"  qwen-image: out={tuple(img.shape)} {'OK' if cond else 'FAIL'}")

    std = _fake_standard_vae()
    latents = torch.randn(1, 4, 16, 16)
    img = decode_anima_image(std, latents, vae_scaling_factor=0.18215)
    cond = img.dim() == 4 and tuple(img.shape) == (1, 3, 128, 128)
    ok &= cond
    print(f"  standard: out={tuple(img.shape)} {'OK' if cond else 'FAIL'}")
    return ok


def check_encode_qwen3_hidden() -> bool:
    print("== encode_qwen3_hidden (shape [1, seq, hidden]) ==")
    hidden_size = 64
    max_len = 32

    class _Tok:
        def __call__(self, text, padding, truncation, max_length, return_tensors):
            return {
                "input_ids": torch.ones(1, max_length, dtype=torch.long),
                "attention_mask": torch.ones(1, max_length, dtype=torch.long),
            }

    class _Enc:
        config = SimpleNamespace(hidden_size=hidden_size)

        def to(self, *args, **kwargs):
            return self

        def __call__(self, input_ids, attention_mask, **kwargs):
            seq = input_ids.shape[1]
            return SimpleNamespace(last_hidden_state=torch.randn(1, seq, hidden_size))

    out = encode_qwen3_hidden(_Enc(), _Tok(), "lulu, 1girl", device="cpu", dtype=torch.float32, max_length=max_len)
    ok = out.dim() == 3 and out.shape[0] == 1 and out.shape[1] == max_len and out.shape[2] == hidden_size
    print(f"  out={tuple(out.shape)} {'OK' if ok else 'FAIL'}")
    return ok


def check_bundle_property() -> bool:
    print("== NativeAnimaInferenceBundle.is_native_qwen3 ==")
    b1 = NativeAnimaInferenceBundle(unet=object(), vae=object(), qwen3_encoder=object(), qwen3_tokenizer=object())
    b2 = NativeAnimaInferenceBundle(unet=object(), vae=object(), qwen3_encoder=None, qwen3_tokenizer=None)
    ok = b1.is_native_qwen3 and (not b2.is_native_qwen3) and b1.text_encoder_1 is None
    print(f"  full={b1.is_native_qwen3} empty={b2.is_native_qwen3} {'OK' if ok else 'FAIL'}")
    return ok


def check_resolve_paths() -> bool:
    print("== resolve_native_anima_paths (convention + override) ==")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "diffusion_models").mkdir()
        (root / "vae").mkdir()
        (root / "text_encoders").mkdir()
        dit = root / "diffusion_models" / "anima-base.safetensors"
        vae = root / "vae" / "qwen_image_vae.safetensors"
        qwen3 = root / "text_encoders" / "qwen3.safetensors"
        for f in (dit, vae, qwen3):
            f.write_bytes(b"x")

        d, v, q = resolve_native_anima_paths(str(dit))
        cond = Path(d) == dit and Path(v) == vae and Path(q) == qwen3
        ok &= cond
        print(f"  from-dit-file: {'OK' if cond else 'FAIL'} (dit={Path(d).name} vae={Path(v).name} qwen3={Path(q).name})")

        d, v, q = resolve_native_anima_paths(str(root))
        cond = Path(d) == dit and Path(v) == vae and Path(q) == qwen3
        ok &= cond
        print(f"  from-root-dir: {'OK' if cond else 'FAIL'}")

        d, v, q = resolve_native_anima_paths(str(dit), {"anima_vae_path": "/custom/vae.sft"})
        cond = v == "/custom/vae.sft" and Path(d) == dit
        ok &= cond
        print(f"  override-vae: {'OK' if cond else 'FAIL'} (vae={v})")
    return ok


def check_rope3d() -> bool:
    print("== AnimaRope3D (split 42/42/44, angles, apply, rotate_half) ==")
    ok = True
    rope = AnimaRope3D(128)
    cond = rope._dim_h == 42 and rope._dim_t == 44
    ok &= cond
    print(f"  split dim_h={rope._dim_h} dim_t={rope._dim_t} {'OK' if cond else 'FAIL'} (expect 42/44)")

    angles = rope.generate(1, 64, 64, device=torch.device("cpu"))
    cond = tuple(angles.shape) == (4096, 128)
    ok &= cond
    print(f"  angles={tuple(angles.shape)} {'OK' if cond else 'FAIL'} (expect 4096,128)")

    q = torch.randn(1, 16, 4096, 128)
    out = AnimaRope3D.apply(q, angles)
    cond = tuple(out.shape) == (1, 16, 4096, 128) and bool(torch.isfinite(out).all())
    ok &= cond
    print(f"  apply out={tuple(out.shape)} finite={bool(torch.isfinite(out).all())} {'OK' if cond else 'FAIL'}")

    # non-interleaved rotate_half: [x0,x1,x2,x3] -> [-x2,-x3,x0,x1]
    rot = _rotate_half_noninterleaved(torch.arange(4.0).view(1, 4))
    cond = rot.flatten().tolist() == [-2.0, -3.0, 0.0, 1.0]
    ok &= cond
    print(f"  rotate_half={rot.flatten().tolist()} {'OK' if cond else 'FAIL'} (expect [-2,-3,0,1])")
    return ok


def check_llm_adapter_shapes() -> bool:
    print("== AnimaLlmAdapter (forward shape + pad-position zeroing) ==")
    adapter = AnimaLlmAdapter(source_dim=32, target_dim=32, model_dim=32,
                              num_layers=2, num_heads=4, vocab_size=100)
    source = torch.randn(2, 7, 32)
    target_ids = torch.randint(0, 100, (2, 5))
    target_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    source_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    out = adapter(source, target_ids, target_attention_mask=target_mask,
                  source_attention_mask=source_mask)
    ok = tuple(out.shape) == (2, 5, 32)
    print(f"  out={tuple(out.shape)} {'OK' if ok else 'FAIL'} (expect 2,5,32)")
    pad_max = float(out[target_mask == 0].abs().max())
    cond = pad_max == 0.0
    ok &= cond
    print(f"  pad rows max-abs={pad_max:.3e} {'OK' if cond else 'FAIL'} (expect 0.0)")
    cond = float(out[target_mask == 1].abs().sum()) > 0.0
    ok &= cond
    print(f"  valid rows nonzero={cond} {'OK' if cond else 'FAIL'}")
    return ok


def check_t5_dir_and_faithful_property() -> bool:
    print("== _resolve_t5_tokenizer_dir + is_faithful property ==")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dit_dir = root / "diffusion_models"
        dit_dir.mkdir()
        dit = dit_dir / "anima-base.safetensors"
        dit.write_bytes(b"x")
        # sibling tokenizer_t5 next to the DiT directory
        t5 = dit_dir / "tokenizer_t5"
        t5.mkdir()
        (t5 / "spiece.model").write_bytes(b"x")
        found = _resolve_t5_tokenizer_dir(str(dit), None)
        cond = Path(found) == t5
        ok &= cond
        print(f"  resolved={Path(found).name} {'OK' if cond else 'FAIL'} (expect tokenizer_t5)")

        # explicit override wins
        explicit = root / "explicit_t5"
        explicit.mkdir()
        (explicit / "tokenizer.json").write_bytes(b"x")
        found = _resolve_t5_tokenizer_dir(str(dit), str(explicit))
        cond = Path(found) == explicit
        ok &= cond
        print(f"  explicit override={'OK' if cond else 'FAIL'}")

    b_faithful = NativeAnimaInferenceBundle(
        unet=object(), vae=object(), qwen3_encoder=object(), qwen3_tokenizer=object(),
        llm_adapter=object(), t5_tokenizer=object(),
    )
    b_stub = NativeAnimaInferenceBundle(
        unet=object(), vae=object(), qwen3_encoder=object(), qwen3_tokenizer=object(),
    )
    cond = b_faithful.is_faithful and (not b_stub.is_faithful)
    ok &= cond
    print(f"  is_faithful full={b_faithful.is_faithful} stub={b_stub.is_faithful} {'OK' if cond else 'FAIL'}")
    return ok


def check_faithful_auto_resolution() -> bool:
    """#133 — tri-state faithful resolves to default-on auto + graceful degrade.

    ``faithful_inference_assets_available`` gates on the llm_adapter weights (a
    stub checkpoint cannot go faithful) then the T5 tokenizer; ``"auto"`` degrades
    to the stub forward — never crashes — when assets are absent, while an explicit
    ``"on"`` stays True (the loader then raises an actionable error on load).
    """
    from safetensors.torch import save_file

    print("== faithful_inference_assets_available + _resolve_faithful_mode (auto/degrade) ==")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ddir = root / "diffusion_models"
        ddir.mkdir()
        stub = ddir / "stub.safetensors"
        save_file({"net.dummy.weight": torch.zeros(2, 2)}, str(stub))
        base = ddir / "base.safetensors"
        save_file({"net.llm_adapter.embed.weight": torch.zeros(4, 4)}, str(base))
        tok = root / "tok_t5"
        tok.mkdir()
        (tok / "spiece.model").write_bytes(b"")

        avail, reason = faithful_inference_assets_available(stub)
        cond = (avail is False) and (reason == "checkpoint_no_llm_adapter")
        ok &= cond
        print(f"  stub-no-adapter -> {avail}/{reason} {'OK' if cond else 'FAIL'}")

        avail, reason = faithful_inference_assets_available(base)
        cond = (avail is False) and (reason == "no_t5_tokenizer")
        ok &= cond
        print(f"  adapter-no-t5 -> {avail}/{reason} {'OK' if cond else 'FAIL'}")

        avail, reason = faithful_inference_assets_available(base, tok)
        cond = (avail is True) and (reason is None)
        ok &= cond
        print(f"  adapter+t5 -> {avail}/{reason} {'OK' if cond else 'FAIL'}")

        m_auto_missing = _resolve_faithful_mode("auto", stub, None, False)
        m_auto_ok = _resolve_faithful_mode("auto", base, tok, False)
        m_on = _resolve_faithful_mode("on", stub, None, False)
        m_off = _resolve_faithful_mode("off", base, tok, False)
        cond = (m_auto_missing is False) and (m_auto_ok is True) and (m_on is True) and (m_off is False)
        ok &= cond
        print(f"  mode auto(miss)={m_auto_missing} auto(ok)={m_auto_ok} on={m_on} off={m_off} "
              f"{'OK' if cond else 'FAIL'}")
    return ok


if __name__ == "__main__":
    checks = [
        check_latent_norm_roundtrip,
        check_resolve_text_embeds,
        check_decode_anima_image,
        check_encode_qwen3_hidden,
        check_bundle_property,
        check_resolve_paths,
        check_rope3d,
        check_llm_adapter_shapes,
        check_t5_dir_and_faithful_property,
        check_faithful_auto_resolution,
    ]
    results = [fn() for fn in checks]
    print()
    print("RESULT:", "ALL PASS" if all(results) else "FAILURES PRESENT")
    sys.exit(0 if all(results) else 1)
