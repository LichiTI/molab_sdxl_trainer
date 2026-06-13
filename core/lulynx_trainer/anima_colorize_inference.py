# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Native-Anima colorize / EasyControl v2 INFERENCE path (clean-room Lulynx).

Training already wires the two-stream patch + per-step condition feed (trainer +
``training_step_forward_loss_backward_handlers``).  This module is the missing
*inference* mirror: load the trained ``easycontrol_v2.*`` adapter out of a LoRA
checkpoint, encode a control image (line-art / grayscale) into condition latents,
and drive the existing :func:`anima_sampler.sample_anima` denoise loop with that
condition re-applied before every DiT forward.

Default-off red-lines (mirroring every other Lulynx reserve):

* **No control image -> bitwise parity.** :func:`colorize_condition_context` is a
  no-op when the adapter or the condition latents are absent, so a plain
  text-to-image render is byte-for-byte identical to today.
* **Per-forward reset.** The patched ``_Block.forward`` *evolves* the condition
  stream across the 28 blocks and republishes it, so the inference loop must
  reset the condition to the original control tokens before every forward (each
  denoise step + each CFG branch).  Training does this via the per-step handler
  ``set_cond``; here a ``forward_pre_hook`` on the DiT does the same, so
  ``sample_anima`` itself needs **zero changes**.
* **Honest degrade.** A checkpoint with no ``easycontrol_v2.*`` keys loads to
  ``None`` and the caller falls back to plain text-to-image (never an error).

Reuses, never reimplements:

* adapter / two-stream patch -> ``easycontrol_v2_adapter`` + ``easycontrol_v2_anima_patch``
* control-image VAE encode    -> ``anima_cache_builder._encode_latents_chunked`` (training path)
* control-image derive         -> ``core.tools.colorize_preprocess._make_control_image``
* qwen-image VAE encode fn     -> ``anima_cache_runtime.build_anima_cache_encode_bundle``

PolyForm Noncommercial. Shares no source with any reference repository.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

ADAPTER_PREFIX = "easycontrol_v2."
# Trainer builds the adapter without an explicit cond_lora_alpha, so the config
# dataclass default (16.0) is what produced every saved checkpoint; cond_scale
# likewise defaults to 1.0. Both are runtime scalars (not saved tensors), so we
# recover them from metadata when present and fall back to these defaults.
_DEFAULT_COND_LORA_ALPHA = 16.0
_DEFAULT_COND_SCALE = 1.0


def _adapter_imports():
    try:
        from .easycontrol_v2_adapter import (
            EasyControlV2Adapter,
            EasyControlV2AdapterConfig,
        )
        from .easycontrol_v2_anima_patch import (
            install_easycontrol_v2_anima_executable_subset_patch,
        )
    except ImportError:  # pragma: no cover - direct-file smoke fallback
        from core.lulynx_trainer.easycontrol_v2_adapter import (
            EasyControlV2Adapter,
            EasyControlV2AdapterConfig,
        )
        from core.lulynx_trainer.easycontrol_v2_anima_patch import (
            install_easycontrol_v2_anima_executable_subset_patch,
        )
    return (
        EasyControlV2Adapter,
        EasyControlV2AdapterConfig,
        install_easycontrol_v2_anima_executable_subset_patch,
    )


# ---------------------------------------------------------------------------
# Adapter load (reconstruct config from saved tensor shapes; metadata optional)
# ---------------------------------------------------------------------------

def _read_prefixed_state_and_metadata(
    adapter_path: str | Path,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Read the ``easycontrol_v2.*`` tensors (prefix stripped) + file metadata.

    Uses ``safe_open`` + ``get_tensor`` so every tensor is materialised (the
    mmap handle is released when the ``with`` block closes — important on
    Windows where a lingering mmap blocks later file ops).
    """
    from safetensors import safe_open

    state: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, str] = {}
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        md = handle.metadata()
        if md:
            metadata = dict(md)
        for key in handle.keys():
            if key.startswith(ADAPTER_PREFIX):
                state[key[len(ADAPTER_PREFIX):]] = handle.get_tensor(key)
    return state, metadata


def _infer_adapter_config(state: Dict[str, torch.Tensor], metadata: Dict[str, str], config_cls):
    """Reverse-engineer the adapter config from the saved tensor shapes.

    Shapes are exact and always present; the two runtime-only scalars
    (``cond_lora_alpha`` / ``cond_scale``) come from metadata when available else
    fall back to the trainer's defaults. ``b_cond_init`` / ``init_zero_out`` are
    init-only (the loaded ``b_cond`` parameter overrides the former, the latter is
    unused after load), so their value here is immaterial.
    """
    cond_w = state.get("cond_proj.weight")
    if cond_w is None:
        return None
    hidden_size = int(cond_w.shape[0])
    cond_channels = int(cond_w.shape[1])

    block_ids = set()
    for key in state:
        if key.startswith("blocks."):
            try:
                block_ids.add(int(key.split(".")[1]))
            except (IndexError, ValueError):
                continue
    num_blocks = (max(block_ids) + 1) if block_ids else 1
    first = min(block_ids) if block_ids else 0

    down = state.get(f"blocks.{first}.qkv.down.weight")
    cond_lora_rank = int(down.shape[0]) if down is not None else 16
    apply_ffn_lora = f"blocks.{first}.ffn1.down.weight" in state

    def _md_float(field: str, default: float) -> float:
        try:
            return float(metadata[f"ss_easycontrol_v2_{field}"])
        except (KeyError, TypeError, ValueError):
            return default

    return config_cls(
        hidden_size=hidden_size,
        cond_channels=cond_channels,
        cond_lora_rank=cond_lora_rank,
        cond_lora_alpha=_md_float("cond_lora_alpha", _DEFAULT_COND_LORA_ALPHA),
        num_blocks=num_blocks,
        cond_scale=_md_float("cond_scale", _DEFAULT_COND_SCALE),
        apply_ffn_lora=apply_ffn_lora,
        init_zero_out=False,
    ).normalized()


def load_easycontrol_v2_adapter_for_inference(
    adapter_path: str | Path,
    *,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
):
    """Load the trained ``easycontrol_v2.*`` adapter from a LoRA checkpoint.

    Returns a ready (eval, on device/dtype) :class:`EasyControlV2Adapter`, or
    ``None`` when the checkpoint carries no ``easycontrol_v2.*`` keys (so the
    caller transparently falls back to plain text-to-image). Never raises on a
    non-colorize checkpoint.
    """
    EasyControlV2Adapter, EasyControlV2AdapterConfig, _ = _adapter_imports()
    dtype = dtype or torch.bfloat16
    try:
        state, metadata = _read_prefixed_state_and_metadata(adapter_path)
    except Exception as exc:  # unreadable file -> degrade to plain t2i
        logger.warning("native-anima colorize: cannot read %s (%s); plain t2i.", adapter_path, exc)
        return None
    if "cond_proj.weight" not in state:
        logger.info(
            "native-anima colorize: %s carries no easycontrol_v2.* keys; plain t2i.",
            Path(adapter_path).name,
        )
        return None

    cfg = _infer_adapter_config(state, metadata, EasyControlV2AdapterConfig)
    adapter = EasyControlV2Adapter(cfg)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(
            "native-anima colorize adapter load: missing=%s unexpected=%s",
            list(missing), list(unexpected),
        )
    adapter.to(device=device, dtype=dtype)
    adapter.eval()
    logger.info(
        "native-anima colorize adapter loaded: hidden=%d cond_ch=%d blocks=%d rank=%d ffn=%s",
        cfg.hidden_size, cfg.cond_channels, cfg.num_blocks, cfg.cond_lora_rank, cfg.apply_ffn_lora,
    )
    return adapter


# ---------------------------------------------------------------------------
# Control-image -> condition latents (training-identical VAE encode)
# ---------------------------------------------------------------------------

def build_cond_vae_encode_fn(
    bundle: Any,
    *,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the SAME qwen-image VAE encode callable training uses for cond.

    Reuses :func:`anima_cache_runtime.build_anima_cache_encode_bundle` so the
    inference cond-latent encode is byte-for-byte the training cond-latent encode
    (5D unsqueeze + qwen-image normalisation). The bundle exposes ``.vae`` and a
    Qwen3 encoder/tokenizer, which is all the encode-bundle builder requires.
    """
    dtype = dtype or torch.bfloat16
    try:
        from .anima_cache_runtime import build_anima_cache_encode_bundle
    except ImportError:  # pragma: no cover - direct-file fallback
        from core.lulynx_trainer.anima_cache_runtime import build_anima_cache_encode_bundle

    class _TokLenCfg:  # token-length attrs are only read by the (unused) text encode
        anima_qwen3_max_token_length = 512
        anima_t5_max_token_length = 512

    encode_bundle = build_anima_cache_encode_bundle(
        model=bundle, device=device, dtype=dtype, config=_TokLenCfg()
    )
    return encode_bundle.vae_encode_fn


def encode_control_image_to_cond_latents(
    vae_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    control_image_path: str | Path,
    *,
    target_resolution: int = 0,
    derive_mode: str = "asis",
    edge_low: int = 100,
    edge_high: int = 200,
) -> torch.Tensor:
    """Encode a control image into condition latents shaped ``[1, 16, h, w]``.

    ``derive_mode``:
      * ``"asis"`` (default): treat the file as an already-prepared control image
        (line-art / grayscale) and encode it directly.
      * ``"lineart"`` / ``"grayscale"``: derive the control image from the input
        first via the SAME :func:`colorize_preprocess._make_control_image` the
        dataset producer uses, then encode it.

    Encoding reuses ``anima_cache_builder._encode_latents_chunked`` (the training
    path), so the produced latents match the cached cond sidecars bit-for-bit.
    """
    try:
        from .anima_cache_builder import _encode_latents_chunked
    except ImportError:  # pragma: no cover - direct-file fallback
        from core.lulynx_trainer.anima_cache_builder import _encode_latents_chunked

    path = Path(control_image_path)
    mode = str(derive_mode or "asis").strip().lower()
    tmp_path: Optional[Path] = None
    encode_path = path
    if mode in ("lineart", "grayscale"):
        from PIL import Image

        try:
            from core.tools.colorize_preprocess import _make_control_image
        except ImportError:  # pragma: no cover - path fallback
            from backend.core.tools.colorize_preprocess import _make_control_image  # type: ignore

        image = Image.open(path).convert("RGB")
        control = _make_control_image(image, mode, low=int(edge_low), high=int(edge_high))
        fd, tmp_name = tempfile.mkstemp(suffix=".png", prefix="lulynx_cond_")
        os.close(fd)
        tmp_path = Path(tmp_name)
        control.save(str(tmp_path))
        encode_path = tmp_path

    try:
        latent = _encode_latents_chunked(
            vae_encode_fn=vae_encode_fn,
            image_path=encode_path,
            chunk_size=0,
            target_resolution=int(target_resolution or 0),
        )  # [16, h, w]
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return latent.unsqueeze(0).contiguous()  # [1, 16, h, w]


# ---------------------------------------------------------------------------
# Condition context: install patch + reset condition before every forward
# ---------------------------------------------------------------------------

@contextmanager
def colorize_condition_context(
    dit_model: Any,
    adapter: Any,
    cond_latents: Optional[torch.Tensor],
    *,
    condition_warmup_forwards: int = 0,
):
    """Drive ``sample_anima`` with an EasyControl-v2 control condition.

    When ``adapter`` or ``cond_latents`` is ``None`` this is a pure no-op (the
    render stays bitwise-identical to plain text-to-image). Otherwise it installs
    the two-stream patch and registers a DiT ``forward_pre_hook`` that resets the
    adapter's condition tokens to the original control tokens before *every*
    forward — the patched blocks evolve and republish the condition within a
    single forward, so without this reset successive denoise steps / CFG branches
    would inherit a stale evolved condition.

    ``condition_warmup_forwards`` skips the condition for the first N DiT forwards
    (setting ``_cond_tokens=None`` → the patched blocks run the bitwise-original
    forward). EasyControl-v2's two-stream condition is timestep-modulated here
    (unlike the timestep-independent KV-cache in the paper), so injecting it in
    the high-noise early steps destabilises the flow-matching trajectory; warming
    up lets the latent form via plain text-to-image first, then engages the
    condition for the low-noise refinement steps. ``0`` (default) keeps the
    inject-every-forward behaviour.

    Yields the patch handle (or ``None`` for the no-op path).
    """
    if adapter is None or cond_latents is None:
        yield None
        return

    _, _, install_patch = _adapter_imports()

    # Align the condition to the adapter's device/dtype, then pre-encode once
    # (the control tokens depend only on the image, never on the latent/timestep).
    param = next(adapter.parameters())
    cond = cond_latents.to(device=param.device, dtype=param.dtype)
    base_tokens = adapter.encode_cond_latents(cond).detach()

    handle = install_patch(dit_model, adapter)
    warmup = max(0, int(condition_warmup_forwards))
    state = {"calls": 0}

    def _reset_condition(_module, args):
        state["calls"] += 1
        if state["calls"] <= warmup:
            # Warm-up: skip the condition so the early (high-noise) steps run as
            # plain text-to-image and the latent can form before the condition
            # engages for the low-noise refinement steps.
            adapter._cond_tokens = None
            return None
        tokens = base_tokens
        try:
            batch = int(args[0].shape[0]) if args else 1
        except (AttributeError, IndexError, TypeError):
            batch = 1
        if tokens.shape[0] == 1 and batch > 1:
            tokens = tokens.expand(batch, -1, -1)
        adapter._cond_tokens = tokens
        return None

    hook = dit_model.register_forward_pre_hook(_reset_condition)
    try:
        yield handle
    finally:
        hook.remove()
        handle.remove()
        adapter.clear_cond()


# ---------------------------------------------------------------------------
# Condition warm-up resolution (skip the high-noise early steps; #214 default)
# ---------------------------------------------------------------------------

# The colorize warm-up sweep (#214) found that skipping the condition for the
# high-noise early denoise steps and engaging it only for the low-noise
# refinement steps removes the early-trajectory damage while keeping a visible
# condition effect. ratio~0.6 (e.g. 12/20 steps) was the quality knee — smaller
# warms up dirtier, larger trends toward plain t2i. This is the colorize default.
_DEFAULT_CONDITION_WARMUP_RATIO = 0.6


def _resolve_condition_warmup_forwards(
    num_inference_steps: int,
    guidance_scale: float,
    warmup_ratio: float,
) -> int:
    """Map ``(steps, guidance, ratio)`` -> number of DiT forwards to run as plain t2i.

    ``warmup_ratio`` of the denoise steps run with no condition (the high-noise
    steps); the rest engage it. Each step is 2 forwards under CFG (guidance>1:
    pos+neg branches) else 1, so the count scales with the branch factor.
    ``steps<=0`` or ``ratio<=0`` -> ``0`` (inject every forward — the legacy
    behaviour, keeping non-warm-up callers byte-identical).
    """
    steps = max(0, int(num_inference_steps or 0))
    ratio = float(warmup_ratio or 0.0)
    if steps <= 0 or ratio <= 0.0:
        return 0
    branches = 2 if float(guidance_scale or 0.0) > 1.0 else 1
    warmup_steps = min(steps, int(round(steps * ratio)))
    return warmup_steps * branches


# ---------------------------------------------------------------------------
# High-level render scope (load adapter + encode control + condition the DiT)
# ---------------------------------------------------------------------------

@contextmanager
def colorize_render_scope(
    model: Any,
    *,
    arch: str,
    adapter_path: str | Path,
    control_image_path: str | Path,
    colorize_mode: str = "asis",
    easycontrol_mode: str = "auto",
    num_inference_steps: int = 0,
    guidance_scale: float = 1.0,
    condition_warmup_ratio: float = _DEFAULT_CONDITION_WARMUP_RATIO,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
):
    """One-call inference scope used by the generation CLI.

    A pure no-op (``yield None``) unless ALL hold: native Anima arch, a control
    image path, a usable adapter checkpoint, and ``easycontrol_mode != "off"``.
    When the checkpoint carries no ``easycontrol_v2.*`` keys it degrades to plain
    text-to-image (and only warns when the operator forced ``easycontrol_mode=on``).
    Otherwise it loads the trained adapter, encodes the control image into the
    training-identical condition latents, and conditions every DiT forward.

    Keeping the load + encode + install behind this single entry lets the CLI add
    the colorize path with a minimal, parity-preserving diff.

    The condition is warmed up by default (``condition_warmup_ratio`` of the
    denoise steps run as plain text-to-image first, then the condition engages for
    the low-noise refinement steps) — the #214 sweep found this removes the
    early-step trajectory damage. ``num_inference_steps`` / ``guidance_scale`` come
    from the sampler call so the warm-up forward count tracks the real schedule.
    """
    mode = str(easycontrol_mode or "auto").strip().lower()
    unet = getattr(model, "unet", None)
    if (
        str(arch).lower() == "newbie"
        or not control_image_path
        or not adapter_path
        or mode == "off"
        or unet is None
    ):
        yield None
        return

    adapter = load_easycontrol_v2_adapter_for_inference(adapter_path, device=device, dtype=dtype)
    if adapter is None:
        if mode == "on":
            logger.warning(
                "easycontrol_v2=on but %s carries no easycontrol_v2.* keys; rendering plain t2i.",
                Path(adapter_path).name,
            )
        yield None
        return

    vae_encode_fn = build_cond_vae_encode_fn(model, device=device, dtype=dtype)
    cond_latents = encode_control_image_to_cond_latents(
        vae_encode_fn, control_image_path, derive_mode=colorize_mode
    )
    warmup_forwards = _resolve_condition_warmup_forwards(
        num_inference_steps, guidance_scale, condition_warmup_ratio
    )
    logger.info(
        "native-anima colorize: condition latents %s from %s (mode=%s, warmup_forwards=%d)",
        tuple(cond_latents.shape), Path(control_image_path).name, colorize_mode, warmup_forwards,
    )
    with colorize_condition_context(
        unet, adapter, cond_latents, condition_warmup_forwards=warmup_forwards
    ) as handle:
        yield handle


__all__ = [
    "ADAPTER_PREFIX",
    "load_easycontrol_v2_adapter_for_inference",
    "build_cond_vae_encode_fn",
    "encode_control_image_to_cond_latents",
    "colorize_condition_context",
    "colorize_render_scope",
]
