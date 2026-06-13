"""Native-Anima inference assembler (clean-room Lulynx).

Assembles the four real components validated during the #132 400-step training
run — the 28-block native DiT executable subset, the qwen-image VAE, and the
Qwen3 text encoder + tokenizer — into one lightweight bundle, *without* going
through ``anima_loader.load()`` (whose ``_load_primary`` always builds the
scaffold DiT, which is inference-dead).

Pure re-use: every weight loader here is an existing, training-validated entry
point —

* DiT  → :func:`anima_native_dit.load_anima_native_executable_subset`
         (the same call ``trainer.py`` drives at ``anima_native_block_count``).
* VAE  → :meth:`AnimaLoader._load_qwen_image_vae_from_single_file`.
* Qwen3 → :meth:`AnimaLoader._load_qwen3_from_single_file`
          + :meth:`AnimaLoader._resolve_qwen3_tokenizer_dir`.

It also exposes :func:`reload_vae_and_qwen3` (VAE + Qwen3 only, no DiT) so the
training-preview path can re-materialise the components cache-first training
released while keeping the live LoRA-injected DiT, and :func:`encode_qwen3_prompt`
shaped for ``sample_anima``'s additive ``prompt_embeds`` seam (Block B).

PolyForm Noncommercial. Shares no source with any reference repository.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

DEFAULT_QWEN3_MAX_TOKENS = 512
# llm_adapter target tokens (T5 SentencePiece ids); pad to a fixed slot count.
DEFAULT_T5_MAX_TOKENS = 512
T5_TOKENIZER_DIRNAME = "tokenizer_t5"


@dataclass
class NativeAnimaInferenceBundle:
    """Lightweight holder for the native-Anima inference stack.

    Exposes the same attribute surface that ``sample_anima`` / ``LoRAInjector``
    / ``anima_inference_cli`` already read on a loaded model (``unet``, ``vae``),
    plus the Qwen3 encoder/tokenizer for native text-encode. The CLIP slots stay
    ``None`` — native-Anima has no CLIP text encoder — which is exactly what the
    additive ``prompt_embeds`` path in ``sample_anima`` keys off.
    """

    unet: Any
    vae: Any
    qwen3_encoder: Any
    qwen3_tokenizer: Any
    device: str = "cuda"
    dtype: Any = None
    qwen3_max_tokens: int = DEFAULT_QWEN3_MAX_TOKENS
    # Faithful native forward: the llm_adapter bridges Qwen3 hidden states +
    # T5 token ids into the DiT cross-attention conditioning. Both stay None on
    # the stub (non-faithful) path so the legacy prompt_embeds route is taken.
    llm_adapter: Any = None
    t5_tokenizer: Any = None
    t5_max_tokens: int = DEFAULT_T5_MAX_TOKENS
    # Native-Anima carries no CLIP/secondary text encoder; keep the slots so
    # callers that introspect a model uniformly (sampler/cli) do not KeyError,
    # but leave them empty so the native prompt-embeds path is taken.
    text_encoder_1: Any = None
    text_encoder_2: Any = None
    tokenizer_1: Any = None
    noise_scheduler: Any = None
    load_report: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_native_qwen3(self) -> bool:
        """True when a usable Qwen3 encoder + tokenizer are present."""
        return self.qwen3_encoder is not None and self.qwen3_tokenizer is not None

    @property
    def is_faithful(self) -> bool:
        """True when the faithful pieces (llm_adapter + T5 tokenizer) are loaded."""
        return self.llm_adapter is not None and self.t5_tokenizer is not None


# ---------------------------------------------------------------------------
# Component loaders (reuse AnimaLoader's single-file readers; no scaffold DiT)
# ---------------------------------------------------------------------------

def _make_loader(device: str, dtype: Any, disable_mmap: bool):
    from .anima_loader import AnimaLoader

    return AnimaLoader(device=device, dtype=dtype, disable_mmap=disable_mmap)


def _first_safetensors(directory: Path) -> str:
    """First ``*.safetensors`` under ``directory`` (sorted), or ``""``."""
    if not directory.is_dir():
        return ""
    files = sorted(directory.glob("*.safetensors"))
    return str(files[0]) if files else ""


def resolve_native_anima_paths(
    model_path: str, model_extra: Optional[Mapping[str, Any]] = None
) -> Tuple[str, str, str]:
    """Resolve ``(dit, vae, qwen3)`` single-file paths for native-Anima inference.

    Explicit ``model_extra`` overrides win; otherwise resolve by the standard
    ``models/anima/{diffusion_models,vae,text_encoders}`` layout. ``model_path``
    may be the DiT ``.safetensors`` itself or the anima root directory. Lives here
    (next to :func:`load_native_anima_for_inference`, the loader it feeds) rather
    than in the CLI so the resolver + loader stay a cohesive pair.
    """
    extra = dict(model_extra or {})
    dit = str(extra.get("anima_native_dit_path") or extra.get("anima_dit_path") or "").strip()
    vae = str(extra.get("anima_vae_path") or "").strip()
    qwen3 = str(extra.get("anima_qwen3_path") or extra.get("anima_text_encoder_path") or "").strip()

    p = Path(model_path)
    if p.is_file():
        dit = dit or str(p)
        root = p.parent.parent if p.parent.name in {"diffusion_models", "unet", "transformer"} else p.parent
    else:
        root = p
    if not dit:
        dit = _first_safetensors(root / "diffusion_models") or _first_safetensors(root)
    if not vae:
        vae = _first_safetensors(root / "vae")
    if not qwen3:
        qwen3 = _first_safetensors(root / "text_encoders")
    return dit, vae, qwen3


def _load_qwen_image_vae(loader, vae_path: str | Path, device: str):
    from diffusers import AutoencoderKLQwenImage

    vae = loader._load_qwen_image_vae_from_single_file(AutoencoderKLQwenImage, Path(vae_path))
    vae.to(device=device)
    return vae


def _load_qwen3(loader, qwen3_path: str | Path, device: str, dtype: Any) -> Tuple[Any, Any]:
    from transformers import AutoTokenizer

    encoder = loader._load_qwen3_from_single_file(Path(qwen3_path))
    encoder.to(device=device, dtype=dtype)
    encoder.requires_grad_(False)
    encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(loader._resolve_qwen3_tokenizer_dir(Path(qwen3_path))))
    return encoder, tokenizer


def _resolve_t5_tokenizer_dir(dit_path: str | Path, explicit: Optional[str | Path]) -> Path:
    """Locate the bundled T5 SentencePiece tokenizer dir for the llm_adapter.

    Tries, in order: an explicit path, ``<dit_dir>/tokenizer_t5``, then a
    ``models/anima/tokenizer_t5`` sibling discovered by walking up from the DiT.
    Only the *target* token ids are produced here — the T5 model is never run.
    """
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    dit_dir = Path(dit_path).resolve().parent
    candidates.append(dit_dir / T5_TOKENIZER_DIRNAME)
    for parent in [dit_dir, *dit_dir.parents]:
        if parent.name == "anima":
            candidates.append(parent / T5_TOKENIZER_DIRNAME)
        candidates.append(parent / "models" / "anima" / T5_TOKENIZER_DIRNAME)
    for cand in candidates:
        if (cand / "spiece.model").exists() or (cand / "tokenizer.json").exists():
            return cand
    raise FileNotFoundError(
        "faithful native forward needs the T5 tokenizer (spiece.model/tokenizer.json); "
        f"looked under {dit_dir / T5_TOKENIZER_DIRNAME} and models/anima/{T5_TOKENIZER_DIRNAME}. "
        "Copy ref configs/t5_old/* there (google/t5-v1_1-xxl, Apache-2.0)."
    )


def _load_t5_tokenizer(tokenizer_dir: Path):
    from transformers import T5TokenizerFast

    return T5TokenizerFast.from_pretrained(str(tokenizer_dir))


def _dit_has_llm_adapter(dit_path: str | Path, *, disable_mmap: bool = False) -> bool:
    """Cheap header peek: does the DiT checkpoint carry the llm_adapter sub-net?

    The faithful forward needs ``net.llm_adapter.embed.weight`` — a stub/preview
    checkpoint omits it (and :func:`load_anima_llm_adapter` would then raise).
    Reads only the safetensors key list, never tensor bytes.
    """
    try:
        from core.lulynx_trainer.safetensors_loader import open_safetensors
    except ImportError:  # pragma: no cover - direct-file usage
        from .safetensors_loader import open_safetensors

    try:
        with open_safetensors(str(Path(dit_path)), framework="pt", device="cpu", disable_mmap=disable_mmap) as handle:
            return "net.llm_adapter.embed.weight" in handle.keys()
    except Exception:  # unreadable / missing file -> not faithful-capable
        return False


def faithful_inference_assets_available(
    dit_path: str | Path,
    t5_tokenizer_dir: Optional[str | Path] = None,
    *,
    disable_mmap: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Can the faithful native forward actually load for this checkpoint?

    Returns ``(available, reason)`` — ``reason`` is ``None`` when available, else
    a short tag for the degrade log. The dominant blocker is a stub/preview
    checkpoint with no ``net.llm_adapter.*`` weights. The T5 *tokenizer* (a ~1 MB
    SentencePiece file — never the T5 *model*, which Anima never runs) is the
    secondary check and resolves from ``models/anima/tokenizer_t5`` on a normal
    install, so it rarely blocks. Pure predicate: no model materialisation.
    """
    if not _dit_has_llm_adapter(dit_path, disable_mmap=disable_mmap):
        return False, "checkpoint_no_llm_adapter"
    try:
        _resolve_t5_tokenizer_dir(dit_path, t5_tokenizer_dir)
    except FileNotFoundError:
        return False, "no_t5_tokenizer"
    return True, None


def _resolve_faithful_mode(
    faithful: bool | str,
    dit_path: str | Path,
    t5_tokenizer_dir: Optional[str | Path],
    disable_mmap: bool,
) -> bool:
    """Map the tri-state ``faithful`` flag to an effective bool.

    ``True``/``False`` are explicit — ``True`` keeps the hard contract (the
    faithful loaders raise on missing assets, giving an actionable error).
    ``"auto"`` enables faithful only when
    :func:`faithful_inference_assets_available`; otherwise it degrades to the
    stub forward with a loud — non-fatal — warning, so *normal rendering stays
    the floor* even on a stub checkpoint or a box missing the T5 tokenizer.
    """
    if isinstance(faithful, str):
        mode = faithful.strip().lower()
        if mode in ("on", "true", "1", "yes"):
            return True
        if mode in ("off", "false", "0", "no", "none", ""):
            return False
        if mode != "auto":  # unknown -> safest is auto (degrade, never crash)
            logger.warning("native-anima faithful=%r unrecognised; treating as 'auto'", faithful)
        available, reason = faithful_inference_assets_available(
            dit_path, t5_tokenizer_dir, disable_mmap=disable_mmap
        )
        if not available:
            logger.warning(
                "native-anima faithful=auto -> degrading to STUB forward (%s). "
                "Renders will be low-fidelity but will NOT crash. Fix: use a "
                "checkpoint carrying net.llm_adapter.* and keep a T5 tokenizer "
                "under models/anima/tokenizer_t5.",
                reason,
            )
        return available
    return bool(faithful)


def reload_vae_and_qwen3(
    vae_path: str | Path,
    qwen3_path: str | Path,
    *,
    device: str = "cuda",
    dtype: Optional[Any] = None,
    disable_mmap: bool = False,
) -> Tuple[Any, Any, Any]:
    """Re-materialise ``(vae, qwen3_encoder, qwen3_tokenizer)`` from single files.

    Used by the training-preview path: cache-first training releases the VAE and
    text encoders after building the latent cache, but the live LoRA-injected DiT
    stays resident — so a preview only needs these two back, not the DiT.
    """
    dtype = dtype or torch.bfloat16
    loader = _make_loader(device, dtype, disable_mmap)
    vae = _load_qwen_image_vae(loader, vae_path, device)
    encoder, tokenizer = _load_qwen3(loader, qwen3_path, device, dtype)
    return vae, encoder, tokenizer


def load_native_anima_for_inference(
    dit_path: str | Path,
    vae_path: str | Path,
    qwen3_path: str | Path,
    *,
    block_count: int = 28,
    device: str = "cuda",
    dtype: Optional[Any] = None,
    disable_mmap: bool = False,
    faithful: bool | str = False,
    t5_tokenizer_dir: Optional[str | Path] = None,
) -> NativeAnimaInferenceBundle:
    """Assemble a native-Anima inference bundle from three single-file weights.

    Args:
        dit_path:   native Anima DiT ``.safetensors`` (e.g. ``anima-base-v1.0``).
        vae_path:   qwen-image VAE ``.safetensors``.
        qwen3_path: Qwen3 text-encoder ``.safetensors``.
        block_count: number of DiT blocks to materialise (default 28, matching
            ``trainer.py``'s ``anima_native_block_count``).
        device / dtype: target placement (dtype defaults to bf16).
        disable_mmap: forward to the safetensors readers (Windows mmap quirks).
        faithful: ``True``/``False`` force the faithful / stub forward. The
            faithful forward enables 3D RoPE on the DiT self-attention plus the
            ``net.llm_adapter.*`` sub-network and its T5 *target* tokenizer (the
            T5 model is never run) so the base model renders structurally. Pass
            ``"auto"`` to enable faithful when the checkpoint carries the adapter
            and a T5 tokenizer resolves, else degrade to the stub forward with a
            loud warning (never crash). Default ``False`` keeps the stub forward
            (bitwise parity with the trainer).
        t5_tokenizer_dir: optional explicit dir for the T5 SentencePiece tokenizer
            (only consulted when ``faithful``); auto-discovered next to the DiT
            otherwise.

    Returns:
        A :class:`NativeAnimaInferenceBundle` ready for ``sample_anima`` /
        ``LoRAInjector`` / ``anima_inference_cli``.
    """
    dtype = dtype or torch.bfloat16
    block_indices = tuple(range(max(int(block_count), 1)))

    # Resolve the tri-state faithful flag once, up front, so the same effective
    # bool flows into both the DiT subset load and the adapter/T5 extras below.
    faithful = _resolve_faithful_mode(faithful, dit_path, t5_tokenizer_dir, disable_mmap)

    # 1) DiT — the SAME executable subset trainer.py drives. assign=True leaves
    #    params on the CPU tensors they were loaded from, so move to device.
    from .anima_native_dit import load_anima_native_executable_subset

    unet, dit_report = load_anima_native_executable_subset(
        dit_path,
        block_indices=block_indices,
        device=device,
        dtype=dtype,
        disable_mmap=disable_mmap,
        faithful=faithful,
    )
    unet.to(device=device, dtype=dtype)
    unet.eval()

    # 2) VAE + 3) Qwen3 — reuse AnimaLoader's single-file loaders directly,
    #    bypassing AnimaLoader.load() (which always builds the scaffold DiT).
    loader = _make_loader(device, dtype, disable_mmap)
    vae = _load_qwen_image_vae(loader, vae_path, device)
    qwen3_encoder, qwen3_tokenizer = _load_qwen3(loader, qwen3_path, device, dtype)

    # 4) Faithful extras — llm_adapter (118 keys already in the DiT checkpoint)
    #    + its T5 target tokenizer. Loaded only on the explicit faithful path so
    #    the legacy stub route stays untouched.
    llm_adapter = None
    t5_tokenizer = None
    adapter_report: Dict[str, Any] = {}
    if faithful:
        from .anima_native_faithful import load_anima_llm_adapter

        llm_adapter, adapter_report = load_anima_llm_adapter(
            dit_path, device=device, dtype=dtype, disable_mmap=disable_mmap
        )
        t5_tokenizer = _load_t5_tokenizer(_resolve_t5_tokenizer_dir(dit_path, t5_tokenizer_dir))

    bundle = NativeAnimaInferenceBundle(
        unet=unet,
        vae=vae,
        qwen3_encoder=qwen3_encoder,
        qwen3_tokenizer=qwen3_tokenizer,
        llm_adapter=llm_adapter,
        t5_tokenizer=t5_tokenizer,
        device=device,
        dtype=dtype,
        load_report={
            "dit": dit_report.to_dict() if hasattr(dit_report, "to_dict") else {},
            "block_count": len(block_indices),
            "vae_class": type(vae).__name__,
            "qwen3_class": type(qwen3_encoder).__name__,
            "faithful": bool(faithful),
            "llm_adapter": adapter_report,
        },
    )
    logger.info(
        "native-anima inference bundle ready: dit=%d blocks, vae=%s, qwen3=%s, faithful=%s",
        len(block_indices),
        type(vae).__name__,
        type(qwen3_encoder).__name__,
        bool(faithful),
    )
    return bundle


# ---------------------------------------------------------------------------
# Qwen3 prompt encode (shaped for sample_anima's prompt_embeds seam)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_qwen3(
    encoder,
    tokenizer,
    prompt: str,
    *,
    device: str,
    dtype: torch.dtype,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize + encode -> ``(hidden [1, seq, hidden], attn_mask [1, seq])``."""
    if encoder is None or tokenizer is None:
        raise RuntimeError("Qwen3 encoder/tokenizer unavailable for native text-encode")
    tokens = tokenizer(
        prompt or "",
        padding="max_length",
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attn = tokens["attention_mask"].to(device)

    encoder.to(device=device, dtype=dtype)
    try:
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
    except TypeError:
        outputs = encoder(input_ids=input_ids, attention_mask=attn)

    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            hidden = hidden_states[-1]
    if hidden is None:
        first = outputs[0]
        hidden_size = int(getattr(getattr(encoder, "config", None), "hidden_size", 0) or 0)
        if hidden_size > 0 and int(first.shape[-1]) == hidden_size:
            hidden = first
    if hidden is None:
        raise RuntimeError("Qwen3 encoder did not return hidden states for native inference")
    return hidden.to(dtype), attn


@torch.no_grad()
def encode_qwen3_hidden(
    encoder,
    tokenizer,
    prompt: str,
    *,
    device: str,
    dtype: torch.dtype,
    max_length: int = DEFAULT_QWEN3_MAX_TOKENS,
) -> torch.Tensor:
    """Encode ``prompt`` into Qwen3 hidden states shaped ``[1, seq, hidden]``.

    Mirrors ``anima_cache_runtime._encode_qwen3`` (padded to ``max_length``,
    ``output_hidden_states``) but keeps the batch dim so the result drops
    straight into ``sample_anima``'s ``prompt_embeds`` seam. An empty/whitespace
    prompt still encodes (pad tokens), so it doubles as the CFG negative.
    """
    hidden, _ = _encode_qwen3(
        encoder, tokenizer, prompt, device=device, dtype=dtype, max_length=max_length
    )
    return hidden


def encode_qwen3_prompt(
    bundle: NativeAnimaInferenceBundle,
    prompt: str,
    *,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """Bundle-flavoured wrapper over :func:`encode_qwen3_hidden`."""
    if not bundle.is_native_qwen3:
        raise RuntimeError("bundle has no Qwen3 encoder/tokenizer for native text-encode")
    return encode_qwen3_hidden(
        bundle.qwen3_encoder,
        bundle.qwen3_tokenizer,
        prompt,
        device=bundle.device,
        dtype=bundle.dtype,
        max_length=int(max_length or bundle.qwen3_max_tokens),
    )


@torch.no_grad()
def encode_native_condition(
    bundle: NativeAnimaInferenceBundle,
    prompt: str,
    *,
    qwen3_max_length: Optional[int] = None,
    t5_max_length: Optional[int] = None,
) -> torch.Tensor:
    """Faithful native conditioning: Qwen3 hidden + T5 ids -> ``crossattn_emb``.

    Runs the llm_adapter once over the prompt (it depends only on the text, never
    on the latent/timestep), producing the cross-attention conditioning the DiT
    expects ``[1, t5_seq, model_dim]``. Drops into ``sample_anima``'s additive
    ``prompt_embeds`` seam exactly like :func:`encode_qwen3_prompt`, so the
    sampler needs no change. An empty prompt encodes to the CFG negative.
    """
    if not bundle.is_faithful:
        raise RuntimeError(
            "bundle is not faithful (load_native_anima_for_inference(faithful=True) "
            "needed for llm_adapter + T5 tokenizer)"
        )
    qwen3_hidden, qwen3_mask = _encode_qwen3(
        bundle.qwen3_encoder,
        bundle.qwen3_tokenizer,
        prompt,
        device=bundle.device,
        dtype=bundle.dtype,
        max_length=int(qwen3_max_length or bundle.qwen3_max_tokens),
    )
    t5_tokens = bundle.t5_tokenizer(
        prompt or "",
        padding="max_length",
        truncation=True,
        max_length=int(t5_max_length or bundle.t5_max_tokens),
        return_tensors="pt",
    )
    t5_ids = t5_tokens["input_ids"].to(bundle.device)
    t5_mask = t5_tokens["attention_mask"].to(bundle.device)
    crossattn = bundle.llm_adapter(
        qwen3_hidden,
        t5_ids,
        target_attention_mask=t5_mask,
        source_attention_mask=qwen3_mask,
    )
    return crossattn.to(bundle.dtype)
