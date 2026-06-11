"""
Offline SD1.5 single-file checkpoint loader.

Loads a standard SD1.5 .safetensors / .ckpt / .pt checkpoint into
diffusers-format components without requiring HuggingFace Hub access.

The CLIP text config is derived entirely from checkpoint tensor shapes.
UNet / VAE tensor remapping is delegated to diffusers'
``download_from_original_stable_diffusion_ckpt`` (MIT-licensed) with all
pre-built config objects injected so the function never contacts the Hub.

Integration boundary:
    ``SD15SingleFileLoader.load()`` returns ``SingleFileSD15Components``,
    which mirrors the shared ``LoadedModel`` contract (minus ``model_arch``).
    Call ``SingleFileSD15Components.to_loaded_model("sd15")`` to obtain a
    fully-populated ``LoadedModel`` instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from diffusers import DDPMScheduler
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)

# ── defaults ───────────────────────────────────────────────────────────
DEFAULT_CONTEXT_LENGTH = 77
DEFAULT_BETA_START = 0.00085
DEFAULT_BETA_END = 0.012
DEFAULT_TIMESTEPS = 1000


# ── data container ─────────────────────────────────────────────────────
@dataclass
class SingleFileSD15Components:
    """Loaded SD1.5 components matching the shared ``LoadedModel`` field layout.

    ``text_encoder_2`` and ``tokenizer_2`` are always ``None`` for SD1.5
    (single CLIP-L text encoder).  Use ``to_loaded_model()`` to obtain a
    ``LoadedModel`` with the ``model_arch`` field populated.
    """
    unet: Any
    text_encoder_1: Any
    text_encoder_2: Optional[Any] = None
    vae: Any = None
    tokenizer_1: Any = None
    tokenizer_2: Optional[Any] = None
    noise_scheduler: Any = None

    def to_loaded_model(self, model_arch: str = "sd15"):
        """Convert to a ``LoadedModel`` instance (from ``model_loader``).

        This is the integration boundary -- callers that work with the
        shared ``LoadedModel`` type should use this method rather than
        accessing fields directly.
        """
        from .model_loader import LoadedModel
        return LoadedModel(
            unet=self.unet,
            text_encoder_1=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            vae=self.vae,
            tokenizer_1=self.tokenizer_1,
            tokenizer_2=self.tokenizer_2,
            noise_scheduler=self.noise_scheduler,
            model_arch=model_arch,
        )


# ── checkpoint probing helpers ─────────────────────────────────────────

def _detect_clip_prefix(checkpoint: Dict[str, torch.Tensor]) -> str:
    """Detect the CLIP text encoder key prefix in an SD1.5 checkpoint.

    Standard LDM checkpoints use ``cond_stage_model.transformer.text_model.*``,
    but fine-tuned or re-exported checkpoints may use shortened prefixes.
    Returns the detected prefix (with trailing dot).
    """
    candidates = [
        "cond_stage_model.transformer.text_model.",
        "cond_stage_model.transformer.",
        "cond_stage_model.",
    ]
    for prefix in candidates:
        if any(k.startswith(prefix) for k in checkpoint):
            return prefix
    raise KeyError(
        "Could not detect CLIP text encoder prefix in checkpoint. "
        "Expected keys starting with 'cond_stage_model.*'. "
        "This may not be a valid SD1.5 checkpoint."
    )


def _detect_unet_prefix(checkpoint: Dict[str, torch.Tensor]) -> str:
    """Detect the UNet key prefix in an SD1.5 checkpoint.

    Returns the detected prefix (with trailing dot).
    """
    candidates = [
        "model.diffusion_model.",
        "unet.",
        "model.",
    ]
    for prefix in candidates:
        if any(k.startswith(prefix) for k in checkpoint):
            return prefix
    raise KeyError(
        "Could not detect UNet prefix in checkpoint. "
        "Expected keys starting with 'model.diffusion_model.*' or 'unet.*'. "
        "This may not be a valid SD1.5 checkpoint."
    )


def _detect_vae_prefix(checkpoint: Dict[str, torch.Tensor]) -> Optional[str]:
    """Detect the VAE key prefix in an SD1.5 checkpoint.

    Returns the prefix (with trailing dot), or ``None`` if no VAE keys are
    found.  Many single-file SD1.5 checkpoints bundle the VAE; some strip it.
    """
    candidates = [
        "first_stage_model.",
        "vae.",
    ]
    for prefix in candidates:
        if any(k.startswith(prefix) for k in checkpoint):
            return prefix
    return None


def _max_block_index(checkpoint: Dict[str, torch.Tensor], prefix: str) -> int:
    """Infer the number of transformer blocks from the highest numeric key suffix.

    Scans all keys starting with *prefix* and returns (max_index + 1) as the
    block count.  Raises ``KeyError`` if no numeric suffixes are found.
    """
    max_index = -1
    for key in checkpoint:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        block_index_text = suffix.split(".", 1)[0]
        if block_index_text.isdigit():
            max_index = max(max_index, int(block_index_text))
    if max_index < 0:
        raise KeyError(
            f"Could not infer block count from checkpoint prefix: '{prefix}'. "
            "No keys with numeric suffixes found."
        )
    return max_index


def _count_keys_with_prefix(checkpoint: Dict[str, torch.Tensor], prefix: str) -> int:
    """Count checkpoint keys starting with *prefix*."""
    return sum(1 for k in checkpoint if k.startswith(prefix))


# ── config builders ────────────────────────────────────────────────────

def _build_clip_config_from_checkpoint(checkpoint: Dict[str, torch.Tensor]) -> CLIPTextConfig:
    """Build a ``CLIPTextConfig`` by reading weight shapes from the checkpoint.

    This avoids any HuggingFace Hub download -- the config is derived entirely
    from the tensors present in the checkpoint file.
    """
    prefix = _detect_clip_prefix(checkpoint)

    # Locate key tensors to infer config dimensions.
    # Try standard LDM layout first, fall back to shorter prefix variants.
    token_emb_key = prefix + "embeddings.token_embedding.weight"
    position_emb_key = prefix + "embeddings.position_embedding.weight"
    fc1_key = prefix + "encoder.layers.0.mlp.fc1.weight"
    layers_prefix = prefix + "encoder.layers."

    if token_emb_key not in checkpoint:
        # Some checkpoints omit the intermediate 'text_model' segment.
        # Try cond_stage_model.transformer.text_model.* explicitly
        alt_prefix = "cond_stage_model.transformer.text_model."
        alt_token = alt_prefix + "embeddings.token_embedding.weight"
        if alt_token in checkpoint:
            prefix = alt_prefix
            token_emb_key = prefix + "embeddings.token_embedding.weight"
            position_emb_key = prefix + "embeddings.position_embedding.weight"
            fc1_key = prefix + "encoder.layers.0.mlp.fc1.weight"
            layers_prefix = prefix + "encoder.layers."

    if token_emb_key not in checkpoint:
        raise KeyError(
            f"Cannot find CLIP token_embedding.weight under detected prefix. "
            f"Checkpoint may use an unexpected key layout."
        )

    token_embedding = checkpoint[token_emb_key]
    position_embedding = checkpoint[position_emb_key]
    fc1_weight = checkpoint[fc1_key]
    hidden_size = position_embedding.shape[1]
    num_hidden_layers = _max_block_index(checkpoint, layers_prefix) + 1

    return CLIPTextConfig(
        vocab_size=token_embedding.shape[0],
        hidden_size=hidden_size,
        intermediate_size=fc1_weight.shape[0],
        projection_dim=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=max(1, hidden_size // 64),
        max_position_embeddings=position_embedding.shape[0],
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        bos_token_id=49406,
        eos_token_id=49407,
        pad_token_id=0,
    )


# ── scheduler ──────────────────────────────────────────────────────────

def _build_training_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=DEFAULT_TIMESTEPS,
        beta_start=DEFAULT_BETA_START,
        beta_end=DEFAULT_BETA_END,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        clip_sample=False,
    )


# ── dtype casting ──────────────────────────────────────────────────────

def _cast_components(components: Dict[str, Any], torch_dtype: torch.dtype) -> None:
    for name in ("unet", "text_encoder_1", "vae"):
        module = components.get(name)
        if module is not None and hasattr(module, "to"):
            module.to(dtype=torch_dtype)


# ── integrity validation ───────────────────────────────────────────────

def _validate_checkpoint_sanity(checkpoint: Dict[str, torch.Tensor]) -> None:
    """Basic integrity checks before attempting conversion."""
    if not checkpoint:
        raise ValueError("Checkpoint is empty.")

    unet_prefix = _detect_unet_prefix(checkpoint)
    clip_prefix = _detect_clip_prefix(checkpoint)

    unet_keys = [k for k in checkpoint if k.startswith(unet_prefix)]
    clip_keys = [k for k in checkpoint if k.startswith(clip_prefix)]

    if len(unet_keys) < 10:
        raise ValueError(
            f"Checkpoint has too few UNet keys ({len(unet_keys)}) under '{unet_prefix}'. "
            "This may not be a valid SD1.5 checkpoint."
        )
    if len(clip_keys) < 5:
        raise ValueError(
            f"Checkpoint has too few CLIP keys ({len(clip_keys)}) under '{clip_prefix}'. "
            "This may not be a valid SD1.5 checkpoint."
        )

    logger.info(
        "Checkpoint sanity OK: %d UNet keys, %d CLIP keys.",
        len(unet_keys), len(clip_keys),
    )


# ── key format detection ──────────────────────────────────────────────

def _is_ldm_format(checkpoint: Dict[str, torch.Tensor]) -> bool:
    """Return True if the checkpoint uses original LDM key prefixes."""
    return any(k.startswith("model.diffusion_model.") for k in checkpoint)


# ── main entry points ──────────────────────────────────────────────────

def load_sd15_single_file_components(
    checkpoint_path: str | Path,
    torch_dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    """Load an SD1.5 single-file checkpoint into diffusers-format components.

    This function is fully offline -- it never contacts HuggingFace Hub.
    The CLIP text config and tokenizer are derived from the checkpoint
    tensors and the built-in ``transformers`` vocabulary.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.safetensors``, ``.ckpt``, or ``.pt`` file.
    torch_dtype:
        Target dtype for model weights.

    Returns
    -------
    dict
        Keys: ``unet``, ``text_encoder_1``, ``text_encoder_2`` (None),
        ``vae``, ``tokenizer_1``, ``tokenizer_2`` (None), ``noise_scheduler``.
    """
    from diffusers.models.modeling_utils import load_state_dict

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix.lower() not in {".safetensors", ".ckpt", ".pt"}:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path.suffix}")

    logger.info("Loading SD1.5 single-file checkpoint (offline): %s", checkpoint_path)

    # ── 1. load raw state dict ─────────────────────────────────────────
    checkpoint = load_state_dict(str(checkpoint_path))
    # Unwrap nested state_dict (some .ckpt files wrap weights).
    while isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    _validate_checkpoint_sanity(checkpoint)

    # ── 2. build components from checkpoint ────────────────────────────
    clip_config = _build_clip_config_from_checkpoint(checkpoint)
    text_encoder = CLIPTextModel(clip_config)

    # SD1.5 uses the standard CLIP tokenizer (49408-token vocab).
    # Try loading from local HF cache first; fall back to None if unavailable.
    # When tokenizer is None, download_from_original_stable_diffusion_ckpt
    # will construct one internally (it only needs the vocab file which ships
    # with transformers).
    tokenizer = None
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            local_files_only=True,
        )
    except Exception:
        logger.warning(
            "CLIPTokenizer not found in local HF cache. "
            "The conversion function will build one from the checkpoint."
        )

    # ── 3. delegate to diffusers conversion ────────────────────────────
    # ``download_from_original_stable_diffusion_ckpt`` handles the actual
    # tensor remapping from LDM / CompVis key layout to diffusers UNet/VAE
    # layouts.  We pass our pre-built text encoder so the function never
    # needs to fetch config files from the Hub.
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        download_from_original_stable_diffusion_ckpt,
    )
    from diffusers import StableDiffusionPipeline

    logger.info("Converting checkpoint via diffusers conversion pipeline...")
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=str(checkpoint_path),
        pipeline_class=StableDiffusionPipeline,
        from_safetensors=checkpoint_path.suffix.lower() == ".safetensors",
        device="cpu",
        local_files_only=True,
        load_safety_checker=False,
        text_encoder=text_encoder,
        # tokenizer= is intentionally omitted when None; the conversion
        # function will use the text_encoder's config to determine tokenizer
        # behavior.  If we have a valid tokenizer object, pass it through.
        **({"tokenizer": tokenizer} if tokenizer is not None else {}),
    )

    # ── 4. cast and assemble result ────────────────────────────────────
    components: Dict[str, Any] = {
        "unet": pipe.unet,
        "text_encoder_1": pipe.text_encoder,
        "text_encoder_2": None,
        "vae": pipe.vae,
        "tokenizer_1": tokenizer if tokenizer is not None else pipe.tokenizer,
        "tokenizer_2": None,
        "noise_scheduler": _build_training_scheduler(),
    }

    _cast_components(components, torch_dtype)

    logger.info("SD1.5 single-file load complete: UNet %s, VAE %s, TE %s",
                type(components["unet"]).__name__,
                type(components["vae"]).__name__,
                type(components["text_encoder_1"]).__name__)

    return components


# ── convenience wrapper (mirrors SDXLSingleFileLoader interface) ───────

class SD15SingleFileLoader:
    """Drop-in SD1.5 single-file loader matching the ``SDXLSingleFileLoader`` interface.

    Usage::

        loader = SD15SingleFileLoader(dtype=torch.float16)
        components = loader.load("path/to/sd15.safetensors")
        # components.unet, components.text_encoder_1, etc.
    """

    def __init__(self, dtype: torch.dtype = torch.float16):
        self.dtype = dtype

    def load(self, checkpoint_path: str | Path) -> SingleFileSD15Components:
        raw = load_sd15_single_file_components(checkpoint_path, torch_dtype=self.dtype)
        return SingleFileSD15Components(
            unet=raw["unet"],
            text_encoder_1=raw["text_encoder_1"],
            text_encoder_2=None,
            vae=raw["vae"],
            tokenizer_1=raw["tokenizer_1"],
            tokenizer_2=None,
            noise_scheduler=raw["noise_scheduler"],
        )
