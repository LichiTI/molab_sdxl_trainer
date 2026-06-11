"""
Lumina Image 2.0 model loader.

Provides a Lumina-specific loading path that follows the shared
``LoadedModel`` contract (from model_loader.py) but adapted for Lumina's
architecture:
  - Single text encoder (Gemma 2B), not dual CLIP.
  - 16-channel latent (Flux-VAE compatible).
  - Flow-matching scheduler (continuous-time, not DDPM/DDIM).
  - Transformer-based denoiser (not UNet).

Integration status:
  - The loading helpers raise ``NotImplementedError`` when diffusers
    does not yet expose a Lumina-compatible transformer class.
  - ``load_lumina_model()`` catches these and returns a report with
    ``SKELETON_MODE`` so callers know exactly what failed.
  - Once diffusers ships ``LuminaTransformer2DModel`` (or a custom
    pipeline class is registered), replace the ``_NotImplemented``
    stubs in each helper with real loading code.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from .lumina_contract import (
    LuminaLimitation,
    LuminaLoadReport,
)

logger = logging.getLogger(__name__)


# ============================================================================
# LuminaModelSpec — loader-internal return type
# ============================================================================

@dataclass
class LuminaModelSpec:
    """Components loaded from a Lumina checkpoint.

    This is the loader's internal return type.  Before handing off to the
    training pipeline, call ``to_loaded_model()`` to convert into the
    shared ``LoadedModel`` contract.
    """

    denoiser: Any        # LuminaTransformer2DModel (mapped to LoadedModel.unet)
    text_encoder: Any    # Gemma 2B
    tokenizer: Any       # SentencePiece tokenizer for Gemma
    vae: Any             # Flux-VAE compatible, 16-channel latent
    noise_scheduler: Any # Flow-matching scheduler
    model_arch: str = "lumina"
    report: Optional[LuminaLoadReport] = None

    def to_loaded_model(self):
        """Convert to the shared ``LoadedModel`` contract.

        The shared ``LoadedModel`` uses ``unet`` for the denoiser.
        For Lumina the denoiser is a transformer — it goes into the
        ``unet`` slot so the training loop can consume it without
        architecture-specific branching.
        """
        from .model_loader import LoadedModel

        return LoadedModel(
            unet=self.denoiser,
            text_encoder_1=self.text_encoder,
            text_encoder_2=None,  # Lumina has no second encoder
            vae=self.vae,
            tokenizer_1=self.tokenizer,
            tokenizer_2=None,     # Lumina has no second tokenizer
            noise_scheduler=self.noise_scheduler,
            model_arch=self.model_arch,
        )


# ============================================================================
# Public API
# ============================================================================

def load_lumina_model(
    model_path: str,
    dtype: Any = None,
    device: str = "cpu",
    vae_path: Optional[str] = None,
) -> Tuple[Optional[LuminaModelSpec], LuminaLoadReport]:
    """Load a Lumina Image 2.0 model from a diffusers-format directory.

    Returns:
        ``(spec, report)`` — ``spec`` is None when loading failed.
        ``report`` always contains honest status of every component.

    Raises:
        Nothing — all failures are captured in the report.
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16

    report = LuminaLoadReport(resolved_model_path=model_path)
    path = Path(model_path)

    if not path.exists() and not _is_hf_id(model_path):
        report.limitations.append(LuminaLimitation.TRANSFORMER_UNAVAILABLE)
        report.notes.append(f"Model path not found: {model_path}")
        logger.error("[LuminaLoader] Model path not found: %s", model_path)
        return None, report

    if _is_hf_id(model_path):
        report.limitations.append(LuminaLimitation.LOADED_FROM_HF_HUB)

    logger.info("[LuminaLoader] Loading from %s", model_path)

    # ── Load transformer (denoiser) ─────────────────────────────────────
    denoiser, transformer_notes = _load_transformer(path, dtype)
    if denoiser is None:
        report.limitations.append(LuminaLimitation.TRANSFORMER_UNAVAILABLE)
    else:
        report.transformer_loaded = True
    report.notes.extend(transformer_notes)

    # ── Load text encoder (Gemma 2B) + tokenizer ───────────────────────
    text_enc, tok, te_notes = _load_text_encoder(path, dtype)
    if text_enc is None:
        report.limitations.append(LuminaLimitation.TEXT_ENCODER_UNAVAILABLE)
    else:
        report.text_encoder_loaded = True
    if tok is None:
        report.limitations.append(LuminaLimitation.TOKENIZER_UNAVAILABLE)
    else:
        report.tokenizer_loaded = True
    report.notes.extend(te_notes)

    # ── Load VAE ───────────────────────────────────────────────────────
    vae, vae_notes = _load_vae(path, dtype, vae_path)
    if vae is None:
        report.limitations.append(LuminaLimitation.VAE_LOAD_FALLBACK)
        if vae_path:
            report.limitations.append(LuminaLimitation.SEPARATE_VAE_NOT_FOUND)
    else:
        report.vae_loaded = True
    report.notes.extend(vae_notes)

    # ── Load scheduler ─────────────────────────────────────────────────
    scheduler, sched_notes = _load_scheduler(path)
    if scheduler is None:
        report.limitations.append(LuminaLimitation.SCHEDULER_UNAVAILABLE)
    else:
        report.scheduler_loaded = True
    report.notes.extend(sched_notes)

    # ── Aggregate result ───────────────────────────────────────────────
    if not report.is_usable:
        report.limitations.append(LuminaLimitation.SKELETON_MODE)
        logger.warning(
            "[LuminaLoader] Skeleton mode — not all components loaded. "
            "Report: %s",
            report.summary(),
        )
        return None, report

    spec = LuminaModelSpec(
        denoiser=denoiser,
        text_encoder=text_enc,
        tokenizer=tok,
        vae=vae,
        noise_scheduler=scheduler,
        model_arch="lumina",
        report=report,
    )

    logger.info("[LuminaLoader] Loaded successfully. Report: %s", report.summary())
    return spec, report


def prepare_lumina_for_training(
    spec: LuminaModelSpec,
    gradient_checkpointing: bool = True,
    train_text_encoder: bool = False,
    device: str = "cuda",
) -> LuminaModelSpec:
    """Prepare a loaded Lumina model for LoRA training.

    Steps:
      1. Freeze VAE (always).
      2. Freeze text encoder (unless ``train_text_encoder`` is True).
      3. Enable gradient checkpointing on the denoiser.
      4. Set denoiser to training mode.
      5. Move denoiser (and optionally text encoder) to device.

    Returns:
        The same spec, mutated in-place.
    """
    # VAE is always frozen
    spec.vae.requires_grad_(False)
    spec.vae.eval()

    # Text encoder: freeze by default (Gemma 2B is ~2B params, expensive)
    if train_text_encoder:
        spec.text_encoder.requires_grad_(True)
        spec.text_encoder.train()
        spec.text_encoder.to(device)
    else:
        spec.text_encoder.requires_grad_(False)
        spec.text_encoder.eval()

    # Denoiser: gradient checkpointing + train mode
    if gradient_checkpointing and hasattr(spec.denoiser, "enable_gradient_checkpointing"):
        spec.denoiser.enable_gradient_checkpointing()
    spec.denoiser.train()
    spec.denoiser.to(device)

    # VAE to device
    spec.vae.to(device)

    logger.info(
        "[LuminaLoader] Prepared for training on %s "
        "(gradient_checkpointing=%s, train_text_encoder=%s)",
        device,
        gradient_checkpointing,
        train_text_encoder,
    )
    return spec


# ============================================================================
# Internal loading helpers
# ============================================================================

def _is_hf_id(path_str: str) -> bool:
    """Check if a string looks like a HuggingFace model ID."""
    return "/" in path_str and not Path(path_str).exists()


def _load_transformer(path: Path, dtype: Any) -> Tuple[Any, list]:
    """Load the Lumina transformer (denoiser).

    Returns ``(model, notes)``.  ``model`` is None on failure.
    """
    notes = []

    # Primary: try diffusers LuminaTransformer2DModel
    try:
        from diffusers import LuminaTransformer2DModel  # type: ignore[import]

        transformer_dir = path / "transformer"
        if transformer_dir.is_dir():
            model = LuminaTransformer2DModel.from_pretrained(
                str(transformer_dir), torch_dtype=dtype,
            )
            return model, notes
    except ImportError:
        notes.append(
            "diffusers.LuminaTransformer2DModel not available. "
            "Install a diffusers version that includes Lumina support, "
            "or register a custom pipeline class."
        )
    except Exception as exc:
        notes.append(f"LuminaTransformer2DModel.from_pretrained failed: {exc}")

    # Fallback: try loading from the root directory (some checkpoints
    # store the transformer weights at the top level).
    try:
        from diffusers import LuminaTransformer2DModel  # type: ignore[import]

        model = LuminaTransformer2DModel.from_pretrained(
            str(path), torch_dtype=dtype,
        )
        return model, notes
    except Exception:
        pass

    notes.append(
        "Transformer loading is not yet implemented for this checkpoint "
        "format.  See TODO comments in _load_transformer()."
    )
    return None, notes


def _load_text_encoder(
    path: Path, dtype: Any,
) -> Tuple[Any, Any, list]:
    """Load the Gemma 2B text encoder and its SentencePiece tokenizer.

    Returns ``(encoder, tokenizer, notes)``.  Either may be None on failure.
    """
    notes = []

    # Primary: diffusers pipeline subfolder layout
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

        te_dir = path / "text_encoder"
        tok_dir = path / "tokenizer"

        if te_dir.is_dir() and tok_dir.is_dir():
            encoder = AutoModel.from_pretrained(str(te_dir), torch_dtype=dtype)
            tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
            return encoder, tokenizer, notes
    except ImportError:
        notes.append(
            "transformers.AutoModel / AutoTokenizer not importable. "
            "Install transformers with Gemma support."
        )
    except Exception as exc:
        notes.append(f"Text encoder loading from subfolders failed: {exc}")

    # Fallback: try the main directory (single-file or merged layout)
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

        encoder = AutoModel.from_pretrained(str(path), torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        return encoder, tokenizer, notes
    except Exception:
        pass

    notes.append(
        "Text encoder loading is not yet implemented for this checkpoint "
        "format.  See TODO comments in _load_text_encoder()."
    )
    return None, None, notes


def _load_vae(
    path: Path, dtype: Any, vae_path: Optional[str] = None,
) -> Tuple[Any, list]:
    """Load the VAE (Flux-VAE compatible, 16-channel latent).

    Returns ``(vae, notes)``.  ``vae`` is None on failure.
    """
    notes = []
    from diffusers import AutoencoderKL

    # Try explicit vae_path first
    if vae_path:
        vae_ref = Path(vae_path)
        if vae_ref.is_file():
            try:
                return AutoencoderKL.from_single_file(str(vae_ref), torch_dtype=dtype), notes
            except Exception as exc:
                notes.append(f"Separate VAE single-file load failed: {exc}")
        elif vae_ref.is_dir():
            try:
                return AutoencoderKL.from_pretrained(str(vae_ref), torch_dtype=dtype), notes
            except Exception as exc:
                notes.append(f"Separate VAE directory load failed: {exc}")
        else:
            notes.append(f"Separate VAE path not found: {vae_path}")

    # Try subfolder layout
    vae_dir = path / "vae"
    if vae_dir.is_dir():
        try:
            return AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=dtype), notes
        except Exception as exc:
            notes.append(f"VAE subfolder load failed: {exc}")

    # Try main directory with subfolder hint
    try:
        return AutoencoderKL.from_pretrained(
            str(path), subfolder="vae", torch_dtype=dtype,
        ), notes
    except Exception as exc:
        notes.append(f"VAE from_pretrained(subfolder='vae') failed: {exc}")

    # Try single-file in root
    for name in ("vae.safetensors", "ae.safetensors"):
        candidate = path / name
        if candidate.is_file():
            try:
                return AutoencoderKL.from_single_file(str(candidate), torch_dtype=dtype), notes
            except Exception:
                pass

    notes.append("All VAE loading paths exhausted.")
    return None, notes


def _load_scheduler(path: Path) -> Tuple[Any, list]:
    """Load the flow-matching scheduler.

    Returns ``(scheduler, notes)``.  ``scheduler`` is None on failure.
    """
    notes = []

    # Primary: FlowMatchEulerDiscreteScheduler (diffusers >= 0.28)
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler  # type: ignore[import]

        sched_dir = path / "scheduler"
        if sched_dir.is_dir():
            return FlowMatchEulerDiscreteScheduler.from_pretrained(str(sched_dir)), notes
        # Some checkpoints store scheduler config at the root
        return FlowMatchEulerDiscreteScheduler.from_pretrained(str(path)), notes
    except ImportError:
        notes.append(
            "diffusers.FlowMatchEulerDiscreteScheduler not available. "
            "Requires diffusers >= 0.28."
        )
    except Exception as exc:
        notes.append(f"FlowMatchEulerDiscreteScheduler load failed: {exc}")

    notes.append(
        "Scheduler loading is not yet implemented for this checkpoint "
        "format.  See TODO comments in _load_scheduler()."
    )
    return None, notes
