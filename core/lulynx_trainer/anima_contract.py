"""
Anima data contract — structured types for the native Anima loader.

These types describe the *result* of loading an Anima model and any
limitations encountered.  They are consumed by the loader (anima_loader.py)
and, later, by the trainer when wiring the native Anima route.

Design note: LoadedModel (in model_loader.py) is the shared cross-family
contract that TrainingLoop and Sampler consume.  The types here are
Anima-specific metadata that sit *alongside* LoadedModel, not inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SecondaryEncoderKind(Enum):
    """Machine-readable tag for what occupies the secondary encoder slot.

    The training loop expects specific output semantics from
    ``text_encoder_2``.  This enum lets the contract declare what is
    actually present so the trainer can branch or fail honestly rather
    than silently breaking.
    """

    #: No secondary encoder loaded.
    NONE = "none"

    #: A CLIP-style encoder (CLIPTextModelWithProjection or equivalent)
    #: that returns ``hidden_states`` and ``text_embeds``.  This is the
    #: shape the current training loop expects from ``text_encoder_2``.
    CLIP = "clip"

    #: A causal-LM encoder (e.g. Qwen3) that returns
    #: ``last_hidden_state`` but NOT ``text_embeds``.  Cannot serve as a
    #: drop-in for the CLIP secondary slot without a dedicated adapter in
    #: the training loop.
    CAUSAL_LM = "causal_lm"


class AnimaLimitation(Enum):
    """Structured limitation flags the loader can surface honestly.

    Each value represents a capability gap that the loader detected at
    load time.  The trainer / UI can read these to warn the user or
    adjust behaviour rather than silently degrading.
    """

    #: The Qwen3 secondary text encoder could not be loaded (missing path,
    #: missing class, or incompatible checkpoint).  The loader fell back to
    #: the CLIP encoder that ships with the primary checkpoint.
    QWEN3_UNAVAILABLE = "qwen3_unavailable"

    #: The user supplied a T5 tokenizer path, but the T5 tokenizer class
    #: was not importable or the path was invalid.  Fell back to the
    #: default CLIP tokenizer.
    T5_TOKENIZER_UNAVAILABLE = "t5_tokenizer_unavailable"

    #: The primary checkpoint was loaded via the SDXL single-file pathway
    #: because a native Anima safetensors loader is not yet implemented.
    #: This is functionally correct for SDXL-compatible Anima checkpoints
    #: but may miss Anima-specific metadata.
    SINGLE_FILE_SDXL_FALLBACK = "single_file_sdxl_fallback"

    #: The requested attention mode hint could not be applied (e.g.
    #: flash-attn2 not installed).  The model loaded with the default
    #: attention backend instead.
    ATTN_MODE_FALLBACK = "attn_mode_fallback"

    #: A Qwen3 (or other causal-LM) encoder was loaded, but it cannot
    #: serve as a drop-in replacement for the CLIP secondary encoder.
    #: The training loop expects ``hidden_states[-2]`` and ``text_embeds``
    #: from text_encoder_2 — outputs that causal-LM models do not
    #: natively produce.  The Qwen3 encoder is stored separately and
    #: available for future adapter-based integration.
    QWEN3_NOT_CLIP_COMPATIBLE = "qwen3_not_clip_compatible"

    #: The loader is running in "scaffold" mode: it reuses the SDXL loader
    #: entirely and only patches model_arch.  No Anima-specific modules
    #: have been loaded.  This is the Phase 1 behaviour.
    SCAFFOLD_MODE = "scaffold_mode"


@dataclass
class AnimaLoadReport:
    """Transparent report of what the Anima loader actually did.

    Returned alongside the LoadedModel so callers can inspect which
    optional components loaded successfully and which fell back.
    """

    #: The resolved primary model path that was actually loaded.
    resolved_model_path: str = ""

    #: What kind of secondary encoder ended up in ``text_encoder_2``.
    #: This is the machine-readable answer to "can I use text_encoder_2
    #: with the current training loop?".
    secondary_encoder_kind: SecondaryEncoderKind = SecondaryEncoderKind.NONE

    #: Whether the Qwen3 secondary text encoder was loaded.
    qwen3_loaded: bool = False

    #: Path to the Qwen3 encoder that was loaded (empty if none).
    qwen3_path: str = ""

    #: Whether a dedicated T5 tokenizer was loaded.
    t5_tokenizer_loaded: bool = False

    #: Path to the T5 tokenizer that was loaded (empty if none).
    t5_tokenizer_path: str = ""

    #: The attention mode that was actually applied.
    applied_attn_mode: str = ""

    #: All limitation flags detected during loading.
    limitations: List[AnimaLimitation] = field(default_factory=list)

    #: Free-form notes for logging / debugging.
    notes: List[str] = field(default_factory=list)

    #: Native single-file checkpoint metadata has been identified as an
    #: Anima `net.*` DiT checkpoint.
    native_introspection_ready: bool = False

    #: Warehouse native module names cover the checkpoint weight/bias keys.
    native_key_map_ready: bool = False

    #: A real-weight executable block0 smoke is available in validation.  This
    #: is not enough to mark training ready; full 28-block conditioning remains
    #: separate.
    native_block0_forward_smoke_ready: bool = False

    #: Full native Anima training is only true after loader, conditioning,
    #: flow objective, backward, and save/reload smokes pass together.
    anima_native_train_ready: bool = False

    def has_limitation(self, kind: AnimaLimitation) -> bool:
        return kind in self.limitations

    @property
    def is_scaffold(self) -> bool:
        """True if the loader used the SDXL scaffold fallback."""
        return AnimaLimitation.SCAFFOLD_MODE in self.limitations

    @property
    def secondary_is_training_compatible(self) -> bool:
        """True if text_encoder_2 can be used with the current training loop.

        The training loop expects CLIP-style outputs from the secondary
        encoder.  A causal-LM encoder (Qwen3) is loaded and available
        but is NOT a drop-in replacement.
        """
        return self.secondary_encoder_kind == SecondaryEncoderKind.CLIP

    def summary(self) -> str:
        """One-line human-readable summary."""
        parts = [f"model={self.resolved_model_path or '(none)'}"]
        parts.append(f"te2_kind={self.secondary_encoder_kind.value}")
        if self.qwen3_loaded:
            parts.append(f"qwen3={self.qwen3_path}")
        if self.t5_tokenizer_loaded:
            parts.append(f"t5tok={self.t5_tokenizer_path}")
        if self.limitations:
            names = ", ".join(lim.value for lim in self.limitations)
            parts.append(f"limitations=[{names}]")
        if self.native_block0_forward_smoke_ready and not self.anima_native_train_ready:
            parts.append("native_block0_smoke=ready")
        return " | ".join(parts)


@dataclass
class AnimaComponents:
    """Raw components loaded from an Anima checkpoint / directory.

    This is the internal return type of the loader's sub-methods.
    The loader converts this into a LoadedModel for the training pipeline.

    ``text_encoder_2`` always holds the CLIP-compatible secondary encoder
    (the one that can go into the LoadedModel's ``text_encoder_2`` slot).
    A Qwen3 encoder is stored separately in ``qwen3_encoder`` and must
    NOT be promoted into ``text_encoder_2`` — the output shapes are
    incompatible with the current training loop.
    """

    unet: Any = None
    text_encoder_1: Any = None       # Primary CLIP encoder
    text_encoder_2: Any = None       # Secondary CLIP encoder (CLIP-2 only)
    vae: Any = None
    tokenizer_1: Any = None          # Primary CLIP tokenizer
    tokenizer_2: Any = None          # Secondary CLIP tokenizer
    noise_scheduler: Any = None

    # Anima-specific optional extras (not part of LoadedModel)
    qwen3_encoder: Any = None        # Qwen3 causal-LM encoder (NOT clip-compatible)
    qwen3_tokenizer: Any = None      # Qwen3 tokenizer
    t5_tokenizer: Any = None         # Dedicated T5 tokenizer, if provided

    report: AnimaLoadReport = field(default_factory=AnimaLoadReport)

