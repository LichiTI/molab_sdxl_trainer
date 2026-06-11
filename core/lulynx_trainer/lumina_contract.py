"""
Lumina Image 2.0 architecture contracts.

This module defines the data contracts that downstream code (model_loader,
training_loop, sampler, config) must satisfy to support Lumina.  Each
contract is a plain dataclass with no runtime dependencies beyond the
standard library.

Contracts defined:
  1. LuminaLimitation        — structured flags for capability gaps.
  2. LuminaLoadReport        — transparent record of what the loader did.
  3. Text encoder contract   — Gemma 2B, not CLIP/T5.
  4. Latent contract         — 16 channels, Flux-VAE compatible.
  5. Scheduler contract      — Continuous-time flow matching.
  6. Target-module contract  — Which nn.Linear layers to inject with LoRA.
  7. Training-loss contract  — Velocity prediction, not epsilon/x0.

NOT implemented here (deferred to shared-core integration):
  - Actual diffusers model loading.
  - Tokenizer instantiation.
  - Forward pass / loss computation.
  - Sampling / inference pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Limitation flags
# ============================================================================

class LuminaLimitation(Enum):
    """Structured limitation flags the loader can surface honestly.

    Each value represents a capability gap detected at load time.  The
    trainer / UI reads these to warn the user or adjust behaviour rather
    than silently degrading.
    """

    #: The Gemma 2B text encoder could not be loaded (missing weights,
    #: missing transformers class, or incompatible checkpoint).  The loader
    #: cannot proceed without a text encoder — this is a hard blocker.
    TEXT_ENCODER_UNAVAILABLE = "text_encoder_unavailable"

    #: The SentencePiece tokenizer for Gemma could not be loaded.
    #: The loader cannot tokenize prompts — this is a hard blocker.
    TOKENIZER_UNAVAILABLE = "tokenizer_unavailable"

    #: The Lumina transformer (denoiser) could not be loaded.  Either
    #: diffusers lacks LuminaTransformer2DModel or the weights are missing.
    #: This is a hard blocker — no training or inference is possible.
    TRANSFORMER_UNAVAILABLE = "transformer_unavailable"

    #: The flow-matching scheduler class was not found in the installed
    #: diffusers version.  The loader used a fallback or raised.
    SCHEDULER_UNAVAILABLE = "scheduler_unavailable"

    #: The VAE could not be loaded from the expected path.  The loader
    #: attempted fallback paths (subfolder, main directory, single-file).
    VAE_LOAD_FALLBACK = "vae_load_fallback"

    #: The user requested a separate VAE path but it was not found.
    #: The loader fell back to the VAE bundled with the model.
    SEPARATE_VAE_NOT_FOUND = "separate_vae_not_found"

    #: The checkpoint was loaded via a HuggingFace model ID rather than
    #: a local directory.  Network access was required.
    LOADED_FROM_HF_HUB = "loaded_from_hf_hub"

    #: The loader is operating in skeleton mode: the high-level structure
    #: is correct but one or more components raised NotImplementedError.
    #: No usable model was produced.
    SKELETON_MODE = "skeleton_mode"


# ============================================================================
# 2. Load report
# ============================================================================

@dataclass
class LuminaLoadReport:
    """Transparent report of what the Lumina loader actually did.

    Returned alongside the loaded components so callers can inspect which
    parts succeeded and which fell back or failed.
    """

    #: The resolved primary model path that was actually loaded.
    resolved_model_path: str = ""

    #: Whether the transformer (denoiser) was loaded successfully.
    transformer_loaded: bool = False

    #: Whether the Gemma 2B text encoder was loaded successfully.
    text_encoder_loaded: bool = False

    #: Whether the SentencePiece tokenizer was loaded successfully.
    tokenizer_loaded: bool = False

    #: Whether the VAE was loaded successfully.
    vae_loaded: bool = False

    #: Whether the flow-matching scheduler was loaded successfully.
    scheduler_loaded: bool = False

    #: All limitation flags detected during loading.
    limitations: List[LuminaLimitation] = field(default_factory=list)

    #: Free-form notes for logging / debugging.
    notes: List[str] = field(default_factory=list)

    def has_limitation(self, kind: LuminaLimitation) -> bool:
        return kind in self.limitations

    @property
    def is_skeleton(self) -> bool:
        """True if the loader could not produce a usable model."""
        return LuminaLimitation.SKELETON_MODE in self.limitations

    @property
    def is_usable(self) -> bool:
        """True if all required components loaded successfully."""
        return (
            self.transformer_loaded
            and self.text_encoder_loaded
            and self.tokenizer_loaded
            and self.vae_loaded
            and self.scheduler_loaded
            and not self.is_skeleton
        )

    @property
    def blocker_count(self) -> int:
        """Number of hard-blocker limitations."""
        _HARD_BLOCKERS = {
            LuminaLimitation.TRANSFORMER_UNAVAILABLE,
            LuminaLimitation.TEXT_ENCODER_UNAVAILABLE,
            LuminaLimitation.TOKENIZER_UNAVAILABLE,
            LuminaLimitation.SCHEDULER_UNAVAILABLE,
            LuminaLimitation.SKELETON_MODE,
        }
        return sum(1 for lim in self.limitations if lim in _HARD_BLOCKERS)

    def summary(self) -> str:
        """One-line human-readable summary."""
        parts = [f"model={self.resolved_model_path or '(none)'}"]
        flags = []
        if self.transformer_loaded:
            flags.append("transformer")
        if self.text_encoder_loaded:
            flags.append("text_enc")
        if self.tokenizer_loaded:
            flags.append("tokenizer")
        if self.vae_loaded:
            flags.append("vae")
        if self.scheduler_loaded:
            flags.append("scheduler")
        if flags:
            parts.append(f"loaded=[{','.join(flags)}]")
        if self.limitations:
            names = ", ".join(lim.value for lim in self.limitations)
            parts.append(f"limitations=[{names}]")
        return " | ".join(parts)


# ============================================================================
# 3. Text Encoder Contract
# ============================================================================

@dataclass(frozen=True)
class LuminaTextEncoderContract:
    """Specification for Lumina's text encoder.

    Lumina Image 2.0 uses **Gemma 2B** as its sole text encoder.
    This is fundamentally different from SDXL (dual CLIP) and Flux (CLIP + T5).

    Implications for integration:
      - The tokenizer is a SentencePiece tokenizer (Gemma family), not
        CLIPTokenizer or T5Tokenizer.
      - The text encoder output is a sequence of hidden states (not pooled
        embeddings from CLIP).  Lumina uses the full sequence for
        cross-attention.
      - There is no ``pooled_prompt_embeds`` equivalent from Gemma.
        If the model needs a pooled vector, it is derived from the
        sequence output (e.g. mean pooling or EOS token).
      - The text encoder has ~2B parameters.  Training it requires
        significant VRAM (~8 GB in bf16 just for the weights).
    """

    # HuggingFace model ID or local path for the text encoder.
    model_id: str = "google/gemma-2b"

    # Tokenizer class to use.
    tokenizer_class: str = "transformers.AutoTokenizer"

    # Hidden dimension of the text encoder.
    hidden_size: int = 2304

    # Maximum sequence length for text prompts.
    max_sequence_length: int = 256

    # Gemma does not produce a CLIP-style pooled_output.
    needs_pooled_output: bool = False

    # Recommended: cache text encoder outputs to disk to save VRAM.
    recommend_cache_to_disk: bool = True


LUMINA_TE_CONTRACT = LuminaTextEncoderContract()


# ============================================================================
# 4. Latent Contract
# ============================================================================

@dataclass(frozen=True)
class LuminaLatentContract:
    """Specification for Lumina's latent space.

    Lumina uses the same VAE family as Flux (16-channel latent, scaling
    factor 0.3611).  The VAE encoder/decoder is architecturally identical
    to Flux's.
    """

    channels: int = 16
    vae_scaling_factor: float = 0.3611
    spatial_compression: int = 8
    patch_size: int = 2


LUMINA_LATENT_CONTRACT = LuminaLatentContract()


# ============================================================================
# 5. Scheduler / Flow-Matching Contract
# ============================================================================

@dataclass(frozen=True)
class LuminaSchedulerContract:
    """Specification for Lumina's noise scheduler.

    Lumina uses continuous-time flow matching.  Key differences from
    DDPM/DDIM:
      - Timesteps are continuous in [0, 1], not discrete integers.
      - The interpolation is: x_t = t * x_1 + (1 - t) * noise.
      - The model predicts velocity v = x_1 - noise.
      - The loss is MSE between predicted and target velocity.
      - There is no beta schedule, no alpha_bar, no posterior.
    """

    scheduler_type: str = "flow_matching"
    t_min: float = 0.0
    t_max: float = 1.0
    recommended_sampling: str = "sigmoid"
    discrete_flow_shift: float = 3.0
    loss_type: str = "velocity"
    default_inference_steps: int = 28
    supports_dynamic_thresholding: bool = False


LUMINA_SCHEDULER_CONTRACT = LuminaSchedulerContract()


# ============================================================================
# 6. Target-Module Contract
# ============================================================================

LUMINA_LORA_TARGETS: Set[str] = {
    # Self-attention
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    # Cross-attention (image attends to text)
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    # Feed-forward network
    "ff.net.0.proj",
    "ff.net.2",
}

# Minimal target set for low-VRAM training (attention only, no FFN).
LUMINA_LORA_TARGETS_MINIMAL: Set[str] = {
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
}

# Extended target set including norms.
LUMINA_LORA_TARGETS_EXTENDED: Set[str] = LUMINA_LORA_TARGETS | {
    "norm1.linear",
    "norm2.linear",
    "attn.norm_q",
    "attn.norm_k",
}


# ============================================================================
# 7. Training-Loss Contract
# ============================================================================

@dataclass(frozen=True)
class LuminaLossContract:
    """Specification for Lumina's training loss.

    Flow-matching velocity prediction:
        1. Sample t ~ sigmoid_distribution(0, 1, shift=discrete_flow_shift).
        2. Construct noisy latent: x_t = t * x_clean + (1 - t) * noise.
        3. Predict velocity: v_pred = model(x_t, t, text_embed).
        4. Target velocity: v_target = x_clean - noise.
        5. Loss = MSE(v_pred, v_target).
    """

    loss_fn: str = "mse"
    prediction_type: str = "velocity"
    loss_weighting: str = "none"
    timestep_sampling: str = "sigmoid"
    discrete_flow_shift: float = 3.0


LUMINA_LOSS_CONTRACT = LuminaLossContract()


# ============================================================================
# Aggregated contract
# ============================================================================

@dataclass(frozen=True)
class LuminaArchitectureContract:
    """Aggregated contract combining all Lumina sub-contracts."""

    text_encoder: LuminaTextEncoderContract = field(
        default_factory=lambda: LUMINA_TE_CONTRACT
    )
    latent: LuminaLatentContract = field(
        default_factory=lambda: LUMINA_LATENT_CONTRACT
    )
    scheduler: LuminaSchedulerContract = field(
        default_factory=lambda: LUMINA_SCHEDULER_CONTRACT
    )
    loss: LuminaLossContract = field(
        default_factory=lambda: LUMINA_LOSS_CONTRACT
    )
    lora_targets: Set[str] = field(
        default_factory=lambda: set(LUMINA_LORA_TARGETS)
    )

    # Model geometry
    num_transformer_blocks: int = 24
    hidden_size: int = 2304
    num_attention_heads: int = 24
    head_dim: int = 96

    # VAE / latent
    latent_channels: int = 16
    vae_scaling_factor: float = 0.3611


LUMINA_ARCH_CONTRACT = LuminaArchitectureContract()


# ============================================================================
# ModelFamily reference spec
# ============================================================================

def get_model_family_spec() -> Dict:
    """Return the ModelFamily kwargs for Lumina.

    This is a reference for the shared-core integration step where
    ``_MODEL_FAMILIES["lumina"]`` must be added to model_family.py.

    Returns:
        Dict with keys matching the ``ModelFamily`` dataclass fields.
    """
    return {
        "unet_target_modules": sorted(LUMINA_LORA_TARGETS),
        "text_encoder_target_modules": [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2",
        ],
        "has_dual_text_encoders": False,
        "uses_pooled_prompt_embeds": False,
        "uses_time_ids": False,
        "default_sampler_pipeline": None,
        "is_stub": True,
        "latent_channels": 16,
        "vae_scaling_factor": 0.3611,
    }
