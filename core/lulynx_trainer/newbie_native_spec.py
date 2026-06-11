from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class NewbieResourcePaths:
    """Resolved filesystem layout for a native Newbie route."""

    root_directory: Path
    transformer_directory: Path
    gemma_directory: Path
    jina_clip_directory: Path
    vae_directory: Path
    scheduler_directory: Optional[Path] = None


@dataclass
class NewbieComponentBundle:
    """Concrete runtime components needed by the native Newbie route."""

    transformer: Any
    gemma_text_encoder: Any
    gemma_tokenizer: Any
    jina_clip_model: Any
    jina_clip_tokenizer: Any
    vae: Any
    scheduler: Any = None
    notes: list[str] = field(default_factory=list)


@dataclass
class NewbieConditionPack:
    """Prompt-side tensors emitted by a native Newbie text pipeline."""

    gemma_hidden_states: Any
    gemma_attention_mask: Any
    clip_pooled_features: Any
    clip_attention_mask: Any = None


@dataclass(frozen=True)
class NewbieTransportHints:
    """Family-specific training hints for transport / flow style routes."""

    latent_channels: int = 16
    latent_scaling_factor: float = 0.3611
    expects_transport_object: bool = True
    expects_flow_targets: bool = True
