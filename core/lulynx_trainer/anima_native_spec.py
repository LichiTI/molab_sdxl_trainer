from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class AnimaResourcePaths:
    """Resolved filesystem inputs for a native Anima route."""

    transformer_checkpoint: Path
    qwen3_directory: Path
    vae_directory_or_file: Path
    tokenizer_directory: Optional[Path] = None
    t5_tokenizer_directory: Optional[Path] = None


@dataclass
class AnimaComponentBundle:
    """Concrete runtime components needed by the native Anima route."""

    text_encoder: Any
    text_tokenizer: Any
    vae: Any
    transformer_handle: Any = None
    t5_tokenizer: Any = None
    notes: list[str] = field(default_factory=list)


@dataclass
class AnimaConditionPack:
    """Prompt-side tensors emitted by a native Anima text pipeline."""

    prompt_embeds: Any
    attention_mask: Any
    t5_input_ids: Any = None
    t5_attention_mask: Any = None


@dataclass(frozen=True)
class AnimaLoaderHints:
    """Non-structural hints for loader behavior."""

    dtype_name: str = "bfloat16"
    device_name: str = "cuda"
    attention_backend: str = "sdpa"
    lazy_transformer: bool = True
