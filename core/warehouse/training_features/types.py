"""Shared types and primitives for the training features Warehouse package."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelArchitecture(str, enum.Enum):
    """Model architectures that affect cache profiles and training paths."""

    SD15 = "sd15"
    SDXL = "sdxl"
    SD3 = "sd3"
    FLUX = "flux"
    ANIMA = "anima"
    NEWBIE = "newbie"
    LUMINA = "lumina"
    LUMINA2 = "lumina2"
    QWEN_IMAGE = "qwen_image"
    HUNYUAN_DIT = "hunyuan_dit"
    HUNYUAN_IMAGE = "hunyuan_image"


class RouteFamily(str, enum.Enum):
    """High-level route families for training type classification."""

    STABLE = "stable"
    ANIMA = "anima"
    NEWBIE = "newbie"
    SDXL = "sdxl"
    SD3 = "sd3"
    FLUX = "flux"
    LUMINA = "lumina"
    LUMINA2 = "lumina2"
    QWEN_IMAGE = "qwen-image"
    HUNYUAN_DIT = "hunyuan-dit"
    HUNYUAN_IMAGE = "hunyuan-image"
    GENERIC = "generic"


class AttentionBackend(str, enum.Enum):
    """Supported attention backends."""

    AUTO = "auto"
    SDPA = "sdpa"
    XFORMERS = "xformers"
    SAGE_ATTN = "sageattn"
    FLASH_ATTN = "flash_attn"
    ROCM = "rocm"
    XPU = "xpu"


class PreflightVerdict(str, enum.Enum):
    """Final go/no-go verdict from a preflight run."""

    GO = "go"
    NO_GO = "no_go"
    WARN = "warn"


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------

class Capability(str, enum.Enum):
    """Named capability flags for route contracts."""

    DORA = "dora"
    BLOCK_SWAP = "block-swap"
    TEXT_ENCODER_CACHE = "text-encoder-cache"
    MIXED_RESOLUTION = "mixed-resolution"
    DISTRIBUTED = "distributed"
    RESUME = "resume"
    CUSTOM_BACKEND = "custom-backend"
    YOLO_MODE = "yolo-mode"
    PISSA = "pissa"
    LOHA = "loha"
    LOKR = "lokr"
    IA3 = "ia3"
    DYLORA = "dylora"
    VRAM_SWAP_TO_RAM = "vram-swap-to-ram"
    LOW_VRAM_OPTIMIZATION = "low-vram-optimization"
    SAGE_ATTENTION = "sage-attention"
    FLASH_ATTENTION = "flash-attention"
    TORCH_COMPILE = "torch-compile"
    CHANNELS_LAST = "channels-last"
    MASKED_LOSS = "masked-loss"
    VALIDATION_SPLIT = "validation-split"
    JSON_CAPTION = "json-caption"
    TWO_PHASE_EXECUTION = "two-phase-execution"
    PERSISTENT_CACHE = "persistent-cache"
    TRANSIENT_CACHE = "transient-cache"


# ---------------------------------------------------------------------------
# Shared result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Severity:
    """A single tagged message with severity level."""

    level: str  # "error" | "warning" | "note"
    message: str
    source: str = ""


@dataclass
class MessageBag:
    """Accumulates errors/warnings/notes across components."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_note(self, msg: str) -> None:
        self.notes.append(msg)

    def merge(self, other: MessageBag) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.notes.extend(other.notes)

    @property
    def is_clean(self) -> bool:
        return len(self.errors) == 0

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }

