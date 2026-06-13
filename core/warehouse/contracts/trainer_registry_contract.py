"""
Y-3 - Trainer Registry & Runtime Dispatch - Interface Contract

Defines the trainer definition schema and registry protocol for mapping
training types to their trainer scripts, validators, and platform overrides.

This module contains NO behavioral implementation.  It specifies:
- What a trainer definition looks like (frozen data model).
- What the registry must support (lookup, validation, platform dispatch).
- What config validation and preflight hooks look like structurally.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RuntimePlatform(enum.Enum):
    """Hardware/runtime platform that may require alternate trainer scripts."""
    DEFAULT = "default"
    AMD_ROCM = "amd_rocm"
    INTEL_XPU = "intel_xpu"
    NVIDIA_CUDA = "nvidia_cuda"


class TrainerRouteKind(enum.Enum):
    """High-level route family for a trainer type."""
    STABLE = "stable"
    SDXL = "sdxl"
    SD3 = "sd3"
    FLUX = "flux"
    ANIMA = "anima"
    NEWBIE = "newbie"
    LUMINA = "lumina"
    LUMINA2 = "lumina2"
    QWEN_IMAGE = "qwen-image"
    HUNYUAN_DIT = "hunyuan-dit"
    HUNYUAN_IMAGE = "hunyuan-image"
    YOLO = "yolo"
    AESTHETIC = "aesthetic"
    CONTROLNET = "controlnet"
    GENERIC = "generic"


# ---------------------------------------------------------------------------
# Hook type aliases (callback signatures)
# ---------------------------------------------------------------------------

# Takes (config_dict) -> optional error string
ConfigValidator = Callable[[dict[str, Any]], str | None]

# Takes (config_dict) -> list of warning strings
WarningBuilder = Callable[[dict[str, Any]], list[str]]

# Takes (config_dict, errors, warnings, notes) -> optional summary dict
PreflightBuilder = Callable[[dict[str, Any], list[str], list[str], list[str]], dict[str, Any] | None]


# ---------------------------------------------------------------------------
# Data Models - Trainer definition (frozen, no behavior)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainerDefinition:
    """Static definition of a training type and its dispatch metadata.

    Describes one entry in the trainer registry: which script to run,
    what platform overrides exist, and what validation/preflight hooks apply.
    """
    train_type: str
    trainer_file: str
    route_kind: str | None = None
    route_label: str | None = None

    # Launch mode
    direct_python: bool = False        # True = run directly, not via accelerate
    direct_cli_args: tuple[str, ...] = ()
    direct_launch_summary: str | None = None

    # Validation hooks (optional callbacks)
    config_validator: ConfigValidator | None = None
    start_warning_builder: WarningBuilder | None = None
    preflight_builder: PreflightBuilder | None = None
    preflight_handles_resume: bool = False

    # Dataset flexibility
    allow_dataset_config_without_train_data_dir: bool = False
    allow_dataset_class_without_train_data_dir: bool = False
    skip_model_validation: bool = False


@dataclass(frozen=True)
class PlatformOverride:
    """Platform-specific trainer script override.

    When a particular runtime platform is detected (e.g. AMD ROCm),
    the registry may substitute an alternate trainer script.
    """
    train_type: str
    platform: RuntimePlatform
    override_trainer_file: str
    detection_signals: tuple[str, ...]  # env var names to check


@dataclass(frozen=True)
class TrainerLookupResult:
    """Result of looking up a trainer definition from the registry."""
    definition: TrainerDefinition
    platform_override_applied: bool
    resolved_trainer_file: str
    platform: RuntimePlatform


@dataclass(frozen=True)
class PreflightReport:
    """Output of running preflight validation for a training type."""
    training_type: str
    can_start: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    notes: tuple[str, ...]
    dataset_summary: dict[str, Any] | None
    distributed_info: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class TrainerRegistryProtocol(Protocol):
    """Queryable registry mapping training types to trainer definitions.

    Implementations MUST:
    - Support lookup by training type string (case-insensitive).
    - Support reverse lookup by trainer file path.
    - Apply platform overrides when detected.
    - Expose the full list of registered training types.
    """

    def get_definition(self, training_type: str) -> TrainerLookupResult | None:
        """Look up a trainer definition, applying platform overrides."""
        ...

    def get_definition_by_file(self, trainer_file: str) -> TrainerLookupResult | None:
        """Reverse lookup: find the definition that uses this trainer file."""
        ...

    def list_training_types(self) -> Sequence[str]:
        """Return all registered training type strings."""
        ...

    def is_direct_python_type(self, training_type: str) -> bool:
        """Return True if this training type bypasses accelerate."""
        ...

    def get_route_kind(self, training_type: str) -> str | None:
        """Return the route kind for a training type."""
        ...


@runtime_checkable
class PlatformDetectorProtocol(Protocol):
    """Detects the current runtime platform for trainer dispatch."""

    def detect_platform(self) -> RuntimePlatform:
        """Return the current platform (AMD ROCm, Intel XPU, etc.)."""
        ...

    def is_amd_rocm_requested(self) -> bool:
        """Return True if AMD ROCm runtime is requested via env or config."""
        ...

    def is_intel_xpu_requested(self) -> bool:
        """Return True if Intel XPU runtime is requested via env or config."""
        ...


@runtime_checkable
class PreflightRunnerProtocol(Protocol):
    """Runs preflight validation for a training configuration.

    Implementations MUST:
    - Validate model path, dataset, resume state, learning rates.
    - Produce structured errors/warnings/notes (not just strings).
    - Call trainer-specific validators when registered.
    """

    def run_preflight(
        self,
        config: dict[str, Any],
        *,
        training_type: str,
        sample_prompt_builder: Callable[..., Any] | None = None,
        attention_fallback_checker: Callable[..., str | None] | None = None,
    ) -> PreflightReport:
        """Run full preflight validation. Returns structured report."""
        ...
