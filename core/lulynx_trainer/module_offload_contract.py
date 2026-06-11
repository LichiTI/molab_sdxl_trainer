"""Shared Warehouse helpers for module_offload config and summaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)


MODULE_OFFLOAD_SCOPE_BACKBONE = "backbone"
MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_1 = "text_encoder_1"
MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_2 = "text_encoder_2"
MODULE_OFFLOAD_SCOPE_ORDER = (
    MODULE_OFFLOAD_SCOPE_BACKBONE,
    MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_1,
    MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_2,
)

MODULE_OFFLOAD_PROFILES: dict[str, tuple[int, int]] = {
    "conservative": (25, 0),
    "balanced": (50, 25),
    "aggressive": (75, 50),
}
MODULE_OFFLOAD_PROFILE_CHOICES = {"custom", *MODULE_OFFLOAD_PROFILES.keys()}

MODULE_OFFLOAD_CONFLICTS: dict[str, tuple[str, str]] = {
    "swap": (
        "module_offload_swap_conflict",
        "Module offload is incompatible with memory swap. Disable swap_granularity/blocks_to_swap or turn module_offload off.",
    ),
    "vram_swap_to_ram": (
        "module_offload_vram_swap_conflict",
        "Module offload is incompatible with vram_swap_to_ram. Use only one CPU offload strategy.",
    ),
    "safe_fallback": (
        "module_offload_safe_fallback_conflict",
        "Module offload is incompatible with safe_fallback. Disable one of them.",
    ),
    "torch_compile": (
        "module_offload_torch_compile_conflict",
        "Module offload is incompatible with torch_compile. Run eager mode or turn module_offload off.",
    ),
    "distributed": (
        "module_offload_distributed_conflict",
        "Module offload v1 supports only single-GPU training. Disable distributed / multi-GPU launch settings.",
    ),
    "deepspeed": (
        "module_offload_deepspeed_conflict",
        "Module offload is incompatible with DeepSpeed in v1.",
    ),
    "pipeline": (
        "module_offload_pipeline_conflict",
        "Module offload is incompatible with ControlNet / IP-Adapter / LLLite pipeline routes in v1.",
    ),
    "gradient_checkpointing": (
        "module_offload_gradient_checkpointing_conflict",
        "Module offload is incompatible with gradient_checkpointing in v1.",
    ),
    "cpu_offload_checkpointing": (
        "module_offload_cpu_offload_checkpointing_conflict",
        "Module offload is incompatible with cpu_offload_checkpointing.",
    ),
    "single_cuda_gpu_required": (
        "module_offload_single_cuda_gpu_required",
        "Module offload v1 requires a single CUDA GPU eager training run.",
    ),
}


def _read_value(source: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def truthy(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def clamp_module_offload_ratio(value: Any, *, default: int = 0) -> int:
    try:
        ratio = int(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        ratio = default
    return max(0, min(100, ratio))


def parse_optional_module_offload_ratio(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return None


def normalize_module_offload_profile(value: Any) -> str:
    profile = str(value or "custom").strip().lower().replace("-", "_")
    return profile if profile in MODULE_OFFLOAD_PROFILE_CHOICES else "custom"


def parse_module_offload_float(value: Any, *, default: float = 0.0, minimum: float = 0.0) -> float:
    try:
        parsed = float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def normalize_module_offload_patterns(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_swap_requested(source: Mapping[str, Any] | Any) -> bool:
    granularity = str(_read_value(source, "swap_granularity", "off") or "off").strip().lower().replace("-", "_")
    try:
        swap_ratio = float(_read_value(source, "swap_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        swap_ratio = 0.0
    try:
        swap_count = int(_read_value(source, "swap_count", 0) or 0)
    except (TypeError, ValueError):
        swap_count = 0
    try:
        blocks_to_swap = int(_read_value(source, "blocks_to_swap", 0) or 0)
    except (TypeError, ValueError):
        blocks_to_swap = 0
    return (granularity != "off" and (swap_ratio > 0.0 or swap_count > 0 or granularity == "auto")) or blocks_to_swap > 0


@dataclass(frozen=True)
class ModuleOffloadConfigView:
    enabled: bool
    ratio: int
    profile: str
    min_param_mb: float
    include_patterns: str
    exclude_patterns: str
    verify_state: bool
    profile_enabled: bool
    prefetch_enabled: bool
    prefetch_mode: str
    backbone_ratio_override: int | None
    text_encoder_ratio_override: int | None
    effective_backbone_ratio: int
    effective_text_encoder_ratio: int
    requested: bool
    enhanced: bool

    def ratio_for_scope(self, scope_name: str) -> int:
        if scope_name == MODULE_OFFLOAD_SCOPE_BACKBONE:
            return self.effective_backbone_ratio
        if scope_name in {MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_1, MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_2}:
            return self.effective_text_encoder_ratio
        return 0


def resolve_module_offload_config(source: Mapping[str, Any] | Any) -> ModuleOffloadConfigView:
    enhanced = truthy(_read_value(source, "module_offload_enhanced", False))
    if enhanced:
        logger.warning(
            "module_offload_enhanced is enabled (experimental). "
            "Activates aggressive offload + prefetch. May cause instability."
        )

    # Enhanced sets defaults; explicit user values take priority.
    enabled = truthy(_read_value(source, "module_offload_enabled", enhanced))
    ratio = clamp_module_offload_ratio(_read_value(source, "module_offload_ratio", 0), default=0)

    profile_enabled_raw = _read_value(source, "module_offload_profile_enabled", None)
    profile_enabled = truthy(profile_enabled_raw) if profile_enabled_raw is not None else enhanced

    profile_raw = _read_value(source, "module_offload_profile", None)
    if enhanced and (not profile_raw or str(profile_raw).strip().lower() in {"", "custom"}):
        profile = "aggressive"
    else:
        profile = normalize_module_offload_profile(profile_raw or "custom")
    if not profile_enabled:
        profile = "custom"

    min_param_mb = parse_module_offload_float(_read_value(source, "module_offload_min_param_mb", 0.0))
    include_patterns = normalize_module_offload_patterns(_read_value(source, "module_offload_include_patterns", ""))
    exclude_patterns = normalize_module_offload_patterns(_read_value(source, "module_offload_exclude_patterns", ""))
    verify_state = truthy(_read_value(source, "module_offload_verify_state", True), default=True)

    prefetch_raw = _read_value(source, "module_offload_prefetch_enabled", None)
    prefetch_enabled = truthy(prefetch_raw) if prefetch_raw is not None else enhanced

    prefetch_mode = str(_read_value(source, "module_offload_prefetch_mode", "experimental") or "experimental").strip().lower()
    if prefetch_mode not in {"experimental"}:
        prefetch_mode = "experimental"
    backbone_override = parse_optional_module_offload_ratio(_read_value(source, "module_offload_backbone_ratio", None))
    text_encoder_override = parse_optional_module_offload_ratio(
        _read_value(source, "module_offload_text_encoder_ratio", None)
    )
    if profile == "custom":
        default_backbone_ratio = ratio
        default_text_encoder_ratio = ratio
    else:
        default_backbone_ratio, default_text_encoder_ratio = MODULE_OFFLOAD_PROFILES[profile]
    effective_backbone_ratio = backbone_override if backbone_override is not None else default_backbone_ratio
    effective_text_encoder_ratio = text_encoder_override if text_encoder_override is not None else default_text_encoder_ratio
    requested = enabled and (effective_backbone_ratio > 0 or effective_text_encoder_ratio > 0)
    return ModuleOffloadConfigView(
        enabled=enabled,
        ratio=ratio,
        profile=profile,
        min_param_mb=min_param_mb,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        verify_state=verify_state,
        profile_enabled=profile_enabled,
        prefetch_enabled=prefetch_enabled,
        prefetch_mode=prefetch_mode,
        backbone_ratio_override=backbone_override,
        text_encoder_ratio_override=text_encoder_override,
        effective_backbone_ratio=effective_backbone_ratio,
        effective_text_encoder_ratio=effective_text_encoder_ratio,
        requested=requested,
        enhanced=enhanced,
    )


def empty_module_offload_scopes(view: ModuleOffloadConfigView) -> dict[str, dict[str, int]]:
    return {
        MODULE_OFFLOAD_SCOPE_BACKBONE: {
            "ratio": view.effective_backbone_ratio,
            "candidate_count": 0,
            "selected_count": 0,
        },
        MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_1: {
            "ratio": view.effective_text_encoder_ratio,
            "candidate_count": 0,
            "selected_count": 0,
        },
        MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_2: {
            "ratio": view.effective_text_encoder_ratio,
            "candidate_count": 0,
            "selected_count": 0,
        },
    }


def build_module_offload_pending_state(
    source: Mapping[str, Any] | Any,
    *,
    reason: str = "pending training loop resolution",
) -> dict[str, Any]:
    view = resolve_module_offload_config(source)
    return {
        "enabled": view.requested,
        "mode": "module_offload" if view.requested else "none",
        "source": "config",
        "reason": reason,
        "warnings": [],
        "ratio": view.ratio,
        "profile": view.profile,
        "min_param_mb": view.min_param_mb,
        "include_patterns": view.include_patterns,
        "exclude_patterns": view.exclude_patterns,
        "verify_state": view.verify_state,
        "prefetch_enabled": view.prefetch_enabled,
        "prefetch_mode": view.prefetch_mode,
        "backbone_ratio": view.effective_backbone_ratio,
        "text_encoder_ratio": view.effective_text_encoder_ratio,
        "scopes": empty_module_offload_scopes(view),
    }


def get_module_offload_conflict(code: str) -> tuple[str, str]:
    return MODULE_OFFLOAD_CONFLICTS[code]

