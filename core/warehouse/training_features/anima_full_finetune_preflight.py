"""Report-only preflight recommendations for Anima full finetune.

Phase 1 trains DiT parameters only.  Text encoders may participate as frozen
conditioning providers through cache-first or online-cache data paths.  The helper keeps
recommendations structured so launcher/UI code can expose them without
mutating a user's request during validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


_OFF_VALUES = {"", "off", "none", "disabled", "false", "0"}
_RESIDENT_VALUES = {"", "resident", "gpu", "off", "none", "disabled"}
_SAFE_PRECISIONS = {"bf16", "fp16"}
_MEMORY_FRIENDLY_OPTIMIZERS = {
    "adamw8bit",
    "pagedadamw8bit",
    "pagedadamw32bit",
    "kahanadamw8bit",
}


def _flag(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _lower(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _compact_patch(patch: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in sorted(patch.items()))


def _is_anima_full_finetune(config: Mapping[str, Any], training_type: str) -> bool:
    schema_id = str(config.get("schema_id") or "").strip().lower()
    route = str(training_type or config.get("training_type") or "").strip().lower()
    model_type = str(config.get("model_type") or config.get("model_arch") or "").strip().lower()
    return schema_id == "anima-finetune" or route == "anima-finetune" or (
        model_type == "anima" and route == "full_finetune"
    )


@dataclass(frozen=True)
class AnimaFullFinetunePreflightProfile:
    applicable: bool
    risk_level: str = "none"
    recommended_config_patch: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "applicable": bool(self.applicable),
            "risk_level": self.risk_level,
            "recommended_config_patch": dict(self.recommended_config_patch),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "reasons": list(self.reasons),
        }


def build_anima_full_finetune_preflight_profile(
    config: Mapping[str, Any],
    training_type: str = "",
) -> AnimaFullFinetunePreflightProfile:
    """Build report-only 16GB-oriented guidance for Anima full finetune."""

    if not _is_anima_full_finetune(config, training_type):
        return AnimaFullFinetunePreflightProfile(applicable=False)

    patch: dict[str, Any] = {}
    warnings: list[str] = []
    notes: list[str] = [
        "Anima full finetune 16GB profile is report-only; preflight does not mutate the launch config.",
    ]
    reasons: list[str] = []

    cache_mode = _lower(config.get("native_cache_mode")) or _lower(config.get("anima_cache_mode"))
    cache_mode = cache_mode or "cache_first"
    frozen_te_online_cache = cache_mode in {"online_cache", "online_cache".replace("_", "-")}
    cache_ready = cache_mode in {"cache_first", "force_cache_only"} and _flag(
        config.get("anima_cached_training"),
        default=True,
    )
    if frozen_te_online_cache:
        notes.append(
            "Anima online_cache uses frozen VAE/text encoders to generate missing latent/text-conditioning cache before the DiT step; TE weights are not trained."
        )
        reasons.append("frozen_te_online_cache")
    elif not cache_ready:
        patch.update({
            "native_cache_mode": "cache_first",
            "anima_cache_mode": "cache_first",
            "anima_cached_training": True,
        })
        warnings.append("Anima full finetune should use cache_first or online_cache; raw online Anima training is not the validated 16GB route.")
        reasons.append("cache_first_required")

    profile = _lower(config.get("native_runtime_profile"))
    if profile in {"", "standard", "auto", "default"}:
        patch["native_runtime_profile"] = "anima_low_vram"
        reasons.append("low_vram_runtime_profile")

    precision = _lower(config.get("mixed_precision"))
    if precision not in _SAFE_PRECISIONS:
        patch["mixed_precision"] = "bf16"
        warnings.append("Anima full finetune on a 16GB target should avoid fp32/no mixed precision; bf16 is the preferred first try.")
        reasons.append("mixed_precision")

    batch = max(_int(config.get("train_batch_size", config.get("batch_size", 1)), 1), 1)
    accum = max(_int(config.get("gradient_accumulation_steps", 1), 1), 1)
    if batch > 1:
        patch["train_batch_size"] = 1
        patch["gradient_accumulation_steps"] = batch * accum
        warnings.append(
            "Anima full finetune 16GB target recommends train_batch_size=1; use gradient accumulation to preserve effective batch."
        )
        reasons.append("micro_batch")

    if not _flag(config.get("gradient_checkpointing")):
        patch["gradient_checkpointing"] = True
        reasons.append("gradient_checkpointing")
    if not _flag(config.get("anima_block_checkpointing")):
        patch["anima_block_checkpointing"] = True
        reasons.append("dit_block_checkpointing")
    if "anima_block_checkpointing" in patch and not config.get("anima_block_checkpointing_mode"):
        patch["anima_block_checkpointing_mode"] = "block"

    optimizer = str(config.get("optimizer_type") or config.get("optimizer") or "").strip()
    optimizer_norm = optimizer.lower().replace("_", "")
    if not optimizer_norm:
        patch["optimizer_type"] = "AdamW8bit"
        reasons.append("optimizer_default")
    elif optimizer_norm == "adamw":
        patch["optimizer_type"] = "AdamW8bit"
        warnings.append("Full DiT finetune has large Adam states; prefer AdamW8bit/PagedAdamW8bit before plain AdamW on 16GB.")
        reasons.append("optimizer_state")
    elif optimizer_norm not in _MEMORY_FRIENDLY_OPTIMIZERS:
        notes.append("Optimizer state may dominate Anima full finetune memory; 8bit or paged AdamW remains the safer 16GB route.")

    residency = _lower(config.get("anima_block_residency"))
    if residency not in _RESIDENT_VALUES:
        warnings.append(
            "Anima full finetune makes DiT weights trainable, so frozen-weight block residency may have little or no active CPU-pinned work in Phase 1. Treat it as diagnostic/future trainable-offload work, not the primary 16GB guardrail."
        )
    if _flag(config.get("anima_block_prefetch")) and residency != "streaming_offload":
        warnings.append("Anima block prefetch only applies to streaming_offload residency.")

    transfer_format = _lower(config.get("pcie_transfer_format"))
    if transfer_format not in _OFF_VALUES:
        notes.append("PCIe transfer format only applies when a CPU-pinned residency path is active; Phase 1 full finetune may report no active packed Linear units.")

    visual_tokens = _int(config.get("anima_fixed_visual_tokens", 0), 0)
    if visual_tokens >= 16384:
        warnings.append("Anima 128x128/16384 visual-token padding is a high-memory profile; start 16GB validation at 4096 tokens or cached native buckets.")

    if _flag(config.get("anima_full_finetune_train_text_encoder_requested")) or _flag(config.get("train_text_encoder")):
        notes.append("Text-encoder full finetune remains Phase 2 and will require a separate memory budget.")
    if _flag(config.get("save_state")):
        notes.append("save_state is useful for exact resume but optimizer snapshots for full finetune can be very large.")
    else:
        notes.append("save_state is off; checkpoint files are lighter, but exact optimizer resume is limited.")

    if patch:
        notes.append("Anima full finetune suggested config patch: " + _compact_patch(patch) + ".")

    risk_level = "high" if warnings else ("medium" if patch else "low")
    return AnimaFullFinetunePreflightProfile(
        applicable=True,
        risk_level=risk_level,
        recommended_config_patch=patch,
        warnings=warnings,
        notes=notes,
        reasons=reasons,
    )


__all__ = [
    "AnimaFullFinetunePreflightProfile",
    "build_anima_full_finetune_preflight_profile",
]
