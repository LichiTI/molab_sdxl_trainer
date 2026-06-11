"""Route-specific acceleration recommendations.

This module is intentionally torch-free.  It keeps model-family choices in one
small table so policy code does not need to guess compile or low-bit defaults
with scattered ``if family == ...`` branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompilePolicySpec:
    runtime: str
    shape_strategy: str
    target_strategy: str
    extra_patch: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfilePatchSpec:
    patch: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FamilyAccelerationSpec:
    family: str
    optimizer_balanced: str = "foreach_adamw"
    optimizer_aggressive: str = "torch_fused"
    attention_safe: str = ""
    attention_balanced: str = ""
    attention_aggressive: str = ""
    compile_aggressive: CompilePolicySpec | None = None
    low_bit_preset: str = "stable_backbone_int8"
    low_bit_notes: tuple[str, ...] = ()
    low_bit_profile: dict[str, ProfilePatchSpec] = field(default_factory=dict)
    cache_base: dict[str, Any] = field(default_factory=dict)
    cache_profile: dict[str, ProfilePatchSpec] = field(default_factory=dict)
    text_cache_profiles: frozenset[str] = frozenset()
    text_cache_to_disk_profiles: frozenset[str] = frozenset()
    runtime_profile: dict[str, str] = field(default_factory=dict)
    checkpoint_profile: dict[str, ProfilePatchSpec] = field(default_factory=dict)
    lora_recompute: dict[str, str] = field(default_factory=dict)
    advanced_optimizer_aggressive: str = ""
    adapter_init_aggressive: ProfilePatchSpec | None = None


_STATIC_COMPILE_GUARDS = {
    "compile_cache_enabled": True,
    "compile_contract_strict": True,
    "compile_static_shape_drop_last": True,
    "compile_require_cache_first": True,
}


_FAMILY_MATRIX: dict[str, FamilyAccelerationSpec] = {
    "sdxl": FamilyAccelerationSpec(
        family="sdxl",
        attention_safe="sdpa",
        attention_balanced="sdpa",
        attention_aggressive="sdpa",
        compile_aggressive=CompilePolicySpec(
            runtime="compile_cache",
            shape_strategy="fixed_pad",
            target_strategy="block",
            extra_patch=dict(_STATIC_COMPILE_GUARDS),
            notes=("SDXL uses U-Net block compile with fixed padded buckets.",),
        ),
        cache_base={"cache_latents": True},
        cache_profile={
            "low_vram": ProfilePatchSpec({"cache_latents_to_disk": True}),
        },
        text_cache_profiles=frozenset({"balanced", "aggressive", "low_vram"}),
        text_cache_to_disk_profiles=frozenset({"low_vram"}),
        checkpoint_profile={
            "low_vram": ProfilePatchSpec({"checkpoint_policy": "offloaded", "gradient_checkpointing": True}),
        },
        low_bit_profile={
            "low_vram": ProfilePatchSpec(
                {
                    "weight_compression_preset": "stable_backbone_int8",
                    "weight_compression_target": "backbone",
                    "weight_compression_format": "torchao_int8",
                },
                ("SDXL low-bit policy keeps compression on frozen U-Net/backbone weights.",),
            ),
        },
        advanced_optimizer_aggressive="lora_plus",
        adapter_init_aggressive=ProfilePatchSpec(
            {
                "adapter_init_strategy": "pissa",
                "pissa_enabled": True,
                "use_pissa": True,
                "pissa_init_iters": 1,
                "pissa_svd_algo": "rsvd",
                "pissa_export_mode": "lora_compatible",
            },
            ("PiSSA is available as an explicit convergence opt-in for SDXL LoRA.",),
        ),
    ),
    "sd15": FamilyAccelerationSpec(
        family="sd15",
        attention_safe="sdpa",
        attention_balanced="sdpa",
        attention_aggressive="sdpa",
        compile_aggressive=CompilePolicySpec(
            runtime="compile_cache",
            shape_strategy="fixed_pad",
            target_strategy="block",
            extra_patch=dict(_STATIC_COMPILE_GUARDS),
            notes=("SD 1.5 reuses the U-Net block compile contract.",),
        ),
        cache_base={"cache_latents": True},
        cache_profile={
            "low_vram": ProfilePatchSpec({"cache_latents_to_disk": True}),
        },
        text_cache_profiles=frozenset({"balanced", "aggressive", "low_vram"}),
        text_cache_to_disk_profiles=frozenset({"low_vram"}),
        checkpoint_profile={
            "low_vram": ProfilePatchSpec({"checkpoint_policy": "offloaded", "gradient_checkpointing": True}),
        },
        low_bit_profile={
            "low_vram": ProfilePatchSpec(
                {
                    "weight_compression_preset": "stable_backbone_int8",
                    "weight_compression_target": "backbone",
                    "weight_compression_format": "torchao_int8",
                },
                ("SD 1.5 low-bit policy keeps compression on frozen U-Net/backbone weights.",),
            ),
        },
        advanced_optimizer_aggressive="lora_plus",
        adapter_init_aggressive=ProfilePatchSpec(
            {
                "adapter_init_strategy": "pissa",
                "pissa_enabled": True,
                "use_pissa": True,
                "pissa_init_iters": 1,
                "pissa_svd_algo": "rsvd",
                "pissa_export_mode": "lora_compatible",
            },
            ("PiSSA is available as an explicit convergence opt-in for SD 1.5 LoRA.",),
        ),
    ),
    "anima": FamilyAccelerationSpec(
        family="anima",
        attention_aggressive="flash2",
        compile_aggressive=CompilePolicySpec(
            runtime="compile_cache",
            shape_strategy="token_flatten",
            target_strategy="inner_forward",
            extra_patch={**_STATIC_COMPILE_GUARDS, "native_token_bucket_compile": True},
            notes=("Anima prefers token-bucket static shapes and inner-forward DiT targets.",),
        ),
        cache_base={"native_cache_mode": "cache_first", "anima_cached_training": True},
        cache_profile={
            "balanced": ProfilePatchSpec({"cached_collate_mode": "pad_sequence"}),
            "aggressive": ProfilePatchSpec({"cached_collate_mode": "pad_sequence"}),
            "low_vram": ProfilePatchSpec({"cached_collate_mode": "pad_sequence"}),
        },
        runtime_profile={"aggressive": "anima_fast", "low_vram": "anima_low_vram"},
        checkpoint_profile={
            "balanced": ProfilePatchSpec({"checkpoint_policy": "full"}),
            "aggressive": ProfilePatchSpec({"checkpoint_policy": "full"}),
            "low_vram": ProfilePatchSpec({"checkpoint_policy": "offloaded", "gradient_checkpointing": True}),
        },
        lora_recompute={"balanced": "auto", "aggressive": "auto", "low_vram": "on"},
        low_bit_profile={
            "low_vram": ProfilePatchSpec(
                {
                    "weight_compression_preset": "stable_backbone_int8",
                    "weight_compression_target": "backbone",
                    "weight_compression_format": "torchao_int8",
                },
                ("Anima low-bit policy targets frozen native DiT linear weights; cache/compile remain mutually exclusive.",),
            ),
        },
        advanced_optimizer_aggressive="lora_plus",
        adapter_init_aggressive=ProfilePatchSpec(
            {
                "adapter_init_strategy": "pissa",
                "pissa_enabled": True,
                "use_pissa": True,
                "pissa_init_iters": 1,
                "pissa_svd_algo": "rsvd",
                "pissa_export_mode": "lora_compatible",
            },
            ("PiSSA is available as an explicit convergence opt-in for Anima native DiT LoRA.",),
        ),
    ),
    "newbie": FamilyAccelerationSpec(
        family="newbie",
        attention_aggressive="flash2",
        compile_aggressive=CompilePolicySpec(
            runtime="compile_cache",
            shape_strategy="token_flatten",
            target_strategy="inner_forward",
            extra_patch={**_STATIC_COMPILE_GUARDS, "native_token_bucket_compile": True},
            notes=("Newbie compile is only recommended with cache-first token buckets.",),
        ),
        cache_base={"native_cache_mode": "cache_first", "use_cache": True},
        cache_profile={
            "balanced": ProfilePatchSpec({"cached_collate_mode": "pad_sequence"}),
            "aggressive": ProfilePatchSpec({"cached_collate_mode": "pad_sequence"}),
            "low_vram": ProfilePatchSpec({"cached_collate_mode": "pad_sequence"}),
        },
        runtime_profile={"aggressive": "aggressive"},
        checkpoint_profile={
            "balanced": ProfilePatchSpec({"checkpoint_policy": "full"}),
            "aggressive": ProfilePatchSpec({"checkpoint_policy": "full"}),
            "low_vram": ProfilePatchSpec({"checkpoint_policy": "offloaded", "gradient_checkpointing": True}),
        },
        lora_recompute={"balanced": "auto", "aggressive": "auto", "low_vram": "on"},
        low_bit_profile={
            "low_vram": ProfilePatchSpec(
                {
                    "weight_compression_preset": "stable_backbone_int8",
                    "weight_compression_target": "backbone",
                    "weight_compression_format": "torchao_int8",
                },
                ("Newbie low-bit policy targets frozen transformer/backbone linear weights.",),
            ),
        },
        advanced_optimizer_aggressive="lora_plus",
        adapter_init_aggressive=ProfilePatchSpec(
            {
                "adapter_init_strategy": "pissa",
                "pissa_enabled": True,
                "use_pissa": True,
                "pissa_init_iters": 1,
                "pissa_svd_algo": "rsvd",
                "pissa_export_mode": "lora_compatible",
            },
            ("PiSSA is available as an explicit convergence opt-in for Newbie LoRA.",),
        ),
    ),
    "flux": FamilyAccelerationSpec(
        family="flux",
        attention_safe="sdpa",
        attention_balanced="sdpa",
        attention_aggressive="sdpa",
        compile_aggressive=CompilePolicySpec(
            runtime="compile_cache",
            shape_strategy="fixed_pad",
            target_strategy="block",
            extra_patch=dict(_STATIC_COMPILE_GUARDS),
            notes=("Flux keeps compile conservative: per-block transformer targets with fixed padding.",),
        ),
        low_bit_notes=("Flux low-bit compression remains experimental and should be A/B checked per model.",),
        low_bit_profile={
            "low_vram": ProfilePatchSpec(
                {
                    "weight_compression_preset": "stable_backbone_int8",
                    "weight_compression_target": "backbone",
                    "weight_compression_format": "torchao_int8",
                },
                ("Flux low-bit policy is limited to frozen transformer/backbone linear weights.",),
            ),
        },
        cache_base={"cache_latents": True},
        cache_profile={
            "balanced": ProfilePatchSpec({"cache_latents_to_disk": True}),
            "aggressive": ProfilePatchSpec({"cache_latents_to_disk": True}),
            "low_vram": ProfilePatchSpec({"cache_latents_to_disk": True}),
        },
        text_cache_profiles=frozenset({"balanced", "aggressive", "low_vram"}),
        text_cache_to_disk_profiles=frozenset({"balanced", "aggressive", "low_vram"}),
        checkpoint_profile={
            "low_vram": ProfilePatchSpec({"checkpoint_policy": "offloaded", "gradient_checkpointing": True}),
        },
        lora_recompute={"balanced": "auto", "aggressive": "auto", "low_vram": "on"},
        advanced_optimizer_aggressive="lora_plus",
        adapter_init_aggressive=ProfilePatchSpec(
            {
                "adapter_init_strategy": "pissa",
                "pissa_enabled": True,
                "use_pissa": True,
                "pissa_init_iters": 1,
                "pissa_svd_algo": "rsvd",
                "pissa_export_mode": "lora_compatible",
            },
            ("PiSSA is available as an explicit convergence opt-in for Flux LoRA.",),
        ),
    ),
    "unknown": FamilyAccelerationSpec(
        family="unknown",
        optimizer_aggressive="foreach_adamw",
        low_bit_notes=("Unknown routes keep low-bit compression manual unless explicitly allowed.",),
        cache_base={"cache_latents": True},
    ),
}


def family_acceleration_spec(family: Any) -> FamilyAccelerationSpec:
    key = str(family or "unknown").strip().lower().replace("-", "_")
    return _FAMILY_MATRIX.get(key, _FAMILY_MATRIX["unknown"])


def optimizer_backend_for(family: Any, profile: str) -> str:
    spec = family_acceleration_spec(family)
    normalized = str(profile or "").strip().lower()
    if normalized == "aggressive":
        return spec.optimizer_aggressive
    if normalized in {"safe", "balanced", "low_vram"}:
        return spec.optimizer_balanced
    return ""


def attention_backend_for(family: Any, profile: str) -> str:
    spec = family_acceleration_spec(family)
    normalized = str(profile or "").strip().lower()
    if normalized == "aggressive":
        return spec.attention_aggressive
    if normalized == "balanced":
        return spec.attention_balanced
    if normalized in {"safe", "low_vram"}:
        return spec.attention_safe
    return ""


def compile_policy_for(family: Any, profile: str) -> CompilePolicySpec | None:
    spec = family_acceleration_spec(family)
    if str(profile or "").strip().lower() != "aggressive":
        return None
    return spec.compile_aggressive


def low_bit_preset_for(family: Any) -> str:
    return family_acceleration_spec(family).low_bit_preset


def low_bit_patch_for(family: Any, profile: str) -> ProfilePatchSpec:
    spec = family_acceleration_spec(family)
    normalized = str(profile or "").strip().lower()
    profile_spec = spec.low_bit_profile.get(normalized)
    if profile_spec:
        return profile_spec
    if spec.low_bit_preset and spec.low_bit_preset != "off":
        return ProfilePatchSpec({"weight_compression_preset": spec.low_bit_preset}, spec.low_bit_notes)
    return ProfilePatchSpec(notes=spec.low_bit_notes)


def cache_patch_for(family: Any, profile: str) -> ProfilePatchSpec:
    spec = family_acceleration_spec(family)
    normalized = str(profile or "").strip().lower()
    patch = dict(spec.cache_base)
    notes: list[str] = []
    profile_spec = spec.cache_profile.get(normalized)
    if profile_spec:
        patch.update(profile_spec.patch)
        notes.extend(profile_spec.notes)
    return ProfilePatchSpec(patch=patch, notes=tuple(notes))


def should_cache_text_encoder_outputs(family: Any, profile: str) -> bool:
    spec = family_acceleration_spec(family)
    return str(profile or "").strip().lower() in spec.text_cache_profiles


def should_cache_text_encoder_outputs_to_disk(family: Any, profile: str) -> bool:
    spec = family_acceleration_spec(family)
    return str(profile or "").strip().lower() in spec.text_cache_to_disk_profiles


def runtime_profile_for(family: Any, profile: str) -> str:
    spec = family_acceleration_spec(family)
    return spec.runtime_profile.get(str(profile or "").strip().lower(), "")


def checkpoint_patch_for(family: Any, profile: str) -> ProfilePatchSpec:
    spec = family_acceleration_spec(family)
    return spec.checkpoint_profile.get(str(profile or "").strip().lower(), ProfilePatchSpec())


def lora_recompute_mode_for(family: Any, profile: str) -> str:
    spec = family_acceleration_spec(family)
    return spec.lora_recompute.get(str(profile or "").strip().lower(), "")


def advanced_optimizer_strategy_for(family: Any, profile: str) -> str:
    spec = family_acceleration_spec(family)
    if str(profile or "").strip().lower() == "aggressive":
        return spec.advanced_optimizer_aggressive
    return ""


def adapter_init_patch_for(family: Any, profile: str) -> ProfilePatchSpec:
    spec = family_acceleration_spec(family)
    if str(profile or "").strip().lower() == "aggressive" and spec.adapter_init_aggressive:
        return spec.adapter_init_aggressive
    return ProfilePatchSpec()


def acceleration_matrix_summary_for(family: Any) -> dict[str, Any]:
    spec = family_acceleration_spec(family)
    compile_spec = spec.compile_aggressive
    return {
        "family": spec.family,
        "attention": {
            "safe": spec.attention_safe,
            "balanced": spec.attention_balanced,
            "aggressive": spec.attention_aggressive,
        },
        "optimizer": {
            "balanced": spec.optimizer_balanced,
            "aggressive": spec.optimizer_aggressive,
            "advanced_aggressive": spec.advanced_optimizer_aggressive,
        },
        "compile_aggressive": (
            {
                "runtime": compile_spec.runtime,
                "shape_strategy": compile_spec.shape_strategy,
                "target_strategy": compile_spec.target_strategy,
                "extra_patch": dict(compile_spec.extra_patch),
            }
            if compile_spec
            else None
        ),
        "low_bit": {
            "default_preset": spec.low_bit_preset,
            "profiles": {key: dict(value.patch) for key, value in spec.low_bit_profile.items()},
            "notes": list(spec.low_bit_notes),
        },
        "cache_profiles": {key: dict(value.patch) for key, value in spec.cache_profile.items()},
        "runtime_profile": dict(spec.runtime_profile),
        "checkpoint_profile": {key: dict(value.patch) for key, value in spec.checkpoint_profile.items()},
        "lora_recompute": dict(spec.lora_recompute),
    }


__all__ = [
    "acceleration_matrix_summary_for",
    "CompilePolicySpec",
    "FamilyAccelerationSpec",
    "ProfilePatchSpec",
    "adapter_init_patch_for",
    "advanced_optimizer_strategy_for",
    "attention_backend_for",
    "cache_patch_for",
    "checkpoint_patch_for",
    "compile_policy_for",
    "family_acceleration_spec",
    "lora_recompute_mode_for",
    "low_bit_patch_for",
    "low_bit_preset_for",
    "optimizer_backend_for",
    "runtime_profile_for",
    "should_cache_text_encoder_outputs",
    "should_cache_text_encoder_outputs_to_disk",
]
