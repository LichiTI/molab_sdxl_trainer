"""Lightweight S/A/B-tier training advisor.

This module consolidates the low-risk parts of the old REapp advisor ideas into
one pre-training report: VRAM estimate, SafeGuard status, LR finder entrypoint,
dataset purifier readiness, and text-encoder residency hints. It intentionally
has no heavyweight model imports and does not mutate training config.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

try:
    from .vram_guardrails import classify_vram_hardware_profile
except Exception:
    try:
        from core.lulynx_trainer.vram_guardrails import classify_vram_hardware_profile
    except Exception:
        module_path = Path(__file__).resolve().with_name("vram_guardrails.py")
        spec = importlib.util.spec_from_file_location("_lulynx_vram_guardrails_for_advisor", module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        classify_vram_hardware_profile = module.classify_vram_hardware_profile

try:
    from .sdxl_lora_low_vram_advisor import recommend_sdxl_lora_low_vram_profile
except Exception:
    try:
        from core.lulynx_trainer.sdxl_lora_low_vram_advisor import recommend_sdxl_lora_low_vram_profile
    except Exception:
        module_path = Path(__file__).resolve().with_name("sdxl_lora_low_vram_advisor.py")
        spec = importlib.util.spec_from_file_location("_lulynx_sdxl_low_vram_for_advisor", module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        recommend_sdxl_lora_low_vram_profile = module.recommend_sdxl_lora_low_vram_profile


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_MASK_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DIT_STREAMING_AUTO_MIN_PARAMETER_COUNT = 262_144
DIT_STREAMING_TARGET_COLD_PARAM_FRACTION = 0.60
DIT_STREAMING_RESIDENT_EDGE_BLOCKS = 1


@dataclass
class AdvisorFinding:
    code: str
    severity: str
    message: str
    suggestion: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class TrainingAdvisorReport:
    summary: Dict[str, Any]
    vram: Dict[str, Any]
    safeguard: Dict[str, Any]
    lr_finder: Dict[str, Any]
    dataset: Dict[str, Any]
    text_encoder: Dict[str, Any]
    compile_token: Dict[str, Any]
    a_tier: Dict[str, Any]
    b_tier: Dict[str, Any]
    findings: List[AdvisorFinding] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "vram": self.vram,
            "safeguard": self.safeguard,
            "lr_finder": self.lr_finder,
            "dataset": self.dataset,
            "text_encoder": self.text_encoder,
            "compile_token": self.compile_token,
            "a_tier": self.a_tier,
            "b_tier": self.b_tier,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, str) and "," in value:
            value = value.split(",", 1)[0]
        return int(float(value))
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _split_csv_tokens(value: Any) -> List[str]:
    raw = str(value or "")
    tokens = []
    for part in raw.replace("\n", ",").split(","):
        item = part.strip()
        if item:
            tokens.append(item)
    return tokens


def _path_exists(value: Any) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    try:
        return Path(raw).exists()
    except OSError:
        return False


def _resolution_pair(config: Any) -> Tuple[int, int]:
    width = _as_int(_get(config, "width", 0), 0)
    height = _as_int(_get(config, "height", 0), 0)
    resolution = _get(config, "resolution", 1024)
    if width > 0 and height > 0:
        return width, height
    if isinstance(resolution, str) and "," in resolution:
        parts = [part.strip() for part in resolution.split(",")]
        if len(parts) >= 2:
            return max(_as_int(parts[0], 1024), 1), max(_as_int(parts[1], 1024), 1)
    side = max(_as_int(resolution, 1024), 1)
    return side, side


def _detect_vram_gb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return float(props.total_memory) / (1024 ** 3)
    except Exception:
        pass
    return None


def _model_family(config: Any) -> str:
    for key in ("model_arch", "architecture", "model_type"):
        raw_value = _get(config, key, "")
        value = str(getattr(raw_value, "value", raw_value) or "").strip().lower()
        if value:
            if "." in value:
                value = value.rsplit(".", 1)[-1]
            return value
    path = str(_get(config, "pretrained_model_name_or_path", "") or "").lower()
    if "anima" in path:
        return "anima"
    if "newbie" in path or "gemma" in path:
        return "newbie"
    if "flux" in path:
        return "flux"
    if "sdxl" in path or "xl" in path:
        return "sdxl"
    return "sdxl"


def _dit_residency_keys(family: str) -> Optional[Tuple[str, str, str, str, str]]:
    if family == "anima":
        return (
            "anima_block_residency",
            "anima_block_residency_min_params",
            "anima_block_checkpointing",
            "anima_block_prefetch",
            "anima_block_prefetch_depth",
        )
    if family == "newbie":
        return (
            "newbie_block_residency",
            "newbie_block_residency_min_params",
            "newbie_block_checkpointing",
            "newbie_block_prefetch",
            "newbie_block_prefetch_depth",
        )
    return None


def _normalize_dit_residency_mode(value: Any) -> str:
    mode = str(value or "resident").strip().lower().replace("-", "_")
    aliases = {
        "off": "resident",
        "gpu": "resident",
        "none": "resident",
        "balanced": "streaming_offload",
        "hot": "streaming_offload",
        "hotaware": "streaming_offload",
        "hot_aware": "streaming_offload",
        "hot_aware_cpu_pinned": "streaming_offload",
        "streaming": "streaming_offload",
        "streaming_cpu_offload": "streaming_offload",
        "streaming_pinned": "streaming_offload",
        "steaming": "streaming_offload",
        "steaming_offload": "streaming_offload",
        "cpu_pinned": "block_cpu_pinned",
        "block_pinned": "block_cpu_pinned",
        "blocks_cpu_pinned": "block_cpu_pinned",
        "linear_cpu_pinned": "block_cpu_pinned",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in {"resident", "streaming_offload", "block_cpu_pinned"} else "resident"


def _dit_runtime_profile(config: Any, *, family: str, width: int, height: int, batch: int, safety: str) -> Dict[str, Any]:
    keys = _dit_residency_keys(family)
    if keys is None:
        return {"available": False, "family": family}

    residency_key, min_params_key, checkpoint_key, prefetch_key, prefetch_depth_key = keys
    mode = _normalize_dit_residency_mode(_get(config, residency_key, "resident"))
    requested_min_params = max(_as_int(_get(config, min_params_key, 0), 0), 0)
    auto_min_parameter_count = mode == "streaming_offload" and requested_min_params == 0
    effective_min_params = DIT_STREAMING_AUTO_MIN_PARAMETER_COUNT if auto_min_parameter_count else requested_min_params
    checkpointing = _as_bool(_get(config, checkpoint_key, False))
    prefetch_enabled = _as_bool(_get(config, prefetch_key, False))
    prefetch_depth = max(_as_int(_get(config, prefetch_depth_key, 1), 1), 0)
    megapixels = (width * height) / (1024 * 1024)
    high_resolution = megapixels >= 1.0
    large_batch = batch >= 4
    fixed_visual_key = f"{family}_fixed_visual_tokens"
    fixed_visual_tokens = max(_as_int(_get(config, fixed_visual_key, 0), 0), 0)
    high_visual_tokens = fixed_visual_tokens >= 4096 or (fixed_visual_tokens <= 0 and high_resolution)
    full_token_resident_pressure = family == "anima" and mode == "resident" and high_resolution and high_visual_tokens
    checkpoint_recommended = (
        mode in {"streaming_offload", "block_cpu_pinned"} or full_token_resident_pressure
    ) and (high_resolution or high_visual_tokens)
    checkpoint_missing = checkpoint_recommended and not checkpointing
    risk = safety in {"danger", "tight"} or (high_resolution and large_batch) or full_token_resident_pressure

    recommendation = mode
    if risk and mode == "resident":
        recommendation = "streaming_offload"
    elif safety == "danger" and mode == "streaming_offload":
        recommendation = "block_cpu_pinned"
    elif mode == "block_cpu_pinned" and not risk:
        recommendation = "streaming_offload"

    strategy = "resident"
    if mode == "streaming_offload":
        strategy = "hot_aware_streaming_auto_threshold" if auto_min_parameter_count else "hot_aware_streaming"
    elif mode == "block_cpu_pinned":
        strategy = "block_cpu_pinned_all_frozen"

    return {
        "available": True,
        "family": family,
        "residency_key": residency_key,
        "min_params_key": min_params_key,
        "checkpoint_key": checkpoint_key,
        "prefetch_key": prefetch_key,
        "prefetch_depth_key": prefetch_depth_key,
        "mode": mode,
        "strategy": strategy,
        "requested_min_parameter_count": requested_min_params,
        "min_parameter_count": effective_min_params,
        "auto_min_parameter_count": auto_min_parameter_count,
        "auto_threshold_floor_parameter_count": DIT_STREAMING_AUTO_MIN_PARAMETER_COUNT,
        "auto_threshold_target_cold_param_fraction": DIT_STREAMING_TARGET_COLD_PARAM_FRACTION,
        "resident_edge_blocks": DIT_STREAMING_RESIDENT_EDGE_BLOCKS if mode == "streaming_offload" else 0,
        "planner_scope": "config_estimate",
        "checkpointing": checkpointing,
        "checkpoint_recommended": checkpoint_recommended,
        "checkpoint_missing": checkpoint_missing,
        "prefetch_enabled": prefetch_enabled,
        "prefetch_depth": prefetch_depth,
        "prefetch_available": mode == "streaming_offload",
        "prefetch_note": (
            "async prefetch is enabled for Streaming Offload; it tries to overlap CPU-pinned frozen Linear H2D copies with the current block."
            if mode == "streaming_offload" and prefetch_enabled
            else (
                "async prefetch is available for Streaming Offload, but remains opt-in while throughput impact is benchmarked."
                if mode == "streaming_offload"
                else ""
            )
        ),
        "fixed_visual_tokens": fixed_visual_tokens,
        "high_resolution": high_resolution,
        "high_visual_tokens": high_visual_tokens,
        "large_batch": large_batch,
        "full_token_resident_pressure": full_token_resident_pressure,
        "risk": risk,
        "recommendation": recommendation,
        "benchmark_basis": (
            "local 2026-05-27 quick 1024 probe: Anima resident oversubscribed 16GB WDDM/shared memory, "
            "while streaming_offload stayed below 5GB for 1bs/2bs"
            if full_token_resident_pressure
            else ""
        ),
        "notes": [
            "resident is the speed path when VRAM is comfortable.",
            "streaming_offload uses the native hot-aware planner: edge blocks and attention/modulation paths stay resident while cold large Linear weights can stream from CPU.",
            "block prefetch is optional and report-only in Advisor; benchmark it before long production runs.",
            "min params 0 means the runtime model-graph planner will auto-select a threshold from cold layer sizes.",
            "block_cpu_pinned plus checkpointing is the emergency low-VRAM path and can be much slower.",
        ],
    }


def _config_namespace_copy(config: Any, keys: Tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(**{key: _get(config, key, None) for key in keys})


def inspect_compile_token_shape(config: Any) -> Dict[str, Any]:
    """Report route-aware compile/token-shape readiness without mutating config."""

    family = _model_family(config)
    if family not in {"anima", "newbie", "sdxl"}:
        return {"available": False, "family": family, "reason": "compile token profile is only tracked for SDXL/Anima/Newbie"}

    compile_keys = (
        "torch_compile",
        "torch_compile_scope",
        "anima_compile_scope",
        "compile_contract_strict",
        "compile_require_cache_first",
        "compile_static_shape_drop_last",
        "compile_anima_full_core_enabled",
        "native_token_bucket_compile",
        "gradient_checkpointing",
        "blocks_to_swap",
        "swap_granularity",
        "swap_count",
        "swap_ratio",
        "newbie_safe_fallback",
        "anima_cached_training",
        "use_cache",
        "anima_fixed_text_tokens",
        "anima_fixed_visual_tokens",
        "newbie_fixed_text_tokens",
        "newbie_fixed_visual_tokens",
    )
    cfg = _config_namespace_copy(config, compile_keys)
    plan = SimpleNamespace(
        torch_compile=_as_bool(_get(config, "torch_compile", False)),
        torch_compile_scope=str(_get(config, "torch_compile_scope", "") or ""),
        anima_compile_scope=str(_get(config, "anima_compile_scope", "") or ""),
    )

    try:
        from .compile_contract import resolve_compile_contract
    except Exception:
        try:
            from core.lulynx_trainer.compile_contract import resolve_compile_contract
        except Exception:
            module_path = Path(__file__).resolve().with_name("compile_contract.py")
            spec = importlib.util.spec_from_file_location("_lulynx_compile_contract_for_advisor", module_path)
            if spec is None or spec.loader is None:
                raise
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            resolve_compile_contract = module.resolve_compile_contract

    decision = resolve_compile_contract(cfg, plan, model_arch=family)
    fixed_text = max(_as_int(_get(config, f"{family}_fixed_text_tokens", 0), 0), 0) if family in {"anima", "newbie"} else 0
    fixed_visual = max(_as_int(_get(config, f"{family}_fixed_visual_tokens", 0), 0), 0) if family in {"anima", "newbie"} else 0
    native_bucket_compile = _as_bool(_get(config, "native_token_bucket_compile", True))
    if family == "anima":
        cache_first = _as_bool(_get(config, "anima_cached_training", True))
    elif family == "newbie":
        cache_first = _as_bool(_get(config, "use_cache", False))
    else:
        cache_first = False
    no_pad_visual_bucket = family in {"anima", "newbie"} and fixed_visual <= 0 and native_bucket_compile and cache_first
    token_safe = decision.compile_active and (
        family == "sdxl"
        or (
            fixed_text > 0
            and (fixed_visual > 0 or no_pad_visual_bucket)
            and (not decision.cache_first_required or cache_first)
        )
    )
    status = "off"
    if decision.compile_active:
        status = "ready" if token_safe else "warning"
    if decision.resolved == "off" and decision.requested != "off":
        status = "disabled"

    notes: List[str] = []
    if no_pad_visual_bucket and decision.compile_active:
        notes.append(f"{family} per-block compile can use no-pad cached visual token buckets.")
    if decision.cache_first_required:
        notes.append("cache-first training is required for this static-shape compile route.")
    if family in {"anima", "newbie"} and fixed_text <= 0 and decision.compile_active:
        notes.append(f"{family}_fixed_text_tokens is required before treating compile as stable.")

    return {
        "available": True,
        "family": family,
        "requested": decision.requested,
        "resolved": decision.resolved,
        "compile_active": decision.compile_active,
        "status": status,
        "static_drop_last": decision.static_drop_last,
        "cache_first_required": decision.cache_first_required,
        "cache_first": cache_first,
        "fixed_text_tokens": fixed_text,
        "fixed_visual_tokens": fixed_visual,
        "native_token_bucket_compile": native_bucket_compile,
        "no_pad_visual_bucket": no_pad_visual_bucket,
        "token_shape_safe": token_safe,
        "reasons": list(decision.reasons),
        "warnings": list(decision.warnings),
        "notes": notes,
    }


def estimate_training_vram(config: Any, available_vram_gb: Optional[float] = None) -> Dict[str, Any]:
    family = _model_family(config)
    width, height = _resolution_pair(config)
    batch = max(_as_int(_get(config, "batch_size", _get(config, "train_batch_size", 1)), 1), 1)
    rank = max(_as_int(_get(config, "network_dim", _get(config, "lora_rank", 32)), 32), 1)
    precision = str(_get(config, "mixed_precision", "bf16") or "bf16").lower()
    gradient_checkpointing = _as_bool(_get(config, "gradient_checkpointing", False))
    blocks_to_swap = max(_as_int(_get(config, "blocks_to_swap", 0), 0), 0)

    base_by_family = {
        "sd15": 4.0,
        "sd1.5": 4.0,
        "sdxl": 7.5,
        "flux": 13.0,
        "anima": 12.0,
        "newbie": 12.0,
    }
    base_gb = base_by_family.get(family, 8.0)
    megapixels = (width * height) / (1024 * 1024)
    activation_gb = 1.35 * megapixels * batch
    if precision in {"fp16", "bf16", "float16", "bfloat16"}:
        activation_gb *= 0.62
    if gradient_checkpointing:
        activation_gb *= 0.52
    if _as_bool(_get(config, "vae_slicing", False)):
        activation_gb *= 0.95
    if _as_bool(_get(config, "attention_slicing", False)):
        activation_gb *= 0.92
    dit_runtime = _dit_runtime_profile(config, family=family, width=width, height=height, batch=batch, safety="unknown")
    if dit_runtime.get("available") and dit_runtime.get("checkpointing"):
        activation_gb *= 0.78

    adapter_gb = 0.012 * rank
    swap_savings_gb = 0.28 * blocks_to_swap
    if _as_bool(_get(config, "module_offload_enabled", False)):
        ratio = max(_as_float(_get(config, "module_offload_ratio", 0.0), 0.0), 0.0)
        swap_savings_gb += min(base_gb * ratio / 100.0 * 0.45, base_gb * 0.4)
    if _as_bool(_get(config, "vram_swap_to_ram", False)):
        swap_savings_gb += max(adapter_gb * 0.6, 0.05)
    if dit_runtime.get("available"):
        mode = str(dit_runtime.get("mode") or "resident")
        if mode == "streaming_offload":
            swap_savings_gb += min(base_gb * 0.18, 2.2)
        elif mode == "block_cpu_pinned":
            swap_savings_gb += min(base_gb * 0.34, 4.2)

    estimated = max(base_gb + activation_gb + adapter_gb - swap_savings_gb, 1.0)
    available = available_vram_gb if available_vram_gb is not None else _detect_vram_gb()
    usage_ratio = estimated / available if available and available > 0 else None
    safety = "unknown"
    if usage_ratio is not None:
        if usage_ratio >= 1.0:
            safety = "danger"
        elif usage_ratio >= 0.9:
            safety = "tight"
        elif usage_ratio >= 0.78:
            safety = "watch"
        else:
            safety = "safe"

    hardware_profile = classify_vram_hardware_profile(
        {
            "available_gb": round(available, 2) if available else None,
            "estimated_gb": round(estimated, 2),
            "usage_ratio": round(usage_ratio, 3) if usage_ratio is not None else None,
        }
    )
    low_vram_profile_advice = recommend_sdxl_lora_low_vram_profile(
        config,
        family=family,
        safety=safety,
        width=width,
        height=height,
        batch=batch,
        hardware_profile=hardware_profile,
        available_gb=available,
    )

    recommendations: List[str] = []
    if dit_runtime.get("available") and dit_runtime.get("checkpoint_missing"):
        recommendations.append(f"enable {family} block checkpointing")
    if low_vram_profile_advice.get("should_patch"):
        recommendations.append(f"set low_vram_profile={low_vram_profile_advice['target']}")
    if safety in {"danger", "tight"}:
        if not gradient_checkpointing:
            recommendations.append("enable gradient_checkpointing")
        if batch > 1:
            recommendations.append(f"reduce batch_size to {max(batch - 1, 1)}")
        dit_runtime = _dit_runtime_profile(config, family=family, width=width, height=height, batch=batch, safety=safety)
        if dit_runtime.get("available"):
            current_mode = str(dit_runtime.get("mode") or "resident")
            if current_mode == "resident":
                recommendations.append(f"enable {family} streaming_offload")
            elif current_mode == "streaming_offload" and safety == "danger":
                recommendations.append(f"try {family} block_cpu_pinned plus block checkpointing")
        elif (
            not low_vram_profile_advice.get("available")
            and not _as_bool(_get(config, "module_offload_enabled", False))
        ):
            recommendations.append("enable module_offload balanced profile")
        if rank > 16:
            recommendations.append(f"try network_dim {max(rank // 2, 8)}")

    dit_runtime = _dit_runtime_profile(config, family=family, width=width, height=height, batch=batch, safety=safety)
    action_plan = build_vram_action_plan(
        config,
        safety,
        batch=batch,
        rank=rank,
        family=family,
        hardware_profile=hardware_profile,
        available_gb=available,
        low_vram_profile_advice=low_vram_profile_advice,
    )
    runtime_probe = build_runtime_probe_hint(config, family=family)

    return {
        "family": family,
        "resolution": {"width": width, "height": height},
        "batch_size": batch,
        "network_dim": rank,
        "mixed_precision": precision,
        "estimated_gb": round(estimated, 2),
        "available_gb": round(available, 2) if available else None,
        "usage_ratio": round(usage_ratio, 3) if usage_ratio is not None else None,
        "safety": safety,
        "device_tier": hardware_profile.get("device_tier"),
        "low_vram_target": hardware_profile.get("low_vram_target"),
        "constrained_vram_target": hardware_profile.get("constrained_vram_target"),
        "shared_vram_detection": hardware_profile.get("shared_vram_detection"),
        "recommendations": recommendations,
        "recommended_config_patch": action_plan["recommended_config_patch"],
        "action_plan": action_plan,
        "hardware_profile": hardware_profile,
        "low_vram_profile_advice": low_vram_profile_advice,
        "dit_runtime": dit_runtime,
        "runtime_probe": runtime_probe,
    }


def build_vram_action_plan(
    config: Any,
    safety: str,
    *,
    batch: int,
    rank: int,
    family: str,
    hardware_profile: Optional[Dict[str, Any]] = None,
    available_gb: Optional[float] = None,
    low_vram_profile_advice: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    steps: List[Dict[str, Any]] = []
    severity = "high" if safety == "danger" else "medium" if safety == "tight" else "low"
    train_te = _as_bool(_get(config, "train_text_encoder", False))
    width, height = _resolution_pair(config)
    dit_runtime = _dit_runtime_profile(config, family=family, width=width, height=height, batch=batch, safety=safety)
    if low_vram_profile_advice is None:
        low_vram_profile_advice = recommend_sdxl_lora_low_vram_profile(
            config,
            family=family,
            safety=safety,
            width=width,
            height=height,
            batch=batch,
            hardware_profile=hardware_profile,
            available_gb=available_gb,
        )

    def _add_dit_residency_step(priority: int) -> bool:
        if not dit_runtime.get("available"):
            return False
        residency_key = str(dit_runtime.get("residency_key") or "")
        checkpoint_key = str(dit_runtime.get("checkpoint_key") or "")
        mode = str(dit_runtime.get("mode") or "resident")
        recommendation = str(dit_runtime.get("recommendation") or mode)
        if not residency_key or recommendation == mode:
            if checkpoint_key and dit_runtime.get("checkpoint_missing"):
                patch[checkpoint_key] = True
                steps.append({
                    "key": checkpoint_key,
                    "priority": priority,
                    "reason": "recompute DiT blocks during backward; Streaming Offload alone only reduces frozen-weight residency",
                })
                return True
            return False
        patch[residency_key] = recommendation
        reason = (
            "native hot-aware Streaming Offload keeps edge/hot paths resident "
            "and streams cold frozen Linear weights before reducing batch size"
        )
        if recommendation == "block_cpu_pinned":
            reason = "emergency DiT low-VRAM mode; expect significant throughput loss"
        steps.append({"key": residency_key, "priority": priority, "reason": reason})
        if checkpoint_key and not bool(dit_runtime.get("checkpointing")):
            patch[checkpoint_key] = True
            steps.append({
                "key": checkpoint_key,
                "priority": priority + 1,
                "reason": "recompute DiT blocks during backward to lower activation peak",
            })
        if recommendation == "streaming_offload":
            prefetch_key = str(dit_runtime.get("prefetch_key") or "")
            prefetch_depth_key = str(dit_runtime.get("prefetch_depth_key") or "")
            if prefetch_key and not _as_bool(_get(config, prefetch_key, False)):
                patch[prefetch_key] = True
                steps.append({"key": prefetch_key, "priority": priority + 2, "reason": "overlap warm Streaming Offload copies with current block compute"})
            if prefetch_depth_key and _as_int(_get(config, prefetch_depth_key, 0), 0) <= 0:
                patch[prefetch_depth_key] = 1
            if not _as_bool(_get(config, "sparse_swap_enabled", False)):
                patch["sparse_swap_enabled"] = True
                steps.append({"key": "sparse_swap_enabled", "priority": priority + 3, "reason": "split cold DiT weights into warm prefetch and cold on-demand buckets"})
            if _as_float(_get(config, "sparse_swap_warm_fraction", 0.0), 0.0) <= 0.0:
                patch["sparse_swap_warm_fraction"] = 0.35
            if (
                _as_bool(_get(config, "enhanced_protection_mode", False))
                and str(_get(config, "pcie_transfer_format", "off") or "off").strip().lower() in {"", "off", "none", "disabled"}
            ):
                patch["pcie_transfer_format"] = "fp8_e4m3"
                steps.append({"key": "pcie_transfer_format", "priority": priority + 4, "reason": "enhanced protection mode allows experimental FP8 PCIe transfer fallback"})
        return True

    if safety not in {"danger", "tight"} and (dit_runtime.get("risk") or dit_runtime.get("checkpoint_missing")):
        _add_dit_residency_step(2)

    if low_vram_profile_advice.get("should_patch"):
        patch["low_vram_profile"] = low_vram_profile_advice["target"]
        steps.append({
            "key": "low_vram_profile",
            "priority": 1,
            "reason": (
                "SDXL LoRA low-VRAM profile combines cache, gradient checkpointing, component residency, "
                "staged resolution, and weight swap before falling back to generic module offload"
            ),
        })

    if safety in {"danger", "tight"}:
        if not _as_bool(_get(config, "gradient_checkpointing", False)):
            patch["gradient_checkpointing"] = True
            steps.append({"key": "gradient_checkpointing", "priority": 1, "reason": "largest low-risk activation memory reduction"})
        if (
            not low_vram_profile_advice.get("available")
            and not _add_dit_residency_step(2)
            and not _as_bool(_get(config, "module_offload_enabled", False))
        ):
            patch.update({
                "module_offload_enabled": True,
                "module_offload_profile_enabled": True,
                "module_offload_profile": "balanced",
            })
            steps.append({"key": "module_offload", "priority": 2, "reason": "moves frozen base modules out of VRAM between uses"})
        if batch > 1:
            patch["batch_size"] = max(batch - 1, 1)
            steps.append({"key": "batch_size", "priority": 3, "reason": "directly reduces activation footprint"})
        if family in {"sdxl", "anima", "newbie"} and not train_te and not _as_bool(_get(config, "cache_text_encoder_outputs", False)):
            patch["cache_text_encoder_outputs"] = True
            steps.append({"key": "cache_text_encoder_outputs", "priority": 4, "reason": "lets text encoder stay colder during most training steps"})
        if rank > 16:
            patch["network_dim"] = max(rank // 2, 8)
            steps.append({"key": "network_dim", "priority": 5, "reason": "reduces adapter parameters if quality budget allows"})

    return {
        "severity": severity,
        "recommended_config_patch": patch,
        "steps": sorted(steps, key=lambda item: item["priority"]),
        "notes": ["Report-only advice; trainer config is not mutated automatically."],
    }


def build_runtime_probe_hint(config: Any, *, family: Optional[str] = None) -> Dict[str, Any]:
    family = family or _model_family(config)
    if family != "anima":
        return {
            "available": False,
            "reason": f"runtime probe is currently implemented for anima, not {family}",
        }

    py = r"backend\env\python-flashattention\python.exe"
    script = r"backend\core\lulynx_trainer\anima_runtime_vram_probe.py"
    return {
        "available": True,
        "script": script,
        "command_cpu": f"{py} {script} --device cpu --dtype fp32 --blocks 1 --cases off,stable_backbone_int8",
        "command_cuda": f"{py} {script} --device cuda --dtype bf16 --blocks 4 --cases off,stable_backbone_int8",
        "notes": [
            "CPU command verifies logic and gradient coverage without requiring VRAM.",
            "CUDA command is the useful pre-flight measurement before a long Anima run.",
        ],
    }


def _try_read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        pass

    try:
        data = path.read_bytes()[:32]
    except OSError:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return None


def _sha256_file(path: Path, max_bytes: int = 64 * 1024 * 1024) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        read_total = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                read_total += len(chunk)
                if read_total > max_bytes:
                    return None
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def inspect_dataset(config: Any, max_scan: int = 5000) -> Dict[str, Any]:
    data_dir = str(_get(config, "train_data_dir", _get(config, "data_dir", "")) or "").strip()
    caption_ext = str(_get(config, "caption_extension", ".txt") or ".txt")
    if not caption_ext.startswith("."):
        caption_ext = "." + caption_ext
    out: Dict[str, Any] = {
        "path": data_dir,
        "exists": False,
        "image_count": 0,
        "caption_count": 0,
        "missing_caption_count": 0,
        "orphan_caption_count": 0,
        "unreadable_image_count": 0,
        "duplicate_image_count": 0,
        "small_image_count": 0,
        "mask_count": 0,
        "missing_mask_count": 0,
        "empty_caption_count": 0,
        "average_tags_per_caption": 0.0,
        "long_caption_count": 0,
        "caption_extension": caption_ext,
        "purifier_ready": False,
        "purifier_mode": "report_only",
        "sample_issues": [],
        "notes": [],
    }
    if not data_dir:
        out["notes"].append("No train_data_dir configured")
        return out
    root = Path(data_dir)
    if not root.exists():
        out["notes"].append("Dataset path does not exist")
        return out
    out["exists"] = True

    images: List[Path] = []
    captions: List[Path] = []
    scanned_files = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        scanned_files += 1
        if scanned_files > max_scan:
            out["notes"].append(f"Scan capped at {max_scan} files")
            break
        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTS:
            images.append(path)
        elif suffix == caption_ext.lower():
            captions.append(path)

    image_keys = {(path.parent, path.stem) for path in images}
    caption_keys = {(path.parent, path.stem) for path in captions}
    missing = [path for path in images if (path.parent, path.stem) not in caption_keys]
    orphan = [path for path in captions if (path.parent, path.stem) not in image_keys]

    hashes: Dict[str, Path] = {}
    duplicate_samples: List[str] = []
    unreadable_samples: List[str] = []
    small_samples: List[str] = []
    unreadable = 0
    small = 0
    duplicates = 0
    mask_count = 0
    missing_mask_samples: List[str] = []
    tag_total = 0
    captioned_with_text = 0
    empty_caption = 0
    long_caption = 0
    caption_by_key = {(path.parent, path.stem): path for path in captions}
    for image_path in images:
        size = _try_read_image_size(image_path)
        if size is None:
            unreadable += 1
            if len(unreadable_samples) < 8:
                unreadable_samples.append(str(image_path.relative_to(root)))
        else:
            width, height = size
            if min(width, height) < 384:
                small += 1
                if len(small_samples) < 8:
                    small_samples.append(str(image_path.relative_to(root)))
        mask_path = _find_mask_sidecar(image_path)
        if mask_path is not None:
            mask_count += 1
        elif len(missing_mask_samples) < 8:
            missing_mask_samples.append(str(image_path.relative_to(root)))
        cap_path = caption_by_key.get((image_path.parent, image_path.stem))
        if cap_path is not None:
            caption_text = _read_text_sidecar(cap_path)
            if caption_text:
                tags = _split_caption_tags(caption_text)
                tag_total += len(tags)
                captioned_with_text += 1
                if len(tags) >= 75 or len(caption_text) >= 600:
                    long_caption += 1
            else:
                empty_caption += 1

        digest = _sha256_file(image_path)
        if digest:
            if digest in hashes:
                duplicates += 1
                if len(duplicate_samples) < 8:
                    duplicate_samples.append(str(image_path.relative_to(root)))
            else:
                hashes[digest] = image_path

    out["image_count"] = len(images)
    out["caption_count"] = len(images) - len(missing)
    out["missing_caption_count"] = len(missing)
    out["orphan_caption_count"] = len(orphan)
    out["unreadable_image_count"] = unreadable
    out["duplicate_image_count"] = duplicates
    out["small_image_count"] = small
    out["mask_count"] = mask_count
    out["missing_mask_count"] = max(len(images) - mask_count, 0)
    out["empty_caption_count"] = empty_caption
    out["average_tags_per_caption"] = round(tag_total / captioned_with_text, 3) if captioned_with_text else 0.0
    out["long_caption_count"] = long_caption
    out["purifier_ready"] = len(images) >= 10
    for label, samples in (("missing_caption", missing), ("orphan_caption", orphan)):
        for sample in samples[:8]:
            out["sample_issues"].append({"type": label, "path": str(sample.relative_to(root))})
    for sample in missing_mask_samples:
        out["sample_issues"].append({"type": "missing_mask", "path": sample})

    for label, samples in (("unreadable_image", unreadable_samples), ("duplicate_image", duplicate_samples), ("small_image", small_samples)):
        for sample in samples:
            out["sample_issues"].append({"type": label, "path": sample})

    if len(images) < 10:
        out["notes"].append("Dataset purifier spectral analysis needs at least 10 images")
    else:
        out["notes"].append("Dataset purifier should use auto/rSVD spectral mode for larger datasets")
    if images and out["missing_caption_count"]:
        out["notes"].append("Some images are missing caption sidecars")
    if out["orphan_caption_count"]:
        out["notes"].append("Some caption files do not match an image stem")
    if out["duplicate_image_count"]:
        out["notes"].append("Duplicate image files detected by sha256")
    if out["unreadable_image_count"]:
        out["notes"].append("Some image files could not be decoded")
    if out["empty_caption_count"]:
        out["notes"].append("Some caption files are empty or unreadable")
    if out["long_caption_count"]:
        out["notes"].append("Some captions are long enough to stress token limits")
    return out


def _find_mask_sidecar(image_path: Path) -> Optional[Path]:
    for suffix in ("_mask", ".mask", "-mask"):
        for ext in _MASK_EXTS:
            candidate = image_path.with_name(f"{image_path.stem}{suffix}{ext}")
            if candidate.exists():
                return candidate
    return None


def _read_text_sidecar(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "cp932", "latin-1"):
        try:
            return path.read_text(encoding=encoding).strip()
        except (UnicodeDecodeError, OSError):
            continue
    return ""


def _split_caption_tags(caption: str) -> List[str]:
    if "," in caption:
        return [part.strip() for part in caption.split(",") if part.strip()]
    return [part.strip() for part in caption.split() if part.strip()]


def inspect_safeguard(config: Any) -> Dict[str, Any]:
    bad_sample_mode = str(_get(config, "so_bad_sample_mode", "report") or "report").strip().lower()
    if bad_sample_mode not in {"report", "move", "quarantine"}:
        bad_sample_mode = "report"
    return {
        "nan_detection": _as_bool(_get(config, "so_enable_nan_detection", True)),
        "loss_spike_detection": _as_bool(_get(config, "so_enable_loss_spike_detection", True)),
        "lr_deadlock_detection": _as_bool(_get(config, "so_enable_lr_deadlock_detection", True)),
        "auto_recovery": _as_bool(_get(config, "so_enable_auto_recovery", True)),
        "bad_sample_culling": _as_bool(_get(config, "so_enable_bad_sample_culling", False)),
        "bad_sample_mode": bad_sample_mode,
        "bad_sample_report_name": str(_get(config, "so_bad_sample_report_name", "safeguard_events.jsonl") or "safeguard_events.jsonl"),
        "bad_sample_max_reported": _as_int(_get(config, "so_bad_sample_max_reported", 32), 32),
        "loss_spike_threshold": _as_float(_get(config, "so_loss_spike_threshold", 10.0), 10.0),
        "max_nan_count": _as_int(_get(config, "so_max_nan_count", 3), 3),
    }


def inspect_lr_finder(config: Any) -> Dict[str, Any]:
    lr = _as_float(_get(config, "learning_rate", 1e-4), 1e-4)
    optimizer = str(_get(config, "optimizer", _get(config, "optimizer_type", "")) or "").lower()
    return {
        "available": True,
        "entrypoint": "backend/core/entry_lr_find.py",
        "current_learning_rate": lr,
        "suggested_range": {"start_lr": max(lr / 1000.0, 1e-7), "end_lr": min(max(lr * 100.0, 1e-5), 1e-1), "steps": 100},
        "notes": ["Use before a long run; LR Finder loads the model and runs a short range test."],
        "optimizer": optimizer,
    }


def inspect_text_encoder(config: Any) -> Dict[str, Any]:
    train_te = _as_bool(_get(config, "train_text_encoder", False))
    cache_te = _as_bool(_get(config, "cache_text_encoder_outputs", False))
    semantic = _as_bool(_get(config, "semantic_tuner_enabled", False))
    family = _model_family(config)
    active = train_te or cache_te or semantic or family in {"sdxl", "newbie", "anima"}
    suggestions: List[str] = []
    if train_te:
        suggestions.append("TE training raises VRAM use; keep SafeGuard and VRAM monitoring enabled")
    elif cache_te:
        suggestions.append("Text encoder cache can allow CPU residency after validation")
    elif family in {"sdxl", "newbie", "anima"}:
        suggestions.append("TE manager can monitor/carry cached encoder outputs for lower VRAM modes")
    return {
        "manager_available": True,
        "expected_active": active,
        "train_text_encoder": train_te,
        "cache_text_encoder_outputs": cache_te,
        "semantic_tuner_enabled": semantic,
        "suggestions": suggestions,
    }



def _build_smart_rank_advice(rank: int, enabled: bool, config: Any) -> Dict[str, Any]:
    min_rank = max(_as_int(_get(config, "smart_rank_min", 4), 4), 1)
    max_rank = max(_as_int(_get(config, "smart_rank_max", 128), 128), min_rank)
    try:
        from core.training_components.smart_rank import advise_rank

        advice = advise_rank(
            current_rank=rank,
            inferred_rank=None,
            stable_rank=_as_float(_get(config, "stable_rank_metric", 0.0), 0.0) or None,
            min_rank=min_rank,
            max_rank=max_rank,
        ).to_dict()
    except Exception:
        if rank >= 96:
            suggested = max(min_rank, min(max_rank, int(math.ceil((rank * 0.5) / 4.0) * 4)))
            severity = "watch"
            reason = "high rank heuristic"
        elif rank >= 64:
            suggested = max(min_rank, min(max_rank, int(math.ceil((rank * 0.75) / 4.0) * 4)))
            severity = "info"
            reason = "medium-high rank heuristic"
        else:
            suggested = max(min_rank, min(rank, max_rank))
            severity = "ok"
            reason = "current rank looks modest"
        advice = {
            "current_rank": rank,
            "suggested_rank": suggested,
            "min_rank": min_rank,
            "max_rank": max_rank,
            "severity": severity,
            "reason": reason,
            "source": "fallback",
            "confidence": 0.25,
            "notes": ["Report-only fallback; no training behavior changed."],
        }
    return {
        "enabled": enabled,
        "mode": "active" if enabled else "advisor_only",
        "interval": max(_as_int(_get(config, "smart_rank_interval", 50), 50), 1),
        "min_rank": min_rank,
        "max_rank": max_rank,
        "current_rank": rank,
        "advice": advice,
        "recommendation": "Prefer report/advisor mode before dynamic rank pruning because optimizer state must stay coherent.",
    }


def _build_memory_vortex_fusion_advice(config: Any, module_offload_enabled: bool, vram_swap_enabled: bool) -> Dict[str, Any]:
    profile = str(_get(config, "module_offload_profile", "custom") or "custom")
    ratio = max(_as_int(_get(config, "module_offload_ratio", 0), 0), 0)
    prefetch_enabled = _as_bool(_get(config, "module_offload_prefetch_enabled", False))
    min_param_mb = max(_as_float(_get(config, "module_offload_min_param_mb", 0.0), 0.0), 0.0)
    include_patterns = str(_get(config, "module_offload_include_patterns", "") or "")
    exclude_patterns = str(_get(config, "module_offload_exclude_patterns", "") or "")

    risk_flags: List[str] = []
    degrade_paths: List[str] = []
    if module_offload_enabled and vram_swap_enabled:
        risk_flags.append("overlapping_cpu_offload_strategies")
        degrade_paths.append("disable vram_swap_to_ram or module_offload before long runs")
    if prefetch_enabled:
        risk_flags.append("experimental_prefetch")
        degrade_paths.append("prefetch auto-degrades to normal module_offload when CUDA/order assumptions fail")
    if module_offload_enabled and ratio == 0 and profile == "custom":
        risk_flags.append("zero_custom_ratio")
        degrade_paths.append("set a profile or non-zero ratio; otherwise no modules are selected")

    return {
        "status": "covered_by_module_offload" if module_offload_enabled else "available_as_advice",
        "module_offload_enabled": module_offload_enabled,
        "vram_swap_to_ram": vram_swap_enabled,
        "profile": profile,
        "ratio": ratio,
        "prefetch_enabled": prefetch_enabled,
        "min_param_mb": min_param_mb,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "absorbed_ideas": [
            "single offload authority instead of parallel Vortex system",
            "runtime H2D materialize statistics",
            "top-module transfer accounting",
            "safe prefetch degradation reason reporting",
            "conflict detection with swap/compile/distributed modes",
        ],
        "risk_flags": risk_flags,
        "degrade_paths": degrade_paths,
        "recommendation": "Keep Vortex-style ideas inside module_offload scheduling/statistics; do not enable overlapping swap systems.",
    }

def inspect_a_tier_features(config: Any, dataset: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    family = _model_family(config)
    rank = max(_as_int(_get(config, "network_dim", _get(config, "lora_rank", 32)), 32), 1)
    smart_rank_enabled = _as_bool(_get(config, "smart_rank_enabled", _get(config, "smart_rank", False)))
    auto_controller_enabled = _as_bool(_get(config, "auto_controller_enabled", False)) or _as_bool(_get(config, "ac_enabled", False))
    ema_enabled = _as_bool(_get(config, "ema_use_ema", _get(config, "use_ema", False)))
    masked_loss_enabled = _as_bool(_get(config, "masked_loss", False))
    alpha_mask_enabled = _as_bool(_get(config, "alpha_mask", False))
    block_weight_enabled = _as_bool(_get(config, "bw_enable", False)) or bool(str(_get(config, "bw_preset", "") or "").strip())
    module_offload_enabled = _as_bool(_get(config, "module_offload_enabled", False))
    vram_swap_enabled = _as_bool(_get(config, "vram_swap_to_ram", False))
    enable_bucket = _as_bool(_get(config, "enable_bucket", True))
    shuffle_caption = _as_bool(_get(config, "shuffle_caption", True))
    keep_tokens = max(_as_int(_get(config, "keep_tokens", 0), 0), 0)
    caption_variants_enabled = _as_bool(_get(config, "caption_variants_enabled", False))

    smart_rank_advice = _build_smart_rank_advice(rank, smart_rank_enabled, config)
    vortex_advice = _build_memory_vortex_fusion_advice(config, module_offload_enabled, vram_swap_enabled)
    dataset = dataset or {}
    mask_count = int(dataset.get("mask_count", 0) or 0)
    image_count = int(dataset.get("image_count", 0) or 0)
    mask_coverage = (mask_count / image_count) if image_count > 0 else 0.0
    caption_avg_tags = float(dataset.get("average_tags_per_caption", 0.0) or 0.0)
    caption_empty = int(dataset.get("empty_caption_count", 0) or 0)
    caption_long = int(dataset.get("long_caption_count", 0) or 0)
    bucket_min = max(_as_int(_get(config, "min_bucket_reso", 256), 256), 1)
    bucket_max = max(_as_int(_get(config, "max_bucket_reso", 2048), 2048), 1)
    bucket_step = max(_as_int(_get(config, "bucket_reso_steps", 64), 64), 1)
    bucket_warnings: List[str] = []
    if bucket_min > bucket_max:
        bucket_warnings.append("min_reso_gt_max_reso")
    if bucket_step % 8 != 0:
        bucket_warnings.append("step_not_multiple_of_8")
    if bucket_max > 4096:
        bucket_warnings.append("very_large_bucket_max")

    ac_axes = []
    if _as_bool(_get(config, "auto_freeze_te", False)) or _as_bool(_get(config, "ac_enable_auto_te_freeze", False)):
        ac_axes.append("auto_te_freeze")
    if _as_bool(_get(config, "smart_early_stop", False)) or _as_bool(_get(config, "ac_enable_smart_early_stopping", False)):
        ac_axes.append("smart_early_stop")
    if _as_bool(_get(config, "smart_lr_decay", False)) or _as_bool(_get(config, "ac_enable_smart_lr_decay", False)):
        ac_axes.append("smart_lr_decay")
    ac_risk = "high" if len(ac_axes) >= 2 else "medium" if ac_axes else "low"
    ema_decay = _as_float(_get(config, "ema_decay", 0.999), 0.999)
    ema_update_after = max(_as_int(_get(config, "ema_update_after_step", 100), 100), 0)
    ema_update_every = max(_as_int(_get(config, "ema_update_every", 1), 1), 1)
    ema_warnings: List[str] = []
    if ema_enabled and not (0.0 < ema_decay < 1.0):
        ema_warnings.append("decay_out_of_range")
    if ema_enabled and ema_update_every > 100:
        ema_warnings.append("very_sparse_updates")
    if ema_enabled and _as_bool(_get(config, "semantic_tuner_enabled", False)):
        ema_warnings.append("semantic_tuner_skips_native_ema")
    bw_in = str(_get(config, "bw_in_weights", "") or "")
    bw_mid = str(_get(config, "bw_mid_weight", "") or "")
    bw_out = str(_get(config, "bw_out_weights", "") or "")
    bw_zero_freeze_requested = any(part.strip() in {"0", "0.0"} for part in (bw_in + "," + bw_mid + "," + bw_out).split(","))
    bw_warnings: List[str] = []
    if block_weight_enabled and not (str(_get(config, "bw_preset", "") or "").strip() or bw_in or bw_mid or bw_out):
        bw_warnings.append("enabled_without_weights")
    if bw_zero_freeze_requested:
        bw_warnings.append("zero_weight_freezes_layers")
    modules: Dict[str, Any] = {
        "memory_vortex_fusion": vortex_advice,
        "block_weight": {
            "enabled": block_weight_enabled,
            "preset": str(_get(config, "bw_preset", "") or ""),
            "in_weights": bw_in,
            "mid_weight": bw_mid,
            "out_weights": bw_out,
            "zero_freeze_requested": bw_zero_freeze_requested,
            "warnings": bw_warnings,
            "recommendation": "Use existing block_weight manager; validate adapter matrix coverage before freezing layers.",
        },
        "smart_rank": smart_rank_advice,
        "auto_controller": {
            "enabled": auto_controller_enabled,
            "auto_te_freeze": "auto_te_freeze" in ac_axes,
            "smart_early_stop": "smart_early_stop" in ac_axes,
            "smart_lr_decay": "smart_lr_decay" in ac_axes,
            "active_axes": ac_axes,
            "automation_risk": ac_risk,
            "warmup_steps": max(_as_int(_get(config, "ac_warmup_steps", 100), 100), 0),
            "lr_decay_factor": _as_float(_get(config, "ac_lr_decay_factor", 0.5), 0.5),
            "recommendation": "Enable one automation axis at a time; keep SafeGuard on while validating LR/early-stop behavior.",
        },
        "ema": {
            "enabled": ema_enabled,
            "decay": ema_decay,
            "update_after_step": ema_update_after,
            "update_every": ema_update_every,
            "warmup": _as_bool(_get(config, "ema_use_ema_warmup", False)),
            "reset_on_resume": _as_bool(_get(config, "ema_reset_on_resume", False)),
            "warnings": ema_warnings,
            "recommendation": "Existing adapter EMA is available; enable for longer LoRA runs after confirming save/load workflow.",
        },
        "masked_loss": {
            "enabled": masked_loss_enabled,
            "alpha_mask": alpha_mask_enabled,
            "strict": _as_bool(_get(config, "strict_masked_loss", False)),
            "mask_count": mask_count,
            "mask_coverage": round(mask_coverage, 4),
            "missing_mask_count": int(dataset.get("missing_mask_count", 0) or 0),
            "recommendation": "Useful for character/local training; strict mode is safer when every sample should carry a mask.",
        },
        "smart_caption": {
            "enabled": shuffle_caption or keep_tokens > 0 or caption_variants_enabled,
            "shuffle_caption": shuffle_caption,
            "keep_tokens": keep_tokens,
            "caption_variants_enabled": caption_variants_enabled,
            "average_tags_per_caption": caption_avg_tags,
            "empty_caption_count": caption_empty,
            "long_caption_count": caption_long,
            "recommendation": "Keep identity/style anchor tags protected with keep_tokens before aggressive shuffle/dropout.",
        },
        "dataset_bucket": {
            "enabled": enable_bucket,
            "mode": str(_get(config, "bucket_selection_mode", "aspect") or "aspect"),
            "min_reso": bucket_min,
            "max_reso": bucket_max,
            "step": bucket_step,
            "warnings": bucket_warnings,
            "recommendation": "Use existing bucket logic; advisor should flag invalid min/max/step rather than adding a parallel loader.",
        },
    }

    recommended_config_patch: Dict[str, Any] = {}
    notes: List[str] = ["A-tier fusion report is advisory only; no training behavior is changed automatically."]
    if smart_rank_advice["advice"]["severity"] in {"watch", "info"} and not smart_rank_enabled:
        recommended_config_patch["smart_rank_enabled"] = False
        notes.append("SmartRank has report-only advice; review suggested_rank before enabling dynamic pruning.")
    if masked_loss_enabled and not alpha_mask_enabled:
        notes.append("masked_loss is enabled; ensure dataset batches actually provide loss_masks or alpha masks.")
    if auto_controller_enabled and not (_as_bool(_get(config, "so_enable_loss_spike_detection", True)) and _as_bool(_get(config, "so_enable_nan_detection", True))):
        notes.append("AutoController works best with SafeGuard NaN and loss-spike detection enabled.")
    if module_offload_enabled and vram_swap_enabled:
        notes.append("module_offload and vram_swap_to_ram overlap; validate residency statistics before long runs.")
    if enable_bucket and modules["dataset_bucket"]["min_reso"] > modules["dataset_bucket"]["max_reso"]:
        notes.append("Bucket min resolution is larger than max resolution.")

    if ema_warnings:
        notes.append("EMA configuration has warnings: " + ", ".join(ema_warnings))
    if bw_warnings:
        notes.append("BlockWeight configuration has warnings: " + ", ".join(bw_warnings))
    if auto_controller_enabled and ac_risk == "high":
        notes.append("AutoController has multiple automation axes enabled; validate one axis at a time.")
    active_count = sum(1 for item in modules.values() if bool(item.get("enabled")) or item.get("status") == "covered_by_module_offload")
    return {
        "status": "advisory",
        "active_count": active_count,
        "modules": modules,
        "recommended_config_patch": recommended_config_patch,
        "notes": notes,
        "migration_policy": "merge_with_existing_systems; do not copy REapp modules wholesale",
        "family": family,
    }


def inspect_b_tier_features(config: Any) -> Dict[str, Any]:
    hutchinson_auto_freeze = _as_bool(_get(config, "hutchinson_auto_freeze", False))
    hutchinson_freeze_ratio = min(max(_as_float(_get(config, "hutchinson_freeze_ratio", 0.5), 0.5), 0.0), 1.0)
    hutchinson_probes = max(_as_int(_get(config, "lulynx_hutchinson_probes", 30), 30), 1)
    grad_accumulation_steps = max(_as_int(_get(config, "gradient_accumulation_steps", 1), 1), 1)

    pcgrad_enabled = _as_bool(_get(config, "pcgrad_enabled", False))
    pcgrad_conflict_threshold = _as_float(_get(config, "pcgrad_conflict_threshold", 0.0), 0.0)
    pcgrad_reduction = str(_get(config, "pcgrad_reduction", "mean") or "mean").strip().lower()
    if pcgrad_reduction not in {"mean", "sum"}:
        pcgrad_reduction = "mean"

    geometric_lock_enabled = _as_bool(_get(config, "lulynx_geometric_lock", False))
    proj_dim = max(_as_int(_get(config, "lulynx_proj_dim", 128), 128), 1)
    sparse_freq = max(_as_int(_get(config, "lulynx_manifold_sparse_freq", 1), 1), 1)
    anchor_layers = _split_csv_tokens(_get(config, "lulynx_anchor_layers", ""))

    ghost_replay_enabled = _as_bool(_get(config, "lulynx_ghost_replay", False))
    ghost_path = str(_get(config, "lulynx_ghost_path", "") or "").strip()
    ghost_path_exists = _path_exists(ghost_path)
    ghost_interval = max(_as_int(_get(config, "lulynx_ghost_interval", 100), 100), 1)
    ghost_weight = max(_as_float(_get(config, "lulynx_ghost_weight", 0.05), 0.05), 0.0)
    ghost_fingerprint_report: Dict[str, Any] = {}
    if ghost_path_exists:
        try:
            from ..training_components.ghost_replay import inspect_ghost_fingerprint
            ghost_fingerprint_report = inspect_ghost_fingerprint(ghost_path)
        except Exception as exc:
            ghost_fingerprint_report = {
                "status": "error",
                "errors": [f"failed to inspect fingerprint: {exc}"],
                "warnings": [],
                "timesteps": [],
                "recorded_layer_count": 0,
                "metadata": {},
            }
    ghost_fingerprint_metadata = ghost_fingerprint_report.get("metadata", {}) if isinstance(ghost_fingerprint_report, dict) else {}
    ghost_fingerprint_arch = str(
        ghost_fingerprint_metadata.get("model_arch")
        or ghost_fingerprint_metadata.get("model_family")
        or ""
    ).strip().lower()
    current_family = _model_family(config)

    modules: Dict[str, Any] = {
        "hutchinson_scan": {
            "implemented": True,
            "requested": hutchinson_auto_freeze,
            "train_chain_wired": True,
            "ui_exposed": True,
            "status": "manual_experimental" if hutchinson_auto_freeze else "available_manual",
            "auto_freeze": hutchinson_auto_freeze,
            "freeze_ratio": round(hutchinson_freeze_ratio, 4),
            "num_probes": hutchinson_probes,
            "recommendation": "Hutchinson scan is wired before optimizer creation as a manual auto-freeze/report path. Keep it off unless intentionally testing layer freezing.",
        },
        "pcgrad": {
            "implemented": True,
            "requested": pcgrad_enabled,
            "train_chain_wired": True,
            "ui_exposed": True,
            "status": "manual_experimental" if pcgrad_enabled else "available_manual",
            "conflict_threshold": pcgrad_conflict_threshold,
            "reduction": pcgrad_reduction,
            "stats_available": True,
            "gradient_accumulation_steps": grad_accumulation_steps,
            "recommendation": "PCGrad is wired into the Warehouse training loop as a manual experimental path. It is most useful when gradient_accumulation_steps is greater than 1.",
        },
        "ghost_replay": {
            "implemented": True,
            "requested": ghost_replay_enabled,
            "train_chain_wired": True,
            "ui_exposed": True,
            "status": "manual_experimental" if ghost_replay_enabled else "available_manual",
            "fingerprint_path": ghost_path,
            "fingerprint_configured": bool(ghost_path),
            "fingerprint_exists": ghost_path_exists,
            "fingerprint_readable": bool(ghost_fingerprint_report.get("readable", False)) if ghost_fingerprint_report else False,
            "fingerprint_status": str(ghost_fingerprint_report.get("status", "missing") or "missing"),
            "fingerprint_errors": list(ghost_fingerprint_report.get("errors", []))[:3] if ghost_fingerprint_report else [],
            "fingerprint_warnings": list(ghost_fingerprint_report.get("warnings", []))[:3] if ghost_fingerprint_report else [],
            "fingerprint_timesteps": list(ghost_fingerprint_report.get("timesteps", [])) if ghost_fingerprint_report else [],
            "fingerprint_layer_count": int(ghost_fingerprint_report.get("recorded_layer_count", 0) or 0) if ghost_fingerprint_report else 0,
            "fingerprint_model_arch": ghost_fingerprint_arch,
            "fingerprint_arch_matches_route": (not ghost_fingerprint_arch) or ghost_fingerprint_arch == current_family,
            "interval": ghost_interval,
            "weight": ghost_weight,
            "recommendation": "Ghost Replay is wired through Warehouse feature hooks and safely no-ops when fingerprints or timestep matches are missing.",
        },
        "manifold_constraint": {
            "implemented": True,
            "requested": geometric_lock_enabled,
            "train_chain_wired": True,
            "ui_exposed": True,
            "status": "manual_experimental" if geometric_lock_enabled else "available_manual",
            "wrapper_hooks_available": True,
            "orchestrator_removed": False,
            "proj_dim": proj_dim,
            "sparse_freq": sparse_freq,
            "anchor_layers": anchor_layers,
            "recommendation": "Geometric lock is wired through a Warehouse feature hook runtime; first captured forward becomes the baseline, then sparse weighted loss is applied.",
        },
    }

    active_count = sum(1 for item in modules.values() if item.get("requested"))
    notes = [
        "B-tier items stay manual and default-off; Advisor only reports status and does not auto-enable them.",
        "PCGrad, Hutchinson, Ghost Replay, and Geometric Lock now have Warehouse integration paths with experimental manual controls.",
    ]
    if hutchinson_auto_freeze:
        notes.append("Hutchinson auto-freeze was requested and will run before optimizer creation.")
    if pcgrad_enabled:
        notes.append("PCGrad was requested and is wired into the main loop; expect the clearest effect when gradient_accumulation_steps is greater than 1.")
    if ghost_replay_enabled and not ghost_path:
        notes.append("Ghost Replay was requested without a fingerprint path.")
    elif ghost_replay_enabled and ghost_fingerprint_report:
        if ghost_fingerprint_report.get("status") == "error":
            notes.append("Ghost Replay fingerprint was found but failed structural validation.")
        elif ghost_fingerprint_report.get("warnings"):
            notes.append("Ghost Replay fingerprint loaded with warnings; check compatibility before long runs.")
    if geometric_lock_enabled:
        notes.append("Geometric lock / manifold constraint is configured through Warehouse feature hooks.")

    return {
        "status": "mixed",
        "active_count": active_count,
        "modules": modules,
        "recommended_config_patch": {},
        "notes": notes,
        "migration_policy": "all_b_tier_manual_default_off_cleanroom_experimental",
    }

def build_training_advisor_report(config: Any, available_vram_gb: Optional[float] = None) -> TrainingAdvisorReport:
    findings: List[AdvisorFinding] = []
    vram = estimate_training_vram(config, available_vram_gb=available_vram_gb)
    safeguard = inspect_safeguard(config)
    lr_finder = inspect_lr_finder(config)
    dataset = inspect_dataset(config)
    text_encoder = inspect_text_encoder(config)
    compile_token = inspect_compile_token_shape(config)
    a_tier = inspect_a_tier_features(config, dataset=dataset)
    b_tier = inspect_b_tier_features(config)

    if vram["safety"] in {"danger", "tight"}:
        findings.append(AdvisorFinding("vram_tight", "error" if vram["safety"] == "danger" else "warning", f"Estimated VRAM is {vram['estimated_gb']}GB", "; ".join(vram["recommendations"])))
    elif vram["safety"] == "watch":
        findings.append(AdvisorFinding("vram_watch", "info", "Estimated VRAM is close to the comfort zone", "Keep ResourceManager thresholds enabled."))

    low_vram_profile_advice = vram.get("low_vram_profile_advice", {})
    if isinstance(low_vram_profile_advice, dict) and low_vram_profile_advice.get("should_patch"):
        findings.append(AdvisorFinding(
            "sdxl_lora_low_vram_profile_recommended",
            "warning" if vram.get("safety") in {"danger", "tight"} else "info",
            f"SDXL LoRA low-VRAM profile recommended: {low_vram_profile_advice.get('target')}",
            (
                "Use the SDXL/LoRA low-VRAM tier before generic module offload; "
                "the trainer will combine cache, gradient checkpointing, component residency, staged resolution, and swap guards at startup."
            ),
        ))

    dit_runtime = vram.get("dit_runtime", {}) if isinstance(vram.get("dit_runtime", {}), dict) else {}
    if dit_runtime.get("available"):
        mode = str(dit_runtime.get("mode") or "resident")
        recommendation = str(dit_runtime.get("recommendation") or mode)
        family = str(dit_runtime.get("family") or _model_family(config))
        checkpoint_key = str(dit_runtime.get("checkpoint_key") or "")
        strategy = str(dit_runtime.get("strategy") or "")
        if mode == "streaming_offload":
            if dit_runtime.get("auto_min_parameter_count"):
                findings.append(AdvisorFinding(
                    "dit_streaming_offload_auto_threshold",
                    "info",
                    f"{family} Streaming Offload will use the hot-aware auto threshold planner",
                    (
                        "At model load time the native planner keeps edge blocks and attention/modulation paths resident, "
                        "then chooses a cold Linear threshold from the actual layer sizes."
                    ),
                ))
            else:
                findings.append(AdvisorFinding(
                    "dit_streaming_offload_hot_aware",
                    "info",
                    f"{family} Streaming Offload strategy is {strategy or 'hot_aware_streaming'}",
                    "The runtime planner streams cold frozen Linear weights while keeping hot paths resident.",
                ))
        if (
            checkpoint_key
            and _as_bool(_get(config, "gradient_checkpointing", False))
            and _as_bool(_get(config, checkpoint_key, False))
        ):
            findings.append(AdvisorFinding(
                "dit_checkpointing_no_double_wrap",
                "info",
                f"{family} DiT block checkpointing is controlling the native DiT path",
                "gradient_checkpointing is treated as a compatibility switch here; enabling both does not double-wrap the same DiT blocks.",
            ))
        if recommendation != mode:
            severity = "warning" if dit_runtime.get("risk") or vram.get("safety") in {"danger", "tight"} else "info"
            findings.append(AdvisorFinding(
                "dit_streaming_offload_recommended",
                severity,
                f"{family} residency is {mode}; Advisor recommends {recommendation}",
                "Use native hot-aware Streaming Offload with DiT block checkpointing for 1024px/4096-token runs; reserve Block CPU pinned + checkpointing for emergency low-VRAM runs.",
            ))
        if dit_runtime.get("checkpoint_missing"):
            findings.append(AdvisorFinding(
                "dit_block_checkpointing_recommended",
                "warning",
                f"{family} {mode} needs DiT block checkpointing at this resolution/token budget",
                "Streaming Offload moves frozen weights, but block checkpointing is needed to reduce backward activation peaks.",
            ))
        elif mode == "block_cpu_pinned":
            findings.append(AdvisorFinding(
                "dit_block_cpu_pinned_slow_path",
                "info",
                f"{family} is using Block CPU pinned residency",
                "This is the lowest-VRAM path and can be much slower; switch to Streaming Offload or resident when VRAM allows.",
            ))

    if compile_token.get("available") and compile_token.get("compile_active"):
        compile_family = str(compile_token.get("family") or _model_family(config))
        if compile_token.get("token_shape_safe"):
            if compile_token.get("no_pad_visual_bucket"):
                findings.append(AdvisorFinding(
                    "dit_compile_no_pad_bucket_ready",
                    "info",
                    f"{compile_family} per-block compile can use no-pad visual token buckets",
                    "Cache-first training and fixed text tokens are active, so visual tokens can stay bucketed instead of padded globally.",
                ))
        else:
            findings.append(AdvisorFinding(
                "dit_compile_token_shape_warning",
                "warning",
                f"{compile_family} compile token shape is not fully stable",
                "; ".join(list(compile_token.get("warnings") or [])[:2] or list(compile_token.get("notes") or [])[:2])
                or "Use cache-first training plus fixed text tokens and fixed/no-pad visual token buckets.",
            ))

    if not safeguard["nan_detection"] or not safeguard["loss_spike_detection"]:
        findings.append(AdvisorFinding("safeguard_partial", "warning", "Some SafeGuard checks are disabled", "Enable NaN and loss spike detection for long runs."))
    if safeguard["bad_sample_culling"] and safeguard["bad_sample_mode"] in {"move", "quarantine"}:
        findings.append(AdvisorFinding("safeguard_bad_sample_move", "warning", "SafeGuard bad-sample mode can move dataset files", "Use report mode first when training data should stay untouched."))
    if dataset["exists"] and dataset["image_count"] > 0 and dataset["missing_caption_count"] > 0:
        findings.append(AdvisorFinding("dataset_missing_captions", "warning", f"{dataset['missing_caption_count']} images appear to miss .txt captions", "Run caption/tag audit before training."))
    if not dataset["exists"]:
        findings.append(AdvisorFinding("dataset_missing", "error", "Training dataset path is missing or empty", "Check train_data_dir before launch."))
    if _as_bool(_get(config, "train_text_encoder", False)) and vram["safety"] in {"tight", "danger"}:
        findings.append(AdvisorFinding("te_vram_pressure", "warning", "Text encoder training may push VRAM over the limit", "Consider caching TE outputs or disabling TE training."))
    if a_tier["modules"]["auto_controller"]["enabled"] and not safeguard["nan_detection"]:
        findings.append(AdvisorFinding("auto_controller_without_nan_guard", "warning", "AutoController is enabled while NaN detection is disabled", "Enable SafeGuard NaN detection before automation-heavy runs."))
    if a_tier["modules"]["masked_loss"]["enabled"] and not a_tier["modules"]["masked_loss"]["alpha_mask"]:
        findings.append(AdvisorFinding("masked_loss_needs_masks", "info", "Masked loss is enabled", "Verify batches include loss_masks or enable alpha_mask/strict checks."))
    bucket = a_tier["modules"]["dataset_bucket"]
    if bucket["enabled"] and bucket["min_reso"] > bucket["max_reso"]:
        findings.append(AdvisorFinding("bucket_resolution_invalid", "error", "Bucket min resolution is larger than max resolution", "Fix min_bucket_reso/max_bucket_reso before launch."))
    if b_tier["modules"]["hutchinson_scan"]["requested"]:
        findings.append(AdvisorFinding("hutchinson_enabled_experimental", "info", "Hutchinson auto-freeze is enabled as a manual experimental path", "It runs before optimizer creation, writes a scan report, and freezes the lowest-entropy trainable tensors by ratio."))
    if b_tier["modules"]["pcgrad"]["requested"]:
        pcgrad_message = "PCGrad is enabled and wired into the Warehouse training loop as an experimental manual feature"
        pcgrad_hint = "Keep pcgrad_enabled manual and default-off; it is most effective with gradient_accumulation_steps > 1."
        if b_tier["modules"]["pcgrad"].get("gradient_accumulation_steps", 1) <= 1:
            pcgrad_hint = "PCGrad is active, but gradient_accumulation_steps is 1 so conflict surgery has less room to help. Keep it for targeted experiments."
        findings.append(AdvisorFinding("pcgrad_enabled_experimental", "info", pcgrad_message, pcgrad_hint))
    ghost_replay = b_tier["modules"]["ghost_replay"]
    if ghost_replay["requested"] and not ghost_replay["fingerprint_configured"]:
        findings.append(AdvisorFinding("ghost_replay_missing_fingerprint", "warning", "Ghost Replay is enabled but no fingerprint path is configured", "Generate or select a fingerprint file before treating Ghost Replay as a usable experiment."))
    elif ghost_replay["requested"] and not ghost_replay["fingerprint_exists"]:
        findings.append(AdvisorFinding("ghost_replay_missing_fingerprint", "warning", "Ghost Replay fingerprint path does not exist or is unreadable", "Point lulynx_ghost_path at a valid .lulynx fingerprint file."))
    elif ghost_replay["requested"] and ghost_replay.get("fingerprint_status") == "error":
        findings.append(AdvisorFinding("ghost_replay_invalid_fingerprint", "warning", "Ghost Replay fingerprint exists but failed validation", "; ".join(ghost_replay.get("fingerprint_errors", [])[:2]) or "Re-record the fingerprint file before enabling Ghost Replay."))
    elif ghost_replay["requested"] and not ghost_replay.get("fingerprint_arch_matches_route", True):
        findings.append(AdvisorFinding("ghost_replay_arch_mismatch", "warning", f"Ghost Replay fingerprint targets {ghost_replay.get('fingerprint_model_arch') or 'another route'} while current route is {_model_family(config)}", "Use a fingerprint recorded from the same model family, or expect replay loss to skip most layers."))
    elif ghost_replay["requested"]:
        findings.append(AdvisorFinding("ghost_replay_requested_experimental", "info", "Ghost Replay fingerprint is configured and wired through Warehouse feature hooks", "If no matching layer/timestep fingerprint is found, the training loop safely skips the replay loss."))
        if ghost_replay.get("fingerprint_warnings"):
            findings.append(AdvisorFinding("ghost_replay_fingerprint_warning", "info", "Ghost Replay fingerprint loaded with compatibility warnings", "; ".join(ghost_replay.get("fingerprint_warnings", [])[:2]) or "Inspect the fingerprint metadata before long runs."))
    if b_tier["modules"]["manifold_constraint"]["requested"]:
        findings.append(AdvisorFinding("manifold_constraint_enabled_experimental", "info", "Geometric lock / manifold constraint is enabled as a manual experimental path", "The first captured forward establishes a baseline; later matching features receive the sparse manifold loss."))

    severity_order = {"error": 3, "warning": 2, "info": 1}
    max_severity = max((severity_order.get(item.severity, 0) for item in findings), default=0)
    summary = {
        "status": "error" if max_severity >= 3 else "warning" if max_severity == 2 else "ok",
        "finding_count": len(findings),
        "s_tier_modules": {
            "vram_advisor": True,
            "safeguard": True,
            "lr_finder": True,
            "dataset_purifier": dataset["purifier_ready"],
            "te_manager": True,
            "compile_token": compile_token.get("available", False),
        },
        "a_tier_modules": {
            "memory_vortex_fusion": True,
            "block_weight": True,
            "smart_rank": True,
            "auto_controller": True,
            "ema": True,
            "masked_loss": True,
            "smart_caption": True,
            "dataset_bucket": True,
        },
        "b_tier_modules": {
            "hutchinson_scan": True,
            "pcgrad": True,
            "ghost_replay": True,
            "manifold_constraint": True,
        },
    }
    return TrainingAdvisorReport(summary, vram, safeguard, lr_finder, dataset, text_encoder, compile_token, a_tier, b_tier, findings)


def write_training_advisor_report(config: Any, output_dir: str | Path, filename: str = "training_advisor_report.json") -> Dict[str, Any]:
    report = build_training_advisor_report(config).to_dict()
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / filename).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

