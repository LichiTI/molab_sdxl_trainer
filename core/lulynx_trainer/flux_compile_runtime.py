"""Flux-specific torch.compile runtime bridge."""

from __future__ import annotations

from typing import Any, Callable

from .compile_contract import resolve_compile_contract
from .compile_target_detector import detect_compile_targets
from .runtime_optimizations import apply_per_block_compile, build_runtime_optimization_plan


_ACTIVE_COMPILE_RUNTIMES = {"cache", "compile", "compile_cache", "cudagraph", "compile_cudagraph"}


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _compile_runtime_active(config: Any) -> bool:
    return _normalize(getattr(config, "compile_runtime", "off")) in _ACTIVE_COMPILE_RUNTIMES


def _activate_runtime_intent(config: Any) -> list[str]:
    if not _compile_runtime_active(config) or _boolish(getattr(config, "torch_compile", False), default=False):
        return []
    reasons = ["compile_runtime activated Flux torch_compile"]
    try:
        setattr(config, "torch_compile", True)
    except Exception:
        return ["compile_runtime requested compile, but config is immutable; torch_compile stayed inactive"]
    scope = _normalize(getattr(config, "torch_compile_scope", ""))
    if scope in {"", "auto", "default"}:
        try:
            setattr(config, "torch_compile_scope", "per_block")
        except Exception:
            pass
    return reasons


def _has_streaming_offload_conflict(config: Any) -> bool:
    component = _normalize(getattr(config, "te_vae_offload_strategy", ""))
    transformer = _normalize(getattr(config, "flux_transformer_offload", ""))
    if component not in {"aggressive", "streaming", "streaming_offload"}:
        return False
    return transformer not in {"", "off", "false", "0", "disabled", "none"}


def _count_compiled_reasons(reasons: list[str], start: int) -> int:
    return sum(
        1
        for reason in reasons[start:]
        if reason.startswith("per_block_compile: compiled ")
    )


def apply_flux_compile_runtime(
    config: Any,
    transformer: Any,
    *,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Apply the conservative Flux per-block compile route when requested."""

    activation_reasons = _activate_runtime_intent(config)
    if not (
        _boolish(getattr(config, "torch_compile", False), default=False)
        or _normalize(getattr(config, "torch_compile_scope", "")) not in {"", "off", "false", "0"}
        or _compile_runtime_active(config)
    ):
        return {}

    plan = build_runtime_optimization_plan(config)
    plan.reasons[:0] = activation_reasons
    decision = resolve_compile_contract(config, plan, model_arch="flux")
    candidates = detect_compile_targets(transformer, route="flux", target_strategy=plan.compile_target_strategy)
    profile: dict[str, Any] = {
        "source": "flux_lora_preview",
        "route": decision.route,
        "requested": decision.requested,
        "resolved": decision.resolved,
        "torch_compile": bool(getattr(plan, "torch_compile", False)),
        "torch_compile_scope": str(getattr(plan, "torch_compile_scope", "") or ""),
        "torch_compile_backend": str(getattr(plan, "torch_compile_backend", "") or ""),
        "compile_shape_strategy": str(getattr(plan, "compile_shape_strategy", "") or ""),
        "compile_target_strategy": str(getattr(plan, "compile_target_strategy", "") or ""),
        "candidate_targets": [candidate.path for candidate in candidates],
        "eligible_targets": sum(1 for candidate in candidates if candidate.eligible),
        "compiled_targets": 0,
        "applied": False,
        "static_drop_last": bool(decision.static_drop_last),
        "cache_first_required": bool(decision.cache_first_required),
    }

    if transformer is None:
        plan.warnings.append("Flux compile requested but transformer is unavailable")
    elif not decision.compile_active or not bool(getattr(plan, "torch_compile", False)):
        plan.reasons.append("Flux compile inactive after route contract resolution")
    elif _has_streaming_offload_conflict(config):
        plan.warnings.append("Flux per-block compile skipped because transformer streaming offload is active")
    else:
        reason_start = len(plan.reasons)
        apply_per_block_compile(transformer, plan, route="flux")
        profile["compiled_targets"] = _count_compiled_reasons(plan.reasons, reason_start)
        profile["applied"] = profile["compiled_targets"] > 0

    profile["reasons"] = list(plan.reasons) + list(decision.reasons)
    profile["warnings"] = list(plan.warnings) + list(decision.warnings)
    if log is not None:
        if profile["applied"]:
            log(f"Flux torch.compile applied to {profile['compiled_targets']} transformer blocks.")
        elif profile["warnings"]:
            log(f"Flux torch.compile not applied: {profile['warnings'][-1]}")
    return profile


__all__ = ["apply_flux_compile_runtime"]
