"""Compile runtime profile helpers shared by route-specific trainers."""

from __future__ import annotations

from typing import Any, Mapping


def _decision_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        return [str(item) for item in list(value)]
    except TypeError:
        return [str(value)]


def _decision_value(decision: Any, key: str, default: Any = None) -> Any:
    if isinstance(decision, Mapping):
        return decision.get(key, default)
    return getattr(decision, key, default)


def compile_contract_to_dict(decision: Any) -> dict[str, Any]:
    if decision is None:
        return {}
    return {
        "route": str(_decision_value(decision, "route", "") or ""),
        "requested": str(_decision_value(decision, "requested", "") or ""),
        "resolved": str(_decision_value(decision, "resolved", "") or ""),
        "compile_active": bool(_decision_value(decision, "compile_active", False)),
        "static_drop_last": bool(_decision_value(decision, "static_drop_last", False)),
        "cache_first_required": bool(_decision_value(decision, "cache_first_required", False)),
        "reasons": _decision_list(_decision_value(decision, "reasons", [])),
        "warnings": _decision_list(_decision_value(decision, "warnings", [])),
    }


def build_compile_runtime_profile(
    *,
    config: Any,
    runtime_plan: Any = None,
    compile_contract: Any = None,
    model_arch: str = "",
    target_profile: Mapping[str, Any] | None = None,
    applied: bool | None = None,
    compiled_targets: int | None = None,
    compile_kind: str = "",
    source: str = "trainer_runtime",
    skip_reason: str = "",
    error: str = "",
) -> dict[str, Any]:
    target = dict(target_profile or {})
    contract = compile_contract_to_dict(compile_contract)
    route = str(model_arch or contract.get("route") or target.get("route") or getattr(config, "model_arch", "") or "")
    resolved = str(contract.get("resolved") or target.get("resolved") or "")
    target_compiled = target.get("compiled_targets")
    if compiled_targets is None:
        try:
            compiled_targets = int(target_compiled or 0)
        except (TypeError, ValueError):
            compiled_targets = 0
    if applied is None:
        applied = bool(target.get("applied", False) or compiled_targets > 0 or resolved == "full_core")
    kind = compile_kind or ("per_block" if resolved == "per_block" else resolved or "off")
    profile: dict[str, Any] = {
        "source": source,
        "route": route,
        "compile_kind": kind,
        "requested_runtime": str(getattr(config, "compile_runtime", "") or ""),
        "torch_compile": bool(getattr(runtime_plan, "torch_compile", getattr(config, "torch_compile", False))),
        "torch_compile_scope": str(getattr(runtime_plan, "torch_compile_scope", getattr(config, "torch_compile_scope", "")) or ""),
        "anima_compile_scope": str(getattr(runtime_plan, "anima_compile_scope", getattr(config, "anima_compile_scope", "")) or ""),
        "torch_compile_backend": str(getattr(runtime_plan, "torch_compile_backend", getattr(config, "torch_compile_backend", "")) or ""),
        "torch_compile_mode": str(getattr(runtime_plan, "torch_compile_mode", getattr(config, "torch_compile_mode", "")) or ""),
        "torch_compile_dynamic": bool(getattr(runtime_plan, "torch_compile_dynamic", getattr(config, "torch_compile_dynamic", False))),
        "torch_compile_fullgraph": bool(getattr(runtime_plan, "torch_compile_fullgraph", getattr(config, "torch_compile_fullgraph", False))),
        "compile_shape_strategy": str(getattr(runtime_plan, "compile_shape_strategy", getattr(config, "compile_shape_strategy", "auto")) or "auto"),
        "compile_target_strategy": str(getattr(runtime_plan, "compile_target_strategy", getattr(config, "compile_target_strategy", "auto")) or "auto"),
        "contract": contract,
        "target_profile": target,
        "candidate_targets": list(target.get("candidate_targets") or []),
        "eligible_targets": int(target.get("eligible_targets", 0) or 0),
        "compiled_targets": int(compiled_targets or 0),
        "applied": bool(applied),
    }
    if skip_reason:
        profile["skip_reason"] = str(skip_reason)
    elif resolved == "off":
        reasons = contract.get("reasons") or []
        profile["skip_reason"] = str(reasons[-1]) if reasons else "compile_contract_resolved_off"
    if error:
        profile["error"] = str(error)
        profile["applied"] = False
    return profile


__all__ = ["build_compile_runtime_profile", "compile_contract_to_dict"]
