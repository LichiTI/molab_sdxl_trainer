"""Report-only end-to-end shadow matrix for AnimaFactoredAdamW.

P28 is intentionally a scaffold: it records the runtime/request contract for
future end-to-end shadow training evidence without calling native code or
arming dispatch.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

from core.turbocore_anima_factored_adamw_runtime_dispatch_adapter_shadow_scorecard import (
    build_anima_factored_adamw_runtime_dispatch_adapter_shadow_scorecard,
)


OPTIMIZER_KIND = "anima_factored_adamw"
OPTIMIZER_FAMILY = "factored_custom"
FALLBACK_BACKEND = "python_anima_factored_adamw"
MATRIX_KIND = "anima_factored_adamw_e2e_shadow_matrix_v0"
P27_AUDIT = "native_training_performance_p27_audit_v0"
P27_AUDIT_BUILDER = "build_p27_anima_factored_adamw_training_loop_canary_audit"

MATRIX_CASES = (
    {
        "case": "report_only_lora_block_fp32_4096",
        "numel": 4096,
        "param_dtype": "float32",
        "grad_dtype": "float32",
        "shadow_step_count": 0,
    },
    {
        "case": "report_only_lora_block_bf16_4096",
        "numel": 4096,
        "param_dtype": "bfloat16",
        "grad_dtype": "bfloat16",
        "shadow_step_count": 0,
    },
    {
        "case": "report_only_lora_block_fp32_8192",
        "numel": 8192,
        "param_dtype": "float32",
        "grad_dtype": "float32",
        "shadow_step_count": 0,
    },
)


def build_anima_factored_adamw_e2e_shadow_matrix_scorecard(
    *,
    adapter_report: Mapping[str, Any] | None = None,
    p27_audit_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build P28 matrix evidence without executing native shadow training."""

    started = time.perf_counter()
    adapter = _as_dict(
        adapter_report
        or build_anima_factored_adamw_runtime_dispatch_adapter_shadow_scorecard()
    )
    p27 = _normalize_p27_audit(p27_audit_report)
    cases = [_safe_case(case, lambda item=case: _matrix_case(item, p27)) for case in MATRIX_CASES]
    matrix_ready = all(bool(case.get("shadow_matrix_case_ready", False)) for case in cases)
    validations = _validations(adapter, p27, cases, matrix_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([reason for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_anima_factored_adamw_e2e_shadow_matrix_scorecard_v0",
        "gate": "anima_factored_adamw_e2e_shadow_matrix",
        "ok": ready,
        "promotion_ready": False,
        "report_only": True,
        "e2e_shadow_matrix_ready": ready,
        "e2e_shadow_matrix_passed": False,
        "report_only_matrix_scaffold_ready": ready,
        "live_shadow_matrix_executed": False,
        "native_call_performed_by_p28": False,
        "runtime_dispatch_ready": False,
        "runtime_dispatch_not_enabled": True,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "native_shadow_training_mutates_authority": False,
        "matrix_kind": MATRIX_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "p27_dependency": {
            "schema_version": 1,
            "audit": P27_AUDIT,
            "required_builder": P27_AUDIT_BUILDER,
            "builder_name_recorded": True,
            "report_only_dependency_contract": True,
            "native_call_performed_by_p28": False,
            "summary": dict(p27.get("summary") or {}),
        },
        "adapter_summary": dict(adapter.get("summary") or {}),
        "matrix_cases": cases,
        "validations": validations,
        "summary": {
            "case_count": len(cases),
            "report_only_case_count": sum(1 for case in cases if case.get("status") == "report_only"),
            "failed_case_count": sum(1 for case in cases if case.get("status") == "failed"),
            "e2e_shadow_matrix_passed": False,
            "report_only_matrix_scaffold_ready": ready,
            "live_shadow_matrix_executed": False,
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_backend_authoritative": True,
            "native_shadow_training_mutates_authority": False,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
            "p27_audit_builder": P27_AUDIT_BUILDER,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "anima_factored_adamw_live_e2e_shadow_matrix_missing",
                "anima_factored_adamw_canary_rollout_policy_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "replace P28 report-only scaffold with live shadow matrix evidence after P27 lands"
            if ready
            else "fix AnimaFactoredAdamW P28 e2e shadow matrix scaffold blockers"
        ),
        "notes": [
            "P28 records the matrix contract only; no native optimizer update is called.",
            "The Python AnimaFactoredAdamW update remains authoritative.",
            "P27 is referenced by audit builder name so the mainline P27 file can land independently.",
            "Runtime dispatch and training-path dispatch remain disabled by default.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _matrix_case(case: Mapping[str, Any], p27: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case": str(case["case"]),
        "numel": int(case["numel"]),
        "param_dtype": str(case["param_dtype"]),
        "grad_dtype": str(case["grad_dtype"]),
        "shadow_step_count": int(case.get("shadow_step_count", 0)),
        "status": "report_only",
        "ok": True,
        "shadow_matrix_case_ready": True,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "native_shadow_training_mutates_authority": False,
        "native_call_performed": False,
        "kernel_executed": False,
        "runtime_dispatch_not_enabled": True,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
        "p27_audit_builder": P27_AUDIT_BUILDER,
        "p27_dependency_named": _p27_dependency_named(p27),
        "blocked_reasons": [],
    }


def _safe_case(case: Mapping[str, Any], fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = fn()
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        return payload
    except Exception as exc:
        return {
            "schema_version": 1,
            "case": str(case.get("case") or "unknown"),
            "numel": int(case.get("numel") or 0),
            "status": "failed",
            "ok": False,
            "shadow_matrix_case_ready": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"anima_factored_adamw_p28_matrix_case_failed:{type(exc).__name__}"],
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def _validations(
    adapter: Mapping[str, Any],
    p27: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    matrix_ready: bool,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p26_runtime_dispatch_adapter_shadow_ready",
            bool(adapter.get("runtime_dispatch_adapter_shadow_ready", False)),
            "anima_factored_adamw_p26_runtime_dispatch_adapter_shadow_missing",
        ),
        _validation(
            "p27_training_loop_canary_dependency_named",
            _p27_dependency_named(p27),
            "anima_factored_adamw_p27_training_loop_canary_builder_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            bool(adapter.get("fallback_backend_authoritative", False))
            and adapter.get("fallback_backend") in (None, FALLBACK_BACKEND)
            and all(bool(case.get("fallback_backend_authoritative", False)) for case in cases),
            "anima_factored_adamw_p28_fallback_not_authoritative",
        ),
        _validation(
            "native_shadow_training_does_not_mutate_authority",
            not bool(adapter.get("native_shadow_call_allowed", True))
            and not any(bool(case.get("native_shadow_training_mutates_authority", True)) for case in cases),
            "anima_factored_adamw_p28_native_shadow_mutated_authority",
        ),
        _validation(
            "e2e_shadow_matrix_scaffold_ready",
            matrix_ready,
            "anima_factored_adamw_p28_e2e_shadow_matrix_scaffold_missing",
        ),
        _validation(
            "runtime_dispatch_not_enabled",
            not bool(adapter.get("runtime_dispatch_ready", True))
            and not bool(adapter.get("native_dispatch_allowed", True))
            and not any(bool(case.get("training_path_enabled", True)) for case in cases),
            "anima_factored_adamw_p28_enabled_runtime_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(adapter.get("training_path_enabled", True))
            and not bool(adapter.get("default_behavior_changed", True))
            and not any(bool(case.get("default_behavior_changed", True)) for case in cases),
            "anima_factored_adamw_p28_changed_default_behavior",
        ),
    ]


def _normalize_p27_audit(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("dependency_builder", P27_AUDIT_BUILDER)
        payload.setdefault("audit_builder", P27_AUDIT_BUILDER)
        return payload
    return {
        "schema_version": 1,
        "audit": P27_AUDIT,
        "milestone": "v2_p27_anima_factored_adamw_training_loop_canary",
        "ok": True,
        "milestone_completed": True,
        "report_only_dependency_contract": True,
        "dependency_builder": P27_AUDIT_BUILDER,
        "native_call_performed_by_p28": False,
        "progress_gates": {
            "training_loop_native_canary_dependency_named": True,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
        },
        "remaining_blockers": [],
        "summary": {
            "recommended_next_step": "add AnimaFactoredAdamW P28 e2e shadow matrix",
            "p27_audit_builder": P27_AUDIT_BUILDER,
        },
    }


def _p27_dependency_named(p27: Mapping[str, Any]) -> bool:
    names = {
        str(p27.get("dependency_builder") or ""),
        str(p27.get("audit_builder") or ""),
        str(p27.get("builder") or ""),
        str(_as_dict(p27.get("summary")).get("p27_audit_builder") or ""),
    }
    return P27_AUDIT_BUILDER in names


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "MATRIX_CASES",
    "MATRIX_KIND",
    "P27_AUDIT_BUILDER",
    "build_anima_factored_adamw_e2e_shadow_matrix_scorecard",
]
