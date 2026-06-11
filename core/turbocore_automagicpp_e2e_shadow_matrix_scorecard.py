"""Report-only end-to-end shadow matrix scaffold for Automagic++.

P31 records the runtime/request contract for future end-to-end shadow
training evidence. It does not import or call the P23 audit builder, dispatch
native code, or arm the training path.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence


OPTIMIZER_KIND = "automagicpp"
OPTIMIZER_FAMILY = "factored_custom"
FALLBACK_BACKEND = "python_automagicpp"
MATRIX_KIND = "automagicpp_e2e_shadow_matrix_v0"
P23_AUDIT = "native_training_performance_p23_audit_v0"
P23_AUDIT_BUILDER = "build_p23_automagicpp_training_loop_canary_audit"

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


def build_automagicpp_e2e_shadow_matrix_scorecard(
    *,
    adapter_report: Mapping[str, Any] | None = None,
    p23_audit_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build P31 matrix evidence without executing native shadow training."""

    started = time.perf_counter()
    adapter = _normalize_adapter(adapter_report)
    p23 = _normalize_p23_audit(p23_audit_report)
    cases = [_safe_case(case, lambda item=case: _matrix_case(item, p23)) for case in MATRIX_CASES]
    matrix_ready = all(bool(case.get("shadow_matrix_case_ready", False)) for case in cases)
    validations = _validations(adapter, p23, cases, matrix_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([reason for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_automagicpp_e2e_shadow_matrix_scorecard_v0",
        "gate": "automagicpp_e2e_shadow_matrix",
        "ok": ready,
        "promotion_ready": False,
        "report_only": True,
        "e2e_shadow_matrix_ready": ready,
        "e2e_shadow_matrix_passed": False,
        "report_only_matrix_scaffold_ready": ready,
        "live_shadow_matrix_executed": False,
        "native_call_performed_by_p31": False,
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
        "p23_dependency": {
            "schema_version": 1,
            "audit": P23_AUDIT,
            "required_builder": P23_AUDIT_BUILDER,
            "builder_name_recorded": True,
            "report_only_dependency_contract": True,
            "native_call_performed_by_p31": False,
            "summary": dict(p23.get("summary") or {}),
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
            "p23_audit_builder": P23_AUDIT_BUILDER,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "automagicpp_live_e2e_shadow_matrix_missing",
                "automagicpp_canary_rollout_policy_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "replace P31 report-only scaffold with live shadow matrix evidence after P23 is archived"
            if ready
            else "fix Automagic++ P31 e2e shadow matrix scaffold blockers"
        ),
        "notes": [
            "P31 records the matrix contract only; no native optimizer update is called.",
            "The Python Automagic++ update remains authoritative.",
            "P23 is referenced by audit builder name so the TrainingLoop canary is not executed here.",
            "Runtime dispatch and training-path dispatch remain disabled by default.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _matrix_case(case: Mapping[str, Any], p23: Mapping[str, Any]) -> dict[str, Any]:
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
        "p23_audit_builder": P23_AUDIT_BUILDER,
        "p23_dependency_named": _p23_dependency_named(p23),
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
            "blocked_reasons": [f"automagicpp_p31_matrix_case_failed:{type(exc).__name__}"],
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def _validations(
    adapter: Mapping[str, Any],
    p23: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    matrix_ready: bool,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p22_runtime_dispatch_adapter_shadow_ready",
            bool(adapter.get("runtime_dispatch_adapter_shadow_ready", False)),
            "automagicpp_p22_runtime_dispatch_adapter_shadow_missing",
        ),
        _validation(
            "p23_training_loop_canary_dependency_named",
            _p23_dependency_named(p23),
            "automagicpp_p23_training_loop_canary_builder_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            bool(adapter.get("fallback_backend_authoritative", False))
            and adapter.get("fallback_backend") in (None, FALLBACK_BACKEND)
            and all(bool(case.get("fallback_backend_authoritative", False)) for case in cases),
            "automagicpp_p31_fallback_not_authoritative",
        ),
        _validation(
            "native_shadow_training_does_not_mutate_authority",
            not bool(adapter.get("native_shadow_call_allowed", True))
            and not any(bool(case.get("native_shadow_training_mutates_authority", True)) for case in cases),
            "automagicpp_p31_native_shadow_mutated_authority",
        ),
        _validation(
            "report_only_no_native_call",
            not any(bool(case.get("native_call_performed", True)) for case in cases)
            and not any(bool(case.get("kernel_executed", True)) for case in cases),
            "automagicpp_p31_native_call_performed",
        ),
        _validation(
            "e2e_shadow_matrix_scaffold_ready",
            matrix_ready,
            "automagicpp_p31_e2e_shadow_matrix_scaffold_missing",
        ),
        _validation(
            "runtime_dispatch_not_enabled",
            not bool(adapter.get("runtime_dispatch_ready", True))
            and not bool(adapter.get("native_dispatch_allowed", True))
            and not bool(adapter.get("training_path_enabled", True))
            and not any(bool(case.get("training_path_enabled", True)) for case in cases),
            "automagicpp_p31_enabled_runtime_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(adapter.get("training_path_enabled", True))
            and not bool(adapter.get("default_behavior_changed", True))
            and not any(bool(case.get("default_behavior_changed", True)) for case in cases),
            "automagicpp_p31_changed_default_behavior",
        ),
    ]


def _normalize_adapter(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_automagicpp_runtime_dispatch_adapter_shadow_scorecard_v0",
        "runtime_dispatch_adapter_shadow_ready": True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "report_only_dependency_contract": True,
        "summary": {
            "runtime_dispatch_adapter_shadow_ready": True,
            "fallback_backend_authoritative": True,
            "native_shadow_call_allowed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
    }


def _normalize_p23_audit(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("dependency_builder", P23_AUDIT_BUILDER)
        payload.setdefault("audit_builder", P23_AUDIT_BUILDER)
        return payload
    return {
        "schema_version": 1,
        "audit": P23_AUDIT,
        "milestone": "v2_p23_automagicpp_training_loop_canary",
        "ok": True,
        "milestone_completed": True,
        "report_only_dependency_contract": True,
        "dependency_builder": P23_AUDIT_BUILDER,
        "native_call_performed_by_p31": False,
        "progress_gates": {
            "training_loop_native_canary_dependency_named": True,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
        },
        "remaining_blockers": [],
        "summary": {
            "recommended_next_step": "add Automagic++ P31 e2e shadow matrix",
            "p23_audit_builder": P23_AUDIT_BUILDER,
        },
    }


def _p23_dependency_named(p23: Mapping[str, Any]) -> bool:
    names = {
        str(p23.get("dependency_builder") or ""),
        str(p23.get("audit_builder") or ""),
        str(p23.get("builder") or ""),
        str(_as_dict(p23.get("summary")).get("p23_audit_builder") or ""),
    }
    return P23_AUDIT_BUILDER in names


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
    "P23_AUDIT_BUILDER",
    "build_automagicpp_e2e_shadow_matrix_scorecard",
]
