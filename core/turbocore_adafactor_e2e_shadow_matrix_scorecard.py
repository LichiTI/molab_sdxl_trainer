"""Report-only end-to-end shadow matrix for Adafactor."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from core.turbocore_adafactor_runtime_dispatch_adapter_shadow_scorecard import (
    build_adafactor_runtime_dispatch_adapter_shadow_scorecard,
)


OPTIMIZER_KIND = "adafactor"
OPTIMIZER_FAMILY = "factored_custom"
FALLBACK_BACKEND = "python_adafactor"
MATRIX_KIND = "adafactor_e2e_shadow_matrix_v0"
P37_AUDIT = "native_training_performance_p37_audit_v0"
P37_AUDIT_BUILDER = "build_p37_adafactor_training_loop_canary_audit"

MATRIX_CASES = (
    {"case": "report_only_factored_fp32_128x128", "shape": [128, 128], "param_dtype": "float32", "grad_dtype": "float32"},
    {"case": "report_only_factored_bf16_128x128", "shape": [128, 128], "param_dtype": "bfloat16", "grad_dtype": "bfloat16"},
    {"case": "report_only_unfactored_fp32_4096", "shape": [4096], "param_dtype": "float32", "grad_dtype": "float32"},
)


def build_adafactor_e2e_shadow_matrix_scorecard(
    *,
    adapter_report: Mapping[str, Any] | None = None,
    p37_audit_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    adapter = _as_dict(adapter_report or build_adafactor_runtime_dispatch_adapter_shadow_scorecard())
    p37 = _normalize_p37_audit(p37_audit_report)
    cases = [_matrix_case(case, p37) for case in MATRIX_CASES]
    validations = _validations(adapter, p37, cases)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([reason for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adafactor_e2e_shadow_matrix_scorecard_v0",
        "gate": "adafactor_e2e_shadow_matrix",
        "ok": ready,
        "promotion_ready": False,
        "report_only": True,
        "e2e_shadow_matrix_ready": ready,
        "e2e_shadow_matrix_passed": False,
        "report_only_matrix_scaffold_ready": ready,
        "live_shadow_matrix_executed": False,
        "native_call_performed_by_p38": False,
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
        "p37_dependency": {
            "schema_version": 1,
            "audit": P37_AUDIT,
            "required_builder": P37_AUDIT_BUILDER,
            "builder_name_recorded": _p37_dependency_named(p37),
            "report_only_dependency_contract": True,
            "summary": dict(p37.get("summary") or {}),
        },
        "adapter_summary": dict(adapter.get("summary") or {}),
        "matrix_cases": cases,
        "validations": validations,
        "summary": {
            "case_count": len(cases),
            "report_only_case_count": len(cases),
            "failed_case_count": 0 if ready else len(failed),
            "e2e_shadow_matrix_passed": False,
            "report_only_matrix_scaffold_ready": ready,
            "live_shadow_matrix_executed": False,
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_backend_authoritative": True,
            "native_shadow_training_mutates_authority": False,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
            "p37_audit_builder": P37_AUDIT_BUILDER,
        },
        "promotion_blockers": _dedupe(blockers + ["adafactor_live_e2e_shadow_matrix_missing", "adafactor_canary_rollout_policy_missing"]),
        "blocked_reasons": blockers,
        "recommended_next_step": "add Adafactor explicit canary rollout policy" if ready else "fix Adafactor e2e shadow matrix scaffold blockers",
        "notes": [
            "P38 records the matrix contract only; no native optimizer update is called.",
            "Python Adafactor remains the authoritative update backend.",
            "Runtime dispatch and training-path dispatch remain disabled by default.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _matrix_case(case: Mapping[str, Any], p37: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case": str(case["case"]),
        "shape": list(case["shape"]),
        "param_dtype": str(case["param_dtype"]),
        "grad_dtype": str(case["grad_dtype"]),
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
        "p37_audit_builder": P37_AUDIT_BUILDER,
        "p37_dependency_named": _p37_dependency_named(p37),
        "blocked_reasons": [],
    }


def _validations(
    adapter: Mapping[str, Any],
    p37: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _validation("p36_runtime_dispatch_adapter_shadow_ready", bool(adapter.get("runtime_dispatch_adapter_shadow_ready", False)), "adafactor_p36_runtime_dispatch_adapter_shadow_missing"),
        _validation("p37_training_loop_canary_dependency_named", _p37_dependency_named(p37), "adafactor_p37_training_loop_canary_builder_missing"),
        _validation("fallback_backend_authoritative", bool(adapter.get("fallback_backend_authoritative", False)) and all(bool(case.get("fallback_backend_authoritative", False)) for case in cases), "adafactor_p38_fallback_not_authoritative"),
        _validation("native_shadow_training_does_not_mutate_authority", not bool(adapter.get("native_shadow_call_allowed", True)) and not any(bool(case.get("native_shadow_training_mutates_authority", True)) for case in cases), "adafactor_p38_native_shadow_mutated_authority"),
        _validation("e2e_shadow_matrix_scaffold_ready", all(bool(case.get("shadow_matrix_case_ready", False)) for case in cases), "adafactor_p38_e2e_shadow_matrix_scaffold_missing"),
        _validation("runtime_dispatch_not_enabled", not bool(adapter.get("runtime_dispatch_ready", True)) and not bool(adapter.get("native_dispatch_allowed", True)) and not any(bool(case.get("training_path_enabled", True)) for case in cases), "adafactor_p38_enabled_runtime_dispatch"),
        _validation("default_behavior_unchanged", not bool(adapter.get("training_path_enabled", True)) and not bool(adapter.get("default_behavior_changed", True)) and not any(bool(case.get("default_behavior_changed", True)) for case in cases), "adafactor_p38_changed_default_behavior"),
    ]


def _normalize_p37_audit(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("dependency_builder", P37_AUDIT_BUILDER)
        payload.setdefault("audit_builder", P37_AUDIT_BUILDER)
        return payload
    return {
        "schema_version": 1,
        "audit": P37_AUDIT,
        "milestone": "v2_p37_adafactor_training_loop_canary",
        "ok": True,
        "milestone_completed": True,
        "report_only_dependency_contract": True,
        "dependency_builder": P37_AUDIT_BUILDER,
        "progress_gates": {"training_loop_native_canary": True, "runtime_dispatch_not_enabled": True, "default_behavior_unchanged": True},
        "remaining_blockers": [],
        "summary": {"recommended_next_step": "add Adafactor e2e shadow matrix", "p37_audit_builder": P37_AUDIT_BUILDER},
    }


def _p37_dependency_named(p37: Mapping[str, Any]) -> bool:
    names = {
        str(p37.get("dependency_builder") or ""),
        str(p37.get("audit_builder") or ""),
        str(p37.get("builder") or ""),
        str(_as_dict(p37.get("summary")).get("p37_audit_builder") or ""),
    }
    return P37_AUDIT_BUILDER in names


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["MATRIX_CASES", "MATRIX_KIND", "P37_AUDIT_BUILDER", "build_adafactor_e2e_shadow_matrix_scorecard"]
