"""Report-only execution matrix for selected fused-backward plugin optimizers."""

from __future__ import annotations

from typing import Any, Mapping


UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "native_kernel_ready",
    "product_native_ready",
    "product_native_dispatch_ready",
)


def build_fused_backward_execution_matrix(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    case_rows = [_row(row) for row in rows]
    ready_rows = [row for row in case_rows if row["fused_backward_abi_implementation_ready"] is True]
    failed = [row for row in case_rows if row["ok"] is not True]
    unsafe = _unsafe_claims(case_rows)
    ready = bool(case_rows) and len(ready_rows) == len(case_rows) and not failed and not unsafe
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_fused_backward_execution_matrix_v0",
        "gate": "plugin_fused_backward_abi_execution_matrix",
        "ok": ready,
        "execution_matrix_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "rows": case_rows,
        "summary": {
            "selected_optimizer_count": len(case_rows),
            "fused_backward_abi_implementation_ready_count": len(ready_rows),
            "resume_parity_matrix_implementation_ready_count": len(ready_rows),
            "fused_backward_replay_case_implementation_ready_count": sum(
                int(row.get("replay_case_implementation_ready_count", 0) or 0) for row in case_rows
            ),
            "native_kernel_preconditions_implementation_ready_count": len(ready_rows),
            "execution_failed_count": len(failed),
            "unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(
            unsafe + [reason for row in failed for reason in row.get("blocked_reasons", [])]
        ),
        "promotion_blockers": ["native_kernel_implementation_missing", "owner_release_hold_missing"],
    }


def promote_fused_backward_row(row: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    name = str(row.get("selected_optimizer_name", "")).strip().lower()
    execution_row = _execution_row(execution, name)
    ready = execution_row.get("fused_backward_abi_implementation_ready") is True
    matrix = dict(_as_dict(row.get("resume_parity_matrix")))
    cases = []
    source_cases = matrix.get("planned_cases", []) if isinstance(matrix.get("planned_cases"), list) else []
    for case in source_cases:
        if isinstance(case, Mapping):
            cases.append({**dict(case), "status": "implementation_ready" if ready else "planned"})
    matrix.update({"matrix_implementation_ready": ready, "planned_cases": cases})
    return {
        **dict(row),
        "batch_status": "fused_backward_abi_replay_ready_report_only" if ready else row.get("batch_status"),
        "fused_backward_abi_implementation_ready": ready,
        "native_kernel_preconditions_implementation_ready": ready,
        "resume_parity_matrix": matrix,
        "execution_matrix_row": execution_row,
        "blocked_reasons": _remaining_blockers(ready),
    }


def _row(source: Mapping[str, Any]) -> dict[str, Any]:
    name = str(source.get("selected_optimizer_name", "")).strip().lower()
    matrix = _as_dict(source.get("resume_parity_matrix"))
    cases = [case for case in matrix.get("planned_cases", []) if isinstance(case, Mapping)]
    case_ready = [_case_ready(case) for case in cases]
    ready = (
        source.get("selector_classified") is True
        and source.get("resume_proven") is True
        and source.get("fused_backward_abi_spec_ready") is True
        and bool(source.get("abi_spec"))
        and matrix.get("matrix_spec_ready") is True
        and bool(cases)
        and all(case_ready)
    )
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "ok": ready,
        "fused_backward_abi_implementation_ready": ready,
        "resume_parity_matrix_implementation_ready": ready,
        "replay_case_implementation_ready_count": sum(1 for value in case_ready if value),
        "native_kernel_preconditions_implementation_ready": ready,
        "execution_reference": "plugin_selector_resume_proven_fused_backward_abi_artifact",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [] if ready else [f"fused_backward_execution_incomplete:{name}"],
    }


def _case_ready(case: Mapping[str, Any]) -> bool:
    return (
        case.get("requires_backward_hook_ownership_token") is True
        and case.get("requires_skip_public_optimizer_step") is True
        and case.get("requires_state_dict_roundtrip") is True
        and case.get("native_dispatch_allowed") is False
    )


def _remaining_blockers(ready: bool) -> list[str]:
    if not ready:
        return ["selected_plugin_fused_backward_resume_parity_matrix_implementation_missing"]
    return ["native_kernel_implementation_missing", "runtime_dispatch_shadow_missing", "owner_release_hold_missing"]


def _execution_row(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("selected_optimizer_name", "")).strip().lower() == name:
            return dict(row)
    return {}


def _unsafe_claims(rows: list[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    for row in rows:
        name = str(row.get("selected_optimizer_name", "unknown"))
        for field in UNSAFE_TRUE_FIELDS:
            if row.get(field) is True:
                out.append(f"unsafe_fused_backward_execution_row:{name}:{field}")
    return _dedupe(out)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_fused_backward_execution_matrix", "promote_fused_backward_row"]
