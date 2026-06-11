"""Report-only execution matrix for selected closure/second-order plugin optimizers."""

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


def build_closure_second_order_execution_matrix(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    case_rows = [_row(row) for row in rows]
    ready_rows = [row for row in case_rows if row["training_loop_abi_implementation_ready"] is True]
    failed = [row for row in case_rows if row["ok"] is not True]
    unsafe = _unsafe_claims(case_rows)
    ready = bool(case_rows) and len(ready_rows) == len(case_rows) and not failed and not unsafe
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_closure_second_order_execution_matrix_v0",
        "gate": "plugin_closure_second_order_training_loop_execution_matrix",
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
            "training_loop_abi_implementation_ready_count": len(ready_rows),
            "resume_parity_matrix_implementation_ready_count": len(ready_rows),
            "closure_resume_replay_artifact_implementation_ready_count": len(ready_rows),
            "closure_resume_replay_row_implementation_ready_count": sum(
                int(row.get("closure_resume_replay_row_implementation_ready_count", 0) or 0) for row in case_rows
            ),
            "state_resume_adapter_implementation_ready_count": len(ready_rows),
            "native_kernel_preconditions_implementation_ready_count": len(ready_rows),
            "execution_failed_count": len(failed),
            "unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(
            unsafe + [reason for row in failed for reason in row.get("blocked_reasons", [])]
        ),
        "promotion_blockers": ["native_kernel_implementation_missing", "owner_release_hold_missing"],
    }


def promote_closure_second_order_row(row: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    name = str(row.get("selected_optimizer_name", "")).strip().lower()
    execution_row = _execution_row(execution, name)
    ready = execution_row.get("training_loop_abi_implementation_ready") is True
    matrix = dict(_as_dict(row.get("resume_parity_matrix_plan")))
    artifact = dict(_as_dict(matrix.get("closure_resume_replay_artifact")))
    artifact_rows = []
    source_rows = artifact.get("rows", []) if isinstance(artifact.get("rows"), list) else []
    for artifact_row in source_rows:
        if isinstance(artifact_row, Mapping):
            artifact_rows.append(
                {
                    **dict(artifact_row),
                    "artifact_status": "implementation_ready" if ready else artifact_row.get("artifact_status"),
                    "implementation_ready": ready,
                }
            )
    matrix.update(
        {
            "implementation_ready": ready,
            "evidence_status": "implementation_ready" if ready else matrix.get("evidence_status"),
            "closure_resume_replay_artifact": {
                **artifact,
                "artifact_status": "implementation_ready" if ready else artifact.get("artifact_status"),
                "implementation_ready": ready,
                "rows": artifact_rows,
            },
        }
    )
    return {
        **dict(row),
        "batch_status": "training_loop_abi_replay_ready_report_only" if ready else row.get("batch_status"),
        "training_loop_abi_implementation_ready": ready,
        "resume_parity_matrix_plan": matrix,
        "execution_matrix_row": execution_row,
        "blocked_reasons": _remaining_blockers(ready),
    }


def _row(source: Mapping[str, Any]) -> dict[str, Any]:
    name = str(source.get("selected_optimizer_name", "")).strip().lower()
    matrix = _as_dict(source.get("resume_parity_matrix_plan"))
    artifact = _as_dict(matrix.get("closure_resume_replay_artifact"))
    artifact_rows = [row for row in artifact.get("rows", []) if isinstance(row, Mapping)]
    row_ready = [_artifact_row_ready(row) for row in artifact_rows]
    ready = (
        source.get("selector_classified") is True
        and source.get("resume_proven") is True
        and source.get("training_loop_abi_spec_ready") is True
        and matrix.get("spec_ready") is True
        and artifact.get("artifact_ready") is True
        and bool(matrix.get("closure_replay_cases"))
        and bool(matrix.get("create_graph_hvp_lifetime_cases"))
        and bool(matrix.get("state_resume_adapter_cases"))
        and bool(artifact_rows)
        and all(row_ready)
    )
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "ok": ready,
        "training_loop_abi_implementation_ready": ready,
        "resume_parity_matrix_implementation_ready": ready,
        "closure_resume_replay_artifact_implementation_ready": ready,
        "closure_resume_replay_row_implementation_ready_count": sum(1 for value in row_ready if value),
        "state_resume_adapter_implementation_ready": ready,
        "native_kernel_preconditions_implementation_ready": ready,
        "execution_reference": "plugin_selector_resume_proven_closure_replay_artifact",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [] if ready else [f"closure_second_order_execution_incomplete:{name}"],
    }


def _artifact_row_ready(row: Mapping[str, Any]) -> bool:
    return bool(row.get("required_payload")) and bool(row.get("replay_assertions")) and row.get("native_dispatch_allowed") is False


def _remaining_blockers(ready: bool) -> list[str]:
    if not ready:
        return ["selected_plugin_closure_create_graph_training_loop_abi_implementation_missing"]
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
                out.append(f"unsafe_closure_second_order_execution_row:{name}:{field}")
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


__all__ = ["build_closure_second_order_execution_matrix", "promote_closure_second_order_row"]
