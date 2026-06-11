"""Report-only execution matrix for selected model/shape-aware plugin optimizers."""

from __future__ import annotations

from typing import Any, Mapping


MODEL_SHAPE_ROUTE_FAMILY = "model_or_shape_aware"
UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "native_kernel_ready",
    "product_native_ready",
    "product_native_dispatch_ready",
)


def build_model_shape_aware_execution_matrix(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Promote selector-resume and param-group artifact rows to ABI evidence."""

    case_rows = [_row(row) for row in rows]
    implementation_ready = [row for row in case_rows if row["param_group_abi_implementation_ready"] is True]
    failed = [row for row in case_rows if row["ok"] is not True]
    unsafe = _unsafe_claims(case_rows)
    ready = bool(case_rows) and len(implementation_ready) == len(case_rows) and not failed and not unsafe
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_model_shape_aware_execution_matrix_v0",
        "gate": "plugin_model_shape_aware_param_group_execution_matrix",
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
        "selected_optimizer_family": MODEL_SHAPE_ROUTE_FAMILY,
        "rows": case_rows,
        "summary": {
            "selected_optimizer_count": len(case_rows),
            "param_group_abi_implementation_ready_count": len(implementation_ready),
            "param_group_resume_replay_matrix_implementation_ready_count": len(implementation_ready),
            "param_group_resume_replay_row_implementation_ready_count": sum(
                int(row.get("matrix_row_implementation_ready_count", 0) or 0) for row in case_rows
            ),
            "execution_failed_count": len(failed),
            "unsafe_claim_count": len(unsafe),
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
        },
        "blocked_reasons": _dedupe(
            unsafe + [reason for row in failed for reason in row.get("blocked_reasons", [])]
        ),
        "promotion_blockers": [
            "native_kernel_implementation_missing",
            "runtime_dispatch_shadow_missing",
            "owner_release_hold_missing",
        ],
        "notes": [
            "This matrix uses selector-proven resume support and planned param-group payload rows as the reference.",
            "It proves ABI/replay contract implementation readiness only; it does not claim a native kernel.",
            "Training path, native dispatch, product readiness, request/schema, and UI exposure remain closed.",
        ],
    }


def _row(source: Mapping[str, Any]) -> dict[str, Any]:
    name = str(source.get("selected_optimizer_name", "")).strip().lower()
    artifact = _as_dict(source.get("param_group_resume_replay_matrix_artifact"))
    matrix_rows = [row for row in artifact.get("rows", []) if isinstance(row, Mapping)]
    row_ready = [
        _matrix_row_ready(row)
        for row in matrix_rows
    ]
    ready = (
        source.get("selector_classified") is True
        and source.get("resume_proven") is True
        and source.get("param_group_abi_spec_ready") is True
        and artifact.get("artifact_ready") is True
        and bool(matrix_rows)
        and all(row_ready)
    )
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "native_route_family": MODEL_SHAPE_ROUTE_FAMILY,
        "ok": ready,
        "param_group_abi_implementation_ready": ready,
        "param_group_resume_replay_matrix_implementation_ready": ready,
        "matrix_row_count": len(matrix_rows),
        "matrix_row_implementation_ready_count": sum(1 for value in row_ready if value),
        "execution_reference": "plugin_selector_resume_proven_param_group_artifact",
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [] if ready else _blocked_reasons(name, source, artifact, matrix_rows, row_ready),
    }


def promote_model_shape_aware_row(row: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    """Return a row with implementation-ready artifact markers applied."""

    name = str(row.get("selected_optimizer_name", "")).strip().lower()
    execution_row = _execution_row(execution, name)
    implementation_ready = execution_row.get("param_group_abi_implementation_ready") is True
    artifact = _as_dict(row.get("param_group_resume_replay_matrix_artifact"))
    artifact_rows = []
    source_rows = artifact.get("rows", []) if isinstance(artifact.get("rows"), list) else []
    for artifact_row in source_rows:
        if isinstance(artifact_row, Mapping):
            artifact_rows.append(
                {
                    **dict(artifact_row),
                    "artifact_status": "implementation_ready" if implementation_ready else "planned_report_only",
                    "implementation_ready": implementation_ready,
                }
            )
    promoted_artifact = {
        **artifact,
        "artifact_status": "implementation_ready" if implementation_ready else artifact.get("artifact_status"),
        "implementation_ready": implementation_ready,
        "rows": artifact_rows,
    }
    return {
        **dict(row),
        "batch_status": "param_group_abi_replay_ready_report_only" if implementation_ready else row.get("batch_status"),
        "param_group_abi_implementation_ready": implementation_ready,
        "param_group_resume_replay_matrix_artifact": promoted_artifact,
        "execution_matrix_row": execution_row,
        "blocked_reasons": _remaining_blockers(row, implementation_ready),
    }


def _remaining_blockers(row: Mapping[str, Any], implementation_ready: bool) -> list[str]:
    if not implementation_ready:
        return [str(reason) for reason in row.get("blocked_reasons", []) or [] if str(reason)]
    return ["native_kernel_implementation_missing", "runtime_dispatch_shadow_missing", "owner_release_hold_missing"]


def _matrix_row_ready(row: Mapping[str, Any]) -> bool:
    return bool(row.get("required_payload")) and bool(row.get("replay_assertions")) and row.get("native_dispatch_allowed") is False


def _blocked_reasons(
    name: str,
    source: Mapping[str, Any],
    artifact: Mapping[str, Any],
    matrix_rows: list[Mapping[str, Any]],
    row_ready: list[bool],
) -> list[str]:
    reasons: list[str] = []
    if source.get("selector_classified") is not True:
        reasons.append(f"model_shape_selector_missing:{name}")
    if source.get("resume_proven") is not True:
        reasons.append(f"model_shape_resume_missing:{name}")
    if source.get("param_group_abi_spec_ready") is not True:
        reasons.append(f"model_shape_param_group_abi_spec_missing:{name}")
    if artifact.get("artifact_ready") is not True:
        reasons.append(f"model_shape_param_group_matrix_artifact_missing:{name}")
    if not matrix_rows or not all(row_ready):
        reasons.append(f"model_shape_param_group_matrix_rows_incomplete:{name}")
    return reasons


def _unsafe_claims(rows: list[Mapping[str, Any]]) -> list[str]:
    claims: list[str] = []
    for row in rows:
        name = str(row.get("selected_optimizer_name", "unknown"))
        for field in UNSAFE_TRUE_FIELDS:
            if row.get(field) is True:
                claims.append(f"unsafe_model_shape_execution_row:{name}:{field}")
    return _dedupe(claims)


def _execution_row(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("selected_optimizer_name", "")).strip().lower() == name:
            return dict(row)
    return {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "MODEL_SHAPE_ROUTE_FAMILY",
    "build_model_shape_aware_execution_matrix",
    "promote_model_shape_aware_row",
]
