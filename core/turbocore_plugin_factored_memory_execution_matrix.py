"""Report-only execution matrix for selected plugin factored-memory optimizers."""

from __future__ import annotations

from typing import Any, Mapping


FACTORED_MEMORY_ROUTE_FAMILY = "factored_memory_layout"
UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "native_kernel_ready",
    "product_native_ready",
    "product_native_dispatch_ready",
)


def build_factored_memory_execution_matrix(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Promote observed plugin layout rows to formula/tensor-binding evidence."""

    case_rows = [_row(row) for row in rows]
    implementation_ready = [row for row in case_rows if row["matrix_implementation_ready"] is True]
    failed = [row for row in case_rows if row["ok"] is not True]
    unsafe = _unsafe_claims(case_rows)
    ready = bool(case_rows) and len(implementation_ready) == len(case_rows) and not failed and not unsafe
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_factored_memory_execution_matrix_v0",
        "gate": "plugin_factored_memory_formula_tensor_binding_execution_matrix",
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
        "selected_optimizer_family": FACTORED_MEMORY_ROUTE_FAMILY,
        "rows": case_rows,
        "summary": {
            "selected_optimizer_count": len(case_rows),
            "formula_step_execution_ready_count": sum(
                1 for row in case_rows if row["formula_step_execution_ready"] is True
            ),
            "resume_next_step_replay_ready_count": sum(
                1 for row in case_rows if row["resume_next_step_replay_ready"] is True
            ),
            "tensor_binding_ready_count": sum(1 for row in case_rows if row["tensor_binding_ready"] is True),
            "matrix_implementation_ready_count": len(implementation_ready),
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
            "This matrix consumes observed plugin state-layout/resume rows as the reference authority.",
            "It proves formula step, resume replay, and tensor-state binding evidence only.",
            "Native dispatch, request/schema/UI exposure, and product readiness remain closed.",
        ],
    }


def _row(layout_row: Mapping[str, Any]) -> dict[str, Any]:
    name = str(layout_row.get("optimizer_name", "")).strip().lower()
    native_layout_abi = _as_dict(layout_row.get("native_layout_abi"))
    after_step = _as_dict(layout_row.get("after_step"))
    state_keys = _strings(native_layout_abi.get("state_keys"))
    tensor_shapes = _as_dict(native_layout_abi.get("tensor_state_shapes"))
    param_group_keys = _strings(after_step.get("param_group_keys"))
    resume_diff = float(layout_row.get("max_resume_diff", 0.0) or 0.0)
    tolerance = float(layout_row.get("tolerance", 1e-5) or 1e-5)

    formula_step_ready = (
        layout_row.get("covers_small_tensor_step") is True
        and layout_row.get("state_layout_status") == "observed_resume_layout"
        and bool(state_keys)
    )
    resume_ready = layout_row.get("covers_resume") is True and resume_diff <= tolerance
    tensor_binding_ready = (
        layout_row.get("native_layout_abi_ready") is True
        and bool(tensor_shapes)
        and layout_row.get("layout_quality_matrix_ready") is True
    )
    matrix_ready = formula_step_ready and resume_ready and tensor_binding_ready
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "native_route_family": FACTORED_MEMORY_ROUTE_FAMILY,
        "ok": matrix_ready,
        "matrix_artifact_ready": True,
        "matrix_implementation_ready": matrix_ready,
        "formula_step_execution_ready": formula_step_ready,
        "resume_next_step_replay_ready": resume_ready,
        "tensor_binding_ready": tensor_binding_ready,
        "state_key_count": len(state_keys),
        "tensor_state_count": len(tensor_shapes),
        "non_tensor_hparam_snapshot_ready": bool(param_group_keys),
        "param_group_key_count": len(param_group_keys),
        "max_resume_diff": resume_diff,
        "tolerance": tolerance,
        "execution_reference": "existing_pytorch_optimizer_plugin_state_layout_probe",
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [] if matrix_ready else _blocked_reasons(name, formula_step_ready, resume_ready, tensor_binding_ready),
    }


def _blocked_reasons(name: str, formula_ready: bool, resume_ready: bool, tensor_ready: bool) -> list[str]:
    reasons: list[str] = []
    if not formula_ready:
        reasons.append(f"factored_memory_formula_step_missing:{name}")
    if not resume_ready:
        reasons.append(f"factored_memory_resume_replay_missing:{name}")
    if not tensor_ready:
        reasons.append(f"factored_memory_tensor_binding_missing:{name}")
    return reasons


def _unsafe_claims(rows: list[Mapping[str, Any]]) -> list[str]:
    claims: list[str] = []
    for row in rows:
        name = str(row.get("selected_optimizer_name", "unknown"))
        for field in UNSAFE_TRUE_FIELDS:
            if row.get(field) is True:
                claims.append(f"unsafe_factored_memory_execution_row:{name}:{field}")
    return _dedupe(claims)


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["FACTORED_MEMORY_ROUTE_FAMILY", "build_factored_memory_execution_matrix"]
