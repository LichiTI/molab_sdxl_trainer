"""Report-only execution matrix for selected adaptive-LR plugin optimizers."""

from __future__ import annotations

from typing import Any, Mapping


UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "native_kernel_ready",
    "product_native_ready",
)


def build_adaptivelr_execution_matrix(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    case_rows = [_row(row) for row in rows]
    ready_rows = [row for row in case_rows if row["state_machine_abi_implementation_ready"] is True]
    failed = [row for row in case_rows if row["ok"] is not True]
    unsafe = _unsafe_claims(case_rows)
    ready = bool(case_rows) and len(ready_rows) == len(case_rows) and not failed and not unsafe
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adaptivelr_execution_matrix_v0",
        "gate": "plugin_adaptivelr_state_machine_execution_matrix",
        "ok": ready,
        "execution_matrix_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "rows": case_rows,
        "summary": {
            "selected_optimizer_count": len(case_rows),
            "state_machine_abi_implementation_ready_count": len(ready_rows),
            "state_machine_replay_matrix_implementation_ready_count": len(ready_rows),
            "state_machine_replay_case_implementation_ready_count": sum(
                int(row.get("replay_case_implementation_ready_count", 0) or 0) for row in case_rows
            ),
            "state_machine_replay_resume_case_implementation_ready_count": sum(
                int(row.get("resume_case_implementation_ready_count", 0) or 0) for row in case_rows
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


def promote_adaptivelr_row(row: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    name = str(row.get("selected_optimizer_name", "")).strip().lower()
    execution_row = _execution_row(execution, name)
    ready = execution_row.get("state_machine_abi_implementation_ready") is True
    abi = dict(_as_dict(row.get("state_machine_abi_spec")))
    abi["implementation_ready"] = ready
    binding = dict(_as_dict(abi.get("plugin_binding")))
    if binding:
        binding["implementation_ready"] = ready
        abi["plugin_binding"] = binding
    for key in (
        "dynamic_lr_scalar_state",
        "d_estimator_global_state",
        "per_step_quality_guard",
        "resume_scope",
        "native_kernel_preconditions",
    ):
        block = dict(_as_dict(abi.get(key)))
        if block:
            block["implementation_ready"] = ready
            abi[key] = block
    matrix = dict(_as_dict(row.get("state_machine_replay_matrix_artifact")))
    matrix.update(
        {
            "implementation_ready": ready,
            "replay_case_status": _case_status(matrix.get("replay_cases", []), ready),
            "resume_replay_case_status": _case_status(matrix.get("resume_replay_cases", []), ready),
        }
    )
    return {
        **dict(row),
        "state_machine_status": "abi_replay_ready_report_only" if ready else row.get("state_machine_status"),
        "state_machine_abi_implementation_ready": ready,
        "native_kernel_preconditions_implementation_ready": ready,
        "state_machine_abi_spec": abi,
        "state_machine_replay_matrix_artifact": matrix,
        "execution_matrix_row": execution_row,
        "blocked_reasons": _remaining_blockers(ready),
    }


def _row(source: Mapping[str, Any]) -> dict[str, Any]:
    name = str(source.get("selected_optimizer_name", "")).strip().lower()
    abi = _as_dict(source.get("state_machine_abi_spec"))
    matrix = _as_dict(source.get("state_machine_replay_matrix_artifact"))
    replay_cases = [case for case in matrix.get("replay_cases", []) if str(case)]
    resume_cases = [case for case in matrix.get("resume_replay_cases", []) if str(case)]
    ready = (
        source.get("selector_classified") is True
        and source.get("resume_proven") is True
        and source.get("selected_state_machine_reference_ready") is True
        and source.get("selected_state_machine_abi_spec_ready") is True
        and _section_ready(abi, "dynamic_lr_scalar_state")
        and _section_ready(abi, "d_estimator_global_state")
        and _section_ready(abi, "per_step_quality_guard")
        and _section_ready(abi, "resume_scope")
        and _section_ready(abi, "native_kernel_preconditions")
        and matrix.get("spec_ready") is True
        and bool(replay_cases)
        and bool(resume_cases)
        and matrix.get("native_dispatch_allowed") is False
    )
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "ok": ready,
        "state_machine_abi_implementation_ready": ready,
        "state_machine_replay_matrix_implementation_ready": ready,
        "replay_case_implementation_ready_count": len(replay_cases) if ready else 0,
        "resume_case_implementation_ready_count": len(resume_cases) if ready else 0,
        "native_kernel_preconditions_implementation_ready": ready,
        "execution_reference": "plugin_selector_resume_proven_adaptivelr_state_machine_artifact",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "blocked_reasons": [] if ready else [f"adaptivelr_execution_incomplete:{name}"],
    }


def _case_status(cases: Any, ready: bool) -> dict[str, str]:
    if not isinstance(cases, list):
        return {}
    return {str(case): "implementation_ready" if ready else "planned" for case in cases if str(case)}


def _remaining_blockers(ready: bool) -> list[str]:
    if not ready:
        return ["selected_plugin_adaptivelr_batch_resume_parity_not_validated"]
    return ["native_kernel_implementation_missing", "runtime_dispatch_shadow_missing", "owner_release_hold_missing"]


def _section_ready(abi: Mapping[str, Any], section: str) -> bool:
    return _as_dict(abi.get(section)).get("spec_ready") is True


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
                out.append(f"unsafe_adaptivelr_execution_row:{name}:{field}")
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


__all__ = ["build_adaptivelr_execution_matrix", "promote_adaptivelr_row"]
