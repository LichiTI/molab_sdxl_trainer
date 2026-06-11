"""Shared family-specific runtime adapter coverage helpers for TurboCore plugins."""

from __future__ import annotations

from typing import Any, Mapping


def build_family_runtime_launch_adapter_coverage(
    *,
    family: str,
    cases: list[Mapping[str, Any]],
    representative_runtime: Mapping[str, Any],
    adapter_kind: str,
    representative_optimizer_name: str,
) -> dict[str, Any]:
    """Build honest per-optimizer adapter coverage from preconditions plus one native launch.

    This does not claim that every optimizer executed its own native math kernel.
    It records that each optimizer row has the runtime adapter inputs needed to
    bind a family-specific launch path, backed by a representative native launch
    for the family.
    """

    representative_ready = representative_runtime.get("runtime_dispatch_rehearsal_ready") is True
    rows = [
        _adapter_row(
            family=family,
            case=case,
            representative_ready=representative_ready,
            adapter_kind=adapter_kind,
            representative_optimizer_name=representative_optimizer_name,
        )
        for case in cases
    ]
    ready_count = sum(1 for row in rows if row["family_specific_runtime_launch_adapter_ready"])
    native_step_count = int(representative_runtime.get("native_step_count", 0) or 0)
    native_kernel_launch_count = int(representative_runtime.get("native_kernel_launch_count", 0) or 0)
    ready = bool(rows) and ready_count == len(rows) and native_step_count > 0 and native_kernel_launch_count > 0
    return {
        "schema_version": 1,
        "adapter": "family_specific_runtime_launch_adapter_coverage_v0",
        "native_route_family": family,
        "adapter_kind": adapter_kind,
        "ok": ready,
        "family_specific_runtime_launch_adapter_ready": ready,
        "representative_optimizer_name": representative_optimizer_name,
        "representative_runtime_dispatch_rehearsal_ready": representative_ready,
        "representative_native_step_count": native_step_count,
        "representative_native_kernel_launch_count": native_kernel_launch_count,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "optimizer_math_native_kernel_executed_for_all_cases": False,
        "rows": rows,
        "summary": {
            "case_count": len(rows),
            "family_specific_runtime_launch_adapter_ready_count": ready_count,
            "representative_runtime_launch_ready_count": 1 if representative_ready else 0,
            "representative_native_step_count": native_step_count,
            "representative_native_kernel_launch_count": native_kernel_launch_count,
            "per_optimizer_native_math_launch_count": sum(
                1 for row in rows if row["per_optimizer_native_math_launch_executed"]
            ),
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
        },
        "notes": [
            "Adapter coverage is not product runtime dispatch approval.",
            "Only the representative canary executes native optimizer math.",
            "Other rows prove adapter binding preconditions and remain product-closed.",
        ],
    }


def _adapter_row(
    *,
    family: str,
    case: Mapping[str, Any],
    representative_ready: bool,
    adapter_kind: str,
    representative_optimizer_name: str,
) -> dict[str, Any]:
    name = str(case.get("selected_optimizer_name") or "")
    precondition_ready = case.get("runtime_precondition_rehearsal_ready") is True
    ready = precondition_ready and representative_ready and bool(name)
    is_representative = name == representative_optimizer_name
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "selected_optimizer_family": family,
        "adapter_kind": adapter_kind,
        "family_specific_runtime_launch_adapter_ready": ready,
        "runtime_precondition_rehearsal_ready": precondition_ready,
        "representative_native_launch_proven": representative_ready,
        "representative_runtime_source_optimizer": representative_optimizer_name,
        "is_representative_runtime_case": is_representative,
        "per_optimizer_native_math_launch_executed": is_representative and representative_ready,
        "optimizer_math_native_kernel_executed": is_representative and representative_ready,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "blocked_reasons": []
        if ready
        else [f"plugin_{name or 'unknown'}_{family}_runtime_launch_adapter_missing"],
    }


def adapter_summary(report: Mapping[str, Any]) -> dict[str, int]:
    summary = report.get("summary")
    return dict(summary) if isinstance(summary, Mapping) else {}


__all__ = ["adapter_summary", "build_family_runtime_launch_adapter_coverage"]
