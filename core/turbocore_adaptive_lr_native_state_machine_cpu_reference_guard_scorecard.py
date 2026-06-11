"""CPU reference guard artifacts for built-in adaptive-LR native ABI skeletons."""

from __future__ import annotations

import math
from typing import Any, Mapping

from core.turbocore_adaptive_lr_native_state_machine_abi_skeleton_scorecard import (
    build_adaptive_lr_native_state_machine_abi_skeleton_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES


SCORECARD = "turbocore_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard_v0"


def build_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard(
    *,
    abi_skeleton_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build default-off CPU guard evidence for adaptive-LR launch plans."""

    skeleton = _as_dict(abi_skeleton_report or build_adaptive_lr_native_state_machine_abi_skeleton_scorecard())
    rows = [_row(case.optimizer.value, skeleton) for case in TARGET_CASES]
    validations = _validations(skeleton, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": SCORECARD,
        "gate": "adaptive_lr_native_state_machine_cpu_reference_guard",
        "ok": ready,
        "promotion_ready": False,
        "native_state_machine_cpu_reference_guard_ready": ready,
        "report_only": True,
        "fallback_authority": "existing_python_or_third_party_optimizer",
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "rows": rows,
        "abi_skeleton_summary": _as_dict(skeleton.get("summary")),
        "validations": validations,
        "summary": {
            "target_count": len(rows),
            "cpu_reference_guard_ready_count": sum(
                1 for row in rows if row.get("cpu_reference_guard_ready") is True
            ),
            "valid_launch_plan_guard_passed_count": sum(
                1 for row in rows if row.get("valid_launch_plan_guard_passed") is True
            ),
            "bad_finite_scalar_guard_rejected_count": sum(
                1 for row in rows if row.get("bad_finite_scalar_guard_rejected") is True
            ),
            "bad_dispatch_guard_rejected_count": sum(
                1 for row in rows if row.get("bad_dispatch_guard_rejected") is True
            ),
            "state_machine_abi_implementation_ready_count": 0,
            "native_kernel_preconditions_implementation_ready_count": 0,
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adaptive_lr_native_state_machine_implementation_missing",
                "adaptive_lr_cuda_kernel_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement adaptive-LR native state-machine implementation stub with dispatch still default-off"
            if ready
            else "fix adaptive-LR CPU reference guard blockers"
        ),
        "notes": [
            "This scorecard validates launch-plan guard behavior using CPU-only artifact checks.",
            "It proves valid plans pass while non-finite scalar and dispatch-enabled plans are rejected.",
            "It does not register native entrypoints, implement CUDA kernels, or alter request/schema/UI behavior.",
        ],
    }


def _row(optimizer_type: str, skeleton_report: Mapping[str, Any]) -> dict[str, Any]:
    source = _skeleton_rows(skeleton_report).get(optimizer_type, {})
    skeleton_ready = source.get("native_state_machine_abi_skeleton_ready") is True
    skeleton = _as_dict(source.get("native_state_machine_abi_skeleton"))
    valid_plan = _launch_plan(optimizer_type, skeleton, dispatch=False, finite=True)
    bad_scalar_plan = _launch_plan(optimizer_type, skeleton, dispatch=False, finite=False)
    bad_dispatch_plan = _launch_plan(optimizer_type, skeleton, dispatch=True, finite=True)
    valid_guard = _guard(valid_plan, skeleton)
    bad_scalar_guard = _guard(bad_scalar_plan, skeleton)
    bad_dispatch_guard = _guard(bad_dispatch_plan, skeleton)
    ready = (
        skeleton_ready
        and valid_guard["ok"] is True
        and bad_scalar_guard["ok"] is False
        and bad_dispatch_guard["ok"] is False
    )
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": str(source.get("family") or _family(optimizer_type)),
        "state_machine_status": "cpu_reference_guard_ready" if ready else "cpu_reference_guard_blocked",
        "native_state_machine_abi_skeleton_ready": skeleton_ready,
        "cpu_reference_guard_ready": ready,
        "valid_launch_plan_guard_passed": valid_guard["ok"] is True,
        "bad_finite_scalar_guard_rejected": bad_scalar_guard["ok"] is False
        and "non_finite_dynamic_scalar" in bad_scalar_guard["blocked_reasons"],
        "bad_dispatch_guard_rejected": bad_dispatch_guard["ok"] is False
        and "dispatch_must_remain_disabled" in bad_dispatch_guard["blocked_reasons"],
        "cpu_reference_guard": valid_guard,
        "negative_guard_cases": {
            "non_finite_scalar": bad_scalar_guard,
            "dispatch_enabled": bad_dispatch_guard,
        },
        "state_machine_abi_implementation_ready": False,
        "native_kernel_preconditions_implementation_ready": False,
        "runtime_authority": "existing_python_or_third_party_optimizer",
        "native_route": "none_report_only",
        "product_native_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": "adaptive_lr_native_state_machine_implementation_stub",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_adaptive_lr_cpu_reference_guard_blocked"],
    }


def _launch_plan(optimizer_type: str, skeleton: Mapping[str, Any], *, dispatch: bool, finite: bool) -> dict[str, Any]:
    family = str(skeleton.get("adaptive_lr_family") or _family(optimizer_type))
    scalars = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "step": 2,
        "d": 1e-5,
        "d0": 1e-6,
        "growth_rate": 1.01,
    }
    if not finite:
        scalars["d"] = math.inf
    return {
        "schema_version": 1,
        "schema": "adaptive_lr_state_machine_launch_plan_v0",
        "optimizer_type": optimizer_type,
        "optimizer_family": family,
        "state_buffer_mapping": _as_dict(skeleton.get("state_buffer_mapping")),
        "scalar_materialization": {
            "scalars": scalars,
            "finite_guard_required": True,
        },
        "quality_guards": ("finite_dynamic_lr", "finite_global_d", "dispatch_disabled"),
        "dispatch_policy": {
            "training_path_enabled": bool(dispatch),
            "runtime_dispatch_ready": bool(dispatch),
            "native_dispatch_allowed": bool(dispatch),
            "product_native_ready": False,
        },
    }


def _guard(plan: Mapping[str, Any], skeleton: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if plan.get("schema") != "adaptive_lr_state_machine_launch_plan_v0":
        blockers.append("invalid_launch_plan_schema")
    required_fields = _strings(_as_dict(skeleton.get("launch_plan_schema")).get("required_fields"))
    for field in required_fields:
        if field not in plan:
            blockers.append(f"missing_launch_plan_field:{field}")
    scalars = _as_dict(_as_dict(plan.get("scalar_materialization")).get("scalars"))
    if not all(_finite_number(value) for value in scalars.values()):
        blockers.append("non_finite_dynamic_scalar")
    dispatch = _as_dict(plan.get("dispatch_policy"))
    if (
        dispatch.get("training_path_enabled") is True
        or dispatch.get("runtime_dispatch_ready") is True
        or dispatch.get("native_dispatch_allowed") is True
    ):
        blockers.append("dispatch_must_remain_disabled")
    required_buffers = _strings(_as_dict(skeleton.get("state_buffer_mapping")).get("required_buffers"))
    reported_buffers = _strings(_as_dict(plan.get("state_buffer_mapping")).get("required_buffers"))
    missing_buffers = [name for name in required_buffers if name not in reported_buffers]
    blockers.extend(f"missing_state_buffer:{name}" for name in missing_buffers)
    return {
        "schema_version": 1,
        "guard": "adaptive_lr_state_machine_cpu_reference_guard_v0",
        "ok": not blockers,
        "report_only": True,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "blocked_reasons": _dedupe(blockers),
    }


def _validations(skeleton: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {case.optimizer.value for case in TARGET_CASES}
    present = {str(row.get("optimizer_type") or "") for row in rows}
    return [
        _validation(
            "abi_skeleton_ready",
            skeleton.get("native_state_machine_abi_skeleton_ready") is True,
            "adaptive_lr_native_abi_skeleton_missing",
        ),
        _validation("optimizer_set_complete", present == expected, "adaptive_lr_cpu_guard_optimizer_set_incomplete"),
        _validation(
            "all_cpu_guards_ready",
            all(row.get("cpu_reference_guard_ready") is True for row in rows),
            "adaptive_lr_cpu_reference_guard_not_ready",
        ),
        _validation(
            "negative_cases_rejected",
            all(row.get("bad_finite_scalar_guard_rejected") is True for row in rows)
            and all(row.get("bad_dispatch_guard_rejected") is True for row in rows),
            "adaptive_lr_cpu_reference_guard_failed_negative_case",
        ),
        _validation(
            "runtime_dispatch_disabled",
            skeleton.get("training_path_enabled") is False
            and skeleton.get("runtime_dispatch_ready") is False
            and skeleton.get("native_dispatch_allowed") is False
            and all(row.get("training_path_enabled") is False for row in rows)
            and all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows),
            "adaptive_lr_cpu_reference_guard_enabled_dispatch",
        ),
    ]


def _skeleton_rows(skeleton: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("optimizer_type") or ""): row
        for row in skeleton.get("rows", [])
        if isinstance(row, Mapping)
    }


def _family(optimizer_type: str) -> str:
    if optimizer_type in {"AutoProdigy", "prodigy", "prodigyplus.ProdigyPlusScheduleFree"}:
        return "adaptive_lr_prodigy"
    return "adaptive_lr_dadapt"


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["SCORECARD", "build_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard"]
