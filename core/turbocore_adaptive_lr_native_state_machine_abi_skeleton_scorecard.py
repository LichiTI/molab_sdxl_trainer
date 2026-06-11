"""Default-off native ABI skeleton contracts for built-in adaptive-LR optimizers."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_adaptive_lr_native_state_machine_abi_preconditions_scorecard import (
    build_adaptive_lr_native_state_machine_abi_preconditions_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES


SCORECARD = "turbocore_adaptive_lr_native_state_machine_abi_skeleton_scorecard_v0"
ENTRYPOINTS = (
    "create_adaptive_lr_state_machine",
    "adaptive_lr_state_machine_materialize_scalars",
    "adaptive_lr_state_machine_reduce_global_d",
    "adaptive_lr_state_machine_build_launch_plan",
    "adaptive_lr_state_machine_snapshot",
    "adaptive_lr_state_machine_load_state",
    "destroy_adaptive_lr_state_machine",
)


def build_adaptive_lr_native_state_machine_abi_skeleton_scorecard(
    *,
    abi_preconditions_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ABI skeleton artifacts without enabling native execution."""

    preconditions = _as_dict(
        abi_preconditions_report or build_adaptive_lr_native_state_machine_abi_preconditions_scorecard()
    )
    rows = [_row(case.optimizer.value, preconditions) for case in TARGET_CASES]
    validations = _validations(preconditions, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": SCORECARD,
        "gate": "adaptive_lr_native_state_machine_abi_skeleton",
        "ok": ready,
        "promotion_ready": False,
        "native_state_machine_abi_skeleton_ready": ready,
        "report_only": True,
        "fallback_authority": "existing_python_or_third_party_optimizer",
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "entrypoints": list(ENTRYPOINTS),
        "rows": rows,
        "abi_precondition_summary": _as_dict(preconditions.get("summary")),
        "validations": validations,
        "summary": {
            "target_count": len(rows),
            "native_state_machine_abi_skeleton_ready_count": sum(
                1 for row in rows if row.get("native_state_machine_abi_skeleton_ready") is True
            ),
            "state_machine_entrypoint_contract_ready_count": sum(
                1 for row in rows if row.get("state_machine_entrypoint_contract_ready") is True
            ),
            "launch_plan_schema_ready_count": sum(
                1 for row in rows if row.get("launch_plan_schema_ready") is True
            ),
            "state_buffer_mapping_contract_ready_count": sum(
                1 for row in rows if row.get("state_buffer_mapping_contract_ready") is True
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
                "adaptive_lr_native_state_machine_cpu_reference_guard_missing",
                "adaptive_lr_cuda_kernel_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement adaptive-LR native state-machine CPU reference guard with dispatch still default-off"
            if ready
            else "fix adaptive-LR native ABI skeleton blockers"
        ),
        "notes": [
            "This scorecard names ABI skeleton entrypoints, state buffers, scalar phases, and launch schema only.",
            "It does not register native functions, implement CUDA kernels, or change request/schema/UI behavior.",
            "The Python or third-party optimizer remains the runtime authority.",
        ],
    }


def _row(optimizer_type: str, preconditions: Mapping[str, Any]) -> dict[str, Any]:
    source = _precondition_rows(preconditions).get(optimizer_type, {})
    precondition_ready = source.get("native_state_machine_abi_precondition_review_ready") is True
    family = str(source.get("family") or _family(optimizer_type))
    skeleton = _skeleton_contract(optimizer_type, family, precondition_ready)
    ready = precondition_ready and skeleton["review_ready"] is True
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": family,
        "state_machine_status": (
            "native_state_machine_abi_skeleton_ready"
            if ready
            else "native_state_machine_abi_skeleton_blocked"
        ),
        "native_state_machine_abi_precondition_review_ready": precondition_ready,
        "native_state_machine_abi_skeleton_ready": ready,
        "state_machine_entrypoint_contract_ready": skeleton["entrypoint_contract"]["review_ready"] is True,
        "launch_plan_schema_ready": skeleton["launch_plan_schema"]["review_ready"] is True,
        "state_buffer_mapping_contract_ready": skeleton["state_buffer_mapping"]["review_ready"] is True,
        "state_machine_abi_implementation_ready": False,
        "native_kernel_preconditions_implementation_ready": False,
        "native_state_machine_abi_skeleton": skeleton,
        "runtime_authority": "existing_python_or_third_party_optimizer",
        "native_route": "none_report_only",
        "product_native_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": "adaptive_lr_native_state_machine_cpu_reference_guard",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_adaptive_lr_native_abi_skeleton_blocked"],
    }


def _skeleton_contract(optimizer_type: str, family: str, ready: bool) -> dict[str, Any]:
    if family == "adaptive_lr_prodigy":
        global_state = ("d", "d0", "d_max", "dlr", "growth_rate", "p0_norm")
        state_buffers = ("param_flat", "grad_flat", "exp_avg", "exp_avg_sq", "s", "p0")
        scalar_phase = "prodigy_global_distance_materialization"
    else:
        global_state = ("d", "d0", "growth_rate", "gsq_weighted", "sk_l1")
        state_buffers = ("param_flat", "grad_flat", "exp_avg", "exp_avg_sq", "sk", "variant_accumulator")
        scalar_phase = "dadapt_global_d_materialization"
    return {
        "schema_version": 1,
        "artifact_kind": "builtin_adaptive_lr_native_state_machine_abi_skeleton_v0",
        "report_only": True,
        "optimizer_type": optimizer_type,
        "adaptive_lr_family": family,
        "review_ready": ready,
        "implementation_ready": False,
        "entrypoint_contract": {
            "review_ready": ready,
            "implementation_ready": False,
            "entrypoints": ENTRYPOINTS,
            "lifecycle": (
                "create",
                "load_state",
                "materialize_scalars",
                "reduce_global_state",
                "build_launch_plan",
                "snapshot",
                "destroy",
            ),
        },
        "state_buffer_mapping": {
            "review_ready": ready,
            "implementation_ready": False,
            "layout": "flat_contiguous_state_machine_buffers",
            "required_buffers": state_buffers,
            "global_state_fields": global_state,
            "resume_source": "optimizer.state_dict",
        },
        "scalar_materialization": {
            "review_ready": ready,
            "implementation_ready": False,
            "phase": scalar_phase,
            "required_scalars": ("lr", "weight_decay", "step", "d", "d0", "growth_rate"),
            "finite_guard_required": True,
        },
        "launch_plan_schema": {
            "review_ready": ready,
            "implementation_ready": False,
            "schema": "adaptive_lr_state_machine_launch_plan_v0",
            "required_fields": (
                "optimizer_type",
                "optimizer_family",
                "state_buffer_mapping",
                "scalar_materialization",
                "quality_guards",
                "dispatch_policy",
            ),
        },
        "dispatch_policy": {
            "review_ready": ready,
            "implementation_ready": False,
            "training_path_enabled": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "product_native_ready": False,
        },
    }


def _validations(preconditions: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {case.optimizer.value for case in TARGET_CASES}
    present = {str(row.get("optimizer_type") or "") for row in rows}
    return [
        _validation(
            "abi_preconditions_ready",
            preconditions.get("native_state_machine_abi_preconditions_ready") is True,
            "adaptive_lr_native_abi_preconditions_missing",
        ),
        _validation(
            "optimizer_set_complete",
            present == expected,
            "adaptive_lr_native_abi_skeleton_optimizer_set_incomplete",
        ),
        _validation(
            "all_skeleton_contracts_ready",
            all(row.get("native_state_machine_abi_skeleton_ready") is True for row in rows),
            "adaptive_lr_native_abi_skeleton_not_ready",
        ),
        _validation(
            "implementation_not_claimed",
            all(row.get("state_machine_abi_implementation_ready") is False for row in rows)
            and all(row.get("native_kernel_preconditions_implementation_ready") is False for row in rows),
            "adaptive_lr_native_abi_skeleton_unexpected_implementation_claim",
        ),
        _validation(
            "runtime_dispatch_disabled",
            preconditions.get("training_path_enabled") is False
            and preconditions.get("runtime_dispatch_ready") is False
            and preconditions.get("native_dispatch_allowed") is False
            and all(row.get("training_path_enabled") is False for row in rows)
            and all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows),
            "adaptive_lr_native_abi_skeleton_enabled_dispatch",
        ),
    ]


def _precondition_rows(preconditions: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("optimizer_type") or ""): row
        for row in preconditions.get("rows", [])
        if isinstance(row, Mapping)
    }


def _family(optimizer_type: str) -> str:
    if optimizer_type in {"AutoProdigy", "prodigy", "prodigyplus.ProdigyPlusScheduleFree"}:
        return "adaptive_lr_prodigy"
    return "adaptive_lr_dadapt"


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ENTRYPOINTS", "SCORECARD", "build_adaptive_lr_native_state_machine_abi_skeleton_scorecard"]
