"""Default-off native ABI preconditions for built-in adaptive-LR optimizers."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import (
    TARGET_CASES,
    build_adaptive_lr_state_machine_replay_executor_scorecard,
)


SCORECARD = "turbocore_adaptive_lr_native_state_machine_abi_preconditions_scorecard_v0"


def build_adaptive_lr_native_state_machine_abi_preconditions_scorecard(
    *,
    replay_executor_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Review native ABI preconditions without claiming implementation or dispatch."""

    executor = _as_dict(
        replay_executor_report or build_adaptive_lr_state_machine_replay_executor_scorecard()
    )
    rows = [_row(case.optimizer.value, executor) for case in TARGET_CASES]
    validations = _validations(executor, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": SCORECARD,
        "gate": "adaptive_lr_native_state_machine_abi_preconditions",
        "ok": ready,
        "promotion_ready": False,
        "native_state_machine_abi_preconditions_ready": ready,
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
        "replay_executor_summary": _as_dict(executor.get("summary")),
        "validations": validations,
        "summary": {
            "target_count": len(rows),
            "native_state_machine_abi_precondition_review_ready_count": sum(
                1 for row in rows if row.get("native_state_machine_abi_precondition_review_ready") is True
            ),
            "native_state_machine_abi_precondition_package_ready_count": sum(
                1 for row in rows if row.get("native_state_machine_abi_precondition_package_ready") is True
            ),
            "native_kernel_precondition_review_ready_count": sum(
                1 for row in rows if row.get("native_kernel_precondition_review_ready") is True
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
                "adaptive_lr_native_state_machine_abi_implementation_missing",
                "adaptive_lr_cuda_kernel_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement adaptive-LR native state-machine ABI skeleton with dispatch still default-off"
            if ready
            else "fix adaptive-LR native ABI precondition blockers"
        ),
        "notes": [
            "This scorecard converts replay-executor evidence into native ABI precondition review packages.",
            "It does not claim ABI implementation, CUDA kernels, request/schema/UI changes, or product dispatch.",
            "The Python or third-party optimizer remains the runtime authority.",
        ],
    }


def _row(optimizer_type: str, executor: Mapping[str, Any]) -> dict[str, Any]:
    source = _executor_rows(executor).get(optimizer_type, {})
    executor_ready = source.get("reference_replay_executor_ready") is True
    family = _family(optimizer_type)
    package = _precondition_package(optimizer_type, family, executor_ready)
    ready = executor_ready and package["review_ready"] is True
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": family,
        "state_machine_status": (
            "native_state_machine_abi_precondition_review_ready"
            if ready
            else "native_state_machine_abi_precondition_blocked"
        ),
        "reference_replay_executor_ready": executor_ready,
        "resume_next_step_parity_passed": source.get("resume_next_step_parity_passed") is True,
        "native_state_machine_abi_precondition_review_ready": ready,
        "native_state_machine_abi_precondition_package_ready": package["review_ready"] is True,
        "native_kernel_precondition_review_ready": package["native_kernel_preconditions"]["review_ready"] is True,
        "state_machine_abi_implementation_ready": False,
        "native_kernel_preconditions_implementation_ready": False,
        "native_state_machine_abi_precondition_package": package,
        "runtime_authority": "existing_python_or_third_party_optimizer",
        "native_route": "none_report_only",
        "product_native_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": "adaptive_lr_native_state_machine_abi_skeleton",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_adaptive_lr_native_abi_preconditions_blocked"],
    }


def _precondition_package(optimizer_type: str, family: str, executor_ready: bool) -> dict[str, Any]:
    scalar_fields = ("lr", "d", "d0", "growth_rate", "step")
    if family == "adaptive_lr_prodigy":
        state_fields = ("exp_avg", "exp_avg_sq", "d", "d_max", "dlr", "s", "p0_norm")
        reduction_policy = "prodigy_distance_reduction_before_param_update"
        variant_preconditions = (
            "materialize dlr and d_max before launching parameter kernels",
            "preserve decoupled weight decay policy per param group",
            "schedule-free variant must expose z-buffer ownership separately",
        )
    else:
        state_fields = ("exp_avg", "exp_avg_sq", "sk", "gsq_weighted", "d", "growth_rate")
        reduction_policy = "dadapt_global_d_reduction_before_param_update"
        variant_preconditions = (
            "materialize variant accumulator buffers before launch",
            "derive lr scalar from d-policy rather than fixed AdamW lr",
            "preserve DAdapt variant-specific denominator and clipping policy",
        )

    return {
        "schema_version": 1,
        "artifact_kind": "builtin_adaptive_lr_native_state_machine_abi_preconditions_v0",
        "report_only": True,
        "optimizer_type": optimizer_type,
        "adaptive_lr_family": family,
        "review_ready": executor_ready,
        "implementation_ready": False,
        "state_machine_boundary": {
            "review_ready": executor_ready,
            "implementation_ready": False,
            "phases": (
                "load_resume_state",
                "materialize_dynamic_scalars",
                "run_global_d_reduction",
                "launch_param_update",
                "write_back_global_state",
            ),
            "fallback_authority": "existing_python_or_third_party_optimizer",
        },
        "scalar_launch_inputs": {
            "review_ready": executor_ready,
            "implementation_ready": False,
            "fields": scalar_fields,
            "materialization_policy": "host_side_before_native_boundary",
        },
        "state_layout_inputs": {
            "review_ready": executor_ready,
            "implementation_ready": False,
            "state_fields": state_fields,
            "resume_source": "optimizer.state_dict",
        },
        "native_kernel_preconditions": {
            "review_ready": executor_ready,
            "implementation_ready": False,
            "reduction_policy": reduction_policy,
            "preconditions": (
                "reference replay executor parity must pass before ABI implementation",
                "finite dynamic lr and global d guards must run before dispatch",
                "state-dict keys must map losslessly to native state buffers",
                *variant_preconditions,
            ),
        },
        "dispatch_policy": {
            "review_ready": executor_ready,
            "implementation_ready": False,
            "training_path_enabled": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "product_native_ready": False,
        },
    }


def _validations(executor: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {case.optimizer.value for case in TARGET_CASES}
    present = {str(row.get("optimizer_type") or "") for row in rows}
    return [
        _validation(
            "replay_executor_ready",
            executor.get("state_machine_replay_executor_ready") is True,
            "adaptive_lr_replay_executor_missing",
        ),
        _validation(
            "optimizer_set_complete",
            present == expected,
            "adaptive_lr_native_abi_precondition_optimizer_set_incomplete",
        ),
        _validation(
            "all_precondition_packages_ready",
            all(row.get("native_state_machine_abi_precondition_package_ready") is True for row in rows),
            "adaptive_lr_native_abi_precondition_package_not_ready",
        ),
        _validation(
            "implementation_not_claimed",
            all(row.get("state_machine_abi_implementation_ready") is False for row in rows)
            and all(row.get("native_kernel_preconditions_implementation_ready") is False for row in rows),
            "adaptive_lr_native_abi_precondition_unexpected_implementation_claim",
        ),
        _validation(
            "runtime_dispatch_disabled",
            executor.get("training_path_enabled") is False
            and executor.get("runtime_dispatch_ready") is False
            and executor.get("native_dispatch_allowed") is False
            and all(row.get("training_path_enabled") is False for row in rows)
            and all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows),
            "adaptive_lr_native_abi_precondition_enabled_dispatch",
        ),
    ]


def _executor_rows(executor: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("optimizer_type") or ""): row
        for row in executor.get("rows", [])
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


__all__ = ["SCORECARD", "build_adaptive_lr_native_state_machine_abi_preconditions_scorecard"]
