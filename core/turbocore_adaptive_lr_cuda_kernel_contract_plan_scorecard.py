"""Default-off CUDA kernel contract plans for built-in adaptive-LR optimizers."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_adaptive_lr_native_state_machine_implementation_stub_scorecard import (
    build_adaptive_lr_native_state_machine_implementation_stub_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES


SCORECARD = "turbocore_adaptive_lr_cuda_kernel_contract_plan_scorecard_v0"
LAUNCH_PLAN_SCHEMA = "adaptive_lr_state_machine_launch_plan_v0"
KERNELS_BY_FAMILY = {
    "adaptive_lr_prodigy": (
        "lulynx_adaptive_lr_prodigy_distance_reduce_v0",
        "lulynx_adaptive_lr_prodigy_apply_v0",
    ),
    "adaptive_lr_dadapt": (
        "lulynx_adaptive_lr_dadapt_global_d_reduce_v0",
        "lulynx_adaptive_lr_dadapt_apply_v0",
    ),
}


def build_adaptive_lr_cuda_kernel_contract_plan_scorecard(
    *,
    implementation_stub_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build review-only CUDA contract evidence without registering kernels."""

    stub = _as_dict(
        implementation_stub_report or build_adaptive_lr_native_state_machine_implementation_stub_scorecard()
    )
    rows = [_row(case.optimizer.value, stub) for case in TARGET_CASES]
    validations = _validations(stub, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": SCORECARD,
        "gate": "adaptive_lr_cuda_kernel_contract_plan",
        "ok": ready,
        "promotion_ready": False,
        "cuda_kernel_contract_plan_ready": ready,
        "runtime_canary_manifest_ready": ready,
        "report_only": True,
        "optimizer_family": "built_in_adaptive_lr",
        "launch_plan_schema": LAUNCH_PLAN_SCHEMA,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "cuda_kernel_implementation_ready": False,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "rows": rows,
        "implementation_stub_summary": _as_dict(stub.get("summary")),
        "validations": validations,
        "summary": _summary(rows),
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adaptive_lr_cuda_kernel_implementation_missing",
                "adaptive_lr_training_tensor_binding_missing",
                "adaptive_lr_runtime_canary_execution_missing",
                "adaptive_lr_product_rollout_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement adaptive-LR CUDA kernels behind the existing default-off runtime boundary"
            if ready
            else "fix adaptive-LR CUDA contract blockers"
        ),
        "notes": [
            "This is a CUDA contract and canary manifest plan only.",
            "It does not compile, register, or launch CUDA kernels.",
            "Request/UI/schema and product native dispatch remain unchanged.",
        ],
    }


def _row(optimizer_type: str, stub_report: Mapping[str, Any]) -> dict[str, Any]:
    source = _stub_rows(stub_report).get(optimizer_type, {})
    stub_ready = source.get("implementation_stub_ready") is True
    family = str(source.get("family") or _family(optimizer_type))
    contract = _contract(optimizer_type, family, stub_ready)
    ready = stub_ready and contract["contract_ready"] is True
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": family,
        "state_machine_status": "cuda_kernel_contract_plan_ready" if ready else "cuda_kernel_contract_plan_blocked",
        "implementation_stub_ready": stub_ready,
        "cuda_kernel_contract_plan_ready": ready,
        "runtime_canary_manifest_ready": ready,
        "launch_plan_schema": LAUNCH_PLAN_SCHEMA,
        "cuda_kernel_implementation_ready": False,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "contract": contract,
        "next_gate": "adaptive_lr_cuda_kernel_implementation",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_adaptive_lr_cuda_contract_plan_blocked"],
    }


def _contract(optimizer_type: str, family: str, ready: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "builtin_adaptive_lr_cuda_kernel_contract_plan_v0",
        "optimizer_type": optimizer_type,
        "adaptive_lr_family": family,
        "contract_ready": ready,
        "implementation_ready": False,
        "launch_plan_schema": LAUNCH_PLAN_SCHEMA,
        "planned_kernel_names": KERNELS_BY_FAMILY[family],
        "required_buffers": (
            "param",
            "grad",
            "exp_avg",
            "exp_avg_sq",
            "adaptive_lr_state",
            "finite_scalar_workspace",
        ),
        "required_scalars": ("lr", "beta1", "beta2", "eps", "weight_decay", "global_d", "dynamic_lr"),
        "quality_guards": (
            "finite_scalar_guard",
            "resume_next_step_parity_guard",
            "dispatch_disabled_guard",
            "python_optimizer_authority_guard",
        ),
        "reduction_boundary": "global_d_or_dynamic_lr_reduce_before_apply_kernel",
        "runtime_canary_manifest": {
            "manifest_ready": ready,
            "runtime_canary_ready": False,
            "runtime_canary_hit": False,
            "canary_shadow_route_only": True,
        },
        "missing_before_dispatch": (
            "cuda_kernel_implementation",
            "runtime_tensor_binding",
            "training_loop_canary",
            "product_rollout_review",
        ),
    }


def _validations(stub_report: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {case.optimizer.value for case in TARGET_CASES}
    present = {str(row.get("optimizer_type") or "") for row in rows}
    return [
        _validation(
            "implementation_stub_ready",
            stub_report.get("native_state_machine_implementation_stub_ready") is True,
            "adaptive_lr_implementation_stub_missing",
        ),
        _validation("optimizer_set_complete", present == expected, "adaptive_lr_cuda_contract_optimizer_set_incomplete"),
        _validation(
            "all_contract_plans_ready",
            all(row.get("cuda_kernel_contract_plan_ready") is True for row in rows),
            "adaptive_lr_cuda_contract_plan_not_ready",
        ),
        _validation(
            "runtime_dispatch_disabled",
            stub_report.get("training_path_enabled") is False
            and stub_report.get("runtime_dispatch_ready") is False
            and stub_report.get("native_dispatch_allowed") is False
            and all(row.get("training_path_enabled") is False for row in rows)
            and all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows),
            "adaptive_lr_cuda_contract_enabled_dispatch",
        ),
    ]


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "target_count": len(rows),
        "cuda_kernel_contract_plan_ready_count": sum(
            1 for row in rows if row.get("cuda_kernel_contract_plan_ready") is True
        ),
        "runtime_canary_manifest_ready_count": sum(
            1 for row in rows if row.get("runtime_canary_manifest_ready") is True
        ),
        "cuda_kernel_implementation_ready_count": 0,
        "runtime_canary_ready_count": 0,
        "runtime_canary_hit_count": 0,
        "product_native_ready_count": 0,
        "runtime_dispatch_ready_count": 0,
        "native_dispatch_allowed_count": 0,
        "training_path_enabled_count": 0,
        "default_behavior_changed_count": 0,
    }


def _stub_rows(stub_report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("optimizer_type") or ""): row
        for row in stub_report.get("rows", [])
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


__all__ = ["SCORECARD", "LAUNCH_PLAN_SCHEMA", "build_adaptive_lr_cuda_kernel_contract_plan_scorecard"]
