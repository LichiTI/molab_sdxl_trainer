"""Default-off implementation stub artifacts for built-in adaptive-LR state machines."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard import (
    build_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES


SCORECARD = "turbocore_adaptive_lr_native_state_machine_implementation_stub_scorecard_v0"
STUB_ENTRYPOINTS = (
    "adaptive_lr_stub_create",
    "adaptive_lr_stub_load_state",
    "adaptive_lr_stub_materialize_scalars",
    "adaptive_lr_stub_validate_launch_plan",
    "adaptive_lr_stub_snapshot",
    "adaptive_lr_stub_destroy",
)


def build_adaptive_lr_native_state_machine_implementation_stub_scorecard(
    *,
    cpu_guard_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build native implementation stub evidence without registering native dispatch."""

    cpu_guard = _as_dict(cpu_guard_report or build_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard())
    rows = [_row(case.optimizer.value, cpu_guard) for case in TARGET_CASES]
    validations = _validations(cpu_guard, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": SCORECARD,
        "gate": "adaptive_lr_native_state_machine_implementation_stub",
        "ok": ready,
        "promotion_ready": False,
        "native_state_machine_implementation_stub_ready": ready,
        "report_only": True,
        "fallback_authority": "existing_python_or_third_party_optimizer",
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "stub_entrypoints": list(STUB_ENTRYPOINTS),
        "rows": rows,
        "cpu_guard_summary": _as_dict(cpu_guard.get("summary")),
        "validations": validations,
        "summary": {
            "target_count": len(rows),
            "implementation_stub_ready_count": sum(
                1 for row in rows if row.get("implementation_stub_ready") is True
            ),
            "stub_entrypoint_contract_ready_count": sum(
                1 for row in rows if row.get("stub_entrypoint_contract_ready") is True
            ),
            "stub_state_transition_contract_ready_count": sum(
                1 for row in rows if row.get("stub_state_transition_contract_ready") is True
            ),
            "stub_dispatch_disabled_assertion_ready_count": sum(
                1 for row in rows if row.get("stub_dispatch_disabled_assertion_ready") is True
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
                "adaptive_lr_cuda_kernel_contract_missing",
                "adaptive_lr_runtime_canary_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "define adaptive-LR CUDA kernel contract and runtime canary plan with dispatch still default-off"
            if ready
            else "fix adaptive-LR implementation stub blockers"
        ),
        "notes": [
            "This scorecard defines a native implementation skeleton contract only.",
            "It does not register PyO3 entrypoints, launch CUDA kernels, or change request/schema/UI behavior.",
            "The Python or third-party optimizer remains the runtime authority.",
        ],
    }


def _row(optimizer_type: str, cpu_guard: Mapping[str, Any]) -> dict[str, Any]:
    source = _guard_rows(cpu_guard).get(optimizer_type, {})
    guard_ready = source.get("cpu_reference_guard_ready") is True
    family = str(source.get("family") or _family(optimizer_type))
    stub = _stub_contract(optimizer_type, family, guard_ready)
    ready = guard_ready and stub["review_ready"] is True
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": family,
        "state_machine_status": "implementation_stub_ready" if ready else "implementation_stub_blocked",
        "cpu_reference_guard_ready": guard_ready,
        "implementation_stub_ready": ready,
        "stub_entrypoint_contract_ready": stub["entrypoint_contract"]["review_ready"] is True,
        "stub_state_transition_contract_ready": stub["state_transition_contract"]["review_ready"] is True,
        "stub_dispatch_disabled_assertion_ready": stub["dispatch_disabled_assertion"]["review_ready"] is True,
        "state_machine_abi_implementation_ready": False,
        "native_kernel_preconditions_implementation_ready": False,
        "implementation_stub": stub,
        "runtime_authority": "existing_python_or_third_party_optimizer",
        "native_route": "none_report_only",
        "product_native_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": "adaptive_lr_cuda_kernel_contract_plan",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_adaptive_lr_implementation_stub_blocked"],
    }


def _stub_contract(optimizer_type: str, family: str, ready: bool) -> dict[str, Any]:
    estimator_phase = "prodigy_distance_estimator" if family == "adaptive_lr_prodigy" else "dadapt_global_d_estimator"
    return {
        "schema_version": 1,
        "artifact_kind": "builtin_adaptive_lr_native_state_machine_implementation_stub_v0",
        "report_only": True,
        "optimizer_type": optimizer_type,
        "adaptive_lr_family": family,
        "review_ready": ready,
        "implementation_ready": False,
        "entrypoint_contract": {
            "review_ready": ready,
            "implementation_ready": False,
            "entrypoints": STUB_ENTRYPOINTS,
            "registration_policy": "not_registered_report_only",
        },
        "state_transition_contract": {
            "review_ready": ready,
            "implementation_ready": False,
            "states": ("created", "state_loaded", "scalars_materialized", "launch_plan_validated", "snapshotted", "destroyed"),
            "estimator_phase": estimator_phase,
            "resume_parity_authority": "trainer_path_replay_executor",
        },
        "dispatch_disabled_assertion": {
            "review_ready": ready,
            "implementation_ready": False,
            "training_path_enabled": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "product_native_ready": False,
        },
    }


def _validations(cpu_guard: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {case.optimizer.value for case in TARGET_CASES}
    present = {str(row.get("optimizer_type") or "") for row in rows}
    return [
        _validation(
            "cpu_reference_guard_ready",
            cpu_guard.get("native_state_machine_cpu_reference_guard_ready") is True,
            "adaptive_lr_cpu_reference_guard_missing",
        ),
        _validation("optimizer_set_complete", present == expected, "adaptive_lr_implementation_stub_optimizer_set_incomplete"),
        _validation(
            "all_implementation_stubs_ready",
            all(row.get("implementation_stub_ready") is True for row in rows),
            "adaptive_lr_implementation_stub_not_ready",
        ),
        _validation(
            "implementation_not_claimed",
            all(row.get("state_machine_abi_implementation_ready") is False for row in rows),
            "adaptive_lr_implementation_stub_unexpected_native_claim",
        ),
        _validation(
            "runtime_dispatch_disabled",
            cpu_guard.get("training_path_enabled") is False
            and cpu_guard.get("runtime_dispatch_ready") is False
            and cpu_guard.get("native_dispatch_allowed") is False
            and all(row.get("training_path_enabled") is False for row in rows)
            and all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows),
            "adaptive_lr_implementation_stub_enabled_dispatch",
        ),
    ]


def _guard_rows(cpu_guard: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("optimizer_type") or ""): row
        for row in cpu_guard.get("rows", [])
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


__all__ = ["SCORECARD", "STUB_ENTRYPOINTS", "build_adaptive_lr_native_state_machine_implementation_stub_scorecard"]
