"""Replay/resume matrix artifact for built-in adaptive-LR state machines."""

from __future__ import annotations

from typing import Any, Mapping

from core.configs import OptimizerType
from core.turbocore_adaptive_lr_state_machine_batch_scorecard import (
    TARGET_OPTIMIZERS,
    build_adaptive_lr_state_machine_batch_scorecard,
)


SCORECARD = "turbocore_adaptive_lr_state_machine_replay_matrix_scorecard_v0"
MATRIX_KIND = "builtin_adaptive_lr_state_machine_replay_matrix_v0"


def build_adaptive_lr_state_machine_replay_matrix_scorecard(
    *,
    batch_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build planned replay/resume matrix artifacts without native dispatch."""

    batch = _as_dict(batch_report or build_adaptive_lr_state_machine_batch_scorecard())
    rows = [_row(optimizer, batch) for optimizer in TARGET_OPTIMIZERS]
    validations = _validations(batch, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": SCORECARD,
        "gate": "adaptive_lr_state_machine_replay_matrix",
        "ok": ready,
        "promotion_ready": False,
        "state_machine_replay_matrix_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "rows": rows,
        "batch_summary": _as_dict(batch.get("summary")),
        "validations": validations,
        "summary": {
            "target_count": len(rows),
            "state_machine_replay_matrix_artifact_ready_count": sum(
                1 for row in rows if _as_dict(row.get("state_machine_replay_matrix_artifact")).get("spec_ready") is True
            ),
            "state_machine_replay_matrix_implementation_ready_count": 0,
            "state_machine_replay_case_planned_count": sum(
                len(_as_dict(row.get("state_machine_replay_matrix_artifact")).get("replay_cases", []))
                for row in rows
            ),
            "state_machine_replay_resume_case_planned_count": sum(
                len(_as_dict(row.get("state_machine_replay_matrix_artifact")).get("resume_replay_cases", []))
                for row in rows
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
                "adaptive_lr_state_machine_abi_implementation_missing",
                "adaptive_lr_resume_replay_not_executed",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement adaptive-LR state-machine ABI replay executor with dispatch still default-off"
            if ready
            else "fix adaptive-LR replay matrix blockers"
        ),
        "notes": [
            "This scorecard plans replay/resume evidence for built-in adaptive-LR optimizers.",
            "It does not claim native kernel implementation readiness.",
            "Runtime dispatch, request/schema/UI exposure, and default behavior remain unchanged.",
        ],
    }


def _row(optimizer: OptimizerType, batch: Mapping[str, Any]) -> dict[str, Any]:
    source = _batch_rows(batch).get(optimizer.value, {})
    family = str(source.get("family") or _family(optimizer))
    abi_ready = source.get("state_machine_abi_spec_ready") is True
    matrix = _matrix_artifact(optimizer.value, family, abi_ready)
    return {
        "schema_version": 1,
        "optimizer_type": optimizer.value,
        "family": family,
        "state_machine_status": "replay_matrix_artifact_ready" if abi_ready else "replay_matrix_blocked",
        "state_machine_reference_ready": source.get("state_machine_reference_ready") is True,
        "state_machine_abi_spec_ready": abi_ready,
        "state_machine_abi_implementation_ready": False,
        "native_kernel_preconditions_spec_ready": source.get("native_kernel_preconditions_spec_ready") is True,
        "native_kernel_preconditions_implementation_ready": False,
        "state_machine_replay_matrix_artifact": matrix,
        "state_machine_replay_matrix_artifact_ready": matrix["spec_ready"] is True,
        "state_machine_replay_matrix_implementation_ready": False,
        "runtime_authority": "existing_python_or_third_party_optimizer",
        "native_route": "none_report_only",
        "product_native_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": _next_gate(family),
        "blocked_reasons": [
            "adaptive_lr_state_machine_abi_implementation_missing",
            "adaptive_lr_batch_resume_replay_not_executed",
            "native_dispatch_gate_not_requested",
        ],
    }


def _matrix_artifact(optimizer_type: str, family: str, spec_ready: bool) -> dict[str, Any]:
    replay_cases = [
        "dynamic_lr_scalar_recomputed_from_saved_state",
        "d_estimator_global_state_replayed_before_param_update",
        "per_step_quality_guard_replay_blocks_bad_d_estimate",
        "lr_scalar_materialized_before_native_boundary",
    ]
    resume_cases = [
        "state_dict_roundtrip_before_step",
        "state_dict_roundtrip_after_step",
        "resume_next_step_matches_python_reference",
    ]
    if family == "adaptive_lr_prodigy":
        replay_cases.extend(["prodigy_global_distance_state_replay", "prodigy_growth_guard_replay"])
        resume_cases.append("prodigy_distance_buffer_resume")
    else:
        replay_cases.extend(["dadapt_variant_accumulator_replay", "dadapt_growth_clip_guard_replay"])
        resume_cases.append("dadapt_variant_accumulator_resume")
    return {
        "schema_version": 1,
        "artifact_kind": MATRIX_KIND,
        "report_only": True,
        "optimizer_type": optimizer_type,
        "adaptive_lr_family": family,
        "spec_ready": spec_ready,
        "implementation_ready": False,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "replay_cases": replay_cases,
        "resume_replay_cases": resume_cases,
        "blocked_until": [
            "state_machine_replay_matrix_implemented",
            "resume_next_step_parity_passed",
            "owner_release_hold",
        ],
        "evidence_status": "planned_report_only",
    }


def _validations(batch: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {optimizer.value for optimizer in TARGET_OPTIMIZERS}
    present = {str(row.get("optimizer_type") or "") for row in rows}
    return [
        _validation(
            "adaptive_lr_batch_ready",
            batch.get("ok") is True and batch.get("state_machine_abi_spec_ready") is True,
            "adaptive_lr_state_machine_batch_missing",
        ),
        _validation(
            "optimizer_set_complete",
            present == expected,
            "adaptive_lr_replay_matrix_optimizer_set_incomplete",
        ),
        _validation(
            "all_matrix_artifacts_ready",
            all(_as_dict(row.get("state_machine_replay_matrix_artifact")).get("spec_ready") is True for row in rows),
            "adaptive_lr_replay_matrix_artifact_not_ready",
        ),
        _validation(
            "implementation_not_claimed",
            all(row.get("state_machine_replay_matrix_implementation_ready") is False for row in rows),
            "adaptive_lr_replay_matrix_unexpected_implementation_claim",
        ),
        _validation(
            "runtime_dispatch_disabled",
            batch.get("training_path_enabled") is False
            and batch.get("runtime_dispatch_ready") is False
            and batch.get("native_dispatch_allowed") is False
            and all(row.get("training_path_enabled") is False for row in rows)
            and all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows),
            "adaptive_lr_replay_matrix_enabled_dispatch",
        ),
    ]


def _batch_rows(batch: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("optimizer_type") or ""): row
        for row in batch.get("rows", [])
        if isinstance(row, Mapping)
    }


def _family(optimizer: OptimizerType) -> str:
    if optimizer in {OptimizerType.AUTO_PRODIGY, OptimizerType.PRODIGY, OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE}:
        return "adaptive_lr_prodigy"
    return "adaptive_lr_dadapt"


def _next_gate(family: str) -> str:
    if family == "adaptive_lr_prodigy":
        return "prodigy_state_machine_abi_replay_executor"
    return "dadapt_state_machine_abi_replay_executor"


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


__all__ = [
    "MATRIX_KIND",
    "SCORECARD",
    "build_adaptive_lr_state_machine_replay_matrix_scorecard",
]
