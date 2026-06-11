"""Smoke checks for native-update native dispatch execution contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_native_update_native_dispatch_execution_contract import (  # noqa: E402
    BLOCKED_DECISION,
    HOLD_DECISION,
    READY_DECISION,
    build_native_update_native_dispatch_execution_contract,
    load_json,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "runs_dispatched",
    "training_path_enabled",
    "training_dispatch",
    "runtime_execution_executed",
    "runtime_dispatch_allowed",
    "runtime_dispatch_enabled",
    "runtime_dispatch_approved",
    "runtime_dispatch_executed",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_approved",
    "native_dispatch_started",
    "native_dispatch_executed",
    "native_dispatch_execution_allowed",
    "native_dispatch_execution_enabled",
    "native_dispatch_execution_executed",
    "native_runtime_handoff_enabled",
    "native_runtime_handoff_executed",
    "native_adapter_handoff_enabled",
    "native_adapter_handoff_executed",
    "native_kernel_handoff_enabled",
    "kernel_launch_allowed",
    "kernel_launch_started",
    "kernel_launch_executed",
    "parity_allowed",
    "parity_executed",
    "parity_check_executed",
    "tensor_transfer_allowed",
    "tensor_transfer_executed",
    "training_step_allowed",
    "training_step_executed",
    "artifact_loaded",
    "execution_replay_executed",
    "ready_for_ui",
    "ui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "rollout_authorization_allowed",
)


def run_smoke() -> dict[str, Any]:
    pending = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract=_runtime_dispatch_contract(),
        native_dispatch_execution_evidence=_native_dispatch_execution_evidence(),
    )
    assert pending["ok"] is True, pending
    assert pending["evidence_ready"] is True, pending
    assert pending["ready_for_native_dispatch_execution_review"] is True, pending
    assert pending["decision"] == HOLD_DECISION, pending
    assert pending["post_native_dispatch_execution_request_fields"] == {}, pending
    assert "native_update_native_dispatch_execution_review_missing" in pending["blocked_reasons"], pending
    _assert_default_off(pending)

    signed = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract=_runtime_dispatch_contract(),
        native_dispatch_execution_evidence=_native_dispatch_execution_evidence(),
        native_dispatch_execution_review=_native_dispatch_execution_review(approve=True),
    )
    assert signed["ok"] is True, signed
    assert signed["native_dispatch_execution_contract_recorded"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["blocked_reasons"] == [], signed
    _assert_default_off(signed)

    missing_dispatch = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract={},
        native_dispatch_execution_evidence=_native_dispatch_execution_evidence(),
    )
    _assert_evidence_blocked(missing_dispatch, "runtime_dispatch_contract_missing")

    missing_evidence = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract=_runtime_dispatch_contract(),
        native_dispatch_execution_evidence={},
    )
    _assert_evidence_blocked(missing_evidence, "native_dispatch_execution_evidence_missing")

    unsafe_dispatch = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract={**_runtime_dispatch_contract(), "native_dispatch_enabled": True},
        native_dispatch_execution_evidence=_native_dispatch_execution_evidence(),
    )
    _assert_evidence_blocked(unsafe_dispatch, "native_dispatch_enabled")

    unsafe_evidence = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract=_runtime_dispatch_contract(),
        native_dispatch_execution_evidence={**_native_dispatch_execution_evidence(), "kernel_launch_executed": True},
    )
    _assert_evidence_blocked(unsafe_evidence, "kernel_launch_executed")

    bad_plan = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract=_runtime_dispatch_contract(),
        native_dispatch_execution_evidence=_native_dispatch_execution_evidence(plan_patch={"native_dispatch_executed": True}),
    )
    _assert_evidence_blocked(bad_plan, "native_dispatch_plan_not_default_off")

    missing_source = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract=_runtime_dispatch_contract(),
        native_dispatch_execution_evidence=_native_dispatch_execution_evidence(plan_patch={"source": ""}),
    )
    _assert_evidence_blocked(missing_source, "native_dispatch_plan_source_missing")

    unsafe_review = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract=_runtime_dispatch_contract(),
        native_dispatch_execution_evidence=_native_dispatch_execution_evidence(),
        native_dispatch_execution_review={
            **_native_dispatch_execution_review(approve=True),
            "approve_kernel_launch_executed": True,
        },
    )
    _assert_review_blocked(unsafe_review, "approve_kernel_launch_executed")

    missing_ack = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract=_runtime_dispatch_contract(),
        native_dispatch_execution_evidence=_native_dispatch_execution_evidence(),
        native_dispatch_execution_review={
            **_native_dispatch_execution_review(approve=True),
            "acknowledge_no_parity_tensor_or_training_step": False,
        },
    )
    _assert_review_blocked(missing_ack, "ack_missing", "parity_tensor")

    real_artifact = _write_real_artifact_case()
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_native_dispatch_execution_contract_smoke",
        "ok": True,
        "pending_decision": pending["decision"],
        "signed_decision": signed["decision"],
        "real_artifact_checked": bool(real_artifact),
        "recommended_next_step": pending["recommended_next_step"],
    }


def _write_real_artifact_case() -> dict[str, Any]:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    dispatch_path = temp_dir / "native_update_runtime_dispatch_contract.json"
    evidence_path = temp_dir / "native_update_native_dispatch_execution_evidence.json"
    contract_path = temp_dir / "native_update_native_dispatch_execution_contract.json"
    evidence = _native_dispatch_execution_evidence(source=str(evidence_path))
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dispatch = load_json(dispatch_path) if dispatch_path.exists() else _runtime_dispatch_contract()
    package = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract=dispatch,
        native_dispatch_execution_evidence=load_json(evidence_path),
    )
    assert package["ok"] is True, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == HOLD_DECISION, package
    assert package["post_native_dispatch_execution_request_fields"] == {}, package
    _assert_default_off(package)
    contract_path.write_text(
        json.dumps(package, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return package


def _assert_default_off(package: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert package[field] is False, (field, package)


def _assert_evidence_blocked(package: dict[str, Any], *needles: str) -> None:
    assert package["ok"] is False, package
    assert package["evidence_ready"] is False, package
    assert package["decision"] == BLOCKED_DECISION, package
    haystack = "\n".join(package["blocked_reasons"])
    for needle in needles:
        assert needle in haystack, package
    _assert_default_off(package)


def _assert_review_blocked(package: dict[str, Any], *needles: str) -> None:
    assert package["ok"] is False, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == BLOCKED_DECISION, package
    haystack = "\n".join(package["blocked_reasons"])
    for needle in needles:
        assert needle in haystack, package
    _assert_default_off(package)


def _runtime_dispatch_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_runtime_dispatch_contract_v0",
        "ok": True,
        "evidence_ready": True,
        "ready_for_runtime_dispatch_review": True,
        "runtime_dispatch_contract_recorded": False,
        "decision": "native_update_runtime_dispatch_contract_hold_for_review_default_off",
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "training_dispatch": False,
        "runtime_dispatch_allowed": False,
        "runtime_dispatch_enabled": False,
        "runtime_dispatch_approved": False,
        "runtime_dispatch_executed": False,
        "native_dispatch_allowed": False,
        "native_dispatch_enabled": False,
        "native_dispatch_executed": False,
        "kernel_launch_allowed": False,
        "kernel_launch_executed": False,
        "training_step_executed": False,
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "post_runtime_dispatch_request_fields": {},
    }


def _native_dispatch_execution_evidence(
    *,
    source: str = "smoke://native_update_native_dispatch_execution_evidence",
    plan_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sections = [
        "runtime_dispatch_contract_reference",
        "native_dispatch_execution_plan_inventory",
        "native_dispatch_authorization_inventory",
        "native_dispatch_precondition_inventory",
        "native_runtime_handoff_boundary",
        "native_dispatch_adapter_boundary",
        "kernel_launch_boundary",
        "parity_boundary",
        "tensor_transfer_boundary",
        "no_native_dispatch_execution_boundary",
        "no_kernel_launch_boundary",
        "no_parity_boundary",
        "no_tensor_transfer_boundary",
        "no_training_step_boundary",
        "no_request_ui_schema_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    plan = _default_off_row("native_update_native_dispatch_execution_plan", source)
    plan.update(plan_patch or {})
    return {
        "schema_version": 1,
        "evidence": "native_update_native_dispatch_execution_evidence_v0",
        "ok": True,
        "native_dispatch_execution_contract_ready": True,
        "report_only": True,
        "contract_only": True,
        "native_dispatch_execution_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "default_off": True,
        "requires_later_kernel_launch_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "source": source,
        "artifact_digest": "smoke-native-dispatch-execution-digest",
        "sections": sections,
        "native_dispatch_execution_plan_inventory": [plan],
        "native_dispatch_authorization_inventory": [_default_off_row("native_update_native_dispatch_authorization", source)],
        "native_dispatch_precondition_inventory": [_default_off_row("native_update_native_dispatch_precondition", source)],
        "native_runtime_handoff_boundary": [_default_off_row("native_update_native_runtime_handoff", source)],
        "native_dispatch_adapter_boundary": [_default_off_row("native_update_native_dispatch_adapter", source)],
        "kernel_launch_boundary": [_default_off_row("native_update_kernel_launch_boundary", source)],
        "parity_boundary": [_default_off_row("native_update_parity_boundary", source)],
        "tensor_transfer_boundary": [_default_off_row("native_update_tensor_transfer_boundary", source)],
        "rollback_policy": [{"id": "native_update_native_dispatch_rollback", "ready": True, "source": source}],
        "observability_policy": [{"id": "native_update_native_dispatch_observability", "ready": True, "source": source}],
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
    }


def _default_off_row(row_id: str, source: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "ready": True,
        "source": source,
        "native_dispatch_allowed": False,
        "native_dispatch_enabled": False,
        "native_dispatch_approved": False,
        "native_dispatch_executed": False,
        "native_runtime_handoff_enabled": False,
        "native_adapter_handoff_enabled": False,
        "kernel_launch_executed": False,
        "parity_executed": False,
        "tensor_transfer_executed": False,
        "training_step_executed": False,
        "request_fields_emitted": False,
    }


def _native_dispatch_execution_review(*, approve: bool) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-05",
        "requested_scope": "native_update_native_dispatch_execution_contract",
        "approve_native_update_native_dispatch_execution_contract": bool(approve),
    }
    for field in (
        "acknowledge_runtime_dispatch_contract_ready",
        "acknowledge_native_dispatch_plan_default_off",
        "acknowledge_native_dispatch_authorization_default_off",
        "acknowledge_native_dispatch_preconditions_default_off",
        "acknowledge_native_runtime_handoff_default_off",
        "acknowledge_native_adapter_kernel_parity_boundaries_default_off",
        "acknowledge_no_native_dispatch_approved_or_executed",
        "acknowledge_no_kernel_launch",
        "acknowledge_no_parity_tensor_or_training_step",
        "acknowledge_no_request_ui_schema_exposure",
        "acknowledge_later_kernel_launch_execution_contract_required",
    ):
        review[field] = True
    return review


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
