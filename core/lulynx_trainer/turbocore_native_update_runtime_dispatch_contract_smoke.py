"""Smoke checks for native-update runtime dispatch contract."""

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

from core.turbocore_native_update_runtime_dispatch_contract import (  # noqa: E402
    BLOCKED_DECISION,
    HOLD_DECISION,
    READY_DECISION,
    build_native_update_runtime_dispatch_contract,
    load_json,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "runs_dispatched",
    "training_path_enabled",
    "training_dispatch",
    "runtime_execution_allowed",
    "runtime_execution_enabled",
    "runtime_execution_started",
    "runtime_execution_executed",
    "runtime_dispatch_allowed",
    "runtime_dispatch_enabled",
    "runtime_dispatch_approved",
    "runtime_dispatch_started",
    "runtime_dispatch_executed",
    "runtime_dispatch_authorized",
    "runtime_dispatch_control_enabled",
    "runtime_dispatch_control_executed",
    "runtime_adapter_enabled",
    "runtime_adapter_handoff_enabled",
    "runtime_state_lock_enabled",
    "runtime_state_refreshed",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_started",
    "native_dispatch_executed",
    "native_kernel_dispatch_enabled",
    "kernel_launch_allowed",
    "kernel_launch_started",
    "kernel_launch_executed",
    "parity_allowed",
    "parity_executed",
    "parity_check_executed",
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
    pending = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract=_runtime_execution_contract(),
        runtime_dispatch_evidence=_runtime_dispatch_evidence(),
    )
    assert pending["ok"] is True, pending
    assert pending["evidence_ready"] is True, pending
    assert pending["ready_for_runtime_dispatch_review"] is True, pending
    assert pending["decision"] == HOLD_DECISION, pending
    assert pending["post_runtime_dispatch_request_fields"] == {}, pending
    assert "native_update_runtime_dispatch_review_missing" in pending["blocked_reasons"], pending
    _assert_default_off(pending)

    signed = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract=_runtime_execution_contract(),
        runtime_dispatch_evidence=_runtime_dispatch_evidence(),
        runtime_dispatch_review=_runtime_dispatch_review(approve=True),
    )
    assert signed["ok"] is True, signed
    assert signed["runtime_dispatch_contract_recorded"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["blocked_reasons"] == [], signed
    _assert_default_off(signed)

    missing_execution = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract={},
        runtime_dispatch_evidence=_runtime_dispatch_evidence(),
    )
    _assert_evidence_blocked(missing_execution, "runtime_execution_contract_missing")

    missing_evidence = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract=_runtime_execution_contract(),
        runtime_dispatch_evidence={},
    )
    _assert_evidence_blocked(missing_evidence, "runtime_dispatch_evidence_missing")

    unsafe_execution = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract={**_runtime_execution_contract(), "runtime_dispatch_enabled": True},
        runtime_dispatch_evidence=_runtime_dispatch_evidence(),
    )
    _assert_evidence_blocked(unsafe_execution, "runtime_dispatch_enabled")

    unsafe_evidence = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract=_runtime_execution_contract(),
        runtime_dispatch_evidence={**_runtime_dispatch_evidence(), "kernel_launch_executed": True},
    )
    _assert_evidence_blocked(unsafe_evidence, "kernel_launch_executed")

    bad_control = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract=_runtime_execution_contract(),
        runtime_dispatch_evidence=_runtime_dispatch_evidence(control_patch={"runtime_dispatch_executed": True}),
    )
    _assert_evidence_blocked(bad_control, "runtime_dispatch_control_not_default_off")

    missing_source = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract=_runtime_execution_contract(),
        runtime_dispatch_evidence=_runtime_dispatch_evidence(control_patch={"source": ""}),
    )
    _assert_evidence_blocked(missing_source, "runtime_dispatch_control_source_missing")

    unsafe_review = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract=_runtime_execution_contract(),
        runtime_dispatch_evidence=_runtime_dispatch_evidence(),
        runtime_dispatch_review={**_runtime_dispatch_review(approve=True), "approve_kernel_launch_executed": True},
    )
    _assert_review_blocked(unsafe_review, "approve_kernel_launch_executed")

    missing_ack = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract=_runtime_execution_contract(),
        runtime_dispatch_evidence=_runtime_dispatch_evidence(),
        runtime_dispatch_review={
            **_runtime_dispatch_review(approve=True),
            "acknowledge_no_parity_or_training_step": False,
        },
    )
    _assert_review_blocked(missing_ack, "ack_missing", "parity_or_training")

    real_artifact = _write_real_artifact_case()
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_runtime_dispatch_contract_smoke",
        "ok": True,
        "pending_decision": pending["decision"],
        "signed_decision": signed["decision"],
        "real_artifact_checked": bool(real_artifact),
        "recommended_next_step": pending["recommended_next_step"],
    }


def _write_real_artifact_case() -> dict[str, Any]:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    execution_path = temp_dir / "native_update_runtime_execution_contract.json"
    evidence_path = temp_dir / "native_update_runtime_dispatch_evidence.json"
    contract_path = temp_dir / "native_update_runtime_dispatch_contract.json"
    evidence = _runtime_dispatch_evidence(source=str(evidence_path))
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    execution = load_json(execution_path) if execution_path.exists() else _runtime_execution_contract()
    package = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract=execution,
        runtime_dispatch_evidence=load_json(evidence_path),
    )
    assert package["ok"] is True, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == HOLD_DECISION, package
    assert package["post_runtime_dispatch_request_fields"] == {}, package
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


def _runtime_execution_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_runtime_execution_contract_v0",
        "ok": True,
        "evidence_ready": True,
        "ready_for_runtime_execution_review": True,
        "runtime_execution_contract_recorded": False,
        "decision": "native_update_runtime_execution_contract_hold_for_review_default_off",
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "training_dispatch": False,
        "runtime_execution_allowed": False,
        "runtime_execution_enabled": False,
        "runtime_execution_started": False,
        "runtime_execution_executed": False,
        "runtime_dispatch_allowed": False,
        "runtime_dispatch_enabled": False,
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
        "post_runtime_execution_request_fields": {},
    }


def _runtime_dispatch_evidence(
    *,
    source: str = "smoke://native_update_runtime_dispatch_evidence",
    control_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sections = [
        "runtime_execution_contract_reference",
        "runtime_dispatch_control_plan_inventory",
        "runtime_dispatch_authorization_inventory",
        "runtime_dispatch_precondition_inventory",
        "runtime_adapter_lock_inventory",
        "runtime_state_lock_inventory",
        "native_dispatch_boundary",
        "kernel_launch_boundary",
        "no_runtime_dispatch_boundary",
        "no_native_dispatch_boundary",
        "no_kernel_launch_boundary",
        "no_parity_boundary",
        "no_training_step_boundary",
        "no_request_ui_schema_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    control = _default_off_row("native_update_runtime_dispatch_control", source)
    control.update(control_patch or {})
    return {
        "schema_version": 1,
        "evidence": "native_update_runtime_dispatch_evidence_v0",
        "ok": True,
        "runtime_dispatch_contract_ready": True,
        "report_only": True,
        "contract_only": True,
        "runtime_dispatch_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "default_off": True,
        "requires_later_native_dispatch_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "source": source,
        "artifact_digest": "smoke-runtime-dispatch-digest",
        "sections": sections,
        "runtime_dispatch_control_plan_inventory": [control],
        "runtime_dispatch_authorization_inventory": [_default_off_row("native_update_runtime_dispatch_authorization", source)],
        "runtime_dispatch_precondition_inventory": [_default_off_row("native_update_runtime_dispatch_precondition", source)],
        "runtime_adapter_lock_inventory": [_default_off_row("native_update_runtime_adapter_lock", source)],
        "runtime_state_lock_inventory": [_default_off_row("native_update_runtime_state_lock", source)],
        "native_dispatch_boundary": [_default_off_row("native_update_native_dispatch_boundary", source)],
        "kernel_launch_boundary": [_default_off_row("native_update_kernel_launch_boundary", source)],
        "rollback_policy": [{"id": "native_update_runtime_dispatch_rollback", "ready": True, "source": source}],
        "observability_policy": [{"id": "native_update_runtime_dispatch_observability", "ready": True, "source": source}],
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
        "runtime_dispatch_allowed": False,
        "runtime_dispatch_enabled": False,
        "runtime_dispatch_approved": False,
        "runtime_dispatch_executed": False,
        "runtime_adapter_enabled": False,
        "runtime_state_refreshed": False,
        "native_dispatch_enabled": False,
        "native_dispatch_executed": False,
        "kernel_launch_executed": False,
        "parity_executed": False,
        "training_step_executed": False,
        "request_fields_emitted": False,
    }


def _runtime_dispatch_review(*, approve: bool) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-05",
        "requested_scope": "native_update_runtime_dispatch_contract",
        "approve_native_update_runtime_dispatch_contract": bool(approve),
    }
    for field in (
        "acknowledge_runtime_execution_contract_ready",
        "acknowledge_runtime_dispatch_control_default_off",
        "acknowledge_runtime_dispatch_authorization_default_off",
        "acknowledge_runtime_dispatch_preconditions_default_off",
        "acknowledge_runtime_adapter_and_state_locks_default_off",
        "acknowledge_native_dispatch_and_kernel_boundaries_default_off",
        "acknowledge_no_runtime_dispatch_approved_or_executed",
        "acknowledge_no_native_dispatch_or_kernel_launch",
        "acknowledge_no_parity_or_training_step",
        "acknowledge_no_request_ui_schema_exposure",
        "acknowledge_later_native_dispatch_execution_contract_required",
    ):
        review[field] = True
    return review


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
