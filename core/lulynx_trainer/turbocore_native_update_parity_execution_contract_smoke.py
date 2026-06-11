"""Smoke checks for native-update parity execution contract."""

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

from core.turbocore_native_update_parity_execution_contract import (  # noqa: E402
    BLOCKED_DECISION,
    HOLD_DECISION,
    READY_DECISION,
    build_native_update_parity_execution_contract,
    load_json,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "runs_dispatched",
    "training_path_enabled",
    "training_dispatch",
    "kernel_launch_executed",
    "parity_allowed",
    "parity_enabled",
    "parity_approved",
    "parity_executed",
    "parity_check_executed",
    "parity_execution_allowed",
    "parity_execution_enabled",
    "parity_execution_executed",
    "parity_input_materialized",
    "parity_tolerance_applied",
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
    pending = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract=_kernel_launch_execution_contract(),
        parity_execution_evidence=_parity_execution_evidence(),
    )
    assert pending["ok"] is True, pending
    assert pending["evidence_ready"] is True, pending
    assert pending["ready_for_parity_execution_review"] is True, pending
    assert pending["decision"] == HOLD_DECISION, pending
    assert pending["post_parity_execution_request_fields"] == {}, pending
    assert "native_update_parity_execution_review_missing" in pending["blocked_reasons"], pending
    _assert_default_off(pending)

    signed = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract=_kernel_launch_execution_contract(),
        parity_execution_evidence=_parity_execution_evidence(),
        parity_execution_review=_parity_execution_review(approve=True),
    )
    assert signed["ok"] is True, signed
    assert signed["parity_execution_contract_recorded"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["blocked_reasons"] == [], signed
    _assert_default_off(signed)

    missing_kernel = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract={},
        parity_execution_evidence=_parity_execution_evidence(),
    )
    _assert_evidence_blocked(missing_kernel, "kernel_launch_contract_missing")

    missing_evidence = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract=_kernel_launch_execution_contract(),
        parity_execution_evidence={},
    )
    _assert_evidence_blocked(missing_evidence, "parity_execution_evidence_missing")

    unsafe_kernel = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract={**_kernel_launch_execution_contract(), "parity_executed": True},
        parity_execution_evidence=_parity_execution_evidence(),
    )
    _assert_evidence_blocked(unsafe_kernel, "parity_executed")

    unsafe_evidence = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract=_kernel_launch_execution_contract(),
        parity_execution_evidence={**_parity_execution_evidence(), "training_step_executed": True},
    )
    _assert_evidence_blocked(unsafe_evidence, "training_step_executed")

    bad_plan = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract=_kernel_launch_execution_contract(),
        parity_execution_evidence=_parity_execution_evidence(plan_patch={"parity_executed": True}),
    )
    _assert_evidence_blocked(bad_plan, "parity_plan_not_default_off")

    missing_source = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract=_kernel_launch_execution_contract(),
        parity_execution_evidence=_parity_execution_evidence(plan_patch={"source": ""}),
    )
    _assert_evidence_blocked(missing_source, "parity_plan_source_missing")

    unsafe_review = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract=_kernel_launch_execution_contract(),
        parity_execution_evidence=_parity_execution_evidence(),
        parity_execution_review={**_parity_execution_review(approve=True), "approve_training_step_executed": True},
    )
    _assert_review_blocked(unsafe_review, "approve_training_step_executed")

    missing_ack = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract=_kernel_launch_execution_contract(),
        parity_execution_evidence=_parity_execution_evidence(),
        parity_execution_review={
            **_parity_execution_review(approve=True),
            "acknowledge_no_tensor_transfer_or_training_step": False,
        },
    )
    _assert_review_blocked(missing_ack, "ack_missing", "tensor_transfer")

    real_artifact = _write_real_artifact_case()
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_parity_execution_contract_smoke",
        "ok": True,
        "pending_decision": pending["decision"],
        "signed_decision": signed["decision"],
        "real_artifact_checked": bool(real_artifact),
        "recommended_next_step": pending["recommended_next_step"],
    }


def _write_real_artifact_case() -> dict[str, Any]:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = temp_dir / "native_update_kernel_launch_execution_contract.json"
    evidence_path = temp_dir / "native_update_parity_execution_evidence.json"
    contract_path = temp_dir / "native_update_parity_execution_contract.json"
    evidence = _parity_execution_evidence(source=str(evidence_path))
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    kernel = load_json(kernel_path) if kernel_path.exists() else _kernel_launch_execution_contract()
    package = build_native_update_parity_execution_contract(
        kernel_launch_execution_contract=kernel,
        parity_execution_evidence=load_json(evidence_path),
    )
    assert package["ok"] is True, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == HOLD_DECISION, package
    assert package["post_parity_execution_request_fields"] == {}, package
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


def _kernel_launch_execution_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_kernel_launch_execution_contract_v0",
        "ok": True,
        "evidence_ready": True,
        "ready_for_kernel_launch_execution_review": True,
        "kernel_launch_execution_contract_recorded": False,
        "decision": "native_update_kernel_launch_execution_contract_hold_for_review_default_off",
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "training_dispatch": False,
        "kernel_launch_executed": False,
        "parity_allowed": False,
        "parity_executed": False,
        "parity_check_executed": False,
        "tensor_transfer_executed": False,
        "training_step_executed": False,
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "post_kernel_launch_execution_request_fields": {},
    }


def _parity_execution_evidence(
    *,
    source: str = "smoke://native_update_parity_execution_evidence",
    plan_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sections = [
        "kernel_launch_execution_contract_reference",
        "parity_execution_plan_inventory",
        "parity_authorization_inventory",
        "parity_precondition_inventory",
        "parity_input_boundary",
        "parity_tolerance_boundary",
        "tensor_transfer_boundary",
        "training_step_boundary",
        "no_parity_execution_boundary",
        "no_tensor_transfer_boundary",
        "no_training_step_boundary",
        "no_request_ui_schema_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    plan = _default_off_row("native_update_parity_execution_plan", source)
    plan.update(plan_patch or {})
    return {
        "schema_version": 1,
        "evidence": "native_update_parity_execution_evidence_v0",
        "ok": True,
        "parity_execution_contract_ready": True,
        "report_only": True,
        "contract_only": True,
        "parity_execution_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "default_off": True,
        "requires_later_training_step_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "source": source,
        "artifact_digest": "smoke-parity-execution-digest",
        "sections": sections,
        "parity_execution_plan_inventory": [plan],
        "parity_authorization_inventory": [_default_off_row("native_update_parity_authorization", source)],
        "parity_precondition_inventory": [_default_off_row("native_update_parity_precondition", source)],
        "parity_input_boundary": [_default_off_row("native_update_parity_input_boundary", source)],
        "parity_tolerance_boundary": [_default_off_row("native_update_parity_tolerance_boundary", source)],
        "tensor_transfer_boundary": [_default_off_row("native_update_tensor_transfer_boundary", source)],
        "training_step_boundary": [_default_off_row("native_update_training_step_boundary", source)],
        "rollback_policy": [{"id": "native_update_parity_rollback", "ready": True, "source": source}],
        "observability_policy": [{"id": "native_update_parity_observability", "ready": True, "source": source}],
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
        "parity_allowed": False,
        "parity_enabled": False,
        "parity_approved": False,
        "parity_executed": False,
        "parity_input_materialized": False,
        "parity_tolerance_applied": False,
        "tensor_transfer_executed": False,
        "training_step_executed": False,
        "request_fields_emitted": False,
    }


def _parity_execution_review(*, approve: bool) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-05",
        "requested_scope": "native_update_parity_execution_contract",
        "approve_native_update_parity_execution_contract": bool(approve),
    }
    for field in (
        "acknowledge_kernel_launch_execution_contract_ready",
        "acknowledge_parity_plan_default_off",
        "acknowledge_parity_authorization_default_off",
        "acknowledge_parity_preconditions_default_off",
        "acknowledge_parity_input_and_tolerance_boundaries_default_off",
        "acknowledge_no_parity_executed",
        "acknowledge_no_tensor_transfer_or_training_step",
        "acknowledge_no_request_ui_schema_exposure",
        "acknowledge_later_training_step_execution_contract_required",
    ):
        review[field] = True
    return review


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
