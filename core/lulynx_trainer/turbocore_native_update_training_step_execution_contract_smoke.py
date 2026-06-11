"""Smoke checks for native-update training-step execution contract."""

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

from core.turbocore_native_update_training_step_execution_contract import (  # noqa: E402
    BLOCKED_DECISION,
    HOLD_DECISION,
    READY_DECISION,
    build_native_update_training_step_execution_contract,
    load_json,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "training_launch_enabled",
    "training_launch_executed",
    "runs_dispatched",
    "training_path_enabled",
    "training_dispatch",
    "parity_executed",
    "training_step_allowed",
    "training_step_enabled",
    "training_step_approved",
    "training_step_executed",
    "training_step_execution_allowed",
    "training_step_execution_enabled",
    "training_step_execution_executed",
    "training_input_materialized",
    "optimizer_update_executed",
    "loss_backward_executed",
    "request_submission_allowed",
    "request_submitted",
    "job_created",
    "queue_enqueued",
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
    pending = build_native_update_training_step_execution_contract(
        parity_execution_contract=_parity_execution_contract(),
        training_step_execution_evidence=_training_step_execution_evidence(),
    )
    assert pending["ok"] is True, pending
    assert pending["evidence_ready"] is True, pending
    assert pending["ready_for_training_step_execution_review"] is True, pending
    assert pending["decision"] == HOLD_DECISION, pending
    assert pending["post_training_step_execution_request_fields"] == {}, pending
    assert "native_update_training_step_execution_review_missing" in pending["blocked_reasons"], pending
    _assert_default_off(pending)

    signed = build_native_update_training_step_execution_contract(
        parity_execution_contract=_parity_execution_contract(),
        training_step_execution_evidence=_training_step_execution_evidence(),
        training_step_execution_review=_training_step_execution_review(approve=True),
    )
    assert signed["ok"] is True, signed
    assert signed["training_step_execution_contract_recorded"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["blocked_reasons"] == [], signed
    _assert_default_off(signed)

    missing_parity = build_native_update_training_step_execution_contract(
        parity_execution_contract={},
        training_step_execution_evidence=_training_step_execution_evidence(),
    )
    _assert_evidence_blocked(missing_parity, "parity_contract_missing")

    missing_evidence = build_native_update_training_step_execution_contract(
        parity_execution_contract=_parity_execution_contract(),
        training_step_execution_evidence={},
    )
    _assert_evidence_blocked(missing_evidence, "training_step_execution_evidence_missing")

    unsafe_parity = build_native_update_training_step_execution_contract(
        parity_execution_contract={**_parity_execution_contract(), "training_step_executed": True},
        training_step_execution_evidence=_training_step_execution_evidence(),
    )
    _assert_evidence_blocked(unsafe_parity, "training_step_executed")

    unsafe_evidence = build_native_update_training_step_execution_contract(
        parity_execution_contract=_parity_execution_contract(),
        training_step_execution_evidence={**_training_step_execution_evidence(), "request_submitted": True},
    )
    _assert_evidence_blocked(unsafe_evidence, "request_submitted")

    bad_plan = build_native_update_training_step_execution_contract(
        parity_execution_contract=_parity_execution_contract(),
        training_step_execution_evidence=_training_step_execution_evidence(plan_patch={"training_step_executed": True}),
    )
    _assert_evidence_blocked(bad_plan, "training_step_plan_not_default_off")

    missing_source = build_native_update_training_step_execution_contract(
        parity_execution_contract=_parity_execution_contract(),
        training_step_execution_evidence=_training_step_execution_evidence(plan_patch={"source": ""}),
    )
    _assert_evidence_blocked(missing_source, "training_step_plan_source_missing")

    unsafe_review = build_native_update_training_step_execution_contract(
        parity_execution_contract=_parity_execution_contract(),
        training_step_execution_evidence=_training_step_execution_evidence(),
        training_step_execution_review={
            **_training_step_execution_review(approve=True),
            "approve_training_launch_executed": True,
        },
    )
    _assert_review_blocked(unsafe_review, "approve_training_launch_executed")

    missing_ack = build_native_update_training_step_execution_contract(
        parity_execution_contract=_parity_execution_contract(),
        training_step_execution_evidence=_training_step_execution_evidence(),
        training_step_execution_review={
            **_training_step_execution_review(approve=True),
            "acknowledge_no_training_launch_or_request_submission": False,
        },
    )
    _assert_review_blocked(missing_ack, "ack_missing", "training_launch")

    real_artifact = _write_real_artifact_case()
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_training_step_execution_contract_smoke",
        "ok": True,
        "pending_decision": pending["decision"],
        "signed_decision": signed["decision"],
        "real_artifact_checked": bool(real_artifact),
        "recommended_next_step": pending["recommended_next_step"],
    }


def _write_real_artifact_case() -> dict[str, Any]:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    parity_path = temp_dir / "native_update_parity_execution_contract.json"
    evidence_path = temp_dir / "native_update_training_step_execution_evidence.json"
    contract_path = temp_dir / "native_update_training_step_execution_contract.json"
    evidence = _training_step_execution_evidence(source=str(evidence_path))
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parity = load_json(parity_path) if parity_path.exists() else _parity_execution_contract()
    package = build_native_update_training_step_execution_contract(
        parity_execution_contract=parity,
        training_step_execution_evidence=load_json(evidence_path),
    )
    assert package["ok"] is True, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == HOLD_DECISION, package
    assert package["post_training_step_execution_request_fields"] == {}, package
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


def _parity_execution_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_parity_execution_contract_v0",
        "ok": True,
        "evidence_ready": True,
        "ready_for_parity_execution_review": True,
        "parity_execution_contract_recorded": False,
        "decision": "native_update_parity_execution_contract_hold_for_review_default_off",
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "training_dispatch": False,
        "parity_executed": False,
        "training_step_allowed": False,
        "training_step_executed": False,
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "post_parity_execution_request_fields": {},
    }


def _training_step_execution_evidence(
    *,
    source: str = "smoke://native_update_training_step_execution_evidence",
    plan_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sections = [
        "parity_execution_contract_reference",
        "training_step_execution_plan_inventory",
        "training_step_authorization_inventory",
        "training_step_precondition_inventory",
        "training_input_boundary",
        "optimizer_update_boundary",
        "loss_backward_boundary",
        "training_launch_boundary",
        "request_ui_schema_boundary",
        "no_training_step_execution_boundary",
        "no_training_launch_boundary",
        "no_request_submission_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    plan = _default_off_row("native_update_training_step_execution_plan", source)
    plan.update(plan_patch or {})
    return {
        "schema_version": 1,
        "evidence": "native_update_training_step_execution_evidence_v0",
        "ok": True,
        "training_step_execution_contract_ready": True,
        "report_only": True,
        "contract_only": True,
        "training_step_execution_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "default_off": True,
        "requires_later_training_launch_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "source": source,
        "artifact_digest": "smoke-training-step-execution-digest",
        "sections": sections,
        "training_step_execution_plan_inventory": [plan],
        "training_step_authorization_inventory": [_default_off_row("native_update_training_step_authorization", source)],
        "training_step_precondition_inventory": [_default_off_row("native_update_training_step_precondition", source)],
        "training_input_boundary": [_default_off_row("native_update_training_input_boundary", source)],
        "optimizer_update_boundary": [_default_off_row("native_update_optimizer_update_boundary", source)],
        "loss_backward_boundary": [_default_off_row("native_update_loss_backward_boundary", source)],
        "training_launch_boundary": [_default_off_row("native_update_training_launch_boundary", source)],
        "request_ui_schema_boundary": [_default_off_row("native_update_request_ui_schema_boundary", source)],
        "rollback_policy": [{"id": "native_update_training_step_rollback", "ready": True, "source": source}],
        "observability_policy": [{"id": "native_update_training_step_observability", "ready": True, "source": source}],
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
        "training_step_allowed": False,
        "training_step_enabled": False,
        "training_step_approved": False,
        "training_step_executed": False,
        "training_input_materialized": False,
        "optimizer_update_executed": False,
        "loss_backward_executed": False,
        "training_launch_executed": False,
        "request_submitted": False,
        "request_fields_emitted": False,
    }


def _training_step_execution_review(*, approve: bool) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-05",
        "requested_scope": "native_update_training_step_execution_contract",
        "approve_native_update_training_step_execution_contract": bool(approve),
    }
    for field in (
        "acknowledge_parity_execution_contract_ready",
        "acknowledge_training_step_plan_default_off",
        "acknowledge_training_step_authorization_default_off",
        "acknowledge_training_step_preconditions_default_off",
        "acknowledge_training_input_optimizer_loss_boundaries_default_off",
        "acknowledge_no_training_step_executed",
        "acknowledge_no_training_launch_or_request_submission",
        "acknowledge_no_request_ui_schema_exposure",
        "acknowledge_later_training_launch_contract_required",
    ):
        review[field] = True
    return review


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
