"""Smoke checks for V5-P53 kernel artifact gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
CORE_ROOT = BACKEND_ROOT / "core"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT), str(CORE_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_kernel_artifact_gate import build_v5_kernel_artifact_gate  # noqa: E402
from lulynx_trainer.turbocore_v5_native_execution_dry_run_parity_boundary_smoke import (  # noqa: E402
    _gate as _p52_gate,
    _p51_ready,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed", "training_launch_allowed", "auto_launch_allowed", "runs_dispatched",
    "default_training_path_enabled", "training_path_enabled", "default_rollout_allowed", "auto_rollout_allowed",
    "ui_exposure_allowed", "product_ui_exposure_allowed", "launcher_exposure_allowed", "webui_exposure_allowed",
    "request_adapter_mapping_allowed", "request_fields_emitted", "request_adapter_registered",
    "request_adapter_enabled", "runtime_adapter_registered", "runtime_adapter_enabled",
    "runtime_execution_allowed", "runtime_execution_executed", "runtime_dispatch_allowed",
    "runtime_dispatch_enabled", "runtime_dispatch_executed", "native_runtime_enabled",
    "native_execution_allowed", "native_execution_enabled", "native_execution_executed",
    "native_execution_dry_run_allowed", "native_execution_dry_run_enabled",
    "native_execution_dry_run_executed", "native_dispatch_allowed", "native_dispatch_enabled",
    "native_dispatch_started", "native_dispatch_executed", "kernel_launch_allowed",
    "kernel_launch_enabled", "kernel_launch_executed", "parity_check_allowed", "parity_check_enabled",
    "parity_check_executed", "parity_result_recorded", "kernel_artifact_registered",
    "kernel_artifact_enabled", "kernel_artifact_gate_allowed", "kernel_artifact_gate_enabled",
    "kernel_artifact_gate_executed", "kernel_implementation_loaded", "kernel_binary_loaded",
    "artifact_loader_enabled", "artifact_loaded", "artifact_abi_loaded", "native_dry_run_result_ingested",
    "artifact_replay_executed", "training_step_allowed", "training_step_enabled", "training_step_executed",
    "generation_request_patch_allowed", "config_adapter_patch_allowed", "runtime_resolver_patch_allowed",
    "execution_resolver_patch_allowed", "training_manager_patch_allowed", "rollout_authorization_allowed",
)
READY_DECISION = "kernel_artifact_gate_recorded_default_off"
BLOCKED_DECISION = "kernel_artifact_gate_blocked_default_off"
HOLD_DECISION = "kernel_artifact_gate_hold_for_signed_review_default_off"
REJECTED_DECISION = "kernel_artifact_gate_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p52_ready = _p52_ready()
    ready = _gate(p52_ready)
    assert ready["ok"] is True, ready
    assert ready["kernel_artifact_gate_ready"] is True, ready
    assert ready["kernel_artifact_gate_review_signed"] is True, ready
    assert ready["kernel_artifact_gate_evidence_recorded"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p52_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p52_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p52_missing = _gate(None)
    _assert_blocked(p52_missing, "p52", "missing")
    p52_not_ready = _gate({**p52_ready, "ok": False, "native_execution_dry_run_parity_boundary_ready": False})
    _assert_blocked(p52_not_ready, "p52", "not_ready")
    p52_decision_mismatch = _gate({**p52_ready, "decision": "wrong"})
    _assert_blocked(p52_decision_mismatch, "p52", "not_ready")
    p52_unsigned_review = _gate({**p52_ready, "native_dry_run_parity_review_signed": False})
    _assert_blocked(p52_unsigned_review, "p52", "not_ready")
    p52_post_fields = _gate({**p52_ready, "post_p52_request_fields": {"bad": True}})
    _assert_blocked(p52_post_fields, "post_p52_request_fields")

    p52_unsafe_cases = {}
    for field in (
        "native_execution_dry_run_executed", "native_execution_executed", "native_dispatch_started",
        "native_dispatch_executed", "kernel_launch_executed", "parity_check_executed",
        "training_step_executed", "kernel_artifact_registered", "kernel_artifact_enabled",
    ):
        report = _gate({**p52_ready, field: True})
        _assert_blocked(report, field)
        p52_unsafe_cases[field] = _summary(report)

    missing_source = _gate(p52_ready, evidence=_without(_evidence(), "source"))
    _assert_blocked(missing_source, "source_missing")
    missing_digest = _gate(p52_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest"))
    _assert_blocked(missing_digest, "digest_missing")
    not_contract_only = _gate(p52_ready, evidence={**_evidence(), "contract_only": False})
    _assert_blocked(not_contract_only, "contract_only")
    not_records_only = _gate(p52_ready, evidence={**_evidence(), "records_evidence_only": False})
    _assert_blocked(not_records_only, "records_evidence_only")
    not_artifact_gate_only = _gate(p52_ready, evidence={**_evidence(), "artifact_gate_only": False})
    _assert_blocked(not_artifact_gate_only, "artifact_gate_only")
    missing_section = _gate(p52_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]})
    _assert_blocked(missing_section, "section_missing")
    evidence_not_ready = _gate(p52_ready, evidence={**_evidence(), "kernel_artifact_gate_ready": False})
    _assert_blocked(evidence_not_ready, "not_ready")

    missing_artifact_inventory = _gate(p52_ready, evidence=_without(_evidence(), "kernel_artifact_inventory"))
    _assert_blocked(missing_artifact_inventory, "kernel_artifact_inventory")
    missing_digest_inventory = _gate(p52_ready, evidence=_without(_evidence(), "kernel_artifact_digest_inventory"))
    _assert_blocked(missing_digest_inventory, "digest")
    missing_abi_inventory = _gate(p52_ready, evidence=_without(_evidence(), "artifact_abi_boundary"))
    _assert_blocked(missing_abi_inventory, "abi")
    missing_loader_inventory = _gate(p52_ready, evidence=_without(_evidence(), "artifact_loader_boundary"))
    _assert_blocked(missing_loader_inventory, "loader")

    artifact_inventory_cases = {}
    for field in (
        "kernel_artifact_registered", "kernel_artifact_enabled", "kernel_implementation_loaded",
        "kernel_binary_loaded", "artifact_loaded", "kernel_launch_enabled", "kernel_launch_executed",
        "native_dispatch_executed", "parity_check_executed", "training_step_executed",
    ):
        report = _inventory_claim("kernel_artifact_inventory", field)
        _assert_blocked(report, field)
        artifact_inventory_cases[field] = _summary(report)

    digest_inventory_cases = {}
    for field in ("kernel_artifact_registered", "kernel_artifact_enabled", "artifact_loaded", "kernel_launch_executed"):
        report = _inventory_claim("kernel_artifact_digest_inventory", field)
        _assert_blocked(report, field)
        digest_inventory_cases[field] = _summary(report)

    abi_inventory_cases = {}
    for field in (
        "artifact_abi_loaded", "kernel_binary_loaded", "kernel_artifact_registered",
        "kernel_artifact_enabled", "native_dispatch_executed", "kernel_launch_executed",
    ):
        report = _inventory_claim("artifact_abi_boundary", field)
        _assert_blocked(report, field)
        abi_inventory_cases[field] = _summary(report)

    loader_inventory_cases = {}
    for field in (
        "artifact_loader_enabled", "artifact_loaded", "kernel_binary_loaded",
        "native_dispatch_executed", "kernel_launch_executed", "parity_check_executed",
    ):
        report = _inventory_claim("artifact_loader_boundary", field)
        _assert_blocked(report, field)
        loader_inventory_cases[field] = _summary(report)

    route_registration = _gate(p52_ready, evidence={**_evidence(), "api_route_registration": {"bad": True}})
    _assert_blocked(route_registration, "api_route_registration")
    review_missing_reviewer = _gate(p52_ready, review={**_review(), "reviewer": ""})
    _assert_blocked(review_missing_reviewer, "reviewer")
    review_missing_reviewed_at = _gate(p52_ready, review={**_review(), "reviewed_at": ""})
    _assert_blocked(review_missing_reviewed_at, "reviewed_at")
    review_scope_mismatch = _gate(p52_ready, review={**_review(), "requested_scope": "wrong"})
    _assert_blocked(review_scope_mismatch, "scope")
    review_missing_ack = _gate(p52_ready, review={**_review(), "acknowledge_no_kernel_artifact_registration": False})
    _assert_blocked(review_missing_ack, "ack_missing", "artifact")

    review_unsafe_cases = {}
    for field in (
        "approve_kernel_artifact_registered", "approve_kernel_artifact_enabled",
        "approve_native_dispatch_executed", "approve_kernel_launch_executed",
        "approve_parity_check_executed", "approve_training_step_executed",
        "approve_training_launch_allowed",
    ):
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        review_unsafe_cases[field] = _summary(report)

    failure_history = _gate(p52_ready, failure_history=[{"reason": "artifact_gate_gap", "open": True, "severity": "high"}])
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(p52_ready, rollback_history=[{"kind": "artifact_gate_rollback", "rollback_required": True}])
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(p52_ready, failure_history=[{"reason": "closed_artifact_warning", "status": "closed", "severity": "high"}])
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p53_kernel_artifact_gate_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p52_missing": _summary(p52_missing),
        "p52_not_ready": _summary(p52_not_ready),
        "p52_decision_mismatch": _summary(p52_decision_mismatch),
        "p52_unsigned_review": _summary(p52_unsigned_review),
        "p52_post_fields": _summary(p52_post_fields),
        "p52_unsafe_cases": p52_unsafe_cases,
        "missing_source": _summary(missing_source),
        "missing_digest": _summary(missing_digest),
        "not_contract_only": _summary(not_contract_only),
        "not_records_only": _summary(not_records_only),
        "not_artifact_gate_only": _summary(not_artifact_gate_only),
        "missing_section": _summary(missing_section),
        "evidence_not_ready": _summary(evidence_not_ready),
        "missing_artifact_inventory": _summary(missing_artifact_inventory),
        "missing_digest_inventory": _summary(missing_digest_inventory),
        "missing_abi_inventory": _summary(missing_abi_inventory),
        "missing_loader_inventory": _summary(missing_loader_inventory),
        "artifact_inventory_cases": artifact_inventory_cases,
        "digest_inventory_cases": digest_inventory_cases,
        "abi_inventory_cases": abi_inventory_cases,
        "loader_inventory_cases": loader_inventory_cases,
        "route_registration": _summary(route_registration),
        "review_missing_reviewer": _summary(review_missing_reviewer),
        "review_missing_reviewed_at": _summary(review_missing_reviewed_at),
        "review_scope_mismatch": _summary(review_scope_mismatch),
        "review_missing_ack": _summary(review_missing_ack),
        "review_unsafe_cases": review_unsafe_cases,
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p52: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_kernel_artifact_gate(
        p52_native_execution_dry_run_parity_boundary=p52,
        kernel_artifact_gate_evidence=_evidence() if evidence is None else evidence,
        kernel_artifact_gate_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p52_ready() -> dict[str, Any]:
    return _p52_gate(_p51_ready())


def _evidence() -> dict[str, Any]:
    sections = [
        "p52_native_execution_dry_run_parity_boundary_reference", "kernel_artifact_inventory",
        "kernel_artifact_digest_inventory", "artifact_abi_boundary", "artifact_loader_boundary",
        "request_adapter_boundary", "no_kernel_artifact_registration_boundary", "no_native_dispatch_boundary",
        "no_kernel_launch_boundary", "no_parity_execution_boundary", "no_training_step_boundary",
        "no_request_fields_boundary", "no_training_launch_boundary", "rollback_policy", "observability_policy",
    ]
    return {
        "evidence_id": "kernel_artifact_gate_v0",
        "evidence_version": "v0",
        "ok": True,
        "kernel_artifact_gate_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "artifact_gate_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_result_ingestion_or_native_dry_run_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "kernel_artifact_inventory": [_artifact_row()],
        "kernel_artifact_digest_inventory": [_digest_row()],
        "artifact_abi_boundary": [_abi_row()],
        "artifact_loader_boundary": [_loader_row()],
        "sha256": "sha256:p53:kernel-artifact-gate:ready",
        "artifact_digest": "sha256:p53:kernel-artifact-gate:ready",
        "source": "temp/turbocore_v5_p53_kernel_artifact_gate.json",
        **_safe_flags(),
    }


def _artifact_row() -> dict[str, Any]:
    return {
        "artifact_id": "future_native_update_kernel",
        "artifact_kind": "future_kernel_artifact_gate_evidence",
        "sha256": "sha256:p53:future-native-update-kernel",
        "abi_id": "turbocore_v5_kernel_abi_v0",
        "loader_id": "turbocore_v5_report_only_loader_v0",
        **_safe_flags(),
    }


def _digest_row() -> dict[str, Any]:
    return {
        "digest_id": "future_native_update_kernel_digest",
        "artifact_id": "future_native_update_kernel",
        "sha256": "sha256:p53:future-native-update-kernel",
        "digest_algorithm": "sha256",
        "source": "temp/turbocore_v5_p53_future_native_update_kernel.digest.json",
        **_safe_flags(),
    }


def _abi_row() -> dict[str, Any]:
    return {
        "boundary_id": "turbocore_v5_kernel_abi_v0",
        "abi_id": "turbocore_v5_kernel_abi_v0",
        "abi_version": "v0",
        "artifact_id": "future_native_update_kernel",
        "loader_id": "turbocore_v5_report_only_loader_v0",
        **_safe_flags(),
    }


def _loader_row() -> dict[str, Any]:
    return {
        "boundary_id": "turbocore_v5_report_only_loader_v0",
        "loader_id": "turbocore_v5_report_only_loader_v0",
        "loader_kind": "future_loader_contract",
        "artifact_id": "future_native_update_kernel",
        **_safe_flags(),
    }


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "kernel_artifact_gate",
        "approve_kernel_artifact_gate": approve,
    }
    for field in DEFAULT_OFF_FIELDS:
        if field != "default_behavior_changed":
            review[f"approve_{field}"] = False
    for field in (
        "acknowledge_p52_native_execution_dry_run_parity_boundary_recorded",
        "acknowledge_default_off_boundary", "acknowledge_no_training_launch",
        "acknowledge_no_ui_exposure", "acknowledge_no_kernel_artifact_registration",
        "acknowledge_no_kernel_artifact_enabled", "acknowledge_no_native_dispatch_executed",
        "acknowledge_no_kernel_launch_executed", "acknowledge_no_parity_executed",
        "acknowledge_no_training_step_executed", "acknowledge_no_request_adapter_enabled",
        "acknowledge_no_request_fields_emitted", "acknowledge_no_default_or_auto_rollout",
        "acknowledge_kernel_artifact_gate_evidence_replayable",
        "acknowledge_later_result_ingestion_or_native_dry_run_execution_contract_required",
        "acknowledge_manual_review_only",
    ):
        review[field] = True
    return review


def _safe_flags() -> dict[str, Any]:
    flags = {field: False for field in DEFAULT_OFF_FIELDS}
    flags.update(
        {
            "blocked_reasons": [], "promotion_blockers": [], "post_p52_request_fields": {},
            "post_p53_request_fields": {}, "post_kernel_artifact_gate_fields": {},
            "native_execution_request": {}, "native_execution_payload": {}, "native_dry_run_request": {},
            "native_dry_run_payload": {}, "native_dispatch_request": {}, "native_dispatch_payload": {},
            "kernel_launch_request": {}, "kernel_launch_payload": {}, "parity_check_request": {},
            "parity_check_payload": {}, "parity_result_payload": {}, "kernel_artifact_request": {},
            "kernel_artifact_payload": {}, "kernel_artifact_registration": {},
            "kernel_artifact_loader_payload": {}, "artifact_loader_payload": {}, "training_step_request": {},
            "training_step_payload": {}, "request_adapter_fields": {}, "request_fields": {},
            "api_route_registration": {}, "backend_route_registration": {}, "ui_route_registration": {},
            "launcher_menu_entry": {}, "webui_tab_entry": {}, "launch_request": {}, "training_request": {},
            "generation_request_patch": {}, "config_adapter_patch": {}, "runtime_resolver_patch": {},
            "execution_resolver_patch": {}, "entry_train_patch": {}, "training_manager_patch": {},
            "auto_launch_plan": {},
        }
    )
    return flags


def _inventory_claim(inventory: str, field: str) -> dict[str, Any]:
    rows = {
        "kernel_artifact_inventory": [_artifact_row()],
        "kernel_artifact_digest_inventory": [_digest_row()],
        "artifact_abi_boundary": [_abi_row()],
        "artifact_loader_boundary": [_loader_row()],
    }
    rows[inventory] = [{**rows[inventory][0], field: True}]
    return _gate(_p52_ready(), evidence={**_evidence(), **rows})


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p52_ready(), review={**_review(), field: True})


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    copied = dict(value)
    copied.pop(key, None)
    return copied


def _without_many(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    copied = dict(value)
    for key in keys:
        copied.pop(key, None)
    return copied


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report
    assert report["post_p53_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["kernel_artifact_gate_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["kernel_artifact_gate_ready"] is False, report
    assert report["decision"] == HOLD_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_reason_fragments(report: dict[str, Any], *fragments: str) -> None:
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "decision": str(report.get("decision") or ""),
        "gate_ready": bool(report.get("kernel_artifact_gate_ready", False)),
        "review_signed": bool(report.get("kernel_artifact_gate_review_signed", False)),
        "rollback_required": bool(report.get("rollback_required", False)),
        "blocked_reasons": _blocked_reasons(report),
    }


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("promotion_blockers") or []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
