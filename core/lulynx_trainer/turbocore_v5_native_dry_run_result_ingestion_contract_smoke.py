"""Smoke checks for V5-P54 native dry-run result ingestion contract."""

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

from core.turbocore_v5_native_dry_run_result_ingestion_contract import (  # noqa: E402
    build_v5_native_dry_run_result_ingestion_contract,
)
from lulynx_trainer.turbocore_v5_kernel_artifact_gate_smoke import (  # noqa: E402
    _gate as _p53_gate,
    _p52_ready,
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
    "native_dispatch_started", "native_dispatch_executed", "native_dry_run_result_ingestion_allowed",
    "native_dry_run_result_ingestion_enabled", "native_dry_run_result_ingestion_executed",
    "native_dry_run_result_ingested", "native_dry_run_result_applied", "kernel_artifact_registered",
    "kernel_artifact_enabled", "kernel_implementation_loaded", "kernel_binary_loaded",
    "artifact_loader_enabled", "artifact_loaded", "artifact_abi_loaded", "artifact_replay_allowed",
    "artifact_replay_enabled", "artifact_replay_executed", "kernel_launch_allowed", "kernel_launch_enabled",
    "kernel_launch_executed", "parity_check_allowed", "parity_check_enabled", "parity_check_executed",
    "parity_result_recorded", "training_step_allowed", "training_step_enabled", "training_step_executed",
    "generation_request_patch_allowed", "config_adapter_patch_allowed", "runtime_resolver_patch_allowed",
    "execution_resolver_patch_allowed", "training_manager_patch_allowed", "rollout_authorization_allowed",
)
READY_DECISION = "native_dry_run_result_ingestion_recorded_default_off"
BLOCKED_DECISION = "native_dry_run_result_ingestion_blocked_default_off"
HOLD_DECISION = "native_dry_run_result_ingestion_hold_for_signed_review_default_off"
REJECTED_DECISION = "native_dry_run_result_ingestion_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p53_ready = _p53_ready()
    ready = _gate(p53_ready)
    assert ready["ok"] is True, ready
    assert ready["native_dry_run_result_ingestion_contract_ready"] is True, ready
    assert ready["native_dry_run_result_review_signed"] is True, ready
    assert ready["native_dry_run_result_ingestion_evidence_recorded"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p53_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p53_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p53_missing = _gate(None)
    _assert_blocked(p53_missing, "p53", "missing")
    p53_not_ready = _gate({**p53_ready, "ok": False, "kernel_artifact_gate_ready": False})
    _assert_blocked(p53_not_ready, "p53", "not_ready")
    p53_decision_mismatch = _gate({**p53_ready, "decision": "wrong"})
    _assert_blocked(p53_decision_mismatch, "p53", "not_ready")
    p53_unsigned_review = _gate({**p53_ready, "kernel_artifact_gate_review_signed": False})
    _assert_blocked(p53_unsigned_review, "p53", "not_ready")
    p53_post_fields = _gate({**p53_ready, "post_p53_request_fields": {"bad": True}})
    _assert_blocked(p53_post_fields, "post_p53_request_fields")

    p53_unsafe_cases = _unsafe_p53_cases(p53_ready)
    evidence_cases = _evidence_cases(p53_ready)
    inventory_cases = _inventory_cases()
    review_cases = _review_cases(p53_ready)
    history_cases = _history_cases(p53_ready)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p54_native_dry_run_result_ingestion_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p53_missing": _summary(p53_missing),
        "p53_not_ready": _summary(p53_not_ready),
        "p53_decision_mismatch": _summary(p53_decision_mismatch),
        "p53_unsigned_review": _summary(p53_unsigned_review),
        "p53_post_fields": _summary(p53_post_fields),
        "p53_unsafe_cases": p53_unsafe_cases,
        **evidence_cases,
        **inventory_cases,
        **review_cases,
        **history_cases,
    }


def _unsafe_p53_cases(p53_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "native_execution_dry_run_executed", "native_dispatch_executed", "kernel_launch_executed",
        "parity_check_executed", "parity_result_recorded", "native_dry_run_result_ingested",
        "artifact_replay_executed", "kernel_artifact_registered", "kernel_artifact_enabled",
        "training_step_executed",
    ):
        report = _gate({**p53_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p53_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p53_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p53_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_contract_only": _gate(p53_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p53_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_result_ingestion_only": _gate(p53_ready, evidence={**_evidence(), "result_ingestion_only": False}),
        "missing_section": _gate(p53_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "evidence_not_ready": _gate(
            p53_ready,
            evidence={**_evidence(), "native_dry_run_result_ingestion_contract_ready": False},
        ),
        "missing_result_manifest": _gate(
            p53_ready,
            evidence=_without(_evidence(), "native_dry_run_result_manifest"),
        ),
        "missing_digest_inventory": _gate(
            p53_ready,
            evidence=_without(_evidence(), "native_dry_run_result_digest_inventory"),
        ),
        "missing_metric_boundary": _gate(
            p53_ready,
            evidence=_without(_evidence(), "native_dry_run_metric_boundary"),
        ),
    }
    fragments = {
        "missing_source": ("source_missing",),
        "missing_digest": ("digest_missing",),
        "not_contract_only": ("contract_only",),
        "not_records_only": ("records_evidence_only",),
        "not_result_ingestion_only": ("result_ingestion_only",),
        "missing_section": ("section_missing",),
        "evidence_not_ready": ("not_ready",),
        "missing_result_manifest": ("result_manifest",),
        "missing_digest_inventory": ("digest",),
        "missing_metric_boundary": ("metric",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases() -> dict[str, Any]:
    return {
        "result_manifest_cases": _unsafe_inventory_cases(
            "native_dry_run_result_manifest",
            (
                "native_execution_dry_run_executed", "native_dispatch_executed", "kernel_launch_executed",
                "parity_check_executed", "parity_result_recorded", "native_dry_run_result_ingested",
                "native_dry_run_result_applied", "artifact_replay_executed", "training_step_executed",
            ),
        ),
        "digest_inventory_cases": _unsafe_inventory_cases(
            "native_dry_run_result_digest_inventory",
            (
                "native_dry_run_result_ingested", "native_dry_run_result_applied", "artifact_replay_executed",
                "kernel_launch_executed",
            ),
        ),
        "metric_boundary_cases": _unsafe_inventory_cases(
            "native_dry_run_metric_boundary",
            (
                "native_dry_run_result_ingested", "native_dry_run_result_applied", "parity_result_recorded",
                "artifact_replay_executed", "kernel_launch_executed", "native_dispatch_executed",
                "training_step_executed",
            ),
        ),
    }


def _review_cases(p53_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "route_registration": _gate(p53_ready, evidence={**_evidence(), "api_route_registration": {"bad": True}}),
        "review_missing_reviewer": _gate(p53_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p53_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p53_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(
            p53_ready,
            review={**_review(), "acknowledge_no_native_dry_run_executed": False},
        ),
    }
    fragments = {
        "route_registration": ("api_route_registration",),
        "review_missing_reviewer": ("reviewer",),
        "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",),
        "review_missing_ack": ("ack_missing", "dry_run"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])

    unsafe_cases = {}
    for field in (
        "approve_native_dry_run_result_ingested", "approve_native_dry_run_result_applied",
        "approve_native_dispatch_executed", "approve_kernel_launch_executed",
        "approve_artifact_replay_executed", "approve_training_step_executed",
        "approve_training_launch_allowed",
    ):
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p53_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p53_ready,
        failure_history=[{"reason": "native_dry_run_result_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(p53_ready, rollback_history=[{"kind": "result_rollback", "rollback_required": True}])
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p53_ready,
        failure_history=[{"reason": "closed_result_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)
    return {
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p53: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_native_dry_run_result_ingestion_contract(
        p53_kernel_artifact_gate=p53,
        native_dry_run_result_evidence=_evidence() if evidence is None else evidence,
        native_dry_run_result_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p53_ready() -> dict[str, Any]:
    return _p53_gate(_p52_ready())


def _evidence() -> dict[str, Any]:
    sections = [
        "p53_kernel_artifact_gate_reference", "native_dry_run_result_manifest",
        "native_dry_run_result_digest_inventory", "native_dry_run_metric_boundary",
        "result_ingestion_boundary", "request_adapter_boundary", "no_native_execution_boundary",
        "no_native_dispatch_boundary", "no_kernel_launch_boundary", "no_parity_execution_boundary",
        "no_training_step_boundary", "no_request_fields_boundary", "no_training_launch_boundary",
        "rollback_policy", "observability_policy",
    ]
    return {
        "evidence_id": "native_dry_run_result_ingestion_contract_v0",
        "evidence_version": "v0",
        "ok": True,
        "native_dry_run_result_ingestion_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "result_ingestion_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_artifact_replay_or_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "native_dry_run_result_manifest": [_result_row()],
        "native_dry_run_result_digest_inventory": [_digest_row()],
        "native_dry_run_metric_boundary": [_metric_row()],
        "sha256": "sha256:p54:native-dry-run-result-ingestion:ready",
        "artifact_digest": "sha256:p54:native-dry-run-result-ingestion:ready",
        "source": "temp/turbocore_v5_p54_native_dry_run_result_ingestion.json",
        **_safe_flags(),
    }


def _result_row() -> dict[str, Any]:
    return {
        "result_id": "future_native_dry_run_result",
        "result_kind": "future_native_dry_run_result_evidence",
        "sha256": "sha256:p54:future-native-dry-run-result",
        "source": "temp/turbocore_v5_p54_future_native_dry_run_result.json",
        **_safe_flags(),
    }


def _digest_row() -> dict[str, Any]:
    return {
        "digest_id": "future_native_dry_run_result_digest",
        "result_id": "future_native_dry_run_result",
        "sha256": "sha256:p54:future-native-dry-run-result",
        "digest_algorithm": "sha256",
        "source": "temp/turbocore_v5_p54_future_native_dry_run_result.digest.json",
        **_safe_flags(),
    }


def _metric_row() -> dict[str, Any]:
    return {
        "metric_id": "future_native_dry_run_result_metric_boundary",
        "result_id": "future_native_dry_run_result",
        "metric_kind": "future_metric_boundary",
        **_safe_flags(),
    }


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "native_dry_run_result_ingestion_contract",
        "approve_native_dry_run_result_ingestion_contract": approve,
    }
    for field in DEFAULT_OFF_FIELDS:
        if field != "default_behavior_changed":
            review[f"approve_{field}"] = False
    for field in (
        "acknowledge_p53_kernel_artifact_gate_recorded", "acknowledge_default_off_boundary",
        "acknowledge_no_training_launch", "acknowledge_no_ui_exposure",
        "acknowledge_no_native_dry_run_executed", "acknowledge_no_native_dispatch_executed",
        "acknowledge_no_kernel_launch_executed", "acknowledge_no_parity_executed",
        "acknowledge_no_training_step_executed", "acknowledge_no_request_adapter_enabled",
        "acknowledge_no_request_fields_emitted", "acknowledge_no_default_or_auto_rollout",
        "acknowledge_native_dry_run_result_ingestion_evidence_replayable",
        "acknowledge_later_artifact_replay_or_execution_contract_required", "acknowledge_manual_review_only",
    ):
        review[field] = True
    return review


def _safe_flags() -> dict[str, Any]:
    flags = {field: False for field in DEFAULT_OFF_FIELDS}
    flags.update(
        {
            "blocked_reasons": [], "promotion_blockers": [], "post_p53_request_fields": {},
            "post_p54_request_fields": {}, "post_native_dry_run_result_ingestion_fields": {},
            "native_execution_request": {}, "native_execution_payload": {}, "native_dry_run_request": {},
            "native_dry_run_payload": {}, "native_dispatch_request": {}, "native_dispatch_payload": {},
            "native_dry_run_result_ingestion_request": {}, "native_dry_run_result_ingestion_payload": {},
            "native_dry_run_result_payload": {}, "result_application_payload": {},
            "kernel_artifact_registration": {}, "kernel_artifact_payload": {}, "artifact_replay_request": {},
            "artifact_replay_payload": {}, "kernel_launch_request": {}, "kernel_launch_payload": {},
            "parity_check_request": {}, "parity_check_payload": {}, "parity_result_payload": {},
            "training_step_request": {}, "training_step_payload": {}, "request_adapter_fields": {},
            "request_fields": {}, "api_route_registration": {}, "backend_route_registration": {},
            "ui_route_registration": {}, "launcher_menu_entry": {}, "webui_tab_entry": {},
            "launch_request": {}, "training_request": {}, "generation_request_patch": {},
            "config_adapter_patch": {}, "runtime_resolver_patch": {}, "execution_resolver_patch": {},
            "entry_train_patch": {}, "training_manager_patch": {}, "auto_launch_plan": {},
        }
    )
    return flags


def _unsafe_inventory_cases(inventory: str, fields: tuple[str, ...]) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _inventory_claim(inventory, field)
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _inventory_claim(inventory: str, field: str) -> dict[str, Any]:
    rows = {
        "native_dry_run_result_manifest": [_result_row()],
        "native_dry_run_result_digest_inventory": [_digest_row()],
        "native_dry_run_metric_boundary": [_metric_row()],
    }
    rows[inventory] = [{**rows[inventory][0], field: True}]
    return _gate(_p53_ready(), evidence={**_evidence(), **rows})


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p53_ready(), review={**_review(), field: True})


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
    assert report["post_p54_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["native_dry_run_result_ingestion_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["native_dry_run_result_ingestion_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("native_dry_run_result_ingestion_contract_ready", False)),
        "review_signed": bool(report.get("native_dry_run_result_review_signed", False)),
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
