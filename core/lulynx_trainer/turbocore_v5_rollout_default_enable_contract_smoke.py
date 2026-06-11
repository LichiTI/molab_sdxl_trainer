"""Smoke checks for V5-P66 rollout/default-enable contract."""

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

from core.turbocore_v5_rollout_default_enable_contract import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P66_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_rollout_default_enable_contract,
)
from lulynx_trainer.turbocore_v5_training_launch_contract_smoke import (  # noqa: E402
    _gate as _p65_gate,
    _p64_ready,
)


READY_DECISION = "rollout_default_enable_contract_recorded_default_off"
BLOCKED_DECISION = "rollout_default_enable_contract_blocked_default_off"
HOLD_DECISION = "rollout_default_enable_contract_hold_for_signed_review_default_off"
REJECTED_DECISION = "rollout_default_enable_contract_rejected_default_off"
INVENTORY_SPECS = (
    ("rollout", "rollout_plan_inventory", "rollout_plan", "p66_rollout_plan"),
    ("authorization", "default_enable_authorization_boundary", "default_enable_authorization", "p66_authorization"),
    ("precondition", "default_enable_precondition_inventory", "default_enable_precondition", "p66_precondition"),
    ("operator", "operator_approval_boundary", "operator_approval", "p66_operator"),
    ("scope", "rollout_scope_boundary", "rollout_scope", "p66_scope"),
    ("policy", "rollout_policy_boundary", "rollout_policy", "p66_policy"),
    ("risk", "risk_acceptance_boundary", "risk_acceptance", "p66_risk"),
    ("monitoring", "monitoring_observability_boundary", "monitoring_observability", "p66_monitoring"),
    ("rollback", "rollback_policy_boundary", "rollback_policy", "p66_rollback"),
    ("adapter", "request_adapter_boundary", "request_adapter_boundary", "p66_adapter"),
    ("ui", "ui_exposure_boundary", "ui_exposure_boundary", "p66_ui"),
    ("launch", "training_launch_boundary", "training_launch_boundary", "p66_launch"),
    ("job", "training_job_boundary", "training_job_boundary", "p66_job"),
    ("run", "run_dispatch_boundary", "run_dispatch_boundary", "p66_run"),
    ("schema", "config_schema_boundary", "config_schema_boundary", "p66_schema"),
)


def run_smoke() -> dict[str, Any]:
    p65_ready = _p65_ready()
    ready = _gate(p65_ready)
    assert ready["ok"] is True, ready
    assert ready["rollout_default_enable_contract_ready"] is True, ready
    assert ready["rollout_default_enable_evidence_recorded"] is True, ready
    assert ready["rollout_default_enable_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p65_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p65_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p65_missing = _gate(None)
    _assert_blocked(p65_missing, "p65", "missing")
    p65_not_ready = _gate({**p65_ready, "ok": False, "training_launch_contract_ready": False})
    _assert_blocked(p65_not_ready, "p65", "not_ready")
    p65_decision_mismatch = _gate({**p65_ready, "decision": "wrong"})
    _assert_blocked(p65_decision_mismatch, "p65", "not_ready")
    p65_post_fields = _gate({**p65_ready, "post_p65_request_fields": {"bad": True}})
    _assert_blocked(p65_post_fields, "post_p65_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p66_rollout_default_enable_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p65_missing": _summary(p65_missing),
        "p65_not_ready": _summary(p65_not_ready),
        "p65_decision_mismatch": _summary(p65_decision_mismatch),
        "p65_post_fields": _summary(p65_post_fields),
        "p65_unsafe_cases": _unsafe_p65_cases(p65_ready),
        **_evidence_cases(p65_ready),
        **_inventory_cases(p65_ready),
        **_review_cases(p65_ready),
        **_history_cases(p65_ready),
    }


def _unsafe_p65_cases(p65_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "rollout_default_enable_allowed", "default_enable_enabled", "default_rollout_allowed",
        "auto_rollout_allowed", "training_path_enabled", "request_adapter_enabled",
        "request_fields_emitted", "ui_exposure_allowed", "training_launch_allowed",
        "training_job_created", "runs_dispatched", "training_run_dispatched",
    ):
        report = _gate({**p65_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p65_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p65_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p65_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p65_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p65_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p65_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p65_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_default_enable_contract": _gate(p65_ready, evidence={**_evidence(), "rollout_default_enable_contract_only": False}),
        "not_manual_only": _gate(p65_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p65_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_product_contract": _gate(
            p65_ready,
            evidence={**_evidence(), "requires_later_product_integration_contract": False},
        ),
        "evidence_not_ready": _gate(p65_ready, evidence={**_evidence(), "rollout_default_enable_contract_ready": False}),
        "missing_section": _gate(p65_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p65_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p65_ready, evidence={**_evidence(), "request_adapter_mapping_allowed": True}),
        "evidence_blocker": _gate(p65_ready, evidence={**_evidence(), "blocked_reasons": ["rollout_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",),
        "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",),
        "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",),
        "not_records_only": ("records_evidence_only",),
        "not_default_enable_contract": ("rollout_default_enable_contract_only",),
        "not_manual_only": ("manual_only",),
        "not_internal_only": ("internal_only",),
        "missing_later_product_contract": ("product_integration",),
        "evidence_not_ready": ("not_ready",),
        "missing_section": ("section_missing",),
        "default_on": ("default_off",),
        "request_adapter_on": ("request_adapter",),
        "evidence_blocker": ("rollout_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p65_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for name, field, kind, item_id in INVENTORY_SPECS:
        missing = _gate(p65_ready, evidence=_without(_evidence(), field))
        not_ready = _gate(p65_ready, evidence={**_evidence(), field: [{**_row(item_id), "ready": False}]})
        source_missing = _gate(p65_ready, evidence={**_evidence(), field: [_without(_row(item_id), "source")]})
        _assert_blocked(missing, f"{kind}_inventory_missing")
        _assert_blocked(not_ready, f"{kind}_not_ready")
        _assert_blocked(source_missing, f"{kind}_source_missing")
        cases[f"missing_{name}"] = _summary(missing)
        cases[f"{name}_not_ready"] = _summary(not_ready)
        cases[f"{name}_missing_source"] = _summary(source_missing)
    return {
        **cases,
        "unsafe_evidence_cases": _unsafe_evidence_cases(p65_ready),
        "unsafe_request_cases": _unsafe_request_cases(p65_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p65_ready),
    }


def _unsafe_evidence_cases(p65_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "rollout_default_enable_allowed", "default_enable_enabled", "default_rollout_allowed",
        "auto_rollout_allowed", "training_path_enabled", "request_adapter_enabled",
        "request_fields_emitted", "ui_exposure_allowed", "training_launch_allowed",
        "training_job_created", "runs_dispatched", "training_run_dispatched",
        "product_integration_enabled", "ui_entry_registered", "default_behavior_changed",
    ):
        report = _gate(p65_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_request_cases(p65_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "api_route_registration", "ui_route_registration", "request_fields", "post_p66_request_fields",
        "rollout_default_enable_contract_request", "rollout_default_enable_contract_payload",
        "rollout_plan_payload", "default_enable_authorization_payload", "default_enable_payload",
        "rollout_policy_payload", "rollout_scope_payload", "risk_acceptance_payload",
        "product_integration_payload", "request_adapter_payload", "ui_exposure_payload",
        "training_launch_payload", "training_job_payload", "training_run_payload",
    ):
        if field not in UNSAFE_NON_EMPTY_FIELDS:
            continue
        report = _gate(p65_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p65_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollout": _unsafe_inventory_claims(
            p65_ready, "rollout_plan_inventory", _row("p66_rollout_plan"),
            ("rollout_plan_executed", "default_rollout_allowed", "auto_rollout_allowed"),
        ),
        "policy": _unsafe_inventory_claims(
            p65_ready, "rollout_policy_boundary", _row("p66_policy"),
            ("rollout_policy_applied", "default_enable_enabled", "training_path_enabled"),
        ),
        "adapter": _unsafe_inventory_claims(
            p65_ready, "request_adapter_boundary", _row("p66_adapter"),
            ("request_adapter_enabled", "request_fields_emitted", "training_launch_allowed"),
        ),
        "ui": _unsafe_inventory_claims(
            p65_ready, "ui_exposure_boundary", _row("p66_ui"),
            ("ui_exposure_allowed", "ui_entry_registered", "request_fields_emitted"),
        ),
        "launch": _unsafe_inventory_claims(
            p65_ready, "training_launch_boundary", _row("p66_launch"),
            ("training_launch_allowed", "training_job_created", "runs_dispatched"),
        ),
    }


def _unsafe_inventory_claims(
    p65_ready: dict[str, Any], inventory: str, row: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p65_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p65_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p65_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p65_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p65_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(p65_ready, review={**_review(), "acknowledge_no_default_enable": False}),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",),
        "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",),
        "review_missing_ack": ("ack_missing", "default_enable"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for base_field in (
        "rollout_default_enable_allowed", "default_enable_enabled", "default_rollout_allowed",
        "auto_rollout_allowed", "request_fields_emitted", "ui_exposure_allowed",
        "training_launch_allowed", "training_job_created", "runs_dispatched",
    ):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p65_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p65_ready,
        failure_history=[{"reason": "rollout_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p65_ready,
        rollback_history=[{"kind": "rollout_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p65_ready,
        failure_history=[{"reason": "closed_rollout_warning", "status": "closed", "severity": "high"}],
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
    p65: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_rollout_default_enable_contract(
        p65_training_launch_contract=p65,
        rollout_default_enable_evidence=_evidence() if evidence is None else evidence,
        rollout_default_enable_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p65_ready() -> dict[str, Any]:
    return _p65_gate(_p64_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    payload = {
        "evidence_id": "rollout_default_enable_contract_v0",
        "evidence_version": "v0",
        "ok": True,
        "rollout_default_enable_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "rollout_default_enable_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_product_integration_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "sha256": "sha256:p66:rollout-default-enable-contract:ready",
        "artifact_digest": "sha256:p66:rollout-default-enable-contract:ready",
        "source": "temp/turbocore_v5_p66_rollout_default_enable_contract.json",
        **_safe_flags(),
    }
    for _name, field, _kind, item_id in INVENTORY_SPECS:
        payload[field] = [_row(item_id)]
    return payload


def _row(item_id: str) -> dict[str, Any]:
    return {"check_id": item_id, "ready": True, "source": f"temp/turbocore_v5_{item_id}.json", **_safe_flags()}


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": P66_SCOPE,
        "approve_rollout_default_enable_contract": approve,
    }
    for field in UNSAFE_TRUE_FIELDS:
        review[f"approve_{field}"] = False
    for field in REQUIRED_REVIEW_ACKS:
        review[field] = True
    return review


def _safe_flags() -> dict[str, Any]:
    flags = {field: False for field in UNSAFE_TRUE_FIELDS}
    flags.update({field: {} for field in UNSAFE_NON_EMPTY_FIELDS})
    flags.update({"blocked_reasons": [], "promotion_blockers": []})
    return flags


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p65_ready(), review={**_review(), field: True})


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
    for field in UNSAFE_TRUE_FIELDS:
        assert report[field] is False, report
    assert report["post_p66_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["rollout_default_enable_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["rollout_default_enable_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("rollout_default_enable_contract_ready", False)),
        "review_signed": bool(report.get("rollout_default_enable_review_signed", False)),
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
