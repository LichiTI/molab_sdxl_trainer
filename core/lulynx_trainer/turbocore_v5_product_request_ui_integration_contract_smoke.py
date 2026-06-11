"""Smoke checks for V5-P67 product/request/UI integration contract."""

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

from core.turbocore_v5_product_request_ui_integration_contract import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P67_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_product_request_ui_integration_contract,
)
from lulynx_trainer.turbocore_v5_rollout_default_enable_contract_smoke import (  # noqa: E402
    _gate as _p66_gate,
    _p65_ready,
)


READY_DECISION = "product_request_ui_integration_contract_recorded_default_off"
BLOCKED_DECISION = "product_request_ui_integration_contract_blocked_default_off"
HOLD_DECISION = "product_request_ui_integration_contract_hold_for_signed_review_default_off"
REJECTED_DECISION = "product_request_ui_integration_contract_rejected_default_off"
INVENTORY_SPECS = (
    ("surface", "product_surface_inventory", "product_surface", "p67_surface"),
    ("schema", "request_schema_inventory", "request_schema", "p67_schema"),
    ("adapter", "request_adapter_mapping_boundary", "request_adapter_mapping", "p67_adapter"),
    ("ui", "ui_exposure_boundary", "ui_exposure_boundary", "p67_ui"),
    ("product", "product_wiring_boundary", "product_wiring", "p67_product"),
    ("launcher", "launcher_boundary", "launcher_boundary", "p67_launcher"),
    ("webui", "webui_boundary", "webui_boundary", "p67_webui"),
    ("router", "backend_router_boundary", "backend_router_boundary", "p67_router"),
    ("training_request", "training_request_boundary", "training_request_boundary", "p67_request"),
    ("default_behavior", "default_behavior_boundary", "default_behavior_boundary", "p67_default"),
    ("operator", "operator_visibility_boundary", "operator_visibility", "p67_operator"),
    ("monitoring", "monitoring_observability_boundary", "monitoring_observability", "p67_monitoring"),
    ("rollback", "rollback_policy_boundary", "rollback_policy", "p67_rollback"),
    ("docs", "docs_release_boundary", "docs_release", "p67_docs"),
)


def run_smoke() -> dict[str, Any]:
    p66_ready = _p66_ready()
    ready = _gate(p66_ready)
    assert ready["ok"] is True, ready
    assert ready["product_request_ui_integration_contract_ready"] is True, ready
    assert ready["product_request_ui_evidence_recorded"] is True, ready
    assert ready["product_request_ui_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p66_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p66_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p66_missing = _gate(None)
    _assert_blocked(p66_missing, "p66", "missing")
    p66_not_ready = _gate({**p66_ready, "ok": False, "rollout_default_enable_contract_ready": False})
    _assert_blocked(p66_not_ready, "p66", "not_ready")
    p66_decision_mismatch = _gate({**p66_ready, "decision": "wrong"})
    _assert_blocked(p66_decision_mismatch, "p66", "not_ready")
    p66_post_fields = _gate({**p66_ready, "post_p66_request_fields": {"bad": True}})
    _assert_blocked(p66_post_fields, "post_p66_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p67_product_request_ui_integration_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p66_missing": _summary(p66_missing),
        "p66_not_ready": _summary(p66_not_ready),
        "p66_decision_mismatch": _summary(p66_decision_mismatch),
        "p66_post_fields": _summary(p66_post_fields),
        "p66_unsafe_cases": _unsafe_p66_cases(p66_ready),
        **_evidence_cases(p66_ready),
        **_inventory_cases(p66_ready),
        **_review_cases(p66_ready),
        **_history_cases(p66_ready),
    }


def _unsafe_p66_cases(p66_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "product_integration_enabled", "ui_route_registered", "ui_exposure_allowed",
        "request_adapter_enabled", "request_fields_emitted", "default_behavior_changed",
        "default_enable_enabled", "default_rollout_allowed", "training_launch_allowed",
        "training_job_created", "runs_dispatched", "runtime_dispatch_executed",
    ):
        report = _gate({**p66_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p66_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p66_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p66_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p66_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p66_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p66_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p66_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_integration_contract": _gate(p66_ready, evidence={**_evidence(), "product_request_ui_integration_contract_only": False}),
        "not_manual_only": _gate(p66_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p66_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_adapter_contract": _gate(
            p66_ready,
            evidence={**_evidence(), "requires_later_request_adapter_activation_contract": False},
        ),
        "evidence_not_ready": _gate(p66_ready, evidence={**_evidence(), "product_request_ui_integration_contract_ready": False}),
        "missing_section": _gate(p66_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p66_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p66_ready, evidence={**_evidence(), "request_adapter_mapping_allowed": True}),
        "evidence_blocker": _gate(p66_ready, evidence={**_evidence(), "blocked_reasons": ["product_ui_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",), "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",), "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",), "not_records_only": ("records_evidence_only",),
        "not_integration_contract": ("product_request_ui_integration_contract_only",),
        "not_manual_only": ("manual_only",), "not_internal_only": ("internal_only",),
        "missing_later_adapter_contract": ("request_adapter_activation",),
        "evidence_not_ready": ("not_ready",), "missing_section": ("section_missing",),
        "default_on": ("default_off",), "request_adapter_on": ("request_adapter",),
        "evidence_blocker": ("product_ui_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p66_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for name, field, kind, item_id in INVENTORY_SPECS:
        missing = _gate(p66_ready, evidence=_without(_evidence(), field))
        not_ready = _gate(p66_ready, evidence={**_evidence(), field: [{**_row(item_id), "ready": False}]})
        source_missing = _gate(p66_ready, evidence={**_evidence(), field: [_without(_row(item_id), "source")]})
        _assert_blocked(missing, f"{kind}_inventory_missing")
        _assert_blocked(not_ready, f"{kind}_not_ready")
        _assert_blocked(source_missing, f"{kind}_source_missing")
        cases[f"missing_{name}"] = _summary(missing)
        cases[f"{name}_not_ready"] = _summary(not_ready)
        cases[f"{name}_missing_source"] = _summary(source_missing)
    return {
        **cases,
        "unsafe_evidence_cases": _unsafe_evidence_cases(p66_ready),
        "unsafe_request_cases": _unsafe_request_cases(p66_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p66_ready),
    }


def _unsafe_evidence_cases(p66_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "product_request_ui_integration_enabled", "product_integration_enabled", "ui_route_registered",
        "ui_entry_registered", "ui_exposure_allowed", "launcher_integration_enabled",
        "webui_integration_enabled", "backend_router_registered", "request_schema_registered",
        "request_adapter_enabled", "request_fields_emitted", "generation_request_fields_added",
        "default_behavior_changed", "default_enable_enabled", "training_launch_allowed",
        "training_job_created", "runs_dispatched",
    ):
        report = _gate(p66_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_request_cases(p66_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "product_request_ui_integration_contract_request", "product_request_ui_integration_contract_payload",
        "product_integration_payload", "product_wiring_payload", "ui_route_registration",
        "ui_route_payload", "ui_exposure_payload", "launcher_payload", "webui_payload",
        "backend_router_payload", "request_schema_payload", "request_adapter_payload",
        "request_fields", "generation_request_payload", "training_request_payload",
        "default_enable_payload", "training_launch_payload", "training_job_payload", "training_run_payload",
    ):
        if field not in UNSAFE_NON_EMPTY_FIELDS:
            continue
        report = _gate(p66_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p66_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "surface": _unsafe_inventory_claims(
            p66_ready, "product_surface_inventory", _row("p67_surface"),
            ("product_integration_enabled", "ui_entry_registered", "request_fields_emitted"),
        ),
        "adapter": _unsafe_inventory_claims(
            p66_ready, "request_adapter_mapping_boundary", _row("p67_adapter"),
            ("request_adapter_enabled", "request_fields_emitted", "generation_request_fields_added"),
        ),
        "ui": _unsafe_inventory_claims(
            p66_ready, "ui_exposure_boundary", _row("p67_ui"),
            ("ui_route_registered", "ui_exposure_allowed", "ui_entry_registered"),
        ),
        "router": _unsafe_inventory_claims(
            p66_ready, "backend_router_boundary", _row("p67_router"),
            ("backend_router_registered", "request_adapter_enabled", "request_fields_emitted"),
        ),
        "request": _unsafe_inventory_claims(
            p66_ready, "training_request_boundary", _row("p67_request"),
            ("training_request_fields_added", "training_launch_allowed", "training_job_created"),
        ),
    }


def _unsafe_inventory_claims(
    p66_ready: dict[str, Any], inventory: str, row: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p66_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p66_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p66_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p66_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p66_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(p66_ready, review={**_review(), "acknowledge_no_ui_route_registered": False}),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",), "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",), "review_missing_ack": ("ack_missing", "ui_route"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for base_field in (
        "product_integration_enabled", "ui_route_registered", "ui_exposure_allowed",
        "request_adapter_enabled", "request_fields_emitted", "training_launch_allowed",
        "training_job_created", "runs_dispatched",
    ):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p66_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p66_ready,
        failure_history=[{"reason": "product_ui_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p66_ready,
        rollback_history=[{"kind": "product_ui_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p66_ready,
        failure_history=[{"reason": "closed_product_ui_warning", "status": "closed", "severity": "high"}],
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
    p66: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_product_request_ui_integration_contract(
        p66_rollout_default_enable_contract=p66,
        product_request_ui_evidence=_evidence() if evidence is None else evidence,
        product_request_ui_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p66_ready() -> dict[str, Any]:
    return _p66_gate(_p65_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    payload = {
        "evidence_id": "product_request_ui_integration_contract_v0",
        "evidence_version": "v0",
        "ok": True,
        "product_request_ui_integration_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "product_request_ui_integration_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_request_adapter_activation_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "sha256": "sha256:p67:product-request-ui-integration-contract:ready",
        "artifact_digest": "sha256:p67:product-request-ui-integration-contract:ready",
        "source": "temp/turbocore_v5_p67_product_request_ui_integration_contract.json",
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
        "requested_scope": P67_SCOPE,
        "approve_product_request_ui_integration_contract": approve,
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
    return _gate(_p66_ready(), review={**_review(), field: True})


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
    assert report["post_p67_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["product_request_ui_integration_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["product_request_ui_integration_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("product_request_ui_integration_contract_ready", False)),
        "review_signed": bool(report.get("product_request_ui_review_signed", False)),
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
