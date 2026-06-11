"""Smoke checks for V5-P68 request-adapter activation contract."""

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

from core.turbocore_v5_request_adapter_activation_contract import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P68_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_request_adapter_activation_contract,
)
from lulynx_trainer.turbocore_v5_product_request_ui_integration_contract_smoke import (  # noqa: E402
    _gate as _p67_gate,
    _p66_ready,
)


READY_DECISION = "request_adapter_activation_contract_recorded_default_off"
BLOCKED_DECISION = "request_adapter_activation_contract_blocked_default_off"
HOLD_DECISION = "request_adapter_activation_contract_hold_for_signed_review_default_off"
REJECTED_DECISION = "request_adapter_activation_contract_rejected_default_off"
INVENTORY_SPECS = (
    ("activation_plan", "request_adapter_activation_plan_inventory", "request_adapter_activation_plan", "p68_plan"),
    ("activation_precondition", "request_adapter_activation_precondition_inventory", "request_adapter_activation_precondition", "p68_precondition"),
    ("adapter_contract", "request_adapter_contract_boundary", "request_adapter_contract", "p68_contract"),
    ("schema", "request_schema_boundary", "request_schema_boundary", "p68_schema"),
    ("generation", "generation_request_boundary", "generation_request_boundary", "p68_generation"),
    ("config", "config_adapter_boundary", "config_adapter_boundary", "p68_config"),
    ("router", "backend_router_boundary", "backend_router_boundary", "p68_router"),
    ("launcher", "launcher_boundary", "launcher_boundary", "p68_launcher"),
    ("webui", "webui_boundary", "webui_boundary", "p68_webui"),
    ("operator", "operator_activation_boundary", "operator_activation_boundary", "p68_operator"),
    ("feature", "feature_flag_boundary", "feature_flag_boundary", "p68_feature"),
    ("observability", "observability_boundary", "observability_boundary", "p68_observability"),
    ("rollback", "rollback_policy_boundary", "rollback_policy", "p68_rollback"),
    ("training_request", "training_request_boundary", "training_request_boundary", "p68_request"),
)


def run_smoke() -> dict[str, Any]:
    p67_ready = _p67_ready()
    ready = _gate(p67_ready)
    assert ready["ok"] is True, ready
    assert ready["request_adapter_activation_contract_ready"] is True, ready
    assert ready["request_adapter_activation_evidence_recorded"] is True, ready
    assert ready["request_adapter_activation_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p67_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p67_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p67_missing = _gate(None)
    _assert_blocked(p67_missing, "p67", "missing")
    p67_not_ready = _gate({**p67_ready, "ok": False, "product_request_ui_integration_contract_ready": False})
    _assert_blocked(p67_not_ready, "p67", "not_ready")
    p67_decision_mismatch = _gate({**p67_ready, "decision": "wrong"})
    _assert_blocked(p67_decision_mismatch, "p67", "not_ready")
    p67_post_fields = _gate({**p67_ready, "post_p67_request_fields": {"bad": True}})
    _assert_blocked(p67_post_fields, "post_p67_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p68_request_adapter_activation_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p67_missing": _summary(p67_missing),
        "p67_not_ready": _summary(p67_not_ready),
        "p67_decision_mismatch": _summary(p67_decision_mismatch),
        "p67_post_fields": _summary(p67_post_fields),
        "p67_unsafe_cases": _unsafe_p67_cases(p67_ready),
        **_evidence_cases(p67_ready),
        **_inventory_cases(p67_ready),
        **_review_cases(p67_ready),
        **_history_cases(p67_ready),
    }


def _unsafe_p67_cases(p67_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "request_adapter_enabled", "request_fields_emitted", "generation_request_fields_added",
        "backend_router_registered", "ui_route_registered", "training_request_submitted",
        "training_launch_allowed", "training_job_created", "runs_dispatched", "runtime_dispatch_executed",
    ):
        report = _gate({**p67_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p67_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p67_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p67_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p67_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p67_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p67_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p67_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_activation_contract": _gate(p67_ready, evidence={**_evidence(), "request_adapter_activation_contract_only": False}),
        "not_manual_only": _gate(p67_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p67_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_execution_contract": _gate(
            p67_ready,
            evidence={**_evidence(), "requires_later_request_adapter_execution_contract": False},
        ),
        "evidence_not_ready": _gate(p67_ready, evidence={**_evidence(), "request_adapter_activation_contract_ready": False}),
        "missing_section": _gate(p67_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p67_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p67_ready, evidence={**_evidence(), "request_adapter_mapping_allowed": True}),
        "evidence_blocker": _gate(p67_ready, evidence={**_evidence(), "blocked_reasons": ["adapter_activation_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",), "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",), "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",), "not_records_only": ("records_evidence_only",),
        "not_activation_contract": ("request_adapter_activation_contract_only",),
        "not_manual_only": ("manual_only",), "not_internal_only": ("internal_only",),
        "missing_later_execution_contract": ("request_adapter_execution",),
        "evidence_not_ready": ("not_ready",), "missing_section": ("section_missing",),
        "default_on": ("default_off",), "request_adapter_on": ("request_adapter",),
        "evidence_blocker": ("adapter_activation_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p67_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for name, field, kind, item_id in INVENTORY_SPECS:
        missing = _gate(p67_ready, evidence=_without(_evidence(), field))
        not_ready = _gate(p67_ready, evidence={**_evidence(), field: [{**_row(item_id), "ready": False}]})
        source_missing = _gate(p67_ready, evidence={**_evidence(), field: [_without(_row(item_id), "source")]})
        _assert_blocked(missing, f"{kind}_inventory_missing")
        _assert_blocked(not_ready, f"{kind}_not_ready")
        _assert_blocked(source_missing, f"{kind}_source_missing")
        cases[f"missing_{name}"] = _summary(missing)
        cases[f"{name}_not_ready"] = _summary(not_ready)
        cases[f"{name}_missing_source"] = _summary(source_missing)
    return {
        **cases,
        "unsafe_evidence_cases": _unsafe_evidence_cases(p67_ready),
        "unsafe_request_cases": _unsafe_request_cases(p67_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p67_ready),
    }


def _unsafe_evidence_cases(p67_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "request_adapter_activation_enabled", "request_adapter_enabled", "request_adapter_registered",
        "request_adapter_contract_registered", "request_fields_emitted", "generation_request_fields_added",
        "generation_request_schema_patched", "config_adapter_patched", "backend_router_registered",
        "ui_route_registered", "adapter_feature_flag_enabled", "training_request_submitted",
        "training_launch_allowed", "training_job_created", "runs_dispatched",
    ):
        report = _gate(p67_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_request_cases(p67_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "request_adapter_activation_contract_request", "request_adapter_activation_contract_payload",
        "request_adapter_activation_payload", "request_adapter_registration_payload",
        "request_adapter_fields_payload", "generation_request_patch_payload", "config_adapter_patch_payload",
        "backend_router_payload", "ui_route_payload", "request_schema_payload", "request_adapter_payload",
        "request_fields", "training_request_payload", "training_launch_payload", "training_job_payload",
    ):
        if field not in UNSAFE_NON_EMPTY_FIELDS:
            continue
        report = _gate(p67_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p67_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "adapter_contract": _unsafe_inventory_claims(
            p67_ready, "request_adapter_contract_boundary", _row("p68_contract"),
            ("request_adapter_registered", "request_adapter_enabled", "request_fields_emitted"),
        ),
        "generation": _unsafe_inventory_claims(
            p67_ready, "generation_request_boundary", _row("p68_generation"),
            ("generation_request_fields_added", "generation_request_schema_patched", "request_fields_emitted"),
        ),
        "config": _unsafe_inventory_claims(
            p67_ready, "config_adapter_boundary", _row("p68_config"),
            ("config_adapter_patched", "request_adapter_enabled", "request_fields_emitted"),
        ),
        "router": _unsafe_inventory_claims(
            p67_ready, "backend_router_boundary", _row("p68_router"),
            ("backend_router_registered", "request_adapter_enabled", "training_request_submitted"),
        ),
        "request": _unsafe_inventory_claims(
            p67_ready, "training_request_boundary", _row("p68_request"),
            ("training_request_submitted", "training_launch_allowed", "training_job_created"),
        ),
    }


def _unsafe_inventory_claims(
    p67_ready: dict[str, Any], inventory: str, row: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p67_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p67_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p67_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p67_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p67_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(p67_ready, review={**_review(), "acknowledge_no_request_adapter_enabled": False}),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",), "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",), "review_missing_ack": ("ack_missing", "request_adapter"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for base_field in (
        "request_adapter_enabled", "request_fields_emitted", "generation_request_fields_added",
        "backend_router_registered", "ui_route_registered", "training_launch_allowed", "training_job_created",
    ):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p67_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p67_ready,
        failure_history=[{"reason": "adapter_activation_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p67_ready,
        rollback_history=[{"kind": "adapter_activation_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p67_ready,
        failure_history=[{"reason": "closed_adapter_warning", "status": "closed", "severity": "high"}],
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
    p67: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_request_adapter_activation_contract(
        p67_product_request_ui_integration_contract=p67,
        request_adapter_activation_evidence=_evidence() if evidence is None else evidence,
        request_adapter_activation_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p67_ready() -> dict[str, Any]:
    return _p67_gate(_p66_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    payload = {
        "evidence_id": "request_adapter_activation_contract_v0",
        "evidence_version": "v0",
        "ok": True,
        "request_adapter_activation_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "request_adapter_activation_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_request_adapter_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "sha256": "sha256:p68:request-adapter-activation-contract:ready",
        "artifact_digest": "sha256:p68:request-adapter-activation-contract:ready",
        "source": "temp/turbocore_v5_p68_request_adapter_activation_contract.json",
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
        "requested_scope": P68_SCOPE,
        "approve_request_adapter_activation_contract": approve,
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
    return _gate(_p67_ready(), review={**_review(), field: True})


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
    assert report["post_p68_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["request_adapter_activation_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["request_adapter_activation_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("request_adapter_activation_contract_ready", False)),
        "review_signed": bool(report.get("request_adapter_activation_review_signed", False)),
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
