"""Smoke checks for V5-P56 execution replay package refresh contract."""

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

from core.turbocore_v5_execution_replay_package_refresh_contract import (  # noqa: E402
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_execution_replay_package_refresh_contract,
)
from lulynx_trainer.turbocore_v5_artifact_replay_verifier_contract_smoke import (  # noqa: E402
    _gate as _p55_gate,
    _p54_ready,
)


READY_DECISION = "execution_replay_package_refresh_recorded_default_off"
BLOCKED_DECISION = "execution_replay_package_refresh_blocked_default_off"
HOLD_DECISION = "execution_replay_package_refresh_hold_for_signed_review_default_off"
REJECTED_DECISION = "execution_replay_package_refresh_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p55_ready = _p55_ready()
    ready = _gate(p55_ready)
    assert ready["ok"] is True, ready
    assert ready["execution_replay_package_refresh_contract_ready"] is True, ready
    assert ready["execution_replay_package_refresh_evidence_recorded"] is True, ready
    assert ready["execution_replay_refresh_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    assert ready["execution_replay_digest_comparisons"][0]["match"] is True, ready
    _assert_default_off(ready)

    missing_review = _gate(p55_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p55_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p55_missing = _gate(None)
    _assert_blocked(p55_missing, "p55", "missing")
    p55_not_ready = _gate({**p55_ready, "ok": False, "artifact_replay_verifier_contract_ready": False})
    _assert_blocked(p55_not_ready, "p55", "not_ready")
    p55_decision_mismatch = _gate({**p55_ready, "decision": "wrong"})
    _assert_blocked(p55_decision_mismatch, "p55", "not_ready")
    p55_post_fields = _gate({**p55_ready, "post_p55_request_fields": {"bad": True}})
    _assert_blocked(p55_post_fields, "post_p55_request_fields")

    p55_unsafe_cases = _unsafe_p55_cases(p55_ready)
    package_cases = _package_cases(p55_ready)
    inventory_cases = _inventory_cases()
    review_cases = _review_cases(p55_ready)
    history_cases = _history_cases(p55_ready)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p56_execution_replay_package_refresh_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p55_missing": _summary(p55_missing),
        "p55_not_ready": _summary(p55_not_ready),
        "p55_decision_mismatch": _summary(p55_decision_mismatch),
        "p55_post_fields": _summary(p55_post_fields),
        "p55_unsafe_cases": p55_unsafe_cases,
        **package_cases,
        **inventory_cases,
        **review_cases,
        **history_cases,
    }


def _unsafe_p55_cases(p55_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "execution_replay_executed", "execution_replay_package_refreshed", "runtime_state_refreshed",
        "artifact_replay_executed", "artifact_loaded", "native_dispatch_executed", "kernel_launch_executed",
        "training_step_executed",
    ):
        report = _gate({**p55_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _package_cases(p55_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p55_ready, package=_without(_package(), "source")),
        "missing_digest": _gate(p55_ready, package=_without_many(_package(), "sha256", "artifact_digest")),
        "not_contract_only": _gate(p55_ready, package={**_package(), "contract_only": False}),
        "not_records_only": _gate(p55_ready, package={**_package(), "records_evidence_only": False}),
        "not_refresh_only": _gate(p55_ready, package={**_package(), "execution_replay_package_refresh_only": False}),
        "missing_section": _gate(p55_ready, package={**_package(), "available_sections": ["rollback_policy"]}),
        "package_not_ready": _gate(p55_ready, package={**_package(), "execution_replay_package_refresh_contract_ready": False}),
        "missing_manifest": _gate(p55_ready, package=_without(_package(), "execution_replay_package_manifest")),
        "missing_digest_inventory": _gate(p55_ready, package=_without(_package(), "execution_replay_digest_inventory")),
        "missing_precondition": _gate(p55_ready, package=_without(_package(), "execution_replay_precondition_inventory")),
        "missing_freshness": _gate(p55_ready, package=_without(_package(), "execution_replay_freshness_inventory")),
        "digest_mismatch": _gate(p55_ready, package={**_package(), "execution_replay_digest_inventory": [{**_digest_row(), "sha256": "sha256:p56:mismatch"}]}),
        "extra_digest_entry": _gate(p55_ready, package={**_package(), "execution_replay_digest_inventory": [_digest_row(), {**_digest_row(), "package_id": "extra_package"}]}),
        "precondition_not_ready": _gate(p55_ready, package={**_package(), "execution_replay_precondition_inventory": [{**_precondition_row(), "ready": False}]}),
        "not_fresh": _gate(p55_ready, package={**_package(), "execution_replay_freshness_inventory": [{**_freshness_row(), "fresh": False}]}),
        "expiry_review_missing": _gate(p55_ready, package={**_package(), "execution_replay_freshness_inventory": [{**_freshness_row(), "expires_review_before_execution": False}]}),
        "route_registration": _gate(p55_ready, package={**_package(), "api_route_registration": {"bad": True}}),
    }
    fragments = {
        "missing_source": ("source_missing",),
        "missing_digest": ("digest_missing",),
        "not_contract_only": ("contract_only",),
        "not_records_only": ("records_evidence_only",),
        "not_refresh_only": ("refresh_only",),
        "missing_section": ("section_missing",),
        "package_not_ready": ("not_ready",),
        "missing_manifest": ("manifest",),
        "missing_digest_inventory": ("digest_inventory",),
        "missing_precondition": ("precondition",),
        "missing_freshness": ("freshness",),
        "digest_mismatch": ("digest_mismatch",),
        "extra_digest_entry": ("manifest_entry_missing",),
        "precondition_not_ready": ("precondition_not_ready",),
        "not_fresh": ("not_fresh",),
        "expiry_review_missing": ("expiry_review",),
        "route_registration": ("api_route_registration",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases() -> dict[str, Any]:
    return {
        "manifest_unsafe_cases": _unsafe_inventory_cases(
            "execution_replay_package_manifest",
            ("execution_replay_executed", "execution_replay_package_refreshed", "runtime_state_refreshed", "artifact_replay_executed", "kernel_launch_executed"),
        ),
        "digest_inventory_unsafe_cases": _unsafe_inventory_cases(
            "execution_replay_digest_inventory",
            ("execution_replay_executed", "execution_replay_package_refreshed", "artifact_loaded"),
        ),
        "freshness_unsafe_cases": _unsafe_inventory_cases(
            "execution_replay_freshness_inventory",
            ("execution_replay_executed", "runtime_state_refreshed", "native_dispatch_executed"),
        ),
    }


def _review_cases(p55_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p55_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p55_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p55_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(p55_ready, review={**_review(), "acknowledge_no_execution_replay_executed": False}),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",),
        "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",),
        "review_missing_ack": ("ack_missing", "execution_replay"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for field in (
        "approve_execution_replay_executed", "approve_execution_replay_package_refreshed",
        "approve_runtime_state_refreshed", "approve_artifact_replay_executed",
        "approve_native_dispatch_executed", "approve_kernel_launch_executed", "approve_training_launch_allowed",
    ):
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p55_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p55_ready,
        failure_history=[{"reason": "execution_replay_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(p55_ready, rollback_history=[{"kind": "execution_replay_rollback", "rollback_required": True}])
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p55_ready,
        failure_history=[{"reason": "closed_execution_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)
    return {"failure_history": _summary(failure_history), "rollback_history": _summary(rollback_history), "closed_failure": _summary(closed_failure)}


def _gate(
    p55: dict[str, Any] | None,
    *,
    package: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_execution_replay_package_refresh_contract(
        p55_artifact_replay_verifier=p55,
        execution_replay_package=_package() if package is None else package,
        execution_replay_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p55_ready() -> dict[str, Any]:
    return _p55_gate(_p54_ready())


def _package() -> dict[str, Any]:
    sections = [
        "p55_artifact_replay_verifier_reference", "execution_replay_package_manifest",
        "execution_replay_digest_inventory", "execution_replay_precondition_inventory",
        "execution_replay_freshness_inventory", "execution_replay_digest_comparison",
        "execution_replay_refresh_boundary", "request_adapter_boundary", "no_execution_replay_boundary",
        "no_artifact_load_boundary", "no_native_execution_boundary", "no_native_dispatch_boundary",
        "no_kernel_launch_boundary", "no_parity_execution_boundary", "no_training_step_boundary",
        "no_request_fields_boundary", "no_training_launch_boundary", "rollback_policy", "observability_policy",
    ]
    return {
        "package_id": "execution_replay_package_refresh_contract_v0",
        "package_version": "v0",
        "ok": True,
        "execution_replay_package_refresh_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "execution_replay_package_refresh_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_execution_readiness_review": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "execution_replay_package_manifest": [_manifest_row()],
        "execution_replay_digest_inventory": [_digest_row()],
        "execution_replay_precondition_inventory": [_precondition_row()],
        "execution_replay_freshness_inventory": [_freshness_row()],
        "sha256": "sha256:p56:execution-replay-refresh:ready",
        "artifact_digest": "sha256:p56:execution-replay-refresh:ready",
        "source": "temp/turbocore_v5_p56_execution_replay_package_refresh.json",
        **_safe_flags(),
    }


def _manifest_row() -> dict[str, Any]:
    return {
        "package_id": "future_execution_replay_package",
        "package_kind": "future_execution_replay_package_evidence",
        "sha256": "sha256:p56:future-execution-replay-package",
        "source": "temp/turbocore_v5_p56_future_execution_replay_package.json",
        **_safe_flags(),
    }


def _digest_row() -> dict[str, Any]:
    return {
        "package_id": "future_execution_replay_package",
        "digest_id": "future_execution_replay_package_digest",
        "sha256": "sha256:p56:future-execution-replay-package",
        "digest_algorithm": "sha256",
        "source": "temp/turbocore_v5_p56_future_execution_replay_package.digest.json",
        **_safe_flags(),
    }


def _precondition_row() -> dict[str, Any]:
    return {"precondition_id": "p56_no_runtime_execution_boundary", "ready": True, "source": "temp/turbocore_v5_p56_precondition.json", **_safe_flags()}


def _freshness_row() -> dict[str, Any]:
    return {
        "refresh_id": "p56_execution_replay_package_freshness",
        "fresh": True,
        "expires_review_before_execution": True,
        "source": "temp/turbocore_v5_p56_freshness.json",
        **_safe_flags(),
    }


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "execution_replay_package_refresh_contract",
        "approve_execution_replay_package_refresh_contract": approve,
    }
    for field in UNSAFE_TRUE_FIELDS:
        if field != "default_behavior_changed":
            review[f"approve_{field}"] = False
    for field in REQUIRED_REVIEW_ACKS:
        review[field] = True
    return review


def _safe_flags() -> dict[str, Any]:
    flags = {field: False for field in UNSAFE_TRUE_FIELDS}
    flags.update({field: {} for field in UNSAFE_NON_EMPTY_FIELDS})
    flags.update({"blocked_reasons": [], "promotion_blockers": []})
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
        "execution_replay_package_manifest": [_manifest_row()],
        "execution_replay_digest_inventory": [_digest_row()],
        "execution_replay_precondition_inventory": [_precondition_row()],
        "execution_replay_freshness_inventory": [_freshness_row()],
    }
    rows[inventory] = [{**rows[inventory][0], field: True}]
    return _gate(_p55_ready(), package={**_package(), **rows})


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p55_ready(), review={**_review(), field: True})


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
    assert report["post_p56_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["execution_replay_package_refresh_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["execution_replay_package_refresh_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("execution_replay_package_refresh_contract_ready", False)),
        "review_signed": bool(report.get("execution_replay_refresh_review_signed", False)),
        "rollback_required": bool(report.get("rollback_required", False)),
        "comparison_count": len(report.get("execution_replay_digest_comparisons") or []),
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
