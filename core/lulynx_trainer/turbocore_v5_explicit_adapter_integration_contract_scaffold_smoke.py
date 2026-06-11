"""Smoke checks for V5-P42 explicit adapter-integration scaffold."""

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

from core.turbocore_v5_explicit_adapter_integration_contract_scaffold import (  # noqa: E402
    build_v5_explicit_adapter_integration_contract_scaffold,
)
from lulynx_trainer.turbocore_v5_request_adapter_mapping_boundary_smoke import (  # noqa: E402
    _gate as _p41_gate,
    _p40_ready,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "auto_launch_allowed",
    "runs_dispatched",
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "ui_exposure_allowed",
    "product_ui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "request_adapter_registered",
    "runtime_adapter_registered",
    "adapter_integration_allowed",
    "adapter_wiring_allowed",
    "generation_request_patch_allowed",
    "config_adapter_patch_allowed",
    "runtime_resolver_patch_allowed",
    "rollout_authorization_allowed",
)
READY_DECISION = "explicit_adapter_integration_contract_scaffold_recorded_default_off"
BLOCKED_DECISION = "explicit_adapter_integration_contract_scaffold_blocked_default_off"
HOLD_DECISION = "explicit_adapter_integration_contract_scaffold_hold_for_signed_review_default_off"
REJECTED_DECISION = "explicit_adapter_integration_contract_scaffold_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p41_ready = _p41_ready()
    ready = _gate(p41_ready)
    assert ready["ok"] is True, ready
    assert ready["adapter_integration_contract_scaffold_ready"] is True, ready
    assert ready["adapter_integration_contract_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p41_ready, review=None)
    _assert_hold(missing_review, "review", "missing")

    rejected_review = _gate(p41_ready, review=_scaffold_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p41_missing = _gate(None)
    _assert_blocked(p41_missing, "p41", "missing")

    p41_not_ready = _gate({**p41_ready, "ok": False, "request_adapter_mapping_boundary_ready": False})
    _assert_blocked(p41_not_ready, "p41", "not_ready")

    p41_decision_mismatch = _gate({**p41_ready, "decision": "wrong"})
    _assert_blocked(p41_decision_mismatch, "p41", "not_ready")

    p41_adapter_registered = _gate({**p41_ready, "request_adapter_registered": True})
    _assert_blocked(p41_adapter_registered, "request_adapter_registered")

    p41_request_fields = _gate({**p41_ready, "request_fields_emitted": True})
    _assert_blocked(p41_request_fields, "request_fields_emitted")

    p41_post_fields = _gate({**p41_ready, "post_p41_request_fields": {"bad": True}})
    _assert_blocked(p41_post_fields, "post_p41_request_fields")

    missing_source = _gate(p41_ready, scaffold=_without(_scaffold_evidence(), "source"))
    _assert_blocked(missing_source, "source_missing")

    missing_digest = _gate(p41_ready, scaffold=_without_many(_scaffold_evidence(), "sha256", "artifact_digest"))
    _assert_blocked(missing_digest, "digest_missing")

    not_report_only = _gate(p41_ready, scaffold={**_scaffold_evidence(), "report_only": False})
    _assert_blocked(not_report_only, "report_only")

    not_scaffold_only = _gate(p41_ready, scaffold={**_scaffold_evidence(), "scaffold_only": False})
    _assert_blocked(not_scaffold_only, "scaffold_only")

    missing_later_contract = _gate(
        p41_ready,
        scaffold={**_scaffold_evidence(), "requires_later_runtime_wiring_contract": False},
    )
    _assert_blocked(missing_later_contract, "later_runtime_wiring_contract")

    missing_section = _gate(p41_ready, scaffold={**_scaffold_evidence(), "available_sections": ["rollback_policy"]})
    _assert_blocked(missing_section, "section_missing")

    missing_contract_inventory = _gate(p41_ready, scaffold=_without(_scaffold_evidence(), "adapter_contract_inventory"))
    _assert_blocked(missing_contract_inventory, "adapter_contract_inventory")

    missing_future_field_inventory = _gate(
        p41_ready,
        scaffold=_without(_scaffold_evidence(), "future_request_field_inventory"),
    )
    _assert_blocked(missing_future_field_inventory, "future_request_field_inventory")

    scaffold_enabled = _gate(
        p41_ready,
        scaffold={**_scaffold_evidence(), "adapter_contract_inventory": [{"contract_id": "native", "scaffold_enabled": True}]},
    )
    _assert_blocked(scaffold_enabled, "scaffold_not_disabled")

    adapter_registered = _contract_claim("adapter_registered")
    _assert_blocked(adapter_registered, "adapter_registered")

    request_fields = _contract_claim("request_fields_emitted")
    _assert_blocked(request_fields, "request_fields_emitted")

    generation_patch = _contract_claim("generation_request_patch_applied")
    _assert_blocked(generation_patch, "generation_request_patch")

    config_patch = _contract_claim("config_adapter_patch_applied")
    _assert_blocked(config_patch, "config_adapter_patch")

    runtime_patch = _contract_claim("runtime_resolver_patch_applied")
    _assert_blocked(runtime_patch, "runtime_resolver_patch")

    future_field_enabled = _future_field_claim("field_enabled")
    _assert_blocked(future_field_enabled, "field_enabled")

    future_field_emitted = _future_field_claim("field_emitted")
    _assert_blocked(future_field_emitted, "field_emitted")

    future_default_value = _future_field_claim("default_value_materialized")
    _assert_blocked(future_default_value, "default_value_materialized")

    review_missing_reviewer = _gate(p41_ready, review={**_scaffold_review(), "reviewer": ""})
    _assert_blocked(review_missing_reviewer, "reviewer")

    review_missing_reviewed_at = _gate(p41_ready, review={**_scaffold_review(), "reviewed_at": ""})
    _assert_blocked(review_missing_reviewed_at, "reviewed_at")

    review_scope_mismatch = _gate(p41_ready, review={**_scaffold_review(), "requested_scope": "wrong"})
    _assert_blocked(review_scope_mismatch, "scope")

    review_missing_ack = _gate(p41_ready, review={**_scaffold_review(), "acknowledge_no_runtime_resolver_patch": False})
    _assert_blocked(review_missing_ack, "ack_missing", "runtime_resolver")

    review_approve_adapter = _unsafe_review("approve_adapter_integration_allowed")
    _assert_blocked(review_approve_adapter, "approve_adapter_integration_allowed")

    review_approve_fields = _unsafe_review("approve_request_fields_emitted")
    _assert_blocked(review_approve_fields, "approve_request_fields_emitted")

    review_approve_runtime = _unsafe_review("approve_runtime_resolver_patch_allowed")
    _assert_blocked(review_approve_runtime, "approve_runtime_resolver_patch_allowed")

    review_approve_launch = _unsafe_review("approve_training_launch_allowed")
    _assert_blocked(review_approve_launch, "approve_training_launch_allowed")

    failure_history = _gate(
        p41_ready,
        failure_history=[{"reason": "adapter_contract_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")

    rollback_history = _gate(p41_ready, rollback_history=[{"kind": "adapter_contract_rollback", "rollback_required": True}])
    _assert_blocked(rollback_history, "rollback_history")

    closed_failure = _gate(
        p41_ready,
        failure_history=[{"reason": "closed_adapter_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p42_explicit_adapter_integration_contract_scaffold_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p41_missing": _summary(p41_missing),
        "p41_not_ready": _summary(p41_not_ready),
        "p41_decision_mismatch": _summary(p41_decision_mismatch),
        "p41_adapter_registered": _summary(p41_adapter_registered),
        "p41_request_fields": _summary(p41_request_fields),
        "p41_post_fields": _summary(p41_post_fields),
        "missing_source": _summary(missing_source),
        "missing_digest": _summary(missing_digest),
        "missing_contract_inventory": _summary(missing_contract_inventory),
        "missing_future_field_inventory": _summary(missing_future_field_inventory),
        "review_approve_runtime": _summary(review_approve_runtime),
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p41: dict[str, Any] | None,
    *,
    scaffold: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _scaffold_review() if review is ... else review
    return build_v5_explicit_adapter_integration_contract_scaffold(
        p41_request_adapter_mapping_boundary=p41,
        adapter_integration_contract_scaffold=_scaffold_evidence() if scaffold is None else scaffold,
        adapter_integration_contract_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p41_ready() -> dict[str, Any]:
    return _p41_gate(_p40_ready())


def _scaffold_evidence() -> dict[str, Any]:
    sections = [
        "p41_mapping_boundary_reference",
        "adapter_contract_inventory",
        "future_request_field_inventory",
        "generation_request_contract_boundary",
        "config_adapter_contract_boundary",
        "runtime_resolver_contract_boundary",
        "no_adapter_registration_boundary",
        "no_request_fields_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    return {
        "evidence_id": "explicit_adapter_integration_contract_scaffold_v0",
        "evidence_version": "v0",
        "ok": True,
        "adapter_integration_contract_scaffold_ready": True,
        "report_only": True,
        "boundary_only": True,
        "scaffold_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_runtime_wiring_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "adapter_contract_inventory": [
            {"contract_id": "native_update_adapter", "scaffold_enabled": False, **_safe_row_flags()},
            {"contract_id": "lora_fused_adapter", "scaffold_enabled": False, **_safe_row_flags()},
        ],
        "future_request_field_inventory": [
            {"field_id": "turbocore_native_update_explicit_adapter_contract", "field_enabled": False},
            {"field_id": "turbocore_runtime_adapter_contract_digest", "field_enabled": False},
        ],
        "sha256": "sha256:p42:scaffold:ready",
        "artifact_digest": "sha256:p42:scaffold:ready",
        "source": "temp/turbocore_v5_p42_scaffold.json",
        **_safe_row_flags(),
    }


def _scaffold_review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "explicit_adapter_integration_contract_scaffold",
        "approve_adapter_integration_contract_scaffold": approve,
    }
    for field in (
        "approve_training_launch_allowed",
        "approve_auto_launch_allowed",
        "approve_runs_dispatched",
        "approve_default_training_path_enabled",
        "approve_training_path_enabled",
        "approve_default_rollout_allowed",
        "approve_auto_rollout_allowed",
        "approve_ui_exposure_allowed",
        "approve_product_ui_exposure_allowed",
        "approve_request_adapter_mapping_allowed",
        "approve_request_fields_emitted",
        "approve_request_adapter_registered",
        "approve_runtime_adapter_registered",
        "approve_adapter_integration_allowed",
        "approve_adapter_wiring_allowed",
        "approve_generation_request_patch_allowed",
        "approve_config_adapter_patch_allowed",
        "approve_runtime_resolver_patch_allowed",
        "approve_rollout_authorization_allowed",
    ):
        review[field] = False
    for field in (
        "acknowledge_p41_mapping_boundary_recorded",
        "acknowledge_default_off_boundary",
        "acknowledge_no_training_launch",
        "acknowledge_no_ui_exposure",
        "acknowledge_no_request_adapter_registration",
        "acknowledge_no_request_fields_emitted",
        "acknowledge_no_generation_request_patch",
        "acknowledge_no_config_adapter_patch",
        "acknowledge_no_runtime_resolver_patch",
        "acknowledge_no_default_or_auto_rollout",
        "acknowledge_scaffold_evidence_replayable",
        "acknowledge_later_runtime_wiring_contract_required",
        "acknowledge_manual_review_only",
    ):
        review[field] = True
    return review


def _safe_row_flags() -> dict[str, Any]:
    return {
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "runs_dispatched": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "ui_exposure_allowed": False,
        "product_ui_exposure_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "runtime_adapter_registered": False,
        "adapter_integration_allowed": False,
        "adapter_wiring_allowed": False,
        "generation_request_patch_allowed": False,
        "config_adapter_patch_allowed": False,
        "runtime_resolver_patch_allowed": False,
        "rollout_authorization_allowed": False,
        "default_behavior_changed": False,
        "blocked_reasons": [],
        "promotion_blockers": [],
    }


def _contract_claim(field: str) -> dict[str, Any]:
    row = {"contract_id": "native", "scaffold_enabled": False, **_safe_row_flags(), field: True}
    return _gate(_p41_ready(), scaffold={**_scaffold_evidence(), "adapter_contract_inventory": [row]})


def _future_field_claim(field: str) -> dict[str, Any]:
    row = {"field_id": "future_field", "field_enabled": False, field: True}
    return _gate(_p41_ready(), scaffold={**_scaffold_evidence(), "future_request_field_inventory": [row]})


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p41_ready(), review={**_scaffold_review(), field: True})


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
    assert report["post_p42_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["adapter_integration_contract_scaffold_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["adapter_integration_contract_scaffold_ready"] is False, report
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
        "scaffold_ready": bool(report.get("adapter_integration_contract_scaffold_ready", False)),
        "review_signed": bool(report.get("adapter_integration_contract_review_signed", False)),
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
