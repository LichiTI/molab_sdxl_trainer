"""Smoke checks for V5-P43 runtime wiring contract boundary."""

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

from core.turbocore_v5_runtime_wiring_contract_boundary import (  # noqa: E402
    build_v5_runtime_wiring_contract_boundary,
)
from lulynx_trainer.turbocore_v5_explicit_adapter_integration_contract_scaffold_smoke import (  # noqa: E402
    _gate as _p42_gate,
    _p41_ready,
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
    "runtime_wiring_allowed",
    "runtime_adapter_wiring_allowed",
    "generation_request_patch_allowed",
    "config_adapter_patch_allowed",
    "runtime_resolver_patch_allowed",
    "execution_resolver_patch_allowed",
    "training_manager_patch_allowed",
    "rollout_authorization_allowed",
)
READY_DECISION = "runtime_wiring_contract_boundary_recorded_default_off"
BLOCKED_DECISION = "runtime_wiring_contract_boundary_blocked_default_off"
HOLD_DECISION = "runtime_wiring_contract_boundary_hold_for_signed_review_default_off"
REJECTED_DECISION = "runtime_wiring_contract_boundary_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p42_ready = _p42_ready()
    ready = _gate(p42_ready)
    assert ready["ok"] is True, ready
    assert ready["runtime_wiring_contract_boundary_ready"] is True, ready
    assert ready["runtime_wiring_contract_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p42_ready, review=None)
    _assert_hold(missing_review, "review", "missing")

    rejected_review = _gate(p42_ready, review=_runtime_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p42_missing = _gate(None)
    _assert_blocked(p42_missing, "p42", "missing")

    p42_not_ready = _gate({**p42_ready, "ok": False, "adapter_integration_contract_scaffold_ready": False})
    _assert_blocked(p42_not_ready, "p42", "not_ready")

    p42_decision_mismatch = _gate({**p42_ready, "decision": "wrong"})
    _assert_blocked(p42_decision_mismatch, "p42", "not_ready")

    p42_runtime_adapter = _gate({**p42_ready, "runtime_adapter_registered": True})
    _assert_blocked(p42_runtime_adapter, "runtime_adapter_registered")

    p42_runtime_patch = _gate({**p42_ready, "runtime_resolver_patch_allowed": True})
    _assert_blocked(p42_runtime_patch, "runtime_resolver_patch_allowed")

    p42_post_fields = _gate({**p42_ready, "post_p42_request_fields": {"bad": True}})
    _assert_blocked(p42_post_fields, "post_p42_request_fields")

    missing_source = _gate(p42_ready, evidence=_without(_runtime_evidence(), "source"))
    _assert_blocked(missing_source, "source_missing")

    missing_digest = _gate(p42_ready, evidence=_without_many(_runtime_evidence(), "sha256", "artifact_digest"))
    _assert_blocked(missing_digest, "digest_missing")

    not_report_only = _gate(p42_ready, evidence={**_runtime_evidence(), "report_only": False})
    _assert_blocked(not_report_only, "report_only")

    not_contract_only = _gate(p42_ready, evidence={**_runtime_evidence(), "contract_only": False})
    _assert_blocked(not_contract_only, "contract_only")

    missing_later_activation = _gate(
        p42_ready,
        evidence={**_runtime_evidence(), "requires_later_runtime_activation_contract": False},
    )
    _assert_blocked(missing_later_activation, "later_runtime_activation_contract")

    missing_section = _gate(p42_ready, evidence={**_runtime_evidence(), "available_sections": ["rollback_policy"]})
    _assert_blocked(missing_section, "section_missing")

    missing_adapter_inventory = _gate(p42_ready, evidence=_without(_runtime_evidence(), "runtime_adapter_inventory"))
    _assert_blocked(missing_adapter_inventory, "runtime_adapter_inventory")

    missing_wiring_inventory = _gate(p42_ready, evidence=_without(_runtime_evidence(), "runtime_wiring_plan_inventory"))
    _assert_blocked(missing_wiring_inventory, "runtime_wiring_plan_inventory")

    adapter_registered = _adapter_claim("runtime_adapter_registered")
    _assert_blocked(adapter_registered, "runtime_adapter_registered")

    adapter_wiring = _adapter_claim("runtime_adapter_wiring_allowed")
    _assert_blocked(adapter_wiring, "runtime_adapter_wiring_allowed")

    wiring_enabled = _wiring_claim("wiring_enabled")
    _assert_blocked(wiring_enabled, "wiring_enabled")

    request_fields = _wiring_claim("request_fields_emitted")
    _assert_blocked(request_fields, "request_fields_emitted")

    generation_patch = _wiring_claim("generation_request_patch_applied")
    _assert_blocked(generation_patch, "generation_request_patch")

    config_patch = _wiring_claim("config_adapter_patch_applied")
    _assert_blocked(config_patch, "config_adapter_patch")

    runtime_patch = _wiring_claim("runtime_resolver_patch_applied")
    _assert_blocked(runtime_patch, "runtime_resolver_patch")

    execution_patch = _wiring_claim("execution_resolver_patch_applied")
    _assert_blocked(execution_patch, "execution_resolver_patch")

    review_missing_reviewer = _gate(p42_ready, review={**_runtime_review(), "reviewer": ""})
    _assert_blocked(review_missing_reviewer, "reviewer")

    review_missing_reviewed_at = _gate(p42_ready, review={**_runtime_review(), "reviewed_at": ""})
    _assert_blocked(review_missing_reviewed_at, "reviewed_at")

    review_scope_mismatch = _gate(p42_ready, review={**_runtime_review(), "requested_scope": "wrong"})
    _assert_blocked(review_scope_mismatch, "scope")

    review_missing_ack = _gate(p42_ready, review={**_runtime_review(), "acknowledge_no_execution_resolver_patch": False})
    _assert_blocked(review_missing_ack, "ack_missing", "execution_resolver")

    review_approve_runtime_adapter = _unsafe_review("approve_runtime_adapter_registered")
    _assert_blocked(review_approve_runtime_adapter, "approve_runtime_adapter_registered")

    review_approve_runtime_wiring = _unsafe_review("approve_runtime_wiring_allowed")
    _assert_blocked(review_approve_runtime_wiring, "approve_runtime_wiring_allowed")

    review_approve_fields = _unsafe_review("approve_request_fields_emitted")
    _assert_blocked(review_approve_fields, "approve_request_fields_emitted")

    review_approve_execution = _unsafe_review("approve_execution_resolver_patch_allowed")
    _assert_blocked(review_approve_execution, "approve_execution_resolver_patch_allowed")

    review_approve_launch = _unsafe_review("approve_training_launch_allowed")
    _assert_blocked(review_approve_launch, "approve_training_launch_allowed")

    failure_history = _gate(
        p42_ready,
        failure_history=[{"reason": "runtime_wiring_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")

    rollback_history = _gate(p42_ready, rollback_history=[{"kind": "runtime_wiring_rollback", "rollback_required": True}])
    _assert_blocked(rollback_history, "rollback_history")

    closed_failure = _gate(
        p42_ready,
        failure_history=[{"reason": "closed_runtime_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p43_runtime_wiring_contract_boundary_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p42_missing": _summary(p42_missing),
        "p42_not_ready": _summary(p42_not_ready),
        "p42_decision_mismatch": _summary(p42_decision_mismatch),
        "p42_runtime_adapter": _summary(p42_runtime_adapter),
        "p42_runtime_patch": _summary(p42_runtime_patch),
        "p42_post_fields": _summary(p42_post_fields),
        "missing_source": _summary(missing_source),
        "missing_digest": _summary(missing_digest),
        "missing_adapter_inventory": _summary(missing_adapter_inventory),
        "missing_wiring_inventory": _summary(missing_wiring_inventory),
        "runtime_patch": _summary(runtime_patch),
        "execution_patch": _summary(execution_patch),
        "review_approve_runtime_wiring": _summary(review_approve_runtime_wiring),
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p42: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _runtime_review() if review is ... else review
    return build_v5_runtime_wiring_contract_boundary(
        p42_adapter_integration_contract_scaffold=p42,
        runtime_wiring_contract_evidence=_runtime_evidence() if evidence is None else evidence,
        runtime_wiring_contract_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p42_ready() -> dict[str, Any]:
    return _p42_gate(_p41_ready())


def _runtime_evidence() -> dict[str, Any]:
    sections = [
        "p42_scaffold_reference",
        "runtime_adapter_inventory",
        "runtime_wiring_plan_inventory",
        "generation_request_boundary",
        "config_adapter_boundary",
        "runtime_resolver_boundary",
        "execution_resolver_boundary",
        "no_runtime_adapter_registration_boundary",
        "no_request_fields_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    return {
        "evidence_id": "runtime_wiring_contract_boundary_v0",
        "evidence_version": "v0",
        "ok": True,
        "runtime_wiring_contract_boundary_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_runtime_activation_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "runtime_adapter_inventory": [
            {"adapter_id": "native_update_runtime_adapter", **_safe_row_flags()},
            {"adapter_id": "lora_fused_runtime_adapter", **_safe_row_flags()},
        ],
        "runtime_wiring_plan_inventory": [
            {"plan_id": "native_update_runtime_wiring", **_safe_row_flags()},
            {"plan_id": "lora_fused_runtime_wiring", **_safe_row_flags()},
        ],
        "sha256": "sha256:p43:runtime-wiring:ready",
        "artifact_digest": "sha256:p43:runtime-wiring:ready",
        "source": "temp/turbocore_v5_p43_runtime_wiring.json",
        **_safe_row_flags(),
    }


def _runtime_review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "runtime_wiring_contract_boundary",
        "approve_runtime_wiring_contract_boundary": approve,
    }
    for field in (
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
        "runtime_wiring_allowed",
        "runtime_adapter_wiring_allowed",
        "generation_request_patch_allowed",
        "config_adapter_patch_allowed",
        "runtime_resolver_patch_allowed",
        "execution_resolver_patch_allowed",
        "training_manager_patch_allowed",
        "rollout_authorization_allowed",
    ):
        review[f"approve_{field}"] = False
    for field in (
        "acknowledge_p42_scaffold_recorded",
        "acknowledge_default_off_boundary",
        "acknowledge_no_training_launch",
        "acknowledge_no_ui_exposure",
        "acknowledge_no_runtime_adapter_registration",
        "acknowledge_no_request_fields_emitted",
        "acknowledge_no_generation_request_patch",
        "acknowledge_no_config_adapter_patch",
        "acknowledge_no_runtime_resolver_patch",
        "acknowledge_no_execution_resolver_patch",
        "acknowledge_no_default_or_auto_rollout",
        "acknowledge_runtime_wiring_evidence_replayable",
        "acknowledge_later_runtime_activation_contract_required",
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
        "runtime_wiring_allowed": False,
        "runtime_adapter_wiring_allowed": False,
        "generation_request_patch_allowed": False,
        "config_adapter_patch_allowed": False,
        "runtime_resolver_patch_allowed": False,
        "execution_resolver_patch_allowed": False,
        "training_manager_patch_allowed": False,
        "rollout_authorization_allowed": False,
        "default_behavior_changed": False,
        "blocked_reasons": [],
        "promotion_blockers": [],
    }


def _adapter_claim(field: str) -> dict[str, Any]:
    row = {"adapter_id": "native", **_safe_row_flags(), field: True}
    return _gate(_p42_ready(), evidence={**_runtime_evidence(), "runtime_adapter_inventory": [row]})


def _wiring_claim(field: str) -> dict[str, Any]:
    row = {"plan_id": "native", **_safe_row_flags(), field: True}
    return _gate(_p42_ready(), evidence={**_runtime_evidence(), "runtime_wiring_plan_inventory": [row]})


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p42_ready(), review={**_runtime_review(), field: True})


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
    assert report["post_p43_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["runtime_wiring_contract_boundary_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["runtime_wiring_contract_boundary_ready"] is False, report
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
        "boundary_ready": bool(report.get("runtime_wiring_contract_boundary_ready", False)),
        "review_signed": bool(report.get("runtime_wiring_contract_review_signed", False)),
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
