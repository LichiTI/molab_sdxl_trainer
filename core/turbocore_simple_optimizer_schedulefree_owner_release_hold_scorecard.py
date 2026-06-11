"""Owner/release approval hold for built-in simple schedule-free canaries."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_simple_optimizer_schedulefree_dispatch_integration_review_scorecard import (
    EXPECTED_OPTIMIZERS,
    FALLBACK_BACKEND,
    build_simple_optimizer_schedulefree_dispatch_integration_review_scorecard,
)


HOLD_KIND = "simple_formula_schedulefree_owner_release_hold_v0"


def build_simple_optimizer_schedulefree_owner_release_hold_scorecard(
    *,
    dispatch_review_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record the approval hold after dispatch review without enabling dispatch."""

    review_report = _as_dict(
        dispatch_review_report or build_simple_optimizer_schedulefree_dispatch_integration_review_scorecard()
    )
    hold = _hold_manifest(review_report)
    validations = _validations(review_report, hold)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    approval_blockers = [
        "simple_schedulefree_owner_approval_missing",
        "simple_schedulefree_release_approval_missing",
        "simple_schedulefree_product_dispatch_not_approved",
    ]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_schedulefree_owner_release_hold_scorecard_v0",
        "gate": "simple_formula_schedulefree_owner_release_hold",
        "ok": ready,
        "promotion_ready": False,
        "owner_release_hold_ready": ready,
        "dispatch_integration_review": review_report.get("dispatch_integration_review") is True,
        "manual_review_required": True,
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "canary_auto_blocked_until_review": True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "experimental_only": True,
        "report_only": True,
        "product_native_dispatch_ready": False,
        "product_native_ready_count": 0,
        "hold_kind": HOLD_KIND,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "hold_manifest": hold,
        "dispatch_review_summary": _as_dict(review_report.get("summary")),
        "validations": validations,
        "summary": {
            "owner_release_hold_ready": ready,
            "dispatch_integration_review": review_report.get("dispatch_integration_review") is True,
            "manual_review_required": True,
            "owner_approval_recorded": False,
            "release_approval_recorded": False,
            "optimizer_count": len(_optimizer_types(review_report)),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(blockers + approval_blockers),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "record explicit owner and release approval before simple schedule-free product dispatch wiring"
            if ready
            else "fix simple schedule-free owner/release hold blockers"
        ),
        "notes": [
            "This gate is a hold state, not a product promotion.",
            "Dispatch review evidence is packaged but owner and release approval are intentionally absent.",
            "Request, schema, UI, and default training dispatch remain unchanged.",
        ],
    }


def _hold_manifest(review_report: Mapping[str, Any]) -> dict[str, Any]:
    optimizer_types = _optimizer_types(review_report)
    return {
        "schema_version": 1,
        "hold_kind": HOLD_KIND,
        "optimizer_family": "simple_formula_schedulefree",
        "optimizer_types": optimizer_types,
        "dispatch_review_gate": str(review_report.get("gate") or ""),
        "dispatch_review_scorecard": str(review_report.get("scorecard") or ""),
        "approval_state": "pending_owner_and_release_approval",
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_approval": ["canary", "auto"],
        "required_approval_artifacts": [
            "owner_approval_record",
            "release_risk_acceptance",
            "runtime_dispatch_wiring_review",
            "train_eval_mode_rollback_acknowledgement",
        ],
        "frozen_product_boundaries": {
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ui_exposure_allowed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
    }


def _validations(review_report: Mapping[str, Any], hold: Mapping[str, Any]) -> list[dict[str, Any]]:
    frozen = _as_dict(hold.get("frozen_product_boundaries"))
    optimizer_types = set(_optimizer_types(review_report))
    return [
        _validation(
            "dispatch_review_ready",
            review_report.get("dispatch_integration_review") is True,
            "simple_schedulefree_dispatch_review_missing",
        ),
        _validation(
            "optimizer_set_complete",
            optimizer_types == EXPECTED_OPTIMIZERS,
            "simple_schedulefree_owner_release_optimizer_set_incomplete",
        ),
        _validation(
            "approval_not_recorded",
            hold.get("approval_state") == "pending_owner_and_release_approval"
            and hold.get("owner_approval_recorded") is False
            and hold.get("release_approval_recorded") is False,
            "simple_schedulefree_owner_release_unexpected_approval",
        ),
        _validation(
            "canary_auto_blocked",
            hold.get("allowed_initial_modes") == ["off", "observe"]
            and hold.get("blocked_modes_until_approval") == ["canary", "auto"],
            "simple_schedulefree_owner_release_allows_dispatch",
        ),
        _validation(
            "product_boundaries_frozen",
            all(frozen.get(field) is False for field in frozen),
            "simple_schedulefree_owner_release_changed_product_boundaries",
        ),
        _validation(
            "review_kept_default_off",
            review_report.get("runtime_dispatch_ready") is False
            and review_report.get("native_dispatch_allowed") is False
            and review_report.get("training_path_enabled") is False
            and review_report.get("request_fields_emitted") is False
            and review_report.get("schema_exposure_allowed") is False
            and review_report.get("ui_exposure_allowed") is False,
            "simple_schedulefree_owner_release_review_enabled_dispatch",
        ),
    ]


def _optimizer_types(report: Mapping[str, Any]) -> list[str]:
    review = _as_dict(report.get("review_package"))
    values = review.get("optimizer_types")
    if not isinstance(values, list):
        return []
    return sorted(str(value) for value in values if str(value))


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["HOLD_KIND", "build_simple_optimizer_schedulefree_owner_release_hold_scorecard"]
