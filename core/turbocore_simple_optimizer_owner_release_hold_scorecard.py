"""Owner/release approval hold for fp32 simple optimizer product canaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.turbocore_simple_optimizer_product_training_canary_scorecard import (
    TARGET_OPTIMIZERS,
    build_simple_optimizer_product_training_canary_scorecard,
)


HOLD_KIND = "simple_formula_owner_release_hold_v0"


def build_simple_optimizer_owner_release_hold_scorecard(
    *,
    product_training_canary_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Record explicit owner/release hold state without enabling dispatch."""

    product_canary = _as_dict(
        product_training_canary_report
        or build_simple_optimizer_product_training_canary_scorecard(workspace_root=workspace_root)
    )
    hold = _hold_manifest(product_canary)
    validations = _validations(product_canary, hold)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    approval_blockers = [
        "simple_formula_owner_approval_missing",
        "simple_formula_release_approval_missing",
        "simple_formula_product_dispatch_not_approved",
    ]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_owner_release_hold_scorecard_v0",
        "gate": "simple_formula_owner_release_hold",
        "ok": ready,
        "promotion_ready": False,
        "owner_release_hold_ready": ready,
        "representative_product_training_canary_ready": product_canary.get(
            "representative_product_training_canary_ready"
        ) is True,
        "manual_review_required": True,
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "product_native_dispatch_ready": False,
        "product_native_ready_count": 0,
        "hold_kind": HOLD_KIND,
        "target_optimizer_types": list(TARGET_OPTIMIZERS),
        "hold_manifest": hold,
        "product_training_canary_summary": _as_dict(product_canary.get("summary")),
        "validations": validations,
        "summary": {
            "owner_release_hold_ready": ready,
            "representative_product_training_canary_ready": product_canary.get(
                "representative_product_training_canary_ready"
            ) is True,
            "optimizer_count": len(_optimizer_types(product_canary)),
            "manual_review_required": True,
            "owner_approval_recorded": False,
            "release_approval_recorded": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(blockers + approval_blockers),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "record explicit fp32 simple optimizer owner and release approval before product dispatch wiring"
            if ready
            else "fix fp32 simple owner/release hold blockers"
        ),
        "notes": [
            "This gate is an approval hold after representative product-training canary evidence.",
            "Owner and release approval are intentionally not recorded by this scorecard.",
            "Request, schema, UI, runtime dispatch, and default training behavior remain unchanged.",
        ],
    }


def _hold_manifest(product_canary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "hold_kind": HOLD_KIND,
        "optimizer_family": "simple_formula",
        "optimizer_types": _optimizer_types(product_canary),
        "product_training_canary_gate": str(product_canary.get("gate") or ""),
        "product_training_canary_scorecard": str(product_canary.get("scorecard") or ""),
        "approval_state": "pending_owner_and_release_approval",
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_approval": ["canary", "auto"],
        "required_approval_artifacts": [
            "owner_approval_record",
            "release_risk_acceptance",
            "runtime_dispatch_wiring_review",
            "rollback_runbook_acknowledgement",
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


def _validations(product_canary: Mapping[str, Any], hold: Mapping[str, Any]) -> list[dict[str, Any]]:
    frozen = _as_dict(hold.get("frozen_product_boundaries"))
    return [
        _validation(
            "product_training_canary_ready",
            product_canary.get("representative_product_training_canary_ready") is True,
            "simple_formula_product_training_canary_missing",
        ),
        _validation(
            "optimizer_set_complete",
            set(_optimizer_types(product_canary)) == set(TARGET_OPTIMIZERS),
            "simple_formula_owner_release_optimizer_set_incomplete",
        ),
        _validation(
            "approval_not_recorded",
            hold.get("approval_state") == "pending_owner_and_release_approval"
            and hold.get("owner_approval_recorded") is False
            and hold.get("release_approval_recorded") is False,
            "simple_formula_owner_release_unexpected_approval",
        ),
        _validation(
            "canary_auto_blocked",
            hold.get("allowed_initial_modes") == ["off", "observe"]
            and hold.get("blocked_modes_until_approval") == ["canary", "auto"],
            "simple_formula_owner_release_allows_dispatch",
        ),
        _validation(
            "product_boundaries_frozen",
            all(frozen.get(field) is False for field in frozen),
            "simple_formula_owner_release_changed_product_boundaries",
        ),
        _validation(
            "source_kept_default_off",
            product_canary.get("runtime_dispatch_ready") is False
            and product_canary.get("native_dispatch_allowed") is False
            and product_canary.get("training_path_enabled") is False
            and product_canary.get("request_fields_emitted") is False
            and product_canary.get("schema_exposure_allowed") is False
            and product_canary.get("ui_exposure_allowed") is False,
            "simple_formula_owner_release_source_enabled_dispatch",
        ),
    ]


def _optimizer_types(report: Mapping[str, Any]) -> list[str]:
    values = report.get("target_optimizer_types")
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


__all__ = ["HOLD_KIND", "build_simple_optimizer_owner_release_hold_scorecard"]
