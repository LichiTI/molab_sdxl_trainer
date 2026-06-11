"""Owner/release hold for selected simple-formula plugin optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_simple_formula_dispatch_integration_review_scorecard import (
    build_plugin_simple_formula_dispatch_integration_review_scorecard,
)


HOLD_KIND = "plugin_simple_formula_owner_release_hold_v0"


def build_plugin_simple_formula_owner_release_hold_scorecard(
    *,
    dispatch_review_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Record a default-off hold after selected simple-formula dispatch review evidence."""

    review = _as_dict(dispatch_review_report or build_plugin_simple_formula_dispatch_integration_review_scorecard())
    summary = _as_dict(review.get("summary"))
    hold = _hold_manifest(review)
    validations = _validations(review, hold)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_simple_formula_owner_release_hold_scorecard_v0",
        "gate": "plugin_simple_formula_owner_release_hold",
        "ok": ready,
        "promotion_ready": False,
        "owner_release_hold_ready": ready,
        "dispatch_review_gate_ready": review.get("review_gate_ready") is True,
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
        "fallback_backend": "pytorch_optimizer_python",
        "fallback_backend_authoritative": True,
        "hold_manifest": hold,
        "dispatch_review_summary": summary,
        "validations": validations,
        "summary": {
            "owner_release_hold_ready": ready,
            "dispatch_review_gate_ready": review.get("review_gate_ready") is True,
            "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "owner_approval_recorded": False,
            "release_approval_recorded": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "plugin_simple_formula_owner_approval_missing",
                "plugin_simple_formula_release_approval_missing",
                "plugin_simple_formula_product_dispatch_not_approved",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add simple-formula request/schema/UI non-exposure guard before product dispatch wiring"
            if ready
            else "fix simple-formula owner/release hold blockers"
        ),
        "notes": [
            "This gate records a hold state only; it does not approve native dispatch.",
            "Selected simple-formula evidence remains report-only until owner/release approval.",
            "Request, schema, UI, runtime dispatch, and training behavior remain unchanged.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _hold_manifest(review: Mapping[str, Any]) -> dict[str, Any]:
    package = _as_dict(review.get("review_package"))
    selected = [str(value) for value in package.get("selected_optimizer_names", []) if str(value)]
    return {
        "schema_version": 1,
        "hold_kind": HOLD_KIND,
        "optimizer_family": "simple_formula",
        "selected_optimizer_names": selected,
        "dispatch_review_gate": str(review.get("gate") or ""),
        "dispatch_review_scorecard": str(review.get("scorecard") or ""),
        "approval_state": "pending_owner_and_release_approval",
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_approval": ["canary", "auto"],
        "required_approval_artifacts": [
            "owner_approval_record",
            "release_risk_acceptance",
            "simple_formula_dispatch_wiring_review",
            "fallback_authority_rollback_acknowledgement",
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


def _validations(review: Mapping[str, Any], hold: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = _as_dict(review.get("summary"))
    frozen = _as_dict(hold.get("frozen_product_boundaries"))
    optimizer_count = int(summary.get("optimizer_count", 0) or 0)
    return [
        _validation(
            "dispatch_review_gate_ready",
            review.get("review_gate_ready") is True,
            "plugin_simple_formula_dispatch_review_missing",
        ),
        _validation(
            "optimizer_set_complete",
            len(hold.get("selected_optimizer_names", [])) == optimizer_count == 18,
            "plugin_simple_formula_optimizer_set_incomplete",
        ),
        _validation(
            "approval_not_recorded",
            hold.get("approval_state") == "pending_owner_and_release_approval"
            and hold.get("owner_approval_recorded") is False
            and hold.get("release_approval_recorded") is False,
            "plugin_simple_formula_unexpected_approval",
        ),
        _validation(
            "canary_auto_blocked",
            hold.get("allowed_initial_modes") == ["off", "observe"]
            and hold.get("blocked_modes_until_approval") == ["canary", "auto"],
            "plugin_simple_formula_hold_allows_dispatch",
        ),
        _validation(
            "product_boundaries_frozen",
            all(frozen.get(field) is False for field in frozen),
            "plugin_simple_formula_changed_product_boundaries",
        ),
        _validation(
            "review_kept_default_off",
            review.get("runtime_dispatch_ready") is False
            and review.get("native_dispatch_allowed") is False
            and review.get("training_path_enabled") is False
            and int(summary.get("runtime_dispatch_ready_count", 0) or 0) == 0
            and int(summary.get("native_dispatch_allowed_count", 0) or 0) == 0
            and int(summary.get("training_path_enabled_count", 0) or 0) == 0
            and int(summary.get("product_native_ready_count", 0) or 0) == 0,
            "plugin_simple_formula_review_enabled_dispatch",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_simple_formula_owner_release_hold_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_plugin_simple_formula_owner_release_hold_scorecard"]
