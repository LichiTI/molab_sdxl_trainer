"""Owner/release approval hold for built-in Muon model-shape-aware routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_muon_dispatch_integration_review_scorecard import (
    build_muon_dispatch_integration_review_scorecard,
)
from core.turbocore_muon_model_shape_aware_family_batch_scorecard import (
    TARGET_OPTIMIZER,
    build_muon_model_shape_aware_family_batch_scorecard,
)


HOLD_KIND = "muon_model_shape_aware_owner_release_hold_v0"


def build_muon_owner_release_hold_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    rollout_policy_report: Mapping[str, Any] | None = None,
    dispatch_review_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Record the Muon approval hold after dispatch review without enabling dispatch."""

    batch = _as_dict(family_batch_report or build_muon_model_shape_aware_family_batch_scorecard())
    review = _as_dict(dispatch_review_report or build_muon_dispatch_integration_review_scorecard())
    rollout = _as_dict(rollout_policy_report or {"canary_rollout_policy_ready": _review_ready(review)})
    hold = _hold_manifest(batch, rollout, review)
    validations = _validations(batch, rollout, review, hold)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    approval_blockers = [
        "muon_owner_approval_missing",
        "muon_release_approval_missing",
        "muon_product_dispatch_not_approved",
    ]
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_muon_owner_release_hold_scorecard_v0",
        "gate": "muon_model_shape_aware_owner_release_hold",
        "ok": ready,
        "promotion_ready": False,
        "owner_release_hold_ready": ready,
        "family_batch_ready": batch.get("muon_model_shape_aware_family_batch_ready") is True,
        "canary_rollout_policy_ready": _rollout_ready(rollout),
        "dispatch_integration_review": _review_ready(review),
        "manual_review_required": True,
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "canary_auto_blocked_until_review": True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
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
        "fallback_backend": "existing_builtin_muon_optimizer",
        "fallback_backend_authoritative": True,
        "hold_manifest": hold,
        "family_batch_summary": _as_dict(batch.get("summary")),
        "rollout_policy_summary": _as_dict(rollout.get("summary")),
        "dispatch_review_summary": _as_dict(review.get("summary")),
        "validations": validations,
        "summary": {
            "owner_release_hold_ready": ready,
            "family_batch_ready": batch.get("muon_model_shape_aware_family_batch_ready") is True,
            "canary_rollout_policy_ready": _rollout_ready(rollout),
            "dispatch_integration_review": _review_ready(review),
            "manual_review_required": True,
            "owner_approval_recorded": False,
            "release_approval_recorded": False,
            "optimizer_count": 1 if TARGET_OPTIMIZER in hold.get("optimizer_types", []) else 0,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "native_kernel_ready": False,
            "training_path_enabled": False,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(blockers + approval_blockers),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "record explicit owner and release approval before Muon product dispatch wiring"
            if ready
            else "fix Muon owner/release hold blockers"
        ),
        "notes": [
            "This gate is a hold state, not a product promotion.",
            "Muon dispatch review evidence is packaged but owner and release approval are intentionally absent.",
            "Request, schema, UI, native dispatch, and default training dispatch remain unchanged.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report

def _hold_manifest(
    batch: Mapping[str, Any],
    rollout: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "hold_kind": HOLD_KIND,
        "optimizer_family": "built_in_muon_model_shape_aware",
        "optimizer_types": [TARGET_OPTIMIZER],
        "family_batch_gate": str(batch.get("gate") or ""),
        "family_batch_scorecard": str(batch.get("scorecard") or ""),
        "rollout_policy_gate": str(rollout.get("gate") or ""),
        "rollout_policy_scorecard": str(rollout.get("scorecard") or ""),
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
            "runtime_dispatch_wiring_review",
            "muon_param_group_shape_rollback_acknowledgement",
        ],
        "frozen_product_boundaries": {
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ui_exposure_allowed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "native_kernel_ready": False,
            "training_path_enabled": False,
        },
    }


def _validations(
    batch: Mapping[str, Any],
    rollout: Mapping[str, Any],
    review: Mapping[str, Any],
    hold: Mapping[str, Any],
) -> list[dict[str, Any]]:
    frozen = _as_dict(hold.get("frozen_product_boundaries"))
    summary = _as_dict(batch.get("summary"))
    return [
        _validation(
            "family_batch_ready",
            batch.get("muon_model_shape_aware_family_batch_ready") is True,
            "muon_model_shape_aware_family_batch_missing",
        ),
        _validation(
            "optimizer_set_complete",
            hold.get("optimizer_types") == [TARGET_OPTIMIZER],
            "muon_owner_release_optimizer_set_incomplete",
        ),
        _validation(
            "dispatch_review_ready",
            _review_ready(review) or int(summary.get("dispatch_integration_review_ready_count", 0) or 0) == 1,
            "muon_dispatch_review_incomplete",
        ),
        _validation(
            "canary_rollout_policy_ready",
            _rollout_ready(rollout),
            "muon_canary_rollout_policy_missing",
        ),
        _validation(
            "approval_not_recorded",
            hold.get("approval_state") == "pending_owner_and_release_approval"
            and hold.get("owner_approval_recorded") is False
            and hold.get("release_approval_recorded") is False,
            "muon_owner_release_unexpected_approval",
        ),
        _validation(
            "canary_auto_blocked",
            hold.get("allowed_initial_modes") == ["off", "observe"]
            and hold.get("blocked_modes_until_approval") == ["canary", "auto"],
            "muon_owner_release_allows_dispatch",
        ),
        _validation(
            "product_boundaries_frozen",
            all(frozen.get(field) is False for field in frozen),
            "muon_owner_release_changed_product_boundaries",
        ),
        _validation(
            "batch_kept_default_off",
            batch.get("runtime_dispatch_ready") is False
            and batch.get("native_dispatch_allowed") is False
            and batch.get("native_kernel_ready") is False
            and batch.get("training_path_enabled") is False
            and int(summary.get("product_native_ready_count", 0) or 0) == 0,
            "muon_owner_release_batch_enabled_dispatch",
        ),
        _validation(
            "rollout_kept_default_off",
            not rollout
            or (
                rollout.get("runtime_dispatch_ready") is False
                and rollout.get("native_dispatch_allowed") is False
                and rollout.get("training_path_enabled") is False
                and rollout.get("canary_auto_enabled") is False
                and int(_as_dict(rollout.get("summary")).get("product_native_ready_count", 0) or 0) == 0
            ),
            "muon_owner_release_rollout_enabled_dispatch",
        ),
        _validation(
            "dispatch_review_kept_default_off",
            not review
            or (
                review.get("runtime_dispatch_ready") is False
                and review.get("native_dispatch_allowed") is False
                and review.get("training_path_enabled") is False
                and review.get("request_fields_emitted") is False
                and review.get("schema_exposure_allowed") is False
                and review.get("ui_exposure_allowed") is False
            ),
            "muon_owner_release_review_enabled_dispatch",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_muon_owner_release_hold_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rollout_ready(rollout: Mapping[str, Any]) -> bool:
    return not rollout or rollout.get("canary_rollout_policy_ready") is True


def _review_ready(review: Mapping[str, Any]) -> bool:
    return bool(review) and review.get("dispatch_integration_review") is True


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["HOLD_KIND", "build_muon_owner_release_hold_scorecard"]
