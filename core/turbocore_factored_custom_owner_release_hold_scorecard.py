"""Owner/release approval hold for built-in factored/custom canaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_factored_custom_optimizer_family_batch_scorecard import (
    build_factored_custom_optimizer_family_batch_scorecard,
)


EXPECTED_OPTIMIZERS = {"adafactor", "Automagic++", "AnimaFactoredAdamW"}
HOLD_KIND = "factored_custom_owner_release_hold_v0"


def build_factored_custom_owner_release_hold_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Record the approval hold after dispatch review without enabling dispatch."""

    batch = _as_dict(family_batch_report or build_factored_custom_optimizer_family_batch_scorecard())
    hold = _hold_manifest(batch)
    validations = _validations(batch, hold)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    approval_blockers = [
        "factored_custom_owner_approval_missing",
        "factored_custom_release_approval_missing",
        "factored_custom_product_dispatch_not_approved",
    ]
    optimizer_types = _optimizer_types(batch)
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_factored_custom_owner_release_hold_scorecard_v0",
        "gate": "factored_custom_owner_release_hold",
        "ok": ready,
        "promotion_ready": False,
        "owner_release_hold_ready": ready,
        "family_batch_ready": batch.get("factored_custom_family_batch_ready") is True,
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
        "fallback_backend": "python_factored_custom",
        "fallback_backend_authoritative": True,
        "hold_manifest": hold,
        "family_batch_summary": _as_dict(batch.get("summary")),
        "validations": validations,
        "summary": {
            "owner_release_hold_ready": ready,
            "family_batch_ready": batch.get("factored_custom_family_batch_ready") is True,
            "manual_review_required": True,
            "owner_approval_recorded": False,
            "release_approval_recorded": False,
            "optimizer_count": len(optimizer_types),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(blockers + approval_blockers),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "record explicit owner and release approval before factored/custom product dispatch wiring"
            if ready
            else "fix factored/custom owner/release hold blockers"
        ),
        "notes": [
            "This gate is a hold state, not a product promotion.",
            "Dispatch review evidence is packaged but owner and release approval are intentionally absent.",
            "Request, schema, UI, and default training dispatch remain unchanged.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _hold_manifest(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "hold_kind": HOLD_KIND,
        "optimizer_family": "built_in_factored_custom",
        "optimizer_types": _optimizer_types(batch),
        "family_batch_gate": str(batch.get("gate") or ""),
        "family_batch_scorecard": str(batch.get("scorecard") or ""),
        "approval_state": "pending_owner_and_release_approval",
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_approval": ["canary", "auto"],
        "required_approval_artifacts": [
            "owner_approval_record",
            "release_risk_acceptance",
            "runtime_dispatch_wiring_review",
            "factored_custom_state_layout_rollback_acknowledgement",
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


def _validations(batch: Mapping[str, Any], hold: Mapping[str, Any]) -> list[dict[str, Any]]:
    frozen = _as_dict(hold.get("frozen_product_boundaries"))
    optimizer_types = set(_optimizer_types(batch))
    summary = _as_dict(batch.get("summary"))
    return [
        _validation(
            "family_batch_ready",
            batch.get("factored_custom_family_batch_ready") is True,
            "factored_custom_family_batch_missing",
        ),
        _validation(
            "optimizer_set_complete",
            optimizer_types == EXPECTED_OPTIMIZERS,
            "factored_custom_owner_release_optimizer_set_incomplete",
        ),
        _validation(
            "dispatch_review_ready_for_all",
            int(summary.get("dispatch_integration_review_ready_count", 0) or 0) == len(EXPECTED_OPTIMIZERS),
            "factored_custom_dispatch_review_incomplete",
        ),
        _validation(
            "approval_not_recorded",
            hold.get("approval_state") == "pending_owner_and_release_approval"
            and hold.get("owner_approval_recorded") is False
            and hold.get("release_approval_recorded") is False,
            "factored_custom_owner_release_unexpected_approval",
        ),
        _validation(
            "canary_auto_blocked",
            hold.get("allowed_initial_modes") == ["off", "observe"]
            and hold.get("blocked_modes_until_approval") == ["canary", "auto"],
            "factored_custom_owner_release_allows_dispatch",
        ),
        _validation(
            "product_boundaries_frozen",
            all(frozen.get(field) is False for field in frozen),
            "factored_custom_owner_release_changed_product_boundaries",
        ),
        _validation(
            "batch_kept_default_off",
            batch.get("runtime_dispatch_ready") is False
            and batch.get("native_dispatch_allowed") is False
            and batch.get("training_path_enabled") is False
            and int(summary.get("product_native_ready_count", 0) or 0) == 0,
            "factored_custom_owner_release_batch_enabled_dispatch",
        ),
    ]


def _optimizer_types(report: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(row.get("optimizer_type") or "")
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and str(row.get("optimizer_type") or "")
    )


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_factored_custom_owner_release_hold_scorecard.json"
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


__all__ = ["EXPECTED_OPTIMIZERS", "HOLD_KIND", "build_factored_custom_owner_release_hold_scorecard"]
