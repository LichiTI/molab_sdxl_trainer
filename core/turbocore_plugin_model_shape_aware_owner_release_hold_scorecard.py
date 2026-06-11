"""Owner/release hold for selected model/shape-aware plugin optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_model_shape_aware_family_batch_scorecard import (
    build_plugin_model_shape_aware_family_batch_scorecard,
)


HOLD_KIND = "plugin_model_shape_aware_owner_release_hold_v0"


def build_plugin_model_shape_aware_owner_release_hold_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Record a default-off hold after model/shape ABI matrices are implementation-ready."""

    batch = _as_dict(family_batch_report or build_plugin_model_shape_aware_family_batch_scorecard())
    summary = _as_dict(batch.get("summary"))
    hold = _hold_manifest(batch)
    validations = _validations(batch, hold)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_model_shape_aware_owner_release_hold_scorecard_v0",
        "gate": "plugin_model_shape_aware_owner_release_hold",
        "ok": ready,
        "promotion_ready": False,
        "owner_release_hold_ready": ready,
        "family_batch_ready": batch.get("selected_model_shape_aware_family_batch_ready") is True,
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
        "family_batch_summary": summary,
        "validations": validations,
        "summary": {
            "owner_release_hold_ready": ready,
            "family_batch_ready": batch.get("selected_model_shape_aware_family_batch_ready") is True,
            "optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
            "param_group_abi_implementation_ready_count": int(
                summary.get("param_group_abi_implementation_ready_count", 0) or 0
            ),
            "param_group_resume_replay_matrix_implementation_ready_count": int(
                summary.get("param_group_resume_replay_matrix_implementation_ready_count", 0) or 0
            ),
            "param_group_resume_replay_row_implementation_ready_count": int(
                summary.get("param_group_resume_replay_row_implementation_ready_count", 0) or 0
            ),
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
                "plugin_model_shape_aware_owner_approval_missing",
                "plugin_model_shape_aware_release_approval_missing",
                "plugin_model_shape_aware_product_dispatch_not_approved",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add model/shape-aware request/schema/UI non-exposure guard before product dispatch wiring"
            if ready
            else "fix model/shape-aware owner/release hold blockers"
        ),
        "notes": [
            "This gate records a hold state only; it does not approve native dispatch.",
            "Model/shape-aware evidence remains report-only until owner/release approval.",
            "Request, schema, UI, runtime dispatch, and training behavior remain unchanged.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _hold_manifest(batch: Mapping[str, Any]) -> dict[str, Any]:
    rows = _rows(batch)
    return {
        "schema_version": 1,
        "hold_kind": HOLD_KIND,
        "optimizer_family": "model_or_shape_aware",
        "selected_optimizer_names": [str(row.get("selected_optimizer_name") or "") for row in rows],
        "approval_state": "pending_owner_and_release_approval",
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_approval": ["canary", "auto"],
        "required_approval_artifacts": [
            "owner_approval_record",
            "release_risk_acceptance",
            "model_shape_param_group_abi_review",
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


def _validations(batch: Mapping[str, Any], hold: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = _as_dict(batch.get("summary"))
    frozen = _as_dict(hold.get("frozen_product_boundaries"))
    optimizer_count = int(summary.get("selected_optimizer_count", 0) or 0)
    return [
        _validation(
            "family_batch_ready",
            batch.get("selected_model_shape_aware_family_batch_ready") is True,
            "plugin_model_shape_aware_family_batch_missing",
        ),
        _validation(
            "optimizer_set_complete",
            len(hold.get("selected_optimizer_names", [])) == optimizer_count == 7,
            "plugin_model_shape_aware_optimizer_set_incomplete",
        ),
        _validation(
            "param_group_matrices_ready",
            int(summary.get("param_group_abi_implementation_ready_count", 0) or 0) == optimizer_count
            and int(summary.get("param_group_resume_replay_matrix_implementation_ready_count", 0) or 0)
            == optimizer_count
            and int(summary.get("param_group_resume_replay_row_implementation_ready_count", 0) or 0) == 29,
            "plugin_model_shape_aware_matrices_not_ready",
        ),
        _validation(
            "approval_not_recorded",
            hold.get("approval_state") == "pending_owner_and_release_approval"
            and hold.get("owner_approval_recorded") is False
            and hold.get("release_approval_recorded") is False,
            "plugin_model_shape_aware_unexpected_approval",
        ),
        _validation(
            "canary_auto_blocked",
            hold.get("allowed_initial_modes") == ["off", "observe"]
            and hold.get("blocked_modes_until_approval") == ["canary", "auto"],
            "plugin_model_shape_aware_hold_allows_dispatch",
        ),
        _validation(
            "product_boundaries_frozen",
            all(frozen.get(field) is False for field in frozen),
            "plugin_model_shape_aware_changed_product_boundaries",
        ),
        _validation(
            "batch_kept_default_off",
            batch.get("runtime_dispatch_ready") is False
            and batch.get("native_dispatch_allowed") is False
            and batch.get("training_path_enabled") is False
            and int(summary.get("plugin_selected_native_ready_count", 0) or 0) == 0,
            "plugin_model_shape_aware_batch_enabled_dispatch",
        ),
    ]


def _rows(batch: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = batch.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_model_shape_aware_owner_release_hold_scorecard.json"
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


__all__ = ["build_plugin_model_shape_aware_owner_release_hold_scorecard"]
