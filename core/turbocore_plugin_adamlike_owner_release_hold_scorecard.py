"""Owner/release hold for selected Adam-like plugin optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_adamlike_family_batch_scorecard import (
    build_plugin_adamlike_family_batch_scorecard,
)


HOLD_KIND = "plugin_adamlike_owner_release_hold_v0"


def build_plugin_adamlike_owner_release_hold_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Record a default-off hold after selected Adam-like canary evidence is ready."""

    batch = _as_dict(family_batch_report) if family_batch_report is not None else _default_family_batch()
    summary = _as_dict(batch.get("summary"))
    hold = _hold_manifest(batch)
    validations = _validations(batch, hold)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adamlike_owner_release_hold_scorecard_v0",
        "gate": "plugin_adamlike_owner_release_hold",
        "ok": ready,
        "promotion_ready": False,
        "owner_release_hold_ready": ready,
        "family_batch_ready": batch.get("selected_adamlike_family_batch_ready") is True,
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
            "family_batch_ready": batch.get("selected_adamlike_family_batch_ready") is True,
            "optimizer_count": int(summary.get("target_count", 0) or summary.get("selected_adamlike_optimizer_count", 0) or 0),
            "selected_native_canary_ready_count": int(summary.get("selected_native_canary_ready_count", 0) or 0),
            "exact_adamw_route_canary_ready_count": int(summary.get("exact_adamw_route_canary_ready_count", 0) or 0),
            "dedicated_route_canary_ready_count": int(summary.get("dedicated_route_canary_ready_count", 0) or 0),
            "e2e_shadow_matrix_ready": summary.get("e2e_shadow_matrix_ready") is True,
            "canary_rollout_policy_ready": summary.get("canary_rollout_policy_ready") is True,
            "pending_selected_optimizer_count": int(summary.get("pending_selected_optimizer_count", 0) or 0),
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
                "plugin_adamlike_owner_approval_missing",
                "plugin_adamlike_release_approval_missing",
                "plugin_adamlike_product_dispatch_not_approved",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add Adam-like request/schema/UI non-exposure guard before product dispatch wiring"
            if ready
            else "fix Adam-like owner/release hold blockers"
        ),
        "notes": [
            "This gate records a hold state only; it does not approve native dispatch.",
            "Selected Adam-like canary evidence remains report-only until owner/release approval.",
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
        "optimizer_family": "adam_like_formula",
        "selected_optimizer_names": [str(row.get("selected_optimizer_name") or "") for row in rows],
        "approval_state": "pending_owner_and_release_approval",
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_approval": ["canary", "auto"],
        "required_approval_artifacts": [
            "owner_approval_record",
            "release_risk_acceptance",
            "adamlike_route_reuse_review",
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
    optimizer_count = int(summary.get("target_count", 0) or summary.get("selected_adamlike_optimizer_count", 0) or 0)
    ready_count = int(summary.get("selected_native_canary_ready_count", 0) or 0)
    return [
        _validation(
            "family_batch_ready",
            batch.get("selected_adamlike_family_batch_ready") is True,
            "plugin_adamlike_family_batch_missing",
        ),
        _validation(
            "optimizer_set_complete",
            len(hold.get("selected_optimizer_names", [])) == optimizer_count == 25,
            "plugin_adamlike_optimizer_set_incomplete",
        ),
        _validation(
            "all_canaries_ready",
            ready_count == optimizer_count
            and int(summary.get("pending_selected_optimizer_count", 0) or 0) == 0
            and summary.get("e2e_shadow_matrix_ready") is True
            and summary.get("canary_rollout_policy_ready") is True,
            "plugin_adamlike_canary_stack_not_ready",
        ),
        _validation(
            "approval_not_recorded",
            hold.get("approval_state") == "pending_owner_and_release_approval"
            and hold.get("owner_approval_recorded") is False
            and hold.get("release_approval_recorded") is False,
            "plugin_adamlike_unexpected_approval",
        ),
        _validation(
            "canary_auto_blocked",
            hold.get("allowed_initial_modes") == ["off", "observe"]
            and hold.get("blocked_modes_until_approval") == ["canary", "auto"],
            "plugin_adamlike_hold_allows_dispatch",
        ),
        _validation(
            "product_boundaries_frozen",
            all(frozen.get(field) is False for field in frozen),
            "plugin_adamlike_changed_product_boundaries",
        ),
        _validation(
            "batch_kept_default_off",
            batch.get("runtime_dispatch_ready") is False
            and batch.get("native_dispatch_allowed") is False
            and batch.get("training_path_enabled") is False
            and int(batch.get("plugin_selected_native_ready_count", 0) or 0) == 0,
            "plugin_adamlike_batch_enabled_dispatch",
        ),
    ]


def _rows(batch: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = batch.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _default_family_batch() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    path = path / "turbocore_plugin_adamlike_family_batch_scorecard.json"
    if path.exists():
        try:
            report = _as_dict(json.loads(path.read_text(encoding="utf-8")))
            summary = _as_dict(report.get("summary"))
            if (
                report.get("selected_adamlike_family_batch_ready") is True
                and int(summary.get("selected_native_canary_ready_count", 0) or 0)
                == int(summary.get("target_count", 0) or 0)
            ):
                return report
        except (OSError, json.JSONDecodeError):
            pass
    return build_plugin_adamlike_family_batch_scorecard(include_live_canaries=True, write_artifact=True)


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_adamlike_owner_release_hold_scorecard.json"
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


__all__ = ["build_plugin_adamlike_owner_release_hold_scorecard"]
