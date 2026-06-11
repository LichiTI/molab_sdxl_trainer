"""Owner/release approval hold for selected plugin optimizer families."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_optimizer_family_batch_scorecard import (
    build_plugin_optimizer_family_batch_scorecard,
)


EXPECTED_FAMILIES = frozenset(
    {
        "adam_like_formula",
        "adaptive_lr_state_machine",
        "closure_or_second_order",
        "custom_formula",
        "factored_memory_layout",
        "fused_backward",
        "model_or_shape_aware",
        "schedule_free_state_machine",
        "simple_formula",
        "state_adapter_special",
    }
)
HOLD_KIND = "plugin_selected_family_owner_release_hold_v0"


def build_plugin_selected_family_owner_release_hold_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
    refresh_family_artifacts: bool = False,
) -> dict[str, Any]:
    """Record a default-off hold after selected family gates are ready."""

    batch = _as_dict(
        family_batch_report
        or build_plugin_optimizer_family_batch_scorecard(
            refresh_family_artifacts=refresh_family_artifacts,
        )
    )
    hold = _hold_manifest(batch)
    validations = _validations(batch, hold)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    approval_blockers = [
        "plugin_selected_family_owner_approval_missing",
        "plugin_selected_family_release_approval_missing",
        "plugin_selected_family_product_dispatch_not_approved",
    ]
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_selected_family_owner_release_hold_scorecard_v0",
        "gate": "plugin_selected_family_owner_release_hold",
        "ok": ready,
        "promotion_ready": False,
        "owner_release_hold_ready": ready,
        "plugin_optimizer_family_batch_ready": batch.get("plugin_optimizer_family_batch_ready") is True,
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
        "fallback_backend": "pytorch_optimizer_python",
        "fallback_backend_authoritative": True,
        "hold_manifest": hold,
        "family_batch_summary": _as_dict(batch.get("summary")),
        "validations": validations,
        "summary": {
            "owner_release_hold_ready": ready,
            "plugin_optimizer_family_batch_ready": batch.get("plugin_optimizer_family_batch_ready") is True,
            "manual_review_required": True,
            "owner_approval_recorded": False,
            "release_approval_recorded": False,
            "family_count": len(_family_names(batch)),
            "plugin_optimizer_count": int(_as_dict(batch.get("summary")).get("plugin_optimizer_count", 0) or 0),
            "selected_optimizer_gate_ready_count": int(
                _as_dict(batch.get("summary")).get("selected_optimizer_gate_ready_count", 0) or 0
            ),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(blockers + approval_blockers),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add plugin selected-family request/schema/UI non-exposure guard before any product dispatch wiring"
            if ready
            else "fix plugin selected-family owner/release hold blockers"
        ),
        "notes": [
            "This gate is a hold state, not a product promotion.",
            "Selected family evidence is packaged but owner and release approval are intentionally absent.",
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
        "optimizer_family": "plugin_selected_families",
        "native_route_families": _family_names(batch),
        "plugin_optimizer_count": int(_as_dict(batch.get("summary")).get("plugin_optimizer_count", 0) or 0),
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
            "plugin_family_fallback_rollback_acknowledgement",
        ],
        "family_rows": _compact_family_rows(batch),
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
    family_names = set(_family_names(batch))
    frozen = _as_dict(hold.get("frozen_product_boundaries"))
    rows = _family_rows(batch)
    return [
        _validation(
            "family_batch_ready",
            batch.get("plugin_optimizer_family_batch_ready") is True,
            "plugin_selected_family_batch_missing",
        ),
        _validation(
            "family_set_complete",
            family_names == EXPECTED_FAMILIES,
            "plugin_selected_family_set_incomplete",
        ),
        _validation(
            "selected_gates_ready",
            int(summary.get("selected_optimizer_gate_ready_count", 0) or 0) == len(EXPECTED_FAMILIES)
            and int(summary.get("selected_optimizer_gate_pending_count", 0) or 0) == 0,
            "plugin_selected_family_gates_not_ready",
        ),
        _validation(
            "plugin_optimizer_count_covered",
            _row_optimizer_count(rows) == int(summary.get("plugin_optimizer_count", 0) or 0) == 124,
            "plugin_selected_family_optimizer_count_mismatch",
        ),
        _validation(
            "approval_not_recorded",
            hold.get("approval_state") == "pending_owner_and_release_approval"
            and hold.get("owner_approval_recorded") is False
            and hold.get("release_approval_recorded") is False,
            "plugin_selected_family_unexpected_approval",
        ),
        _validation(
            "canary_auto_blocked",
            hold.get("allowed_initial_modes") == ["off", "observe"]
            and hold.get("blocked_modes_until_approval") == ["canary", "auto"],
            "plugin_selected_family_hold_allows_dispatch",
        ),
        _validation(
            "product_boundaries_frozen",
            all(frozen.get(field) is False for field in frozen),
            "plugin_selected_family_changed_product_boundaries",
        ),
        _validation(
            "batch_kept_default_off",
            batch.get("runtime_dispatch_ready") is False
            and batch.get("native_dispatch_allowed") is False
            and batch.get("training_path_enabled") is False
            and int(summary.get("plugin_selected_native_ready_count", 0) or 0) == 0
            and int(summary.get("plugin_selected_runtime_dispatch_ready_count", 0) or 0) == 0,
            "plugin_selected_family_batch_enabled_dispatch",
        ),
    ]


def _compact_family_rows(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _family_rows(batch):
        rows.append(
            {
                "native_route_family": str(row.get("native_route_family") or ""),
                "plugin_optimizer_count": int(row.get("plugin_optimizer_count", 0) or 0),
                "selected_optimizer_gate": str(row.get("selected_optimizer_gate") or ""),
                "selected_optimizer_gate_ready": row.get("selected_optimizer_gate_ready") is True,
                "next_gate": str(row.get("next_gate") or ""),
                "runtime_dispatch_ready": False,
                "native_dispatch_allowed": False,
                "training_path_enabled": False,
            }
        )
    return rows


def _family_names(batch: Mapping[str, Any]) -> list[str]:
    return sorted(str(row.get("native_route_family") or "") for row in _family_rows(batch))


def _family_rows(batch: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = batch.get("family_rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping) and str(row.get("native_route_family") or "")]


def _row_optimizer_count(rows: list[Mapping[str, Any]]) -> int:
    return sum(int(row.get("plugin_optimizer_count", 0) or 0) for row in rows)


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_selected_family_owner_release_hold_scorecard.json"
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


__all__ = [
    "EXPECTED_FAMILIES",
    "HOLD_KIND",
    "build_plugin_selected_family_owner_release_hold_scorecard",
]
