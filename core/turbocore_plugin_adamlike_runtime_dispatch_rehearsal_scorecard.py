"""Batch runtime dispatch rehearsal for selected plugin Adam-like optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_adamlike_family_batch_scorecard import (
    build_plugin_adamlike_family_batch_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_NAME = "turbocore_plugin_adamlike_runtime_dispatch_rehearsal_scorecard.json"


def build_plugin_adamlike_runtime_dispatch_rehearsal_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Normalize the Adam-like live canary batch as runtime rehearsal evidence."""

    family = _as_dict(
        family_batch_report
        if family_batch_report is not None
        else build_plugin_adamlike_family_batch_scorecard(include_live_canaries=True, write_artifact=True)
    )
    rows = [_case_row(row) for row in family.get("rows", []) if isinstance(row, Mapping)]
    blockers = _dedupe(reason for row in rows for reason in row.get("blocked_reasons", []) or [])
    summary = _as_dict(family.get("summary"))
    ready = (
        family.get("selected_adamlike_family_batch_ready") is True
        and bool(rows)
        and all(row.get("runtime_dispatch_rehearsal_ready") is True for row in rows)
        and not blockers
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adamlike_runtime_dispatch_rehearsal_scorecard_v0",
        "gate": "plugin_adamlike_runtime_dispatch_rehearsal",
        "roadmap": ROADMAP,
        "ok": ready,
        "promotion_ready": False,
        "runtime_dispatch_rehearsal_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_dispatch_ready": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "internal_rehearsal_executed": True,
        "selected_optimizer_family": "adam_like_formula",
        "family_batch_scorecard": _compact_family_batch(family),
        "cases": rows,
        "summary": {
            "selected_optimizer_count": int(summary.get("target_count", len(rows)) or len(rows)),
            "case_count": len(rows),
            "runtime_dispatch_rehearsal_ready_count": sum(
                1 for row in rows if row.get("runtime_dispatch_rehearsal_ready") is True
            ),
            "training_executor_called_count": sum(1 for row in rows if row.get("training_executor_called") is True),
            "native_step_count": sum(int(row.get("native_step_count", 0) or 0) for row in rows),
            "native_kernel_launch_count": sum(int(row.get("native_kernel_launch_count", 0) or 0) for row in rows),
            "skip_pytorch_count": sum(1 for row in rows if row.get("should_call_pytorch_optimizer_step") is False),
            "exact_adamw_route_canary_ready_count": int(
                summary.get("exact_adamw_route_canary_ready_count", 0) or 0
            ),
            "dedicated_route_canary_ready_count": int(summary.get("dedicated_route_canary_ready_count", 0) or 0),
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "plugin_adamlike_owner_release_review_missing",
                "plugin_adamlike_product_training_route_not_bound",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "bind selected plugin Adam-like rehearsal evidence into guarded product-training canary"
            if ready
            else "fix selected plugin Adam-like runtime dispatch rehearsal blockers"
        ),
        "notes": [
            "This wraps the existing selected Adam-like live canary batch as one runtime rehearsal artifact.",
            "Request, schema, UI, runtime dispatch, and training defaults remain closed.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _case_row(row: Mapping[str, Any]) -> dict[str, Any]:
    reports = _as_dict(row.get("compact_reports"))
    training_loop = _as_dict(reports.get("training_loop"))
    training_summary = _as_dict(training_loop.get("summary"))
    native_step_count = int(training_summary.get("native_step_count", 0) or 0)
    native_kernel_launch_count = int(training_summary.get("native_kernel_launch_count", 0) or 0)
    ready = row.get("selected_native_canary_ready") is True
    return {
        "schema_version": 1,
        "ok": ready,
        "selected_optimizer_name": str(row.get("selected_optimizer_name") or ""),
        "selected_optimizer_family": "adam_like_formula",
        "native_route": str(row.get("native_route") or ""),
        "route_kind": str(row.get("route_kind") or ""),
        "runtime_dispatch_rehearsal_ready": ready,
        "training_executor_called": native_step_count > 0,
        "native_step_count": native_step_count,
        "native_kernel_launch_count": native_kernel_launch_count,
        "native_step_executed": native_step_count > 0,
        "native_kernel_launched": native_kernel_launch_count > 0,
        "should_call_pytorch_optimizer_step": False if native_step_count > 0 else True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "stage_status": dict(_as_dict(row.get("stage_status"))),
        "blocked_reasons": [] if ready else [f"plugin_{row.get('selected_optimizer_name')}_runtime_dispatch_rehearsal_missing"],
    }


def _compact_family_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_adamlike_family_batch_ready": report.get("selected_adamlike_family_batch_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "selected_native_canary_ready_count": int(summary.get("selected_native_canary_ready_count", 0) or 0),
        "plugin_selected_native_ready_count": int(summary.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / ARTIFACT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ARTIFACT_NAME", "ROADMAP", "build_plugin_adamlike_runtime_dispatch_rehearsal_scorecard"]
