"""Batch runtime dispatch rehearsal for selected plugin schedule-free optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_schedulefree_family_batch_scorecard import (
    build_plugin_schedulefree_family_batch_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_NAME = "turbocore_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard.json"


def build_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Normalize selected schedule-free live canaries as runtime rehearsal evidence."""

    family = _as_dict(
        family_batch_report
        if family_batch_report is not None
        else build_plugin_schedulefree_family_batch_scorecard(write_artifact=True)
    )
    stage_summaries = _as_dict(family.get("stage_summaries"))
    cases = [
        _case_row("schedulefreeadamw", stage_summaries.get("schedulefreeadamw_training_loop_canary")),
        _case_row("schedulefreesgd", stage_summaries.get("schedulefreesgd_training_loop_canary")),
        _case_row("schedulefreeradam", stage_summaries.get("schedulefreeradam_training_loop_canary")),
    ]
    blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
    summary = _as_dict(family.get("summary"))
    ready = (
        family.get("selected_schedulefree_family_batch_ready") is True
        and all(case.get("runtime_dispatch_rehearsal_ready") is True for case in cases)
        and not blockers
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard_v0",
        "gate": "plugin_schedulefree_runtime_dispatch_rehearsal",
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
        "selected_optimizer_family": "schedule_free_state_machine",
        "family_batch_scorecard": _compact_family_batch(family),
        "cases": cases,
        "summary": {
            "selected_optimizer_count": int(summary.get("selected_optimizer_count", len(cases)) or len(cases)),
            "case_count": len(cases),
            "runtime_dispatch_rehearsal_ready_count": sum(
                1 for case in cases if case.get("runtime_dispatch_rehearsal_ready") is True
            ),
            "training_executor_called_count": sum(1 for case in cases if case.get("training_executor_called") is True),
            "native_step_count": sum(int(case.get("native_step_count", 0) or 0) for case in cases),
            "native_kernel_launch_count": sum(int(case.get("native_kernel_launch_count", 0) or 0) for case in cases),
            "skip_pytorch_count": sum(1 for case in cases if case.get("should_call_pytorch_optimizer_step") is False),
            "e2e_shadow_case_count": int(summary.get("e2e_shadow_case_count", 0) or 0),
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "plugin_schedulefree_owner_release_review_missing",
                "plugin_schedulefree_product_training_route_not_bound",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "bind selected plugin schedule-free rehearsal evidence into guarded product-training canary"
            if ready
            else "fix selected plugin schedule-free runtime dispatch rehearsal blockers"
        ),
        "notes": [
            "This wraps selected schedule-free live canaries as one runtime rehearsal artifact.",
            "Request, schema, UI, runtime dispatch, and training defaults remain closed.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _case_row(selected_optimizer_name: str, stage: Any) -> dict[str, Any]:
    stage_report = _as_dict(stage)
    summary = _as_dict(stage_report.get("summary"))
    native_step_count = int(summary.get("native_step_count", 0) or 0)
    native_kernel_launch_count = int(summary.get("native_kernel_launch_count", 0) or 0)
    ready = stage_report.get("ok") is True and native_step_count > 0 and native_kernel_launch_count > 0
    return {
        "schema_version": 1,
        "ok": ready,
        "selected_optimizer_name": selected_optimizer_name,
        "selected_optimizer_family": "schedule_free_state_machine",
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
        "source_scorecard": str(stage_report.get("scorecard") or ""),
        "blocked_reasons": [] if ready else [f"plugin_{selected_optimizer_name}_runtime_dispatch_rehearsal_missing"],
    }


def _compact_family_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_schedulefree_family_batch_ready": report.get("selected_schedulefree_family_batch_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "selected_native_canary_ready_count": int(summary.get("selected_native_canary_ready_count", 0) or 0),
        "e2e_shadow_case_count": int(summary.get("e2e_shadow_case_count", 0) or 0),
        "plugin_selected_native_ready_count": int(report.get("plugin_selected_native_ready_count", 0) or 0),
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


__all__ = ["ARTIFACT_NAME", "ROADMAP", "build_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard"]
