"""Batch runtime dispatch rehearsal for selected plugin adaptive-LR optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_adaptive_lr_training_loop_canary_scorecard import (
    build_adaptive_lr_training_loop_canary_scorecard,
)
from core.turbocore_plugin_adaptivelr_family_batch_scorecard import (
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_adaptivelr_family_batch_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_NAME = "turbocore_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard.json"


def build_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    training_loop_canary_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Map adaptive-LR plugin optimizers onto family runtime canary evidence."""

    family_batch = _as_dict(
        family_batch_report
        if family_batch_report is not None
        else build_plugin_adaptivelr_family_batch_scorecard(write_artifact=True)
    )
    training_loop = _as_dict(
        training_loop_canary_report
        if training_loop_canary_report is not None
        else build_adaptive_lr_training_loop_canary_scorecard()
    )
    family_cases = _family_cases(training_loop)
    family_rows = _family_rows(family_batch)
    cases = [_case_row(name, family_rows.get(name, {}), family_cases) for name in TARGET_PLUGIN_OPTIMIZERS]
    blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
    representative_native_steps = {
        str(case.get("representative_family") or "")
        for case in cases
        if case.get("representative_native_step_executed") is True
    }
    representative_kernel_launches = {
        str(case.get("representative_family") or "")
        for case in cases
        if case.get("representative_native_kernel_launched") is True
    }
    ready = (
        family_batch.get("selected_adaptivelr_family_batch_ready") is True
        and training_loop.get("training_loop_canary_ready") is True
        and all(case.get("runtime_dispatch_rehearsal_ready") is True for case in cases)
        and not blockers
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard_v0",
        "gate": "plugin_adaptivelr_runtime_dispatch_rehearsal",
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
        "selected_optimizer_family": "adaptive_lr_state_machine",
        "family_batch_scorecard": _compact_family_batch(family_batch),
        "training_loop_canary": _compact_training_loop(training_loop),
        "cases": cases,
        "summary": {
            "selected_optimizer_count": len(cases),
            "case_count": len(cases),
            "runtime_dispatch_rehearsal_ready_count": sum(
                1 for case in cases if case.get("runtime_dispatch_rehearsal_ready") is True
            ),
            "training_executor_called_count": sum(
                1 for case in cases if case.get("training_executor_called") is True
            ),
            "native_step_count": len(representative_native_steps),
            "native_kernel_launch_count": len(representative_kernel_launches),
            "mapped_selected_native_step_count": sum(
                1 for case in cases if case.get("representative_native_step_executed") is True
            ),
            "mapped_selected_native_kernel_launch_count": sum(
                1 for case in cases if case.get("representative_native_kernel_launched") is True
            ),
            "representative_family_case_count": len(family_cases),
            "skip_pytorch_count": sum(1 for case in cases if case.get("should_call_pytorch_optimizer_step") is False),
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "plugin_adaptivelr_owner_release_review_missing",
                "plugin_adaptivelr_product_training_route_not_bound",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "bind selected plugin adaptive-LR rehearsal evidence into guarded product-training canary"
            if ready
            else "fix selected plugin adaptive-LR runtime dispatch rehearsal blockers"
        ),
        "notes": [
            "This maps six selected plugin adaptive-LR optimizers onto two family runtime canaries.",
            "The native step counts are representative family launches, not product dispatch approval.",
            "Request, schema, UI, runtime dispatch, and training defaults remain closed.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _case_row(
    selected_optimizer_name: str,
    family_row: Mapping[str, Any],
    family_cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    family = _family(selected_optimizer_name)
    canary = _as_dict(family_cases.get(family))
    native_step = canary.get("native_step_executed") is True
    native_kernel = canary.get("native_kernel_launched") is True
    skipped_pytorch = canary.get("should_call_pytorch_optimizer_step") is False
    ready = (
        family_row.get("state_machine_abi_implementation_ready") is True
        and native_step
        and native_kernel
        and skipped_pytorch
        and canary.get("ok") is True
    )
    return {
        "schema_version": 1,
        "ok": ready,
        "selected_optimizer_name": selected_optimizer_name,
        "selected_optimizer_family": "adaptive_lr_state_machine",
        "representative_family": family,
        "builtin_reference_optimizer_type": str(family_row.get("builtin_reference_optimizer_type") or ""),
        "runtime_dispatch_rehearsal_ready": ready,
        "training_executor_called": canary.get("training_executor_called") is True,
        "training_executor_ok": canary.get("training_executor_ok") is True,
        "representative_native_step_executed": native_step,
        "representative_native_kernel_launched": native_kernel,
        "should_call_pytorch_optimizer_step": skipped_pytorch is False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "source_scorecard": "turbocore_adaptive_lr_training_loop_canary_scorecard_v0",
        "blocked_reasons": [] if ready else [f"plugin_{selected_optimizer_name}_adaptivelr_runtime_rehearsal_missing"],
    }


def _family(name: str) -> str:
    return "adaptive_lr_prodigy" if name == "prodigy" else "adaptive_lr_dadapt"


def _family_cases(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(case.get("family") or ""): case
        for case in report.get("family_cases", [])
        if isinstance(case, Mapping)
    }


def _family_rows(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("selected_optimizer_name") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }


def _compact_family_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_adaptivelr_family_batch_ready": report.get("selected_adaptivelr_family_batch_ready") is True,
        "report_only": report.get("report_only") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "state_machine_abi_implementation_ready_count": int(
            summary.get("selected_state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("selected_native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_training_loop(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "training_loop_canary_ready": report.get("training_loop_canary_ready") is True,
        "runtime_dispatch_shadow_ready": report.get("runtime_dispatch_shadow_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "family_case_count": int(summary.get("family_case_count", 0) or 0),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
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


__all__ = ["ARTIFACT_NAME", "ROADMAP", "build_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard"]
