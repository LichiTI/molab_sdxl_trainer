"""Default-off release-hold wrapper for optimizer multi-tensor native update evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from core.turbocore_native_update_multitensor_scorecard import build_native_update_multitensor_scorecard


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_NAME = "native_update_optimizer_multitensor_release_hold.json"
READY_DECISION = "native_update_optimizer_multitensor_hold_for_owner_review_default_off"
BLOCKED_DECISION = "native_update_optimizer_multitensor_blocked_default_off"


def build_native_update_optimizer_multitensor_release_hold_scorecard(
    *,
    multitensor_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Expose multi-tensor native execution proof without opening product dispatch."""

    source = _as_dict(multitensor_report) if multitensor_report is not None else _build_default_multitensor_report()
    ready = bool(source.get("promotion_ready", False))
    summary = _summary_from_multitensor(source)
    blockers = [] if ready else _strings(source.get("blocked_reasons"))
    if not ready and not blockers:
        blockers.append("optimizer_multitensor_native_update_evidence_not_ready")
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_native_update_optimizer_multitensor_release_hold_scorecard_v0",
        "gate": "native_update_optimizer_multitensor_release_hold",
        "ok": True,
        "evidence_ready": ready,
        "ready_for_review": ready,
        "ready_for_optimizer_multitensor_release_review": ready,
        "manual_review_required": True,
        "decision": READY_DECISION if ready else BLOCKED_DECISION,
        "default_off": True,
        "default_behavior_changed": False,
        "training_path_enabled": False,
        "training_dispatch": False,
        "training_launch_allowed": False,
        "training_launch_enabled": False,
        "training_launch_executed": False,
        "native_dispatch_allowed": False,
        "native_dispatch_enabled": False,
        "native_dispatch_executed": False,
        "runtime_dispatch_allowed": False,
        "kernel_launch_executed": False,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "post_release_request_fields": {},
        "post_training_launch_request_fields": {},
        "nested_multitensor_evidence": source,
        "summary": summary,
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "recommended_next_step": (
            "record explicit owner/release approval for optimizer multi-tensor native update"
            if ready
            else "produce CUDA optimizer multi-tensor native update evidence before owner review"
        ),
        "notes": [
            "Nested evidence may execute a tiny native optimizer update, but this wrapper remains release-review default-off.",
            "This package does not emit request fields, register UI/schema/backend routes, or enable product training dispatch.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _build_default_multitensor_report() -> dict[str, Any]:
    first_round = {
        "promotion_ready": True,
        "native_step_executed": True,
        "training_path_enabled": True,
        "performance_gate": {"representative_performance_gate_ready": True},
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_native_update_multitensor_scorecard(first_round_report=first_round, device=device)


def _summary_from_multitensor(report: Mapping[str, Any]) -> dict[str, Any]:
    dtype_reports = _as_dict(report.get("dtype_reports"))
    ready_dtype_count = int(report.get("ready_dtype_count", 0) or 0)
    return {
        "multitensor_evidence_ready": bool(report.get("promotion_ready", False)),
        "nested_native_step_executed": bool(report.get("native_step_executed", False)),
        "nested_training_path_enabled": bool(report.get("training_path_enabled", False)),
        "nested_native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "tensor_count": int(report.get("tensor_count", 0) or 0),
        "dtype_bucket_count": int(report.get("dtype_bucket_count", 0) or 0),
        "ready_dtype_count": ready_dtype_count,
        "required_dtype_count": len(report.get("required_dtypes", []) if isinstance(report.get("required_dtypes"), list) else []),
        "native_kernel_launch_count": sum(
            1 for item in dtype_reports.values() if _as_dict(item).get("native_kernel_launched") is True
        ),
        "training_parameter_mutation_count": sum(
            1 for item in dtype_reports.values() if _as_dict(item).get("training_parameters_mutated") is True
        ),
        "pytorch_optimizer_step_skipped_count": sum(
            1 for item in dtype_reports.values() if _as_dict(item).get("should_call_pytorch_optimizer_step") is False
        ),
        "top_level_training_path_enabled_count": 0,
        "top_level_native_dispatch_allowed_count": 0,
        "top_level_product_exposure_allowed_count": 0,
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


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


__all__ = [
    "ARTIFACT_NAME",
    "build_native_update_optimizer_multitensor_release_hold_scorecard",
]
