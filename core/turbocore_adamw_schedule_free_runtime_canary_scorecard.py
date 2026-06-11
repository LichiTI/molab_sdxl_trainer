"""Report-only runtime canary manifest for AdamWScheduleFree native scratch route."""

from __future__ import annotations

from typing import Any

from core.services.native_module_loader import native_with_entrypoints
from core.turbocore_adamw_schedule_free_native_scratch_kernel_scorecard import (
    ENTRYPOINT,
    build_adamw_schedule_free_native_scratch_kernel_scorecard,
)


def build_adamw_schedule_free_runtime_canary_scorecard() -> dict[str, Any]:
    """Expose a default-off runtime canary manifest without training dispatch."""

    native = native_with_entrypoints(ENTRYPOINT)
    scratch = build_adamw_schedule_free_native_scratch_kernel_scorecard()
    ready = native is not None and scratch.get("native_scratch_kernel_parity_ready") is True
    blockers = []
    if native is None:
        blockers.append("adamw_schedule_free_native_scratch_entrypoint_missing")
    if scratch.get("native_scratch_kernel_parity_ready") is not True:
        blockers.append("adamw_schedule_free_native_scratch_kernel_parity_missing")
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_schedule_free_runtime_canary_scorecard_v0",
        "gate": "adamw_schedule_free_runtime_canary_manifest",
        "ok": ready,
        "promotion_ready": False,
        "runtime_canary_manifest_ready": ready,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "optimizer_kind": "adamw_schedule_free",
        "optimizer_family": "adamw_schedule_free",
        "native_route": "rust_cuda_adamw_schedule_free_scratch_v0",
        "entrypoint": ENTRYPOINT,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "canary_shadow_route_only": True,
        "native_scratch_kernel": {
            "ok": scratch.get("ok") is True,
            "native_scratch_kernel_parity_ready": scratch.get("native_scratch_kernel_parity_ready") is True,
            "native_kernel_ready": scratch.get("native_kernel_ready") is True,
            "summary": dict(scratch.get("summary") or {}),
        },
        "summary": {
            "runtime_canary_manifest_ready": ready,
            "runtime_canary_ready": False,
            "entrypoint_count": 1 if native is not None else 0,
            "native_scratch_kernel_ready": scratch.get("native_scratch_kernel_parity_ready") is True,
            "training_path_enabled": False,
        },
        "promotion_blockers": blockers + ["training_loop_canary_missing", "product_rollout_review_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add AdamWScheduleFree TrainingLoop native canary with dispatch still disabled"
            if ready
            else "build AdamWScheduleFree native scratch entrypoint before runtime canary"
        ),
        "notes": [
            "This manifest is report-only and does not enable native dispatch.",
            "The native scratch kernel is not a product training route.",
        ],
    }


__all__ = ["build_adamw_schedule_free_runtime_canary_scorecard"]
