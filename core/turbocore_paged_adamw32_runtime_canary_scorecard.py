"""Report-only runtime canary manifest for fp32 PagedAdamW variants."""

from __future__ import annotations

from typing import Any

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINTS = (
    "create_flat_adamw_tensor_binding_session",
    "tensor_binding_session_cuda_adamw_tensor_probe",
    "destroy_tensor_binding_session",
)


def build_paged_adamw32_runtime_canary_scorecard() -> dict[str, Any]:
    native = native_with_entrypoints(*ENTRYPOINTS)
    ready = native is not None
    blockers = [] if ready else ["paged_adamw32_adamw_tensor_binding_entrypoints_missing"]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw32_runtime_canary_scorecard_v0",
        "gate": "paged_adamw32_runtime_canary_manifest",
        "ok": ready,
        "promotion_ready": False,
        "runtime_canary_manifest_ready": ready,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "optimizer_kinds": ["paged_adamw", "paged_adamw32bit"],
        "optimizer_family": "adamw_paged",
        "native_route": "rust_cuda_adamw_tensor_binding_v0",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "canary_shadow_route_only": True,
        "summary": {
            "runtime_canary_manifest_ready": ready,
            "runtime_canary_ready": False,
            "entrypoint_count": len(ENTRYPOINTS) if ready else 0,
        },
        "promotion_blockers": blockers + ["training_loop_canary_missing", "product_rollout_review_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run PagedAdamW/PagedAdamW32bit TrainingLoop native canary"
            if ready
            else "build native AdamW tensor-binding entrypoints"
        ),
        "notes": [
            "PagedAdamW and PagedAdamW32bit expose fp32 state1/state2 buffers that map to AdamW exp_avg/exp_avg_sq roles.",
            "This manifest is report-only and does not enable native dispatch.",
        ],
    }


__all__ = ["build_paged_adamw32_runtime_canary_scorecard"]
