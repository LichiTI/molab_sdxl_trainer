"""Review-only gate for CUDA graph static-shape training integration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_cuda_graph_observe_manifest_scorecard import (
    build_cuda_graph_observe_manifest_scorecard,
)


FEATURE = "cuda_graph_static_shape_training"
REVIEW_KIND = "cuda_graph_training_integration_review_v0"


def build_cuda_graph_training_integration_review_scorecard(
    *,
    observe_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    """Build a manual-review gate without enabling CUDA graph training."""

    mode = _normalize_mode(native_training_mode)
    observe = dict(
        observe_report
        or build_cuda_graph_observe_manifest_scorecard(
            native_training_mode=mode,
            run_live_probe=True,
        )
    )
    review = _review_package(observe, mode)
    validations = _validations(observe, review)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_cuda_graph_training_integration_review_scorecard_v0",
        "gate": "p6n_cuda_graph_training_integration_review",
        "ok": ready,
        "promotion_ready": ready,
        "review_gate_ready": ready,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "experimental_only": True,
        "feature": FEATURE,
        "review_kind": REVIEW_KIND,
        "native_training_mode": mode,
        "review_package": review,
        "observe_summary": dict(observe.get("summary") or {}),
        "validations": validations,
        "summary": {
            "review_gate_ready": ready,
            "manual_review_required": True,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "fallback_backend": review["rollback_policy"]["fallback_backend"],
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit review before wiring CUDA graph training loop integration"
            if ready
            else "fix CUDA graph integration review gate blockers"
        ),
        "notes": [
            "This gate turns the observe manifest into a training integration review package.",
            "It does not capture or replay the real trainer loop.",
            "Canary and auto modes remain blocked until a human integration review approves wiring.",
        ],
    }


def _review_package(observe: Mapping[str, Any], mode: str) -> dict[str, Any]:
    manifest = observe.get("manifest") if isinstance(observe.get("manifest"), Mapping) else {}
    static_contract = manifest.get("static_contract") if isinstance(manifest.get("static_contract"), Mapping) else {}
    incompatibilities = list(manifest.get("runtime_incompatibilities", []) or [])
    return {
        "schema_version": 1,
        "review_kind": REVIEW_KIND,
        "feature": FEATURE,
        "native_training_mode": mode,
        "candidate_recorded": bool(manifest.get("candidate_recorded", False)),
        "manual_review_required": True,
        "dispatch_review_outcome": "pending_manual_review",
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "static_shape_requirements": {
            "requires_static_batch": True,
            "requires_static_resolution": True,
            "requires_static_dtype": True,
            "requires_fixed_token_counts": True,
            "requires_fixed_microbatch_shape": True,
            "shape_mismatch_blocked": bool(static_contract.get("shape_mismatch_blocked", False)),
        },
        "runtime_incompatibilities": incompatibilities,
        "required_disabled_features": [
            "block_offload",
            "module_offload",
            "cpu_offload_checkpointing",
            "safe_fallback",
            "torch_compile_active",
            "dynamic_batch_or_resolution",
        ],
        "training_loop_contract": {
            "warmup_before_capture": True,
            "capture_after_first_static_batch": True,
            "reset_capture_on_shape_or_dtype_change": True,
            "fallback_on_capture_failure": True,
            "fallback_on_nonfinite_loss": True,
            "fallback_on_optimizer_or_scheduler_mutation_mismatch": True,
        },
        "rollback_policy": {
            "fallback_backend": "standardcore_eager_training_loop",
            "fallback_authoritative": True,
            "rollback_on_shape_change": True,
            "rollback_on_capture_failure": True,
            "rollback_on_replay_parity_failure": True,
            "rollback_on_training_exception": True,
        },
        "audit_fields": [
            "native_training_mode",
            "model_arch",
            "batch_size",
            "resolution",
            "dtype",
            "fixed_token_counts",
            "gradient_accumulation_steps",
            "route_decision",
            "fallback_reason",
            "rollback_reason",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
    }


def _validations(observe: Mapping[str, Any], review: Mapping[str, Any]) -> list[dict[str, Any]]:
    static = review.get("static_shape_requirements") if isinstance(review.get("static_shape_requirements"), Mapping) else {}
    rollback = review.get("rollback_policy") if isinstance(review.get("rollback_policy"), Mapping) else {}
    incompatibilities = set(str(item) for item in review.get("runtime_incompatibilities", []) or [])
    required_disabled = set(str(item) for item in review.get("required_disabled_features", []) or [])
    return [
        _validation(
            "p6f_observe_manifest_ready",
            bool(observe.get("observe_manifest_ready", False)),
            "cuda_graph_observe_manifest_missing",
        ),
        _validation(
            "static_shape_review_requirements_present",
            bool(static.get("requires_static_batch", False))
            and bool(static.get("requires_static_resolution", False))
            and bool(static.get("requires_fixed_microbatch_shape", False))
            and bool(static.get("shape_mismatch_blocked", False)),
            "cuda_graph_static_shape_review_requirements_missing",
        ),
        _validation(
            "incompatibility_matrix_complete",
            required_disabled.issubset(incompatibilities),
            "cuda_graph_incompatibility_matrix_incomplete",
        ),
        _validation(
            "rollback_manifest_present",
            bool(rollback.get("fallback_authoritative", False))
            and bool(rollback.get("rollback_on_shape_change", False))
            and bool(rollback.get("rollback_on_capture_failure", False))
            and bool(rollback.get("rollback_on_replay_parity_failure", False)),
            "cuda_graph_training_review_missing_rollback",
        ),
        _validation(
            "manual_review_blocks_canary_auto",
            bool(review.get("manual_review_required", False))
            and review.get("allowed_initial_modes") == ["off", "observe"]
            and review.get("blocked_modes_until_review") == ["canary", "auto"],
            "cuda_graph_training_review_allows_dispatch_before_review",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(review.get("runtime_dispatch_ready", True))
            and not bool(review.get("native_dispatch_allowed", True))
            and not bool(review.get("training_path_enabled", True)),
            "cuda_graph_training_review_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(observe.get("training_path_enabled", True))
            and not bool(observe.get("default_behavior_changed", True)),
            "cuda_graph_training_review_changed_default_behavior",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _normalize_mode(value: str) -> str:
    normalized = str(value or "observe").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "observe"


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_cuda_graph_training_integration_review_scorecard"]
