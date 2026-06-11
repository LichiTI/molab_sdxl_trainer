"""Review-only gate for native data pipeline trainer integration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_native_data_pipeline_canary_rollout_policy_scorecard import (
    build_native_data_pipeline_canary_rollout_policy_scorecard,
)


FEATURE = "native_data_pipeline_trainer_integration"
REVIEW_KIND = "native_data_pipeline_training_integration_review_v0"


def build_native_data_pipeline_training_integration_review_scorecard(
    *,
    policy_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    """Build a manual review package without enabling trainer data dispatch."""

    mode = _normalize_mode(native_training_mode)
    policy_report = dict(
        policy_report
        or build_native_data_pipeline_canary_rollout_policy_scorecard(
            native_training_mode=mode,
        )
    )
    review = _review_package(policy_report, mode)
    validations = _validations(policy_report, review)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_native_data_pipeline_training_integration_review_scorecard_v0",
        "gate": "p6p_native_data_pipeline_training_integration_review",
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
        "policy_summary": dict(policy_report.get("summary") or {}),
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
            "requires explicit review before wiring native data pipeline trainer integration"
            if ready
            else "fix native data pipeline integration review gate blockers"
        ),
        "notes": [
            "This gate prepares the trainer data-path integration review only.",
            "The StandardCore Python dataloader remains authoritative.",
            "Canary and auto modes remain blocked until explicit integration review approves wiring.",
        ],
    }


def _review_package(policy_report: Mapping[str, Any], mode: str) -> dict[str, Any]:
    policy = policy_report.get("policy") if isinstance(policy_report.get("policy"), Mapping) else {}
    rollback = policy.get("rollback_policy") if isinstance(policy.get("rollback_policy"), Mapping) else {}
    return {
        "schema_version": 1,
        "review_kind": REVIEW_KIND,
        "feature": FEATURE,
        "native_training_mode": mode,
        "manual_review_required": True,
        "dispatch_review_outcome": "pending_manual_review",
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "trainer_hook_contract": {
            "dataset_build_hook": "core.lulynx_trainer.trainer.Trainer.train.dataset_build",
            "caption_dataset_hook": "core.lulynx_trainer.dataset_loader.CaptionDataset",
            "cached_dataset_hooks": [
                "core.lulynx_trainer.anima_cached_dataset.AnimaCachedDataset",
                "core.lulynx_trainer.newbie_cached_dataset.NewbieCachedDataset",
            ],
            "dataloader_hook": "core.lulynx_trainer.trainer.Trainer.dataloader",
            "training_loop_epoch_hook": "core.lulynx_trainer.training_loop.TrainingLoop.train_epoch",
            "training_loop_step_hook": "core.lulynx_trainer.training_loop.TrainingLoop.train_step",
            "staged_resolution_hooks": [
                "Trainer._maybe_switch_anima_staged_resolution_dataset",
                "Trainer._maybe_switch_lora_staged_resolution_dataset",
            ],
        },
        "batch_semantic_contract": {
            "descriptor_order_parity_required": True,
            "drop_last_parity_required": True,
            "shuffle_seed_parity_required": True,
            "bucket_resolution_parity_required": True,
            "caption_mode_parity_required": True,
            "loss_weight_keys_preserved": True,
            "filenames_preserved": True,
        },
        "h2d_ownership_contract": {
            "fallback_owns_training_batch": True,
            "native_pipeline_owns_device_tensor": False,
            "copy_independent_required": True,
            "non_blocking_semantics_audited": True,
            "device_tensor_mutation_must_not_update_source": True,
        },
        "cache_and_stage_contract": {
            "cache_first_boundary_preserved": True,
            "cached_latent_resolution_manifest_required": True,
            "staged_resolution_switch_requires_dataloader_rebuild": True,
            "dataset_set_current_epoch_preserved": True,
            "dataset_set_global_step_preserved": True,
        },
        "source_policy_contract": {
            "fallback_backend": rollback.get("fallback_backend", "standardcore_python_data_path"),
            "fallback_authoritative": bool(rollback.get("fallback_authoritative", False)),
            "descriptor_parity_rollback": bool(rollback.get("rollback_on_descriptor_parity_failure", False)),
            "h2d_ownership_rollback": bool(rollback.get("rollback_on_h2d_ownership_failure", False)),
            "queue_stall_rollback": bool(rollback.get("rollback_on_queue_stall_regression", False)),
        },
        "rollback_policy": {
            "fallback_backend": "standardcore_python_dataloader",
            "fallback_authoritative": True,
            "rollback_on_descriptor_parity_failure": True,
            "rollback_on_h2d_ownership_failure": True,
            "rollback_on_nonfinite_batch_tensor": True,
            "rollback_on_queue_stall_regression": True,
            "rollback_on_staged_resolution_boundary": True,
            "rollback_on_cache_manifest_mismatch": True,
        },
        "audit_fields": [
            "native_training_mode",
            "dataset_kind",
            "cache_first",
            "sample_manifest_uri",
            "batch_size",
            "resolution_bucket",
            "caption_mode",
            "prefetch_depth",
            "chunk_size",
            "h2d_transfer_mode",
            "route_decision",
            "fallback_backend",
            "rollback_reason",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
    }


def _validations(policy_report: Mapping[str, Any], review: Mapping[str, Any]) -> list[dict[str, Any]]:
    hooks = review.get("trainer_hook_contract") if isinstance(review.get("trainer_hook_contract"), Mapping) else {}
    batch = review.get("batch_semantic_contract") if isinstance(review.get("batch_semantic_contract"), Mapping) else {}
    h2d = review.get("h2d_ownership_contract") if isinstance(review.get("h2d_ownership_contract"), Mapping) else {}
    cache = review.get("cache_and_stage_contract") if isinstance(review.get("cache_and_stage_contract"), Mapping) else {}
    rollback = review.get("rollback_policy") if isinstance(review.get("rollback_policy"), Mapping) else {}
    return [
        _validation(
            "p6l_canary_rollout_policy_ready",
            bool(policy_report.get("canary_rollout_policy_ready", False)),
            "native_data_pipeline_canary_rollout_policy_missing",
        ),
        _validation(
            "trainer_hook_contract_present",
            bool(hooks.get("dataset_build_hook"))
            and bool(hooks.get("dataloader_hook"))
            and bool(hooks.get("training_loop_epoch_hook"))
            and bool(hooks.get("training_loop_step_hook")),
            "native_data_pipeline_trainer_hook_contract_missing",
        ),
        _validation(
            "batch_semantic_contract_present",
            bool(batch.get("descriptor_order_parity_required", False))
            and bool(batch.get("bucket_resolution_parity_required", False))
            and bool(batch.get("filenames_preserved", False)),
            "native_data_pipeline_batch_semantic_contract_missing",
        ),
        _validation(
            "h2d_ownership_contract_present",
            bool(h2d.get("fallback_owns_training_batch", False))
            and not bool(h2d.get("native_pipeline_owns_device_tensor", True))
            and bool(h2d.get("copy_independent_required", False)),
            "native_data_pipeline_h2d_ownership_contract_missing",
        ),
        _validation(
            "cache_stage_contract_present",
            bool(cache.get("cache_first_boundary_preserved", False))
            and bool(cache.get("staged_resolution_switch_requires_dataloader_rebuild", False)),
            "native_data_pipeline_cache_stage_contract_missing",
        ),
        _validation(
            "rollback_manifest_present",
            bool(rollback.get("fallback_authoritative", False))
            and bool(rollback.get("rollback_on_descriptor_parity_failure", False))
            and bool(rollback.get("rollback_on_h2d_ownership_failure", False))
            and bool(rollback.get("rollback_on_queue_stall_regression", False)),
            "native_data_pipeline_training_review_missing_rollback",
        ),
        _validation(
            "manual_review_blocks_canary_auto",
            bool(review.get("manual_review_required", False))
            and review.get("allowed_initial_modes") == ["off", "observe"]
            and review.get("blocked_modes_until_review") == ["canary", "auto"],
            "native_data_pipeline_training_review_allows_dispatch_before_review",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(review.get("runtime_dispatch_ready", True))
            and not bool(review.get("native_dispatch_allowed", True))
            and not bool(review.get("training_path_enabled", True)),
            "native_data_pipeline_training_review_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(policy_report.get("training_path_enabled", True))
            and not bool(policy_report.get("default_behavior_changed", True)),
            "native_data_pipeline_training_review_changed_default_behavior",
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


__all__ = ["build_native_data_pipeline_training_integration_review_scorecard"]
