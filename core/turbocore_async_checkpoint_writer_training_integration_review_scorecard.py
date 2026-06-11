"""Review-only gate for trainer async checkpoint writer integration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_async_checkpoint_writer_observe_manifest_scorecard import (
    build_async_checkpoint_writer_observe_manifest_scorecard,
)


FEATURE = "async_checkpoint_writer_trainer_integration"
REVIEW_KIND = "async_checkpoint_writer_training_integration_review_v0"


def build_async_checkpoint_writer_training_integration_review_scorecard(
    *,
    observe_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    """Build a manual review package without changing trainer checkpoint saves."""

    mode = _normalize_mode(native_training_mode)
    observe = dict(
        observe_report
        or build_async_checkpoint_writer_observe_manifest_scorecard(
            native_training_mode=mode,
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
        "scorecard": "turbocore_async_checkpoint_writer_training_integration_review_scorecard_v0",
        "gate": "p6o_async_checkpoint_writer_training_integration_review",
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
            "requires explicit review before wiring trainer async checkpoint writer integration"
            if ready
            else "fix async checkpoint writer integration review gate blockers"
        ),
        "notes": [
            "This gate prepares the trainer checkpoint integration review only.",
            "Trainer _save_model, _save_state, _load_state, manifest, and pruning behavior remain unchanged.",
            "Canary and auto modes stay blocked until an explicit integration review approves wiring.",
        ],
    }


def _review_package(observe: Mapping[str, Any], mode: str) -> dict[str, Any]:
    manifest = observe.get("manifest") if isinstance(observe.get("manifest"), Mapping) else {}
    rollback = manifest.get("rollback_policy") if isinstance(manifest.get("rollback_policy"), Mapping) else {}
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
        "trainer_hook_contract": {
            "rank0_only_required": True,
            "save_model_hook": "core.lulynx_trainer.trainer.Trainer._save_model",
            "save_state_hook": "core.lulynx_trainer.trainer.Trainer._save_state",
            "load_state_hook": "core.lulynx_trainer.trainer.Trainer._load_state",
            "manifest_hook": "core.lulynx_trainer.trainer.Trainer._write_run_manifest",
            "event_hook": "core.lulynx_trainer.trainer.Trainer._emit_runtime_event",
            "retention_hook": "core.lulynx_trainer.trainer.Trainer._prune_saved_artifacts",
        },
        "checkpoint_lifecycle_contract": {
            "model_artifact_atomic_before_manifest": True,
            "state_artifact_atomic_before_resume": True,
            "final_wait_required_before_resume": True,
            "retention_after_completed_jobs_only": True,
            "save_to_and_output_dir_paths_preserved": True,
            "merged_export_out_of_scope_until_review": True,
            "huggingface_upload_out_of_scope_until_review": True,
        },
        "resume_parity_matrix": {
            "model_artifact_digest_required": True,
            "state_artifact_digest_required": True,
            "state_load_uses_safe_torch_load": True,
            "rng_state_roundtrip_required": True,
            "optimizer_scheduler_state_roundtrip_required": True,
            "turbocore_update_state_roundtrip_required": True,
        },
        "source_observe_contract": {
            "fallback_writer": rollback.get("fallback_writer", "standardcore_python_sync_checkpoint_save"),
            "fallback_authoritative": bool(rollback.get("fallback_authoritative", False)),
            "checksum_rollback": bool(rollback.get("rollback_on_checksum_mismatch", False)),
            "incomplete_job_rollback": bool(rollback.get("rollback_on_incomplete_job", False)),
            "temp_leftover_rollback": bool(rollback.get("rollback_on_temp_leftovers", False)),
        },
        "rollback_policy": {
            "fallback_backend": "standardcore_trainer_sync_checkpoint_save",
            "fallback_authoritative": True,
            "rollback_on_async_job_failure": True,
            "rollback_on_checksum_mismatch": True,
            "rollback_on_resume_probe_failure": True,
            "rollback_on_temp_leftovers": True,
            "rollback_on_retention_conflict": True,
            "rollback_on_distributed_rank_mismatch": True,
        },
        "audit_fields": [
            "native_training_mode",
            "run_id",
            "rank",
            "checkpoint_step",
            "epoch",
            "artifact_kind",
            "save_path",
            "state_path",
            "writer_provider",
            "job_id",
            "route_decision",
            "fallback_reason",
            "rollback_reason",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
    }


def _validations(observe: Mapping[str, Any], review: Mapping[str, Any]) -> list[dict[str, Any]]:
    hook_contract = review.get("trainer_hook_contract") if isinstance(review.get("trainer_hook_contract"), Mapping) else {}
    lifecycle = (
        review.get("checkpoint_lifecycle_contract")
        if isinstance(review.get("checkpoint_lifecycle_contract"), Mapping)
        else {}
    )
    resume = review.get("resume_parity_matrix") if isinstance(review.get("resume_parity_matrix"), Mapping) else {}
    rollback = review.get("rollback_policy") if isinstance(review.get("rollback_policy"), Mapping) else {}
    return [
        _validation(
            "p6m_async_checkpoint_writer_observe_manifest_ready",
            bool(observe.get("observe_manifest_ready", False)),
            "async_checkpoint_writer_observe_manifest_missing",
        ),
        _validation(
            "trainer_hook_contract_present",
            bool(hook_contract.get("rank0_only_required", False))
            and bool(hook_contract.get("save_model_hook"))
            and bool(hook_contract.get("save_state_hook"))
            and bool(hook_contract.get("load_state_hook"))
            and bool(hook_contract.get("retention_hook")),
            "async_checkpoint_writer_trainer_hook_contract_missing",
        ),
        _validation(
            "checkpoint_lifecycle_contract_present",
            bool(lifecycle.get("model_artifact_atomic_before_manifest", False))
            and bool(lifecycle.get("state_artifact_atomic_before_resume", False))
            and bool(lifecycle.get("retention_after_completed_jobs_only", False)),
            "async_checkpoint_writer_lifecycle_contract_missing",
        ),
        _validation(
            "resume_parity_matrix_present",
            bool(resume.get("state_load_uses_safe_torch_load", False))
            and bool(resume.get("rng_state_roundtrip_required", False))
            and bool(resume.get("optimizer_scheduler_state_roundtrip_required", False)),
            "async_checkpoint_writer_resume_parity_matrix_missing",
        ),
        _validation(
            "rollback_manifest_present",
            bool(rollback.get("fallback_authoritative", False))
            and bool(rollback.get("rollback_on_async_job_failure", False))
            and bool(rollback.get("rollback_on_checksum_mismatch", False))
            and bool(rollback.get("rollback_on_resume_probe_failure", False)),
            "async_checkpoint_writer_training_review_missing_rollback",
        ),
        _validation(
            "manual_review_blocks_canary_auto",
            bool(review.get("manual_review_required", False))
            and review.get("allowed_initial_modes") == ["off", "observe"]
            and review.get("blocked_modes_until_review") == ["canary", "auto"],
            "async_checkpoint_writer_training_review_allows_dispatch_before_review",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(review.get("runtime_dispatch_ready", True))
            and not bool(review.get("native_dispatch_allowed", True))
            and not bool(review.get("training_path_enabled", True)),
            "async_checkpoint_writer_training_review_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(observe.get("training_path_enabled", True))
            and not bool(observe.get("default_behavior_changed", True)),
            "async_checkpoint_writer_training_review_changed_default_behavior",
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


__all__ = ["build_async_checkpoint_writer_training_integration_review_scorecard"]
