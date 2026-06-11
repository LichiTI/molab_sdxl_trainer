"""Report-only observe manifest for async checkpoint writer candidates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_async_checkpoint_writer_scorecard import build_async_checkpoint_writer_scorecard


FEATURE = "async_checkpoint_writer"
MANIFEST_KIND = "async_checkpoint_writer_observe_manifest_v0"


def build_async_checkpoint_writer_observe_manifest_scorecard(
    *,
    writer_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    """Build a runtime-shaped manifest without wiring trainer checkpoint saves."""

    mode = _normalize_mode(native_training_mode)
    writer = dict(writer_report or build_async_checkpoint_writer_scorecard())
    decision = _route_decision(writer, mode)
    manifest = _manifest(writer, decision, mode)
    validations = _validations(writer, decision, manifest, mode)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_async_checkpoint_writer_observe_manifest_scorecard_v0",
        "gate": "p6m_async_checkpoint_writer_observe_manifest",
        "ok": ready,
        "promotion_ready": ready,
        "observe_manifest_ready": ready,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "experimental_only": True,
        "feature": FEATURE,
        "manifest_kind": MANIFEST_KIND,
        "native_training_mode": mode,
        "route_decision": decision,
        "manifest": manifest,
        "writer_summary": dict(writer.get("summary") or {}),
        "validations": validations,
        "summary": {
            "observe_manifest_ready": ready,
            "decision": decision.get("decision"),
            "reason": decision.get("reason"),
            "candidate_recorded": bool(manifest.get("candidate_recorded", False)),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit review before wiring trainer async checkpoint writer"
            if ready
            else "fix async checkpoint writer observe manifest blockers"
        ),
        "notes": [
            "Observe mode records an async checkpoint writer candidate but never dispatches it.",
            "Trainer checkpoint save/load paths remain authoritative and unchanged.",
            "Canary and auto modes remain blocked until an explicit integration review.",
        ],
    }


def _route_decision(writer: Mapping[str, Any], mode: str) -> dict[str, Any]:
    writer_ready = bool(writer.get("promotion_ready", False))
    summary = writer.get("summary") if isinstance(writer.get("summary"), Mapping) else {}
    proof_ready = bool(summary.get("atomic_commit_ok", False)) and bool(summary.get("parity_ok", False))
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
        candidate = False
    elif not writer_ready or not proof_ready:
        decision = "fallback"
        reason = "async_checkpoint_writer_not_ready"
        candidate = False
    elif mode == "observe":
        decision = "would_select_async_checkpoint_writer_observe_but_dispatch_disabled"
        reason = "observe_mode_records_candidate_only"
        candidate = True
    else:
        decision = "blocked_before_async_checkpoint_writer_canary"
        reason = "async_checkpoint_writer_integration_review_required"
        candidate = True
    return {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "feature": FEATURE,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "request_supported": bool(writer_ready and proof_ready),
        "candidate_recorded": candidate,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "missing_before_dispatch": [
            "manual_integration_review",
            "trainer_checkpoint_hook_contract",
            "resume_parity_matrix",
            "rollback_manifest",
        ],
    }


def _manifest(writer: Mapping[str, Any], decision: Mapping[str, Any], mode: str) -> dict[str, Any]:
    summary = writer.get("summary") if isinstance(writer.get("summary"), Mapping) else {}
    capabilities = writer.get("capabilities") if isinstance(writer.get("capabilities"), Mapping) else {}
    return {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "feature": FEATURE,
        "native_training_mode": mode,
        "candidate_recorded": bool(decision.get("candidate_recorded", False)),
        "writer_summary": {
            "async_writer_ready": bool(summary.get("async_writer_ready", False)),
            "native_copy_used": bool(summary.get("native_copy_used", False)),
            "submit_nonblocking_ok": bool(summary.get("submit_nonblocking_ok", False)),
            "atomic_commit_ok": bool(summary.get("atomic_commit_ok", False)),
            "parity_ok": bool(summary.get("parity_ok", False)),
        },
        "capabilities": {
            "native_importable": bool(capabilities.get("native_importable", False)),
            "native_stream_copy_entrypoint": bool(capabilities.get("native_stream_copy_entrypoint", False)),
            "python_fallback": bool(capabilities.get("python_fallback", True)),
        },
        "artifact_contract": {
            "atomic_temp_commit": True,
            "final_wait_required_before_resume": True,
            "metadata_bytes_supported": True,
            "artifact_copy_supported": True,
            "checksum_parity_required": True,
            "partial_temp_cleanup_required": True,
        },
        "rollback_policy": {
            "fallback_writer": "standardcore_python_sync_checkpoint_save",
            "fallback_authoritative": True,
            "rollback_on_checksum_mismatch": True,
            "rollback_on_incomplete_job": True,
            "rollback_on_resume_probe_failure": True,
            "rollback_on_temp_leftovers": True,
        },
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "runtime_incompatibilities": [
            "trainer_checkpoint_hook_unreviewed",
            "distributed_checkpoint_coordination_unreviewed",
            "resume_atomicity_unreviewed",
            "partial_artifact_cleanup_unreviewed",
        ],
        "audit_fields": [
            "native_training_mode",
            "run_id",
            "checkpoint_step",
            "artifact_kind",
            "output_dir",
            "writer_provider",
            "route_decision",
            "fallback_reason",
            "rollback_reason",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
    }


def _validations(
    writer: Mapping[str, Any],
    decision: Mapping[str, Any],
    manifest: Mapping[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    rollback = manifest.get("rollback_policy") if isinstance(manifest.get("rollback_policy"), Mapping) else {}
    return [
        _validation(
            "p6g_async_checkpoint_writer_ready",
            bool(writer.get("promotion_ready", False)),
            "async_checkpoint_writer_scorecard_missing",
        ),
        _validation(
            "observe_route_decision_ready",
            mode != "off"
            and decision.get("decision")
            in {
                "would_select_async_checkpoint_writer_observe_but_dispatch_disabled",
                "blocked_before_async_checkpoint_writer_canary",
            },
            "async_checkpoint_writer_observe_route_decision_missing",
        ),
        _validation(
            "manifest_records_checkpoint_contract",
            bool(manifest.get("artifact_contract")) and bool(manifest.get("audit_fields")),
            "async_checkpoint_writer_observe_manifest_contract_missing",
        ),
        _validation(
            "fallback_and_rollback_present",
            bool(rollback.get("fallback_authoritative", False))
            and bool(rollback.get("rollback_on_checksum_mismatch", False))
            and bool(rollback.get("rollback_on_incomplete_job", False))
            and bool(rollback.get("rollback_on_temp_leftovers", False)),
            "async_checkpoint_writer_observe_manifest_missing_rollback",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(decision.get("runtime_dispatch_ready", True))
            and not bool(manifest.get("native_dispatch_allowed", True))
            and not bool(manifest.get("training_path_enabled", True)),
            "async_checkpoint_writer_observe_manifest_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(writer.get("training_path_enabled", True))
            and not bool(writer.get("default_behavior_changed", True)),
            "async_checkpoint_writer_observe_manifest_changed_default_behavior",
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


__all__ = ["build_async_checkpoint_writer_observe_manifest_scorecard"]
