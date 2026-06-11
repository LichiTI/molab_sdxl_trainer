"""Shared default-off rollout proposal bridge for DiT frontier A/B routes."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .cdm_qta_lora_ab_review import build_cdm_qta_lora_default_off_rollout_proposal
from .diffcr_ab_review import build_diffcr_default_off_rollout_proposal
from .dit_blockskip_ab_review import build_dit_blockskip_default_off_rollout_proposal
from .dit_compute_reducer_default_off_rollout_proposal import (
    build_dit_compute_reducer_default_off_rollout_proposal,
)
from .dit_local_window_attention_ab_review import build_local_window_attention_default_off_rollout_proposal
from .sra2_haste_ab_review import build_sra2_haste_default_off_rollout_proposal
from .tlora_ab_default_off_rollout_proposal import build_tlora_ab_default_off_rollout_proposal


def build_dit_frontier_ab_default_off_rollout_bridge(
    *,
    feature_id: str,
    quality_review_bridge: Mapping[str, Any],
    rollout_proposal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    feature = _feature(feature_id)
    bridge = dict(quality_review_bridge)
    proposal = dict(rollout_proposal or {})
    child_quality = dict(bridge.get("child_quality_review") or {})
    blockers: list[str] = []

    if bridge.get("scorecard") != "dit_frontier_ab_quality_review_bridge_v0":
        blockers.append("unexpected_quality_review_bridge")
    if not bool(bridge.get("quality_review_bridge_ready", bridge.get("ok", False))):
        blockers.append("quality_review_bridge_not_ready")
    if _feature(bridge.get("feature_id")) != feature:
        blockers.append("feature_id_mismatch")
    if not child_quality:
        blockers.append("child_quality_review_missing")
    if _unsafe_flags(bridge, child_quality, proposal):
        blockers.append("unsafe_rollout_bridge_input_flag")

    builder = _builder(feature)
    if builder is None:
        blockers.append("unsupported_feature_id")

    child: dict[str, Any] = {}
    if not blockers and builder is not None:
        child = builder(child_quality, proposal)
        if not bool(child.get("rollout_proposal_ready", child.get("ok", False))):
            blockers.append("feature_rollout_proposal_not_ready")
        if _unsafe_flags(child):
            blockers.append("unsafe_feature_rollout_proposal_flag")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_frontier_ab_default_off_rollout_bridge_v0",
        "ok": ready,
        "default_off_rollout_bridge_ready": ready,
        "feature_id": feature,
        "child_scorecard": str(child.get("scorecard") or ""),
        "child_rollout_proposal": child,
        **_safe_flags(),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_execution_started": False,
        "ab_execution_completed": False,
        "ab_dispatch_executed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "trainer_wiring_executed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "hold default-off rollout until explicit runtime activation review"
            if ready
            else "complete signed quality review and rollout proposal before runtime activation review"
        ),
    }


def _builder(feature: str) -> Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]] | None:
    return {
        "sra2_haste": _sra2,
        "cdm_qta_lora": _cdm,
        "diffcr": _diffcr,
        "dit_blockskip": _blockskip,
        "dit_local_window_attention": _local_window,
        "dit_compute_reducer": _compute_reducer,
        "tlora_ab": _tlora,
    }.get(feature)


def _sra2(quality_decision: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    return build_sra2_haste_default_off_rollout_proposal(
        quality_decision=quality_decision,
        rollout_proposal=proposal,
    )


def _cdm(quality_decision: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    return build_cdm_qta_lora_default_off_rollout_proposal(
        quality_decision=quality_decision,
        rollout_proposal=proposal,
    )


def _diffcr(quality_decision: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    return build_diffcr_default_off_rollout_proposal(
        quality_decision=quality_decision,
        rollout_proposal=proposal,
    )


def _blockskip(quality_decision: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    return build_dit_blockskip_default_off_rollout_proposal(
        quality_decision=quality_decision,
        rollout_proposal=proposal,
    )


def _local_window(quality_decision: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    return build_local_window_attention_default_off_rollout_proposal(quality_decision, proposal)


def _compute_reducer(quality_decision: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    return build_dit_compute_reducer_default_off_rollout_proposal(
        quality_decision=quality_decision,
        rollout_proposal=proposal,
    )


def _tlora(quality_decision: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    return build_tlora_ab_default_off_rollout_proposal(
        quality_decision=quality_decision,
        rollout_proposal=proposal,
    )


def _feature(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _safe_flags() -> dict[str, bool]:
    return {
        "ab_execution_allowed": False,
        "ab_dispatch_allowed": False,
        "trainer_wiring_allowed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "default_rollout_allowed",
        "auto_rollout_allowed",
        "runtime_activation_enabled",
        "request_fields_emitted",
        "request_adapter_registered",
        "trainer_wiring_allowed",
        "trainer_wiring_executed",
        "ab_execution_allowed",
        "ab_execution_started",
        "ab_execution_completed",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
        "request_payload_materialized",
        "request_payload_submitted",
        "execution_job_created",
        "training_job_created",
        "job_record_written",
        "queue_enqueued",
        "training_runtime_started",
        "training_process_started",
        "operator_training_launch_allowed",
        "operator_training_launch_executed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = ["build_dit_frontier_ab_default_off_rollout_bridge"]
