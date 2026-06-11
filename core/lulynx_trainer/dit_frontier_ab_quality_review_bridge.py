"""Shared quality-review bridge for DiT frontier A/B result ingestion."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .cdm_qta_lora_ab_review import build_cdm_qta_lora_quality_review_decision
from .diffcr_ab_review import build_diffcr_quality_review_decision
from .dit_blockskip_ab_review import build_dit_blockskip_quality_review_decision
from .dit_compute_reducer_quality_review_decision import build_dit_compute_reducer_quality_review_decision
from .dit_local_window_attention_ab_review import build_local_window_attention_quality_review_decision
from .sra2_haste_ab_review import build_sra2_haste_quality_review_decision
from .tlora_ab_quality_review_decision import build_tlora_ab_quality_review_decision


def build_dit_frontier_ab_quality_review_bridge(
    *,
    feature_id: str,
    result_ingestion_bridge: Mapping[str, Any],
    quality_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    feature = _feature(feature_id)
    bridge = dict(result_ingestion_bridge)
    review = dict(quality_review or {})
    child_ingestion = dict(bridge.get("child_result_ingestion") or {})
    blockers: list[str] = []

    if bridge.get("scorecard") != "dit_frontier_ab_result_ingestion_bridge_v0":
        blockers.append("unexpected_result_ingestion_bridge")
    if not bool(bridge.get("result_ingestion_bridge_ready", bridge.get("ok", False))):
        blockers.append("result_ingestion_bridge_not_ready")
    if _feature(bridge.get("feature_id")) != feature:
        blockers.append("feature_id_mismatch")
    if not child_ingestion:
        blockers.append("child_result_ingestion_missing")
    if _unsafe_flags(bridge, child_ingestion, review):
        blockers.append("unsafe_quality_bridge_input_flag")

    builder = _builder(feature)
    if builder is None:
        blockers.append("unsupported_feature_id")

    child: dict[str, Any] = {}
    if not blockers and builder is not None:
        child = builder(child_ingestion, review)
        if not bool(child.get("quality_review_ready", child.get("ok", False))):
            blockers.append("feature_quality_review_not_ready")
        if _unsafe_flags(child):
            blockers.append("unsafe_feature_quality_review_flag")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_frontier_ab_quality_review_bridge_v0",
        "ok": ready,
        "quality_review_bridge_ready": ready,
        "feature_id": feature,
        "child_scorecard": str(child.get("scorecard") or ""),
        "child_quality_review": child,
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
            "prepare feature-specific default-off rollout proposal"
            if ready
            else "complete bridged result ingestion and signed quality review before rollout"
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


def _sra2(result_ingestion: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    return build_sra2_haste_quality_review_decision(result_ingestion=result_ingestion, quality_review=review)


def _cdm(result_ingestion: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    return build_cdm_qta_lora_quality_review_decision(result_ingestion=result_ingestion, quality_review=review)


def _diffcr(result_ingestion: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    return build_diffcr_quality_review_decision(result_ingestion=result_ingestion, quality_review=review)


def _blockskip(result_ingestion: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    return build_dit_blockskip_quality_review_decision(result_ingestion=result_ingestion, quality_review=review)


def _local_window(result_ingestion: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    return build_local_window_attention_quality_review_decision(result_ingestion, review)


def _compute_reducer(result_ingestion: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    return build_dit_compute_reducer_quality_review_decision(result_ingestion=result_ingestion, quality_review=review)


def _tlora(result_ingestion: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    return build_tlora_ab_quality_review_decision(result_ingestion=result_ingestion, quality_review=review)


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


__all__ = ["build_dit_frontier_ab_quality_review_bridge"]
