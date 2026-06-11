"""Bridge shared DiT frontier A/B result bundles into existing ingestion gates."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .cdm_qta_lora_ab_review import build_cdm_qta_lora_ab_result_ingestion
from .diffcr_ab_review import build_diffcr_ab_result_ingestion
from .dit_blockskip_ab_review import build_dit_blockskip_ab_result_ingestion
from .dit_compute_reducer_ab_result_ingestion import build_dit_compute_reducer_ab_result_ingestion
from .dit_local_window_attention_ab_review import ingest_local_window_attention_ab_results
from .sra2_haste_ab_review import build_sra2_haste_ab_result_ingestion
from .tlora_ab_dispatch_result_ingestion import build_tlora_ab_dispatch_result_ingestion


_RESULT_READY_KEYS = (
    "ab_result_ingestion_ready",
    "result_ingestion_ready",
)


def build_dit_frontier_ab_result_ingestion_bridge(
    *,
    feature_id: str,
    evidence_package: Mapping[str, Any],
    result_bundle_summaries: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    feature = _feature(feature_id)
    package = dict(evidence_package)
    bundle = dict(result_bundle_summaries)
    summaries = [dict(item) for item in bundle.get("result_summaries", ()) if isinstance(item, Mapping)]
    blockers: list[str] = []

    if bundle.get("scorecard") != "dit_frontier_ab_result_summaries_v0":
        blockers.append("unexpected_result_bundle_summaries")
    if not bool(bundle.get("result_summaries_ready", bundle.get("ok", False))):
        blockers.append("result_bundle_summaries_not_ready")
    if _feature(bundle.get("feature_id")) != feature:
        blockers.append("feature_id_mismatch")
    if _unsafe_flags(package, bundle):
        blockers.append("unsafe_bridge_input_flag")
    if not summaries:
        blockers.append("result_summaries_missing")
    builder = _builder(feature)
    if builder is None:
        blockers.append("unsupported_feature_id")

    child: dict[str, Any] = {}
    if not blockers and builder is not None:
        child = builder(package, summaries, thresholds)
        if not _child_ready(child):
            blockers.append("feature_result_ingestion_not_ready")
        if _unsafe_flags(child):
            blockers.append("unsafe_feature_result_ingestion_flag")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_frontier_ab_result_ingestion_bridge_v0",
        "ok": ready,
        "result_ingestion_bridge_ready": ready,
        "feature_id": feature,
        "child_scorecard": str(child.get("scorecard") or ""),
        "child_result_ingestion": child,
        "result_summary_count": len(summaries),
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
            "prepare signed feature quality review using the bridged result ingestion"
            if ready
            else "complete result bundle summaries and feature evidence before bridge ingestion"
        ),
    }


def _builder(
    feature: str,
) -> Callable[[Mapping[str, Any], list[dict[str, Any]], Mapping[str, Any] | None], dict[str, Any]] | None:
    return {
        "sra2_haste": _sra2,
        "cdm_qta_lora": _cdm,
        "diffcr": _diffcr,
        "dit_blockskip": _blockskip,
        "dit_local_window_attention": _local_window,
        "dit_compute_reducer": _compute_reducer,
        "tlora_ab": _tlora,
    }.get(feature)


def _sra2(package: Mapping[str, Any], summaries: list[dict[str, Any]], thresholds: Mapping[str, Any] | None) -> dict:
    return build_sra2_haste_ab_result_ingestion(
        evidence_package=package,
        result_summaries=summaries,
        thresholds=thresholds,
    )


def _cdm(package: Mapping[str, Any], summaries: list[dict[str, Any]], thresholds: Mapping[str, Any] | None) -> dict:
    return build_cdm_qta_lora_ab_result_ingestion(
        evidence_package=package,
        result_summaries=summaries,
        thresholds=thresholds,
    )


def _diffcr(package: Mapping[str, Any], summaries: list[dict[str, Any]], thresholds: Mapping[str, Any] | None) -> dict:
    return build_diffcr_ab_result_ingestion(
        evidence_package=package,
        result_summaries=summaries,
        thresholds=thresholds,
    )


def _blockskip(
    package: Mapping[str, Any],
    summaries: list[dict[str, Any]],
    thresholds: Mapping[str, Any] | None,
) -> dict:
    return build_dit_blockskip_ab_result_ingestion(
        evidence_package=package,
        result_summaries=summaries,
        thresholds=thresholds,
    )


def _local_window(
    package: Mapping[str, Any],
    summaries: list[dict[str, Any]],
    thresholds: Mapping[str, Any] | None,
) -> dict:
    evidence = dict(package)
    if thresholds:
        evidence["threshold_policy"] = dict(thresholds)
    return ingest_local_window_attention_ab_results(evidence, summaries)


def _compute_reducer(
    package: Mapping[str, Any],
    summaries: list[dict[str, Any]],
    thresholds: Mapping[str, Any] | None,
) -> dict:
    return build_dit_compute_reducer_ab_result_ingestion(
        evidence_package=package,
        result_summaries=summaries,
        thresholds=thresholds,
    )


def _tlora(package: Mapping[str, Any], summaries: list[dict[str, Any]], thresholds: Mapping[str, Any] | None) -> dict:
    return build_tlora_ab_dispatch_result_ingestion(
        dispatch_manifest=package,
        case_results=summaries,
        thresholds=thresholds,
    )


def _child_ready(child: Mapping[str, Any]) -> bool:
    return bool(child.get("ok", False)) and any(bool(child.get(key, False)) for key in _RESULT_READY_KEYS)


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
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = ["build_dit_frontier_ab_result_ingestion_bridge"]
