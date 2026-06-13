"""Fail-closed Newbie BlockSkip quality drift/render review."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REPORT = "bubble_newbie_blockskip_quality_drift_review_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
FAMILY = "newbie"
SUPPORTED_QUALITY_REVIEW_TYPES = {"render_review", "tensor_drift", "loss_curve_ab"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _loss_blockers(quality_stability_review: Mapping[str, Any]) -> list[str]:
    summary = _mapping(quality_stability_review.get("summary"))
    return [str(item) for item in _sequence(summary.get("blockers")) if str(item)]


def build_newbie_blockskip_quality_drift_review(
    *,
    quality_stability_review: Mapping[str, Any],
    compute_bound_policy: Mapping[str, Any],
    quality_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Review quality drift evidence without running training, CUDA, or render."""

    stability_summary = _mapping(quality_stability_review.get("summary"))
    policy = _mapping(compute_bound_policy.get("policy"))
    evidence = _mapping(quality_evidence)
    review_type = str(evidence.get("review_type") or "").strip()
    throughput_ready = bool(stability_summary.get("throughput_repeat_ready"))
    loss_quality_ready = bool(stability_summary.get("loss_quality_ready"))
    policy_ready = bool(compute_bound_policy.get("policy_ready"))
    render_pair_count = _safe_int(evidence.get("render_pair_count"))
    quality_drift = _safe_float(evidence.get("quality_drift"), 0.0)
    max_quality_drift = _safe_float(evidence.get("max_quality_drift"), 0.01)
    loss_curve_delta = _safe_float(evidence.get("loss_curve_delta"), 0.0)
    max_loss_curve_delta = _safe_float(evidence.get("max_loss_curve_delta"), 0.25)
    evidence_present = bool(evidence)
    quality_drift_ready = "quality_drift" in evidence
    loss_curve_ready = "loss_curve_delta" in evidence
    shape_stable = evidence.get("shape_stable") is True
    disabled_parity_ok = evidence.get("disabled_parity_ok") is True
    checkpoint_semantics_ok = evidence.get("checkpoint_semantics_ok") is True
    residual_reuse_parity_ok = evidence.get("residual_reuse_parity_ok") is True
    reviewer_present = bool(str(evidence.get("reviewer") or "").strip())
    digest_present = bool(str(evidence.get("artifact_digest") or evidence.get("result_digest") or "").strip())
    unsafe_flags = [
        key
        for key in (
            "release_claim_allowed",
            "safe_to_auto_start",
            "trainer_wiring_allowed",
            "runtime_activation_enabled",
            "default_behavior_changed",
        )
        if evidence.get(key) is True
    ]

    blockers: list[str] = []
    if not throughput_ready:
        blockers.append("blockskip_throughput_repeat_not_ready")
    if not loss_quality_ready:
        blockers.append("loss_quality_gate_not_ready")
    if not policy_ready:
        blockers.append("compute_bound_policy_not_ready")
    if policy.get("compute_bound_exception_allowed") is True:
        blockers.append("unexpected_compute_bound_exception_allowed")
    if not evidence_present:
        blockers.append("quality_drift_or_render_review_missing")
    if review_type and review_type not in SUPPORTED_QUALITY_REVIEW_TYPES:
        blockers.append("unsupported_quality_review_type")
    if review_type == "render_review" and render_pair_count <= 0:
        blockers.append("render_pair_count_missing")
    if not quality_drift_ready and not loss_curve_ready:
        blockers.append("quality_drift_measurement_missing")
    if quality_drift_ready and quality_drift > max_quality_drift:
        blockers.append("quality_drift_above_threshold")
    if loss_curve_ready and abs(loss_curve_delta) > max_loss_curve_delta:
        blockers.append("loss_curve_delta_above_threshold")
    for key, ok in (
        ("shape_stable", shape_stable),
        ("disabled_parity_ok", disabled_parity_ok),
        ("checkpoint_semantics_ok", checkpoint_semantics_ok),
        ("residual_reuse_parity_ok", residual_reuse_parity_ok),
    ):
        if evidence_present and not ok:
            blockers.append(f"{key}_missing")
    if evidence_present and not reviewer_present:
        blockers.append("reviewer_missing")
    if evidence_present and not digest_present:
        blockers.append("artifact_digest_missing")
    blockers.extend(f"unsafe_quality_evidence_flag:{key}" for key in unsafe_flags)

    blocker_set = []
    for blocker in [*_loss_blockers(quality_stability_review), *blockers]:
        if blocker not in blocker_set:
            blocker_set.append(blocker)

    evidence_ready = evidence_present and not blockers
    status = (
        "newbie_blockskip_quality_drift_review_ready_nonrelease"
        if evidence_ready
        else "newbie_blockskip_quality_drift_review_blocked"
    )
    return {
        "report": REPORT,
        "schema_version": 1,
        "roadmap": ROADMAP,
        "family": FAMILY,
        "candidate": "dit_compute_reducer:blockskip_skip25",
        "status": status,
        "review_ready": evidence_ready,
        "quality_evidence_present": evidence_present,
        "classification": "blockskip_quality_drift_or_render_review_fail_closed",
        "summary": {
            "completed_seed_pair_count": _safe_int(quality_stability_review.get("completed_seed_pair_count")),
            "throughput_repeat_ready": throughput_ready,
            "loss_quality_ready": loss_quality_ready,
            "compute_bound_policy_ready": policy_ready,
            "quality_review_type": review_type,
            "render_pair_count": render_pair_count,
            "quality_drift_ready": quality_drift_ready,
            "quality_drift": _round(quality_drift),
            "max_quality_drift": _round(max_quality_drift),
            "loss_curve_ready": loss_curve_ready,
            "loss_curve_delta": _round(loss_curve_delta),
            "max_loss_curve_delta": _round(max_loss_curve_delta),
            "shape_stable": shape_stable,
            "disabled_parity_ok": disabled_parity_ok,
            "checkpoint_semantics_ok": checkpoint_semantics_ok,
            "residual_reuse_parity_ok": residual_reuse_parity_ok,
            "blocker_count": len(blocker_set),
            "blockers": blocker_set,
        },
        "quality_contract": {
            "supported_review_types": sorted(SUPPORTED_QUALITY_REVIEW_TYPES),
            "requires_reviewer": True,
            "requires_artifact_digest": True,
            "requires_shape_stability": True,
            "requires_disabled_parity": True,
            "requires_checkpoint_semantics": True,
            "requires_residual_reuse_parity": True,
            "release_claim_allowed_after_success": False,
        },
        "next_actions": [
            {
                "id": "run_blockskip_quality_drift_ab_or_render_review",
                "kind": "quality_gate_followup",
                "requires_gpu_heavy_run": True,
                "release_claim_allowed": False,
                "release_claim_allowed_after_success": False,
                "safe_to_auto_start": False,
            },
            {
                "id": "keep_blockskip_default_off_until_signed_quality_review",
                "kind": "release_gate_policy",
                "requires_gpu_heavy_run": False,
                "release_claim_allowed": False,
                "release_claim_allowed_after_success": False,
                "safe_to_auto_start": False,
            },
        ],
        "release_claim": {
            "eligible": False,
            "scope": "not_eligible",
            "reason": "BlockSkip quality drift/render review is required before any compute-bound exception or release claim",
        },
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_gpu_heavy": True,
    }


__all__ = [
    "FAMILY",
    "REPORT",
    "ROADMAP",
    "SUPPORTED_QUALITY_REVIEW_TYPES",
    "build_newbie_blockskip_quality_drift_review",
]
