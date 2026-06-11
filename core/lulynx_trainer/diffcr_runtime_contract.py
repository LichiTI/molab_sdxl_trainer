"""Default-off runtime contract for DiffCR token compression."""

from __future__ import annotations

from typing import Any, Mapping


def build_diffcr_runtime_contract(
    *,
    cached_token_ab: Mapping[str, Any],
    runtime_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = dict(cached_token_ab)
    policy = dict(runtime_policy or {})
    plan = dict(report.get("plan") or {})
    cache = dict(report.get("cache_replay") or {})
    blockers: list[str] = []

    if report.get("scorecard") != "diffcr_cached_token_ab_replay_v0":
        blockers.append("unexpected_cached_token_ab")
    if not bool(report.get("cached_token_ab_ready", report.get("ok", False))):
        blockers.append("cached_token_ab_not_ready")
    if _unsafe_flags(report, policy):
        blockers.append("unsafe_child_flag")
    if not bool(plan.get("enabled", False)):
        blockers.append("compression_plan_not_enabled")
    if int(plan.get("token_count") or 0) != int(cache.get("token_count") or 0):
        blockers.append("token_count_mismatch")
    if int(plan.get("compressed_count") or 0) <= 0:
        blockers.append("compressed_count_missing")
    if float(plan.get("estimated_attention_fraction", 1.0) or 1.0) >= 1.0:
        blockers.append("attention_fraction_not_reduced")
    if not bool(policy.get("shape_fingerprint_required", False)):
        blockers.append("shape_fingerprint_requirement_missing")
    if not bool(policy.get("assignment_determinism_required", False)):
        blockers.append("assignment_determinism_requirement_missing")
    if not bool(policy.get("expand_original_shape_required", False)):
        blockers.append("expand_original_shape_requirement_missing")
    if float(cache.get("valid_token_fraction", 1.0) or 1.0) < 1.0 and not bool(
        policy.get("padding_mask_preservation_required", False)
    ):
        blockers.append("padding_mask_preservation_requirement_missing")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "diffcr_runtime_contract_v0",
        "ok": ready,
        "runtime_contract_ready": ready,
        "runtime_compression_allowed": False,
        "family": str(cache.get("family") or ""),
        "token_grid": list(cache.get("token_grid") or []),
        "token_count": int(cache.get("token_count") or 0),
        "compressed_count": int(plan.get("compressed_count") or 0),
        "estimated_attention_fraction": float(plan.get("estimated_attention_fraction", 1.0) or 1.0),
        "shape_fingerprint": {
            "latent_shape": list(cache.get("latent_shape") or []),
            "token_grid": list(cache.get("token_grid") or []),
            "token_count": int(cache.get("token_count") or 0),
            "hidden_size": int(cache.get("hidden_size") or 0),
        },
        "runtime_requirements": {
            "shape_fingerprint_required": bool(policy.get("shape_fingerprint_required", False)),
            "assignment_determinism_required": bool(policy.get("assignment_determinism_required", False)),
            "expand_original_shape_required": bool(policy.get("expand_original_shape_required", False)),
            "padding_mask_preservation_required": bool(policy.get("padding_mask_preservation_required", False)),
        },
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run DiffCR expand/assignment parity audit before trainer wiring"
            if ready
            else "complete DiffCR runtime compression contract requirements"
        ),
    }


def build_diffcr_expand_parity_audit(
    *,
    runtime_contract: Mapping[str, Any],
    observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(runtime_contract)
    report = dict(observed or {})
    blockers: list[str] = []

    if contract.get("scorecard") != "diffcr_runtime_contract_v0":
        blockers.append("unexpected_runtime_contract")
    if not bool(contract.get("runtime_contract_ready", contract.get("ok", False))):
        blockers.append("runtime_contract_not_ready")
    if _unsafe_flags(contract, report):
        blockers.append("unsafe_child_flag")
    if int(report.get("token_count") or 0) != int(contract.get("token_count") or 0):
        blockers.append("token_count_mismatch")
    if int(report.get("compressed_count") or 0) != int(contract.get("compressed_count") or 0):
        blockers.append("compressed_count_mismatch")
    if not bool(report.get("assignment_deterministic", False)):
        blockers.append("assignment_not_deterministic")
    if not bool(report.get("expanded_shape_matches_original", False)):
        blockers.append("expanded_shape_mismatch")
    if not bool(report.get("shape_fingerprint_matched", False)):
        blockers.append("shape_fingerprint_not_matched")
    if contract["runtime_requirements"]["padding_mask_preservation_required"] and not bool(
        report.get("padding_mask_preserved", False)
    ):
        blockers.append("padding_mask_not_preserved")
    if float(report.get("max_expand_abs_diff", 0.0) or 0.0) > float(
        report.get("max_allowed_expand_abs_diff", 1e-6) or 1e-6
    ):
        blockers.append("expand_abs_diff_above_threshold")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "diffcr_expand_parity_audit_v0",
        "ok": ready,
        "expand_parity_ready": ready,
        "runtime_contract_ready": bool(contract.get("runtime_contract_ready", contract.get("ok", False))),
        "token_count": int(report.get("token_count") or 0),
        "compressed_count": int(report.get("compressed_count") or 0),
        "max_expand_abs_diff": float(report.get("max_expand_abs_diff", 0.0) or 0.0),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "feed DiffCR expand parity evidence into A/B quality review"
            if ready
            else "fix DiffCR expand parity blockers before trainer wiring"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "request_fields_emitted",
        "request_adapter_registered",
        "ab_dispatch_allowed",
        "training_launch_allowed",
        "training_launch_executed",
        "runs_dispatched",
        "trainer_wiring_allowed",
        "runtime_activation_enabled",
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "build_diffcr_expand_parity_audit",
    "build_diffcr_runtime_contract",
]
