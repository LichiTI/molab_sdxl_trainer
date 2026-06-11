"""Default-off checkpoint/residual contract for DiT BlockSkip."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_dit_blockskip_checkpoint_contract(
    *,
    blockskip_scorecard: Mapping[str, Any],
    checkpoint_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scorecard = dict(blockskip_scorecard)
    policy = dict(checkpoint_policy or {})
    plan = dict(scorecard.get("plan") or {})
    decisions = [dict(item) for item in plan.get("decisions", ()) if isinstance(item, Mapping)]
    skipped = [item for item in decisions if bool(item.get("skip", False))]
    blockers: list[str] = []

    if scorecard.get("scorecard") != "dit_blockskip_training_spike_v0":
        blockers.append("unexpected_blockskip_scorecard")
    if not bool(scorecard.get("probe_ready", scorecard.get("ok", False))):
        blockers.append("blockskip_probe_not_ready")
    if _unsafe_flags(scorecard, policy):
        blockers.append("unsafe_child_flag")
    if not bool(plan.get("enabled", False)):
        blockers.append("blockskip_plan_not_enabled")
    if not skipped:
        blockers.append("skipped_blocks_missing")
    if not bool(policy.get("residual_reuse_policy_recorded", False)):
        blockers.append("residual_reuse_policy_missing")
    if not bool(policy.get("checkpoint_metadata_required", False)):
        blockers.append("checkpoint_metadata_requirement_missing")
    if not bool(policy.get("resume_recomputes_or_restores_residuals", False)):
        blockers.append("resume_residual_strategy_missing")
    if bool(policy.get("persist_raw_residual_tensors", False)):
        blockers.append("persist_raw_residual_tensors_not_allowed")
    if not bool(policy.get("shape_fingerprint_required", False)):
        blockers.append("shape_fingerprint_requirement_missing")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_blockskip_checkpoint_contract_v0",
        "ok": ready,
        "checkpoint_contract_ready": ready,
        "runtime_checkpoint_integration_allowed": False,
        "skipped_block_indices": [int(item.get("block_index") or 0) for item in skipped],
        "residual_reuse": bool((plan.get("policy") or {}).get("reuse_residual", False)),
        "checkpoint_metadata": {
            "blockskip_policy": dict(plan.get("policy") or {}),
            "skipped_block_indices": [int(item.get("block_index") or 0) for item in skipped],
            "shape_fingerprint_required": True,
            "residual_strategy": "recompute_or_restore_by_policy",
            "persist_raw_residual_tensors": False,
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
            "run BlockSkip resume parity audit before trainer wiring"
            if ready
            else "complete BlockSkip checkpoint/residual metadata contract"
        ),
    }


def build_dit_blockskip_resume_parity_audit(
    *,
    checkpoint_contract: Mapping[str, Any],
    resume_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(checkpoint_contract)
    report = dict(resume_report or {})
    expected = tuple(int(item) for item in contract.get("skipped_block_indices", ()) or ())
    observed = tuple(int(item) for item in report.get("resumed_skipped_block_indices", ()) or ())
    blockers: list[str] = []

    if contract.get("scorecard") != "dit_blockskip_checkpoint_contract_v0":
        blockers.append("unexpected_checkpoint_contract")
    if not bool(contract.get("checkpoint_contract_ready", contract.get("ok", False))):
        blockers.append("checkpoint_contract_not_ready")
    if _unsafe_flags(contract, report):
        blockers.append("unsafe_child_flag")
    if tuple(sorted(expected)) != tuple(sorted(observed)):
        blockers.append("skipped_block_indices_mismatch")
    if not bool(report.get("shape_fingerprint_matched", False)):
        blockers.append("shape_fingerprint_not_matched")
    if not bool(report.get("resume_next_step_loss_parity", False)):
        blockers.append("resume_next_step_loss_parity_missing")
    if not bool(report.get("residual_reuse_parity", False)):
        blockers.append("residual_reuse_parity_missing")
    if bool(report.get("raw_residual_tensors_persisted", False)):
        blockers.append("raw_residual_tensors_persisted")
    if float(report.get("max_loss_delta", 0.0) or 0.0) > float(report.get("max_allowed_loss_delta", 1e-6) or 1e-6):
        blockers.append("loss_delta_above_threshold")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_blockskip_resume_parity_audit_v0",
        "ok": ready,
        "resume_parity_ready": ready,
        "checkpoint_contract_ready": bool(contract.get("checkpoint_contract_ready", contract.get("ok", False))),
        "expected_skipped_block_indices": list(expected),
        "observed_skipped_block_indices": list(observed),
        "max_loss_delta": float(report.get("max_loss_delta", 0.0) or 0.0),
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
            "feed BlockSkip checkpoint/resume parity into A/B quality review"
            if ready
            else "fix BlockSkip resume parity blockers before trainer wiring"
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
    "build_dit_blockskip_checkpoint_contract",
    "build_dit_blockskip_resume_parity_audit",
]
