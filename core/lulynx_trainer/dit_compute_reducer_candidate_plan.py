"""Candidate planning for default-off DiT compute reducer A/B work."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_POLICY_FIELDS = ("owner", "review_id", "candidate_scope", "rollback_plan")


def build_dit_compute_reducer_candidate_plan(
    *,
    trainer_gate: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
) -> dict[str, Any]:
    gate = dict(trainer_gate)
    policy = dict(selection_policy)
    rows = [dict(row) for row in gate.get("rows", ()) if isinstance(row, Mapping)]
    min_reduction = _clamp_fraction(policy.get("min_compute_reduction", 0.05))
    max_candidates = max(int(policy.get("max_candidates") or len(rows) or 1), 1)
    blockers: list[str] = []

    if gate.get("scorecard") != "dit_compute_reducer_trainer_gate_v0":
        blockers.append("unexpected_compute_reducer_trainer_gate")
    if not bool(gate.get("contract_ready", gate.get("ok", False))):
        blockers.append("compute_reducer_contract_not_ready")
    if not bool(gate.get("trainer_ready", False)):
        blockers.append("compute_reducer_trainer_gate_not_ready")
    if _unsafe_flags(gate, policy):
        blockers.append("unsafe_child_flag")
    if any(_unsafe_flags(row) for row in rows):
        blockers.append("unsafe_child_flag")
    for name in REQUIRED_POLICY_FIELDS:
        if not str(policy.get(name) or "").strip():
            blockers.append(f"selection_policy_field_missing:{name}")
    if not bool(policy.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(policy.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(policy.get("acknowledge_no_trainer_wiring", False)):
        blockers.append("trainer_wiring_ack_missing")
    if not bool(policy.get("requires_later_ab_execution_contract", False)):
        blockers.append("later_ab_execution_contract_missing")
    if policy.get("trainer_wiring_allowed") is not False:
        blockers.append("trainer_wiring_allowed_must_be_false")
    if policy.get("ab_dispatch_allowed") is not False:
        blockers.append("ab_dispatch_allowed_must_be_false")

    ranked = sorted(
        (_candidate_row(row, min_reduction) for row in rows),
        key=lambda row: (-float(row["compute_reduction"]), str(row["reducer_id"])),
    )
    eligible = [row for row in ranked if row["eligible"]][:max_candidates]
    if not eligible:
        blockers.append("no_compute_reducer_candidate_selected")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_candidate_plan_v0",
        "ok": ready,
        "candidate_plan_ready": ready,
        "candidate_scope": str(policy.get("candidate_scope") or ""),
        "owner": str(policy.get("owner") or ""),
        "min_compute_reduction": float(min_reduction),
        "max_candidates": int(max_candidates),
        "selected_reducers": [row["reducer_id"] for row in eligible],
        "candidate_rows": ranked,
        "trainer_wiring_allowed": False,
        "trainer_wiring_executed": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare manual A/B evidence package for selected compute reducers"
            if ready
            else "collect complete compute-reducer quality/performance gates before A/B planning"
        ),
    }


def _candidate_row(row: Mapping[str, Any], min_reduction: float) -> dict[str, Any]:
    compute_fraction = _clamp_fraction(row.get("estimated_compute_fraction", 1.0))
    reduction = max(0.0, 1.0 - compute_fraction)
    blocked = tuple(str(item) for item in row.get("blocked_reasons", ()) or ())
    eligible = (
        bool(row.get("present", False))
        and bool(row.get("probe_ready", False))
        and bool(row.get("enabled", False))
        and not blocked
        and reduction >= min_reduction
        and not _unsafe_flags(row)
    )
    reasons: list[str] = []
    if not bool(row.get("present", False)):
        reasons.append("missing_evidence")
    if not bool(row.get("probe_ready", False)):
        reasons.append("probe_not_ready")
    if not bool(row.get("enabled", False)):
        reasons.append("reducer_not_enabled")
    if blocked:
        reasons.extend(blocked)
    if reduction < min_reduction:
        reasons.append("compute_reduction_below_threshold")
    if _unsafe_flags(row):
        reasons.append("unsafe_candidate_flag")
    return {
        "reducer_id": str(row.get("reducer_id") or ""),
        "eligible": bool(eligible),
        "estimated_compute_fraction": float(compute_fraction),
        "compute_reduction": float(reduction),
        "blocked_reasons": reasons,
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "default_rollout_allowed",
        "auto_rollout_allowed",
        "trainer_wiring_allowed",
        "trainer_wiring_executed",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


def _clamp_fraction(value: Any) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["build_dit_compute_reducer_candidate_plan"]
