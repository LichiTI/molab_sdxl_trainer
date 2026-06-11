"""Smoke checks for V5 longer-replicate failure-history gate."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_longer_replicate_failure_history_gate import (  # noqa: E402
    build_v5_longer_replicate_failure_history_gate,
)


def run_smoke() -> dict[str, Any]:
    happy = _build_gate(
        p25_decision=_p25_decision("approved"),
        longer_replicates=_longer_replicates(),
        failure_history=_failure_history(),
    )
    assert happy["ok"] is True, happy
    _assert_default_off(happy)

    pending = _build_gate(
        p25_decision=_p25_decision("pending"),
        longer_replicates=_longer_replicates(),
        failure_history=_failure_history(),
    )
    _assert_blocked(pending, "p25")

    rejected = _build_gate(
        p25_decision=_p25_decision("rejected"),
        longer_replicates=_longer_replicates(),
        failure_history=_failure_history(),
    )
    _assert_blocked(rejected, "p25")

    too_few = _build_gate(
        p25_decision=_p25_decision("approved"),
        longer_replicates=_longer_replicates(count=2),
        failure_history=_failure_history(),
    )
    _assert_blocked(too_few, "replicate")

    default_min_runs = _build_gate_default_thresholds(
        p25_decision=_p25_decision("approved"),
        longer_replicates=_longer_replicates(),
        failure_history=_failure_history(),
    )
    _assert_blocked(default_min_runs, "replicate")

    slow = _build_gate(
        p25_decision=_p25_decision("approved"),
        longer_replicates=_longer_replicates(speedups=[1.08, 1.03, 1.07, 1.09]),
        failure_history=_failure_history(),
    )
    _assert_blocked(slow, "speedup")

    spread = _build_gate(
        p25_decision=_p25_decision("approved"),
        longer_replicates=_longer_replicates(speedups=[1.6, 1.05, 1.06, 1.07]),
        failure_history=_failure_history(),
    )
    _assert_blocked(spread, "spread")

    malformed_p25 = _build_gate(
        p25_decision={**_p25_decision("approved"), "decision_record_ready": False, "owner_rollout_review_signed": False},
        longer_replicates=_longer_replicates(),
        failure_history=_failure_history(),
    )
    _assert_blocked(malformed_p25, "decision_record", "signed")

    blocked_run = _build_gate(
        p25_decision=_p25_decision("approved"),
        longer_replicates=[*_longer_replicates(count=3), _replicate(index=4, speedup=1.08, blocked=True)],
        failure_history=_failure_history(),
    )
    _assert_blocked(blocked_run, "replicate", "blocked")

    failure_history = _build_gate(
        p25_decision=_p25_decision("approved"),
        longer_replicates=_longer_replicates(),
        failure_history=_failure_history(
            events=[
                _history_event("native_error_open", status="open", severity="medium"),
                _history_event("state_sync_mismatch", status="resolved", severity="high"),
                _history_event("rollback_cooldown", status="cooldown", severity="medium"),
            ]
        ),
    )
    _assert_blocked(failure_history, "open", "high", "cooldown")

    summary_history = _build_gate(
        p25_decision=_p25_decision("approved"),
        longer_replicates=_longer_replicates(),
        failure_history={"events": [], "open_failure_count": 1, "cooldown_active": True},
    )
    _assert_blocked(summary_history, "open", "cooldown")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_longer_replicate_failure_history_gate_smoke",
        "ok": True,
        "happy_decision": _decision(happy),
        "pending_blocked_reasons": _blocked_reasons(pending),
        "rejected_blocked_reasons": _blocked_reasons(rejected),
        "too_few_blocked_reasons": _blocked_reasons(too_few),
        "default_min_runs_blocked_reasons": _blocked_reasons(default_min_runs),
        "slow_blocked_reasons": _blocked_reasons(slow),
        "spread_blocked_reasons": _blocked_reasons(spread),
        "malformed_p25_blocked_reasons": _blocked_reasons(malformed_p25),
        "blocked_run_blocked_reasons": _blocked_reasons(blocked_run),
        "failure_history_blocked_reasons": _blocked_reasons(failure_history),
        "summary_history_blocked_reasons": _blocked_reasons(summary_history),
    }


def _build_gate(
    *,
    p25_decision: dict[str, Any],
    longer_replicates: list[dict[str, Any]],
    failure_history: dict[str, Any],
) -> dict[str, Any]:
    values = _argument_values(p25_decision, longer_replicates, failure_history)
    signature = inspect.signature(build_v5_longer_replicate_failure_history_gate)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return build_v5_longer_replicate_failure_history_gate(
            p25_decision=p25_decision,
            longer_replicates=longer_replicates,
            failure_history=failure_history,
        )

    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is inspect.Parameter.empty:
            missing_required.append(name)
    assert not missing_required, f"unsupported required gate arguments: {missing_required}"
    return build_v5_longer_replicate_failure_history_gate(**kwargs)


def _argument_values(
    p25_decision: dict[str, Any],
    longer_replicates: list[dict[str, Any]],
    failure_history: dict[str, Any],
) -> dict[str, Any]:
    replicate_evidence = {
        "schema_version": 1,
        "evidence": "v5_longer_replicate_fixture",
        "samples": longer_replicates,
        "replicates": longer_replicates,
        "run_count": len(longer_replicates),
    }
    threshold_values = {
        "min_longer_replicate_runs": 4,
        "min_replicate_runs": 4,
        "min_run_count": 4,
        "min_end_to_end_speedup": 1.05,
        "min_representative_speedup": 1.05,
        "min_speedup": 1.05,
        "min_speedup_threshold": 1.05,
    }
    return {
        **threshold_values,
        "p25_decision": p25_decision,
        "owner_rollout_review_decision": p25_decision,
        "owner_rollout_decision": p25_decision,
        "signed_owner_rollout_review_decision": p25_decision,
        "previous_decision": p25_decision,
        "longer_replicates": longer_replicates,
        "longer_replicate_samples": longer_replicates,
        "longer_replicate_runs": longer_replicates,
        "replicate_samples": longer_replicates,
        "replicate_runs": longer_replicates,
        "longer_replicate_evidence": replicate_evidence,
        "replicate_evidence": replicate_evidence,
        "failure_history": failure_history,
        "wider_failure_history": failure_history,
        "rollback_history": failure_history,
    }


def _build_gate_default_thresholds(
    *,
    p25_decision: dict[str, Any],
    longer_replicates: list[dict[str, Any]],
    failure_history: dict[str, Any],
) -> dict[str, Any]:
    return build_v5_longer_replicate_failure_history_gate(
        owner_rollout_review_decision=p25_decision,
        longer_replicate_evidence={
            "schema_version": 1,
            "samples": longer_replicates,
            "run_count": len(longer_replicates),
        },
        failure_history=failure_history,
        rollback_history=failure_history,
    )


def _assert_default_off(report: dict[str, Any]) -> None:
    assert report["default_rollout_allowed"] is False, report
    assert report["request_adapter_mapping_allowed"] is False, report
    assert report["request_fields_emitted"] is False, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    _assert_default_off(report)
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("promotion_blockers") or []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return []


def _decision(report: dict[str, Any]) -> str:
    return str(report.get("rollout_decision") or report.get("gate_decision") or report.get("decision") or "")


def _p25_decision(state: str) -> dict[str, Any]:
    approved = state == "approved"
    rejected = state == "rejected"
    pending = state == "pending"
    decision = {
        "approved": "owner_rollout_review_recorded_default_off",
        "rejected": "owner_rollout_review_rejected_default_off",
        "pending": "hold_for_signed_owner_rollout_review",
    }[state]
    blocked = ["v5_p25_signed_owner_rollout_review_missing"] if pending else []
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_owner_rollout_review_decision_v0",
        "gate": "v5_owner_rollout_review_decision",
        "ok": approved or rejected,
        "decision_record_ready": approved or rejected,
        "owner_rollout_review_recorded": approved or rejected,
        "owner_rollout_review_signed": approved or rejected,
        "approved_for_next_stage": approved,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "rollout_decision": decision,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
    }


def _longer_replicates(
    *,
    count: int | None = None,
    speedups: list[float] | None = None,
) -> list[dict[str, Any]]:
    samples = list(speedups or [1.09, 1.08, 1.07, 1.11])
    if count is not None:
        samples = samples[:count]
    return [_replicate(index=index, speedup=speedup) for index, speedup in enumerate(samples, start=1)]


def _replicate(*, index: int, speedup: float, blocked: bool = False) -> dict[str, Any]:
    baseline_ms = 1000.0
    native_ms = baseline_ms / float(speedup)
    return {
        "schema_version": 1,
        "run_id": f"v5_p26_longer_replicate_{index:02d}",
        "success": True,
        "duration_minutes": 45,
        "steps_completed": 768,
        "representative_end_to_end_speedup": float(speedup),
        "speedup_vs_baseline": float(speedup),
        "performance": {
            "representative_end_to_end_speedup": float(speedup),
            "speedup_vs_baseline": float(speedup),
        },
        "summary": {
            "success": True,
            "steps_completed": 768,
            "baseline_mean_step_ms": baseline_ms,
            "native_mean_step_ms": native_ms,
            "native_dispatch_executed": True,
            "checkpoint_resume_native_state_boundary": True,
            "rollback_events": [],
        },
        "rollback_events": ["runtime_blocked"] if blocked else [],
        "blocked_reasons": ["replicate_blocked_fixture"] if blocked else [],
        "default_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
    }


def _failure_history(events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "history": "v5_wider_failure_history_fixture",
        "events": list(events or []),
        "open_failure_count": 0 if not events else sum(1 for event in events if event.get("status") == "open"),
        "cooldown_active": any(event.get("status") == "cooldown" for event in events or []),
    }


def _history_event(event: str, *, status: str, severity: str) -> dict[str, Any]:
    return {
        "event": event,
        "status": status,
        "severity": severity,
        "opened_at": "2026-06-01",
        "cooldown_until": "2026-06-08" if status == "cooldown" else "",
        "rollback_required": status == "cooldown",
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
