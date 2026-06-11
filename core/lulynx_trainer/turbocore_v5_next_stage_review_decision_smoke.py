"""Smoke checks for V5 next-stage review decision contract."""

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
from core.turbocore_v5_next_stage_review_decision import (  # noqa: E402
    build_v5_next_stage_review_decision,
)


DEFAULT_OFF_FIELDS = (
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
)


def run_smoke() -> dict[str, Any]:
    p26_ready = _p26_gate(ready=True)
    assert p26_ready["ok"] is True, p26_ready
    assert _decision(p26_ready) == "longer_replicate_failure_history_review_ready", p26_ready
    assert p26_ready["manual_next_stage_review_allowed"] is True, p26_ready
    _assert_default_off(p26_ready)

    p26_not_ready = _p26_gate(ready=False)
    assert p26_not_ready["ok"] is False, p26_not_ready
    assert _decision(p26_not_ready) != "longer_replicate_failure_history_review_ready", p26_not_ready
    _assert_default_off(p26_not_ready)

    pending = _build_next_stage_decision(p26_gate=p26_ready, next_stage_review=None)
    _assert_blocked(pending, "signed", "review")
    assert _decision(pending) == "hold_for_signed_next_stage_review", pending

    approved_not_ready = _build_next_stage_decision(
        p26_gate=p26_not_ready,
        next_stage_review=_next_stage_review(approve=True),
    )
    _assert_blocked(approved_not_ready, "p26")

    p26_request_adapter = _build_next_stage_decision(
        p26_gate={**p26_ready, "request_adapter_mapping_allowed": True},
        next_stage_review=_next_stage_review(approve=True),
    )
    _assert_blocked(p26_request_adapter, "p26", "request", "adapter")

    p26_post_fields = _build_next_stage_decision(
        p26_gate={**p26_ready, "post_gate_request_fields": {"native_dispatch": True}},
        next_stage_review=_next_stage_review(approve=True),
    )
    _assert_blocked(p26_post_fields, "post")

    p26_default_rollout = _build_next_stage_decision(
        p26_gate={**p26_ready, "default_rollout_allowed": True},
        next_stage_review=_next_stage_review(approve=True),
    )
    _assert_blocked(p26_default_rollout, "p26", "default")

    p26_bad_p25 = _build_next_stage_decision(
        p26_gate={
            **p26_ready,
            "p25_decision_summary": {
                **p26_ready["p25_decision_summary"],
                "owner_rollout_review_signed": False,
            },
        },
        next_stage_review=_next_stage_review(approve=True),
    )
    _assert_blocked(p26_bad_p25, "p25")

    approved = _build_next_stage_decision(
        p26_gate=p26_ready,
        next_stage_review=_next_stage_review(approve=True),
    )
    assert approved["ok"] is True, approved
    assert _decision(approved) == "signed_next_stage_review_recorded_default_off", approved
    _assert_default_off(approved)

    rejected = _build_next_stage_decision(
        p26_gate=p26_ready,
        next_stage_review=_next_stage_review(approve=False),
    )
    assert rejected["ok"] is True, rejected
    assert _decision(rejected) == "signed_next_stage_review_rejected_default_off", rejected
    _assert_default_off(rejected)

    default_rollout = _build_next_stage_decision(
        p26_gate=p26_ready,
        next_stage_review=_next_stage_review(approve=True, approve_default_rollout=True),
    )
    _assert_blocked(default_rollout, "default")

    auto_rollout = _build_next_stage_decision(
        p26_gate=p26_ready,
        next_stage_review=_next_stage_review(approve=True, approve_auto_rollout=True),
    )
    _assert_blocked(auto_rollout, "default")

    default_training = _build_next_stage_decision(
        p26_gate=p26_ready,
        next_stage_review=_next_stage_review(approve=True, approve_default_training=True),
    )
    _assert_blocked(default_training, "default")

    missing_request_adapter_ack = _build_next_stage_decision(
        p26_gate=p26_ready,
        next_stage_review=_without_review_field(
            _next_stage_review(approve=True),
            "acknowledge_no_request_adapter_mapping",
        ),
    )
    _assert_blocked(missing_request_adapter_ack, "request", "adapter")

    missing_manual_only_ack = _build_next_stage_decision(
        p26_gate=p26_ready,
        next_stage_review=_without_review_field(
            _next_stage_review(approve=True),
            "acknowledge_manual_review_only",
        ),
    )
    _assert_blocked(missing_manual_only_ack, "manual")

    missing_runtime_evidence_ack = _build_next_stage_decision(
        p26_gate=p26_ready,
        next_stage_review=_without_review_field(
            _next_stage_review(approve=True),
            "acknowledge_runtime_evidence_complete",
        ),
    )
    _assert_blocked(missing_runtime_evidence_ack, "runtime", "evidence")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_next_stage_review_decision_smoke",
        "ok": True,
        "p26_ready_decision": _decision(p26_ready),
        "p26_not_ready_decision": _decision(p26_not_ready),
        "pending_decision": _decision(pending),
        "approved_not_ready_decision": _decision(approved_not_ready),
        "p26_request_adapter_blocked_reasons": _blocked_reasons(p26_request_adapter),
        "p26_post_fields_blocked_reasons": _blocked_reasons(p26_post_fields),
        "p26_default_rollout_blocked_reasons": _blocked_reasons(p26_default_rollout),
        "p26_bad_p25_blocked_reasons": _blocked_reasons(p26_bad_p25),
        "approved_decision": _decision(approved),
        "rejected_decision": _decision(rejected),
        "default_rollout_blocked_reasons": _blocked_reasons(default_rollout),
        "auto_rollout_blocked_reasons": _blocked_reasons(auto_rollout),
        "default_training_blocked_reasons": _blocked_reasons(default_training),
        "missing_request_adapter_ack_blocked_reasons": _blocked_reasons(missing_request_adapter_ack),
        "missing_manual_only_ack_blocked_reasons": _blocked_reasons(missing_manual_only_ack),
        "missing_runtime_evidence_ack_blocked_reasons": _blocked_reasons(missing_runtime_evidence_ack),
    }


def _build_next_stage_decision(
    *,
    p26_gate: dict[str, Any],
    next_stage_review: dict[str, Any] | None,
) -> dict[str, Any]:
    values = _argument_values(p26_gate, next_stage_review)
    signature = inspect.signature(build_v5_next_stage_review_decision)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return build_v5_next_stage_review_decision(
            p26_gate=p26_gate,
            next_stage_review=next_stage_review,
        )

    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is inspect.Parameter.empty:
            missing_required.append(name)
    assert not missing_required, f"unsupported required next-stage decision arguments: {missing_required}"
    return build_v5_next_stage_review_decision(**kwargs)


def _argument_values(
    p26_gate: dict[str, Any],
    next_stage_review: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "p26_gate": p26_gate,
        "p26_review_gate": p26_gate,
        "p26_decision": p26_gate,
        "longer_replicate_failure_history_gate": p26_gate,
        "longer_replicate_failure_history_review_gate": p26_gate,
        "longer_replicate_failure_history_gate_report": p26_gate,
        "previous_gate": p26_gate,
        "previous_decision": p26_gate,
        "next_stage_review": next_stage_review,
        "signed_next_stage_review": next_stage_review,
        "owner_next_stage_review": next_stage_review,
        "next_stage_owner_review": next_stage_review,
        "review": next_stage_review,
    }


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    _assert_default_off(report)
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("promotion_blockers") or []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _decision(report: dict[str, Any]) -> str:
    return str(
        report.get("rollout_decision")
        or report.get("next_stage_review_decision")
        or report.get("review_decision")
        or report.get("gate_decision")
        or report.get("decision")
        or ""
    )


def _p26_gate(*, ready: bool) -> dict[str, Any]:
    count = 5 if ready else 2
    return build_v5_longer_replicate_failure_history_gate(
        owner_rollout_review_decision=_p25_decision(),
        longer_replicate_evidence={
            "schema_version": 1,
            "evidence": "v5_p26_longer_replicate_fixture",
            "samples": _longer_replicates(count=count),
            "run_count": count,
        },
        failure_history=_history(),
        rollback_history=_history(),
    )


def _p25_decision() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_owner_rollout_review_decision_v0",
        "gate": "v5_owner_rollout_review_decision",
        "ok": True,
        "decision_record_ready": True,
        "owner_rollout_review_recorded": True,
        "owner_rollout_review_signed": True,
        "approved_for_next_stage": True,
        "rejected_for_default_off_hold": False,
        "rollback_required": False,
        "rollout_decision": "owner_rollout_review_recorded_default_off",
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "blocked_reasons": [],
        "promotion_blockers": [],
    }


def _longer_replicates(*, count: int) -> list[dict[str, Any]]:
    speedups = [1.08, 1.07, 1.06, 1.09, 1.10][:count]
    return [_replicate(index=index, speedup=speedup) for index, speedup in enumerate(speedups, start=1)]


def _replicate(*, index: int, speedup: float) -> dict[str, Any]:
    baseline_ms = 1000.0
    native_ms = baseline_ms / float(speedup)
    return {
        "schema_version": 1,
        "run_id": f"v5_p26_next_stage_replicate_{index:02d}",
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
        "rollback_events": [],
        "blocked_reasons": [],
        "default_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
    }


def _history() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "history": "v5_p26_empty_history_fixture",
        "events": [],
        "open_failure_count": 0,
        "cooldown_active": False,
    }


def _next_stage_review(
    *,
    approve: bool,
    approve_default_rollout: bool = False,
    approve_auto_rollout: bool = False,
    approve_default_training: bool = False,
) -> dict[str, Any]:
    return {
        "reviewer": "next_stage_fixture",
        "reviewed_at": "2026-06-01",
        "requested_scope": "manual_next_stage_review",
        "approve_next_stage": bool(approve),
        "approve_next_stage_review": bool(approve),
        "approve_next_stage_manual_experiment": bool(approve),
        "approve_keep_p26_evidence": bool(approve),
        "approve_keep_longer_replicate_failure_history_evidence": bool(approve),
        "approve_default_training": bool(approve_default_training),
        "approve_default_training_path_enabled": bool(approve_default_training),
        "approve_default_rollout": bool(approve_default_rollout),
        "approve_default_rollout_allowed": bool(approve_default_rollout),
        "approve_auto_rollout": bool(approve_auto_rollout),
        "approve_auto_rollout_allowed": bool(approve_auto_rollout),
        "acknowledge_no_request_adapter_mapping": True,
        "acknowledge_p26_gate_ready": True,
        "acknowledge_runtime_evidence_complete": True,
        "acknowledge_manual_review_only": True,
        "acknowledge_longer_replicate_evidence": True,
        "acknowledge_failure_history_clear": True,
        "acknowledge_rollback_history_clear": True,
    }


def _without_review_field(review: dict[str, Any], field: str) -> dict[str, Any]:
    out = dict(review)
    out.pop(field, None)
    return out


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
