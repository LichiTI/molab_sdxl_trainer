"""Smoke checks for V5-P29 owner next-stage evidence package."""

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

from core.turbocore_v5_longer_replicate_evidence_collector import (  # noqa: E402
    build_v5_longer_replicate_evidence_bundle,
)
from core.turbocore_v5_longer_replicate_failure_history_gate import (  # noqa: E402
    build_v5_longer_replicate_failure_history_gate,
)
from core.turbocore_v5_next_stage_review_decision import (  # noqa: E402
    build_v5_next_stage_review_decision,
)
from core.turbocore_v5_owner_next_stage_package import (  # noqa: E402
    build_v5_owner_next_stage_package,
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
    p28_ready = _p28_bundle(ready=True)
    assert p28_ready["ok"] is True, p28_ready
    assert p28_ready["longer_replicate_evidence_ready"] is True, p28_ready
    _assert_default_off(p28_ready)

    p26_ready = _p26_gate(ready=True)
    assert p26_ready["ok"] is True, p26_ready
    assert p26_ready["longer_replicate_failure_history_gate_ready"] is True, p26_ready
    _assert_default_off(p26_ready)

    p27_approved = _p27_decision(p26_ready, state="approved")
    assert p27_approved["ok"] is True, p27_approved
    assert p27_approved["decision_record_ready"] is True, p27_approved
    _assert_default_off(p27_approved)

    ready_package = _build_package(
        p28_bundle=p28_ready,
        p26_gate=p26_ready,
        p27_decision=p27_approved,
    )
    assert ready_package["ok"] is True, ready_package
    assert ready_package["package_ready"] is True, ready_package
    _assert_default_off(ready_package)

    p28_not_ready_package = _build_package(
        p28_bundle=_p28_bundle(ready=False),
        p26_gate=p26_ready,
        p27_decision=p27_approved,
    )
    _assert_blocked(p28_not_ready_package)

    p26_not_ready_package = _build_package(
        p28_bundle=p28_ready,
        p26_gate=_p26_gate(ready=False),
        p27_decision=p27_approved,
    )
    _assert_blocked(p26_not_ready_package)

    p27_pending_package = _build_package(
        p28_bundle=p28_ready,
        p26_gate=p26_ready,
        p27_decision=_p27_decision(p26_ready, state="pending"),
    )
    _assert_blocked(p27_pending_package)

    p27_rejected_package = _build_package(
        p28_bundle=p28_ready,
        p26_gate=p26_ready,
        p27_decision=_p27_decision(p26_ready, state="rejected"),
    )
    _assert_blocked(p27_rejected_package)

    p27_invalid_default_package = _build_package(
        p28_bundle=p28_ready,
        p26_gate=p26_ready,
        p27_decision=_p27_decision(p26_ready, state="invalid_default"),
    )
    _assert_blocked(p27_invalid_default_package)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p29_owner_next_stage_package_smoke",
        "ok": True,
        "ready": _case_summary(ready_package),
        "p28_not_ready": _case_summary(p28_not_ready_package),
        "p26_not_ready": _case_summary(p26_not_ready_package),
        "p27_pending": _case_summary(p27_pending_package),
        "p27_rejected": _case_summary(p27_rejected_package),
        "p27_invalid_default": _case_summary(p27_invalid_default_package),
    }


def _build_package(
    *,
    p28_bundle: dict[str, Any],
    p26_gate: dict[str, Any],
    p27_decision: dict[str, Any],
) -> dict[str, Any]:
    values = _package_argument_values(p28_bundle, p26_gate, p27_decision)
    signature = inspect.signature(build_v5_owner_next_stage_package)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )

    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is inspect.Parameter.empty:
            missing_required.append(name)

    assert not missing_required, f"unsupported required P29 package arguments: {missing_required}"
    if accepts_kwargs:
        kwargs.setdefault("p28_collector_bundle", p28_bundle)
        kwargs.setdefault("p26_gate", p26_gate)
        kwargs.setdefault("p27_decision", p27_decision)
    return build_v5_owner_next_stage_package(**kwargs)


def _package_argument_values(
    p28_bundle: dict[str, Any],
    p26_gate: dict[str, Any],
    p27_decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "p28_bundle": p28_bundle,
        "p28_collector_bundle": p28_bundle,
        "p28_evidence_bundle": p28_bundle,
        "p28_longer_replicate_evidence": p28_bundle,
        "collector_bundle": p28_bundle,
        "longer_replicate_evidence": p28_bundle,
        "longer_replicate_evidence_bundle": p28_bundle,
        "p26_gate": p26_gate,
        "p26_review_gate": p26_gate,
        "p26_failure_history_gate": p26_gate,
        "p26_longer_replicate_failure_history_gate": p26_gate,
        "longer_replicate_failure_history_gate": p26_gate,
        "longer_replicate_failure_history_review_gate": p26_gate,
        "p27_decision": p27_decision,
        "p27_next_stage_decision": p27_decision,
        "p27_review_decision": p27_decision,
        "signed_next_stage_review_decision": p27_decision,
        "next_stage_review_decision": p27_decision,
        "signed_approved_decision": p27_decision,
    }


def _p28_bundle(*, ready: bool) -> dict[str, Any]:
    run_count = 5 if ready else 2
    return build_v5_longer_replicate_evidence_bundle(
        run_payloads=_run_payloads(count=run_count),
    )


def _p26_gate(*, ready: bool) -> dict[str, Any]:
    run_count = 5 if ready else 2
    return build_v5_longer_replicate_failure_history_gate(
        owner_rollout_review_decision=_p25_decision(),
        longer_replicate_evidence={
            "schema_version": 1,
            "evidence": "v5_p29_p26_longer_replicate_fixture",
            "samples": _run_payloads(count=run_count),
            "run_count": run_count,
        },
        failure_history=_empty_history("failure"),
        rollback_history=_empty_history("rollback"),
    )


def _p27_decision(p26_gate: dict[str, Any], *, state: str) -> dict[str, Any]:
    if state == "pending":
        review: dict[str, Any] | None = None
    elif state == "rejected":
        review = _next_stage_review(approve=False)
    elif state == "invalid_default":
        review = _next_stage_review(approve=True, approve_default_training=True)
    elif state == "approved":
        review = _next_stage_review(approve=True)
    else:
        raise AssertionError(f"unsupported P27 state fixture: {state}")
    return build_v5_next_stage_review_decision(
        p26_gate=p26_gate,
        next_stage_review=review,
    )


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report


def _assert_blocked(report: dict[str, Any]) -> None:
    assert report["ok"] is False, report
    _assert_default_off(report)


def _case_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "package_ready": bool(report.get("package_ready", False)),
        "decision": _decision(report),
        "blocked_reasons": _blocked_reasons(report),
    }


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    for key in ("blocked_reasons", "promotion_blockers", "package_blockers"):
        value = report.get(key)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if str(item)]
    return []


def _decision(report: dict[str, Any]) -> str:
    return str(
        report.get("decision")
        or report.get("package_decision")
        or report.get("gate_decision")
        or report.get("next_stage_review_decision")
        or report.get("rollout_review_decision")
        or ""
    )


def _run_payloads(*, count: int) -> list[dict[str, Any]]:
    speedups = [1.08, 1.07, 1.06, 1.09, 1.10][:count]
    return [_run_payload(index=index, speedup=speedup) for index, speedup in enumerate(speedups, start=1)]


def _run_payload(*, index: int, speedup: float) -> dict[str, Any]:
    baseline_ms = 1000.0
    native_ms = baseline_ms / float(speedup)
    return {
        "schema_version": 1,
        "run_id": f"v5_p29_longer_replicate_{index:02d}",
        "status": "completed",
        "ok": True,
        "success": True,
        "ready": True,
        "longer_replicate_candidate": True,
        "steps_completed": 768,
        "representative_steps": 768,
        "end_to_end_speedup": float(speedup),
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
        "failure_events": [],
        "blocked_reasons": [],
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_gate_request_fields": {},
    }


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


def _next_stage_review(
    *,
    approve: bool,
    approve_default_training: bool = False,
) -> dict[str, Any]:
    return {
        "reviewer": "v5_p29_owner_next_stage_fixture",
        "reviewed_at": "2026-06-01",
        "requested_scope": "manual_next_stage_review",
        "approve_next_stage": bool(approve),
        "approve_next_stage_review": bool(approve),
        "approve_next_stage_manual_experiment": bool(approve),
        "approve_keep_p26_evidence": bool(approve),
        "approve_keep_longer_replicate_failure_history_evidence": bool(approve),
        "approve_default_training": bool(approve_default_training),
        "approve_default_training_path_enabled": bool(approve_default_training),
        "approve_default_rollout": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout": False,
        "approve_auto_rollout_allowed": False,
        "approve_request_adapter_mapping_allowed": False,
        "approve_request_fields_emitted": False,
        "acknowledge_no_request_adapter_mapping": True,
        "acknowledge_p26_gate_ready": True,
        "acknowledge_runtime_evidence_complete": True,
        "acknowledge_manual_review_only": True,
        "acknowledge_longer_replicate_evidence": True,
        "acknowledge_failure_history_clear": True,
        "acknowledge_rollback_history_clear": True,
    }


def _empty_history(kind: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "history": f"v5_p29_empty_{kind}_history_fixture",
        "events": [],
        "open_failure_count": 0,
        "cooldown_active": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
