"""Smoke checks for V5-P28 longer replicate evidence collector."""

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


MIN_READY_RUNS = 3
MIN_STEPS = 768
MIN_SPEEDUP = 1.05

DEFAULT_REQUEST_FALSE_FIELDS = (
    "default_behavior_changed",
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
)


def run_smoke() -> dict[str, Any]:
    ready_runs = _ready_runs()
    ready = _build_bundle(ready_runs)
    assert ready["ok"] is True, ready
    assert ready["longer_replicate_evidence_ready"] is True, ready
    _assert_default_request_off(ready)

    ready_samples = _samples(ready)
    assert len(ready_samples) == MIN_READY_RUNS, ready
    p26_gate = _p26_gate_consumes(ready)
    assert p26_gate["ok"] is True, p26_gate
    assert p26_gate["longer_replicate_summary"]["speedup_sample_count"] == MIN_READY_RUNS, p26_gate

    too_few = _build_bundle(ready_runs[:2])
    _assert_blocked(too_few, "few", "sample")

    low_speedup = _build_bundle(
        [
            _run_payload(index=1, speedup=1.08),
            _run_payload(index=2, speedup=1.02),
            _run_payload(index=3, speedup=1.07),
        ]
    )
    _assert_blocked(low_speedup, "speedup")

    steps_short = _build_bundle(
        [
            _run_payload(index=1, speedup=1.08),
            _run_payload(index=2, speedup=1.07, steps=512),
            _run_payload(index=3, speedup=1.09),
        ]
    )
    _assert_blocked(steps_short, "step")

    rollback = _build_bundle(
        [
            _run_payload(index=1, speedup=1.08),
            _run_payload(index=2, speedup=1.07, rollback_events=["performance_regression"]),
            _run_payload(index=3, speedup=1.09),
        ]
    )
    _assert_blocked(rollback, "rollback")

    default_request_violation = _build_bundle(
        [
            _run_payload(index=1, speedup=1.08),
            _run_payload(index=2, speedup=1.07, default_request_violation=True),
            _run_payload(index=3, speedup=1.09),
        ]
    )
    _assert_blocked(default_request_violation, "default", "request", "adapter")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p28_longer_replicate_evidence_collector_smoke",
        "ok": True,
        "ready_sample_count": len(ready_samples),
        "p26_gate_decision": _decision(p26_gate),
        "too_few_blocked_reasons": _blocked_reasons(too_few),
        "low_speedup_blocked_reasons": _blocked_reasons(low_speedup),
        "steps_short_blocked_reasons": _blocked_reasons(steps_short),
        "rollback_blocked_reasons": _blocked_reasons(rollback),
        "default_request_violation_blocked_reasons": _blocked_reasons(default_request_violation),
    }


def _build_bundle(run_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    values = _argument_values(run_payloads)
    signature = inspect.signature(build_v5_longer_replicate_evidence_bundle)
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
    assert not missing_required, f"unsupported required evidence collector arguments: {missing_required}"
    if accepts_kwargs:
        kwargs.setdefault("run_payloads", run_payloads)
        kwargs.setdefault("min_longer_replicate_runs", MIN_READY_RUNS)
        kwargs.setdefault("min_representative_steps", MIN_STEPS)
        kwargs.setdefault("min_end_to_end_speedup", MIN_SPEEDUP)
    return build_v5_longer_replicate_evidence_bundle(**kwargs)


def _argument_values(run_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    threshold_values = {
        "min_ready_runs": MIN_READY_RUNS,
        "min_longer_replicate_runs": MIN_READY_RUNS,
        "min_replicate_runs": MIN_READY_RUNS,
        "min_runs": MIN_READY_RUNS,
        "min_run_count": MIN_READY_RUNS,
        "min_sample_count": MIN_READY_RUNS,
        "min_samples": MIN_READY_RUNS,
        "min_steps": MIN_STEPS,
        "min_steps_completed": MIN_STEPS,
        "min_completed_steps": MIN_STEPS,
        "min_representative_steps": MIN_STEPS,
        "min_end_to_end_speedup": MIN_SPEEDUP,
        "min_representative_speedup": MIN_SPEEDUP,
        "min_speedup": MIN_SPEEDUP,
        "min_speedup_threshold": MIN_SPEEDUP,
    }
    return {
        **threshold_values,
        "run_payloads": run_payloads,
        "payloads": run_payloads,
        "runs": run_payloads,
        "run_summaries": run_payloads,
        "ready_run_payloads": run_payloads,
        "ready_runs": run_payloads,
        "samples": run_payloads,
        "replicate_samples": run_payloads,
        "replicate_runs": run_payloads,
        "longer_replicate_samples": run_payloads,
        "longer_replicate_runs": run_payloads,
        "longer_replicates": run_payloads,
    }


def _p26_gate_consumes(bundle: dict[str, Any]) -> dict[str, Any]:
    return build_v5_longer_replicate_failure_history_gate(
        owner_rollout_review_decision=_p25_decision(),
        longer_replicate_evidence=bundle,
        failure_history=_empty_history(),
        rollback_history=_empty_history(),
        min_longer_replicate_runs=MIN_READY_RUNS,
        min_end_to_end_speedup=MIN_SPEEDUP,
    )


def _assert_default_request_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_REQUEST_FALSE_FIELDS:
        assert report.get(field) is False, report
    assert report.get("post_gate_request_fields", {}) == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["longer_replicate_evidence_ready"] is False, report
    _assert_default_request_off(report)
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    for key in ("blocked_reasons", "promotion_blockers", "evidence_blockers"):
        value = report.get(key)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if str(item)]
    return []


def _samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    value = report.get("samples")
    assert isinstance(value, list), report
    assert all(isinstance(item, dict) for item in value), report
    return value


def _decision(report: dict[str, Any]) -> str:
    return str(
        report.get("rollout_decision")
        or report.get("gate_decision")
        or report.get("decision")
        or ""
    )


def _ready_runs() -> list[dict[str, Any]]:
    return [
        _run_payload(index=1, speedup=1.08),
        _run_payload(index=2, speedup=1.07),
        _run_payload(index=3, speedup=1.09),
    ]


def _run_payload(
    *,
    index: int,
    speedup: float,
    steps: int = MIN_STEPS,
    rollback_events: list[str] | None = None,
    default_request_violation: bool = False,
) -> dict[str, Any]:
    baseline_ms = 1000.0
    native_ms = baseline_ms / float(speedup)
    rollback_values = list(rollback_events or [])
    default_enabled = bool(default_request_violation)
    return {
        "schema_version": 1,
        "run_id": f"v5_p28_longer_replicate_{index:02d}",
        "status": "completed",
        "ok": True,
        "success": True,
        "ready": True,
        "longer_replicate_candidate": True,
        "steps_completed": int(steps),
        "representative_steps": int(steps),
        "expected_steps": MIN_STEPS,
        "duration_minutes": 45,
        "end_to_end_speedup": float(speedup),
        "representative_end_to_end_speedup": float(speedup),
        "speedup_vs_baseline": float(speedup),
        "performance": {
            "end_to_end_speedup": float(speedup),
            "representative_end_to_end_speedup": float(speedup),
            "speedup_vs_baseline": float(speedup),
        },
        "summary": {
            "success": True,
            "steps_completed": int(steps),
            "baseline_mean_step_ms": baseline_ms,
            "native_mean_step_ms": native_ms,
            "native_dispatch_executed": True,
            "checkpoint_resume_native_state_boundary": True,
            "rollback_events": rollback_values,
        },
        "rollback_events": rollback_values,
        "failure_events": [],
        "blocked_reasons": [],
        "default_behavior_changed": default_enabled,
        "default_training_path_enabled": default_enabled,
        "training_path_enabled": default_enabled,
        "default_rollout_allowed": default_enabled,
        "auto_rollout_allowed": default_enabled,
        "request_adapter_mapping_allowed": default_enabled,
        "request_fields_emitted": default_enabled,
        "post_gate_request_fields": (
            {"turbocoreNativeUpdateCanaryScope": "unexpected_default_request_fixture"}
            if default_enabled
            else {}
        ),
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


def _empty_history() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "history": "v5_p28_empty_history_fixture",
        "events": [],
        "open_failure_count": 0,
        "cooldown_active": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
