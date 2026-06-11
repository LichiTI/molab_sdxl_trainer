"""Audit helpers for explicit borrowed-stream TurboCore canaries."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


BORROWED_STREAM_POLICY = "borrowed_stream_event_chain"
BORROWED_STREAM_POLICY_NOT_ALLOWED = "borrowed_stream_policy_not_allowed"


def benchmark_requested_sync_policy(summary: Mapping[str, Any]) -> str:
    benchmark = _as_dict(summary.get("benchmark"))
    return str(
        benchmark.get("turbocore_native_update_runtime_synchronization_policy", "")
        or benchmark.get("native_update_runtime_synchronization_policy", "")
        or ""
    )


def audit_training_executor_reports(
    dispatch_runtime_reports: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    rows = [_training_executor_report(item) for item in dispatch_runtime_reports or []]
    rows = [item for item in rows if item]
    attempted = [item for item in rows if bool(item.get("attempted", False))]
    called = [item for item in rows if bool(item.get("called", False))]
    ok = [item for item in rows if bool(item.get("ok", False))]
    executed = [
        item
        for item in rows
        if bool(item.get("native_step_executed", False))
        or bool(item.get("native_kernel_launched", False))
    ]
    last = called[-1] if called else (attempted[-1] if attempted else (rows[-1] if rows else {}))
    result = _as_dict(last.get("result"))
    error = str(result.get("error", "") or last.get("error", "") or "")
    blockers = []
    native_policy_blockers: list[str] = []
    launch_evidence_blockers: list[str] = []
    stream_guard_blockers: list[str] = []
    lease_blockers: list[str] = []
    native_policy: dict[str, Any] = {}
    launch_evidence: dict[str, Any] = {}
    for item in rows:
        blockers.extend(_strings(item.get("blocked_reasons")))
        result = _as_dict(item.get("result"))
        err = str(result.get("error", "") or "")
        if BORROWED_STREAM_POLICY_NOT_ALLOWED in err:
            blockers.append(BORROWED_STREAM_POLICY_NOT_ALLOWED)
        native_report = _as_dict(result.get("native_report"))
        native_policy = _as_dict(result.get("borrowed_stream_policy")) or _as_dict(
            native_report.get("borrowed_stream_policy")
        ) or native_policy
        launch_evidence = _as_dict(result.get("borrowed_stream_launch_evidence")) or _as_dict(
            native_report.get("borrowed_stream_launch_evidence")
        ) or launch_evidence
        native_policy_blockers.extend(_strings(native_policy.get("blocked_reasons")))
        launch_evidence_blockers.extend(_strings(launch_evidence.get("blocked_reasons")))
        stream_guard_blockers.extend(_strings(launch_evidence.get("stream_guard_blocked_reasons")))
        lease_blockers.extend(_strings(launch_evidence.get("lease_blocked_reasons")))
    return {
        "reports": len(rows),
        "attempted_reports": len(attempted),
        "called_reports": len(called),
        "ok_reports": len(ok),
        "executed_reports": len(executed),
        "last_reason": str(last.get("reason", "") or ""),
        "last_error": error,
        "blocked_reasons": _dedupe(
            blockers + native_policy_blockers + launch_evidence_blockers + stream_guard_blockers + lease_blockers
        ),
        "native_policy_blocked_reasons": _dedupe(native_policy_blockers),
        "launch_evidence_blocked_reasons": _dedupe(launch_evidence_blockers),
        "stream_guard_blocked_reasons": _dedupe(stream_guard_blockers),
        "lease_blocked_reasons": _dedupe(lease_blockers),
        "native_policy_allowed": native_policy.get("allowed") if native_policy else None,
        "runtime_stream_guard_evidence_ready": launch_evidence.get("runtime_stream_guard_evidence_ready")
        if launch_evidence
        else None,
        "runtime_stream_lifetime_lease_ready": launch_evidence.get("runtime_stream_lifetime_lease_ready")
        if launch_evidence
        else None,
        "stream_handle_nonzero": launch_evidence.get("stream_handle_nonzero")
        if launch_evidence
        else native_policy.get("stream_handle_nonzero") if native_policy else None,
        "event_chain_verified": launch_evidence.get("event_chain_verified") if launch_evidence else None,
        "stream_lifetime_bound": launch_evidence.get("stream_lifetime_bound") if launch_evidence else None,
        "policy_not_allowed": BORROWED_STREAM_POLICY_NOT_ALLOWED in error
        or BORROWED_STREAM_POLICY_NOT_ALLOWED in blockers
        or bool(native_policy and native_policy.get("requested") and not native_policy.get("allowed")),
    }


def borrowed_stream_case_audit(summary: Mapping[str, Any]) -> dict[str, Any]:
    policy = str(summary.get("native_dispatch_requested_runtime_synchronization_policy", "") or "")
    requested = bool(summary.get("native_dispatch_requested_borrowed_stream_event_chain", False))
    native_requested = bool(summary.get("native_dispatch_requested", False))
    executed = bool(summary.get("native_dispatch_executed", False))
    runtime_blockers = _strings(summary.get("native_dispatch_runtime_blocked_reasons"))
    training_blockers = _strings(summary.get("native_dispatch_training_executor_blocked_reasons"))
    native_policy_blockers = _strings(summary.get("native_dispatch_borrowed_stream_native_policy_blocked_reasons"))
    launch_evidence_blockers = _strings(
        summary.get("native_dispatch_borrowed_stream_launch_evidence_blocked_reasons")
    )
    stream_guard_blockers = _strings(summary.get("native_dispatch_borrowed_stream_stream_guard_blocked_reasons"))
    lease_blockers = _strings(summary.get("native_dispatch_borrowed_stream_lease_blocked_reasons"))
    gate_blockers = _strings(summary.get("gate_blocked_reasons"))
    readiness_blockers = _strings(summary.get("readiness_blockers"))
    performance_blockers = _strings(summary.get("performance_gate_blocked_reasons"))
    blocked_before_native_step = bool(requested and native_requested and not executed)
    stage = ""
    if requested:
        if executed:
            stage = "executed"
        elif training_blockers or summary.get("native_dispatch_training_executor_last_error"):
            stage = "training_executor"
        elif runtime_blockers:
            stage = "dispatch_runtime"
        elif _strings(summary.get("native_dispatch_execution_blocked_reasons")):
            stage = "execution_plan"
        elif gate_blockers or readiness_blockers:
            stage = "gate_or_readiness"
        else:
            stage = "requested_not_executed"
    return {
        "requested_policy": policy,
        "borrowed_stream_requested": requested,
        "native_dispatch_requested": native_requested,
        "native_dispatch_executed": executed,
        "blocked_before_native_step": blocked_before_native_step,
        "block_stage": stage,
        "policy_not_allowed": bool(summary.get("native_dispatch_borrowed_stream_policy_not_allowed", False)),
        "runtime_blockers": runtime_blockers,
        "training_executor_blockers": training_blockers,
        "native_policy_blockers": native_policy_blockers,
        "launch_evidence_blockers": launch_evidence_blockers,
        "stream_guard_blockers": stream_guard_blockers,
        "lease_blockers": lease_blockers,
        "gate_blockers": gate_blockers,
        "readiness_blockers": readiness_blockers,
        "performance_blockers": performance_blockers,
        "blocked_reasons": _dedupe(
            training_blockers
            + native_policy_blockers
            + launch_evidence_blockers
            + stream_guard_blockers
            + lease_blockers
            + runtime_blockers
            + _strings(summary.get("native_dispatch_execution_blocked_reasons"))
            + gate_blockers
            + readiness_blockers
            + performance_blockers
        ),
    }


def summarize_borrowed_stream_matrix(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    audits: dict[str, dict[str, Any]] = {}
    for entry in entries:
        summary = _as_dict(entry.get("summary"))
        if not summary:
            continue
        audit = borrowed_stream_case_audit(summary)
        if not audit["borrowed_stream_requested"]:
            continue
        name = str(_as_dict(entry.get("case")).get("name", "") or "")
        if name:
            audits[name] = audit
    return {
        "native_dispatch_requested_borrowed_stream_cases": list(audits.keys()),
        "native_dispatch_borrowed_stream_executed_cases": [
            name for name, audit in audits.items() if bool(audit.get("native_dispatch_executed", False))
        ],
        "native_dispatch_borrowed_stream_blocked_cases": [
            name for name, audit in audits.items() if bool(audit.get("blocked_before_native_step", False))
        ],
        "native_dispatch_borrowed_stream_policy_not_allowed_cases": [
            name for name, audit in audits.items() if bool(audit.get("policy_not_allowed", False))
        ],
        "native_dispatch_borrowed_stream_block_stage_by_case": {
            name: str(audit.get("block_stage", "") or "")
            for name, audit in audits.items()
            if audit.get("block_stage")
        },
        "native_dispatch_borrowed_stream_blockers_by_case": {
            name: list(audit.get("blocked_reasons", []))
            for name, audit in audits.items()
            if audit.get("blocked_reasons")
        },
        "native_dispatch_borrowed_stream_audit_by_case": audits,
    }


def _training_executor_report(report: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _as_dict(report)
    return _as_dict(runtime.get("training_executor"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = [
    "BORROWED_STREAM_POLICY",
    "audit_training_executor_reports",
    "benchmark_requested_sync_policy",
    "borrowed_stream_case_audit",
    "summarize_borrowed_stream_matrix",
]
