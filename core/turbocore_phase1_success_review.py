"""Phase-1 success review for the TurboCore native optimizer roadmap.

The reviewer is intentionally report-only.  It aggregates existing parity,
performance, and layout-cost evidence into the Phase 1 success criteria without
opening native dispatch, training launch, request fields, or UI/schema paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_controlled_rollout_policy_evidence_gate_utils import (  # noqa: E402
    as_dict as _as_dict,
    dedupe as _dedupe,
    string_list as _string_list,
)


SCOPE = "turbocore_phase1_success_review"
SUCCESS_DECISION = "turbocore_phase1_success_review_ready_default_off"
BLOCKED_DECISION = "turbocore_phase1_success_review_blocked_default_off"
UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "training_path_enabled",
    "training_dispatch",
    "training_activation_allowed",
    "runtime_dispatch_allowed",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_executed",
    "kernel_launch_executed",
    "request_submitted",
    "job_created",
    "queue_enqueued",
    "run_record_written",
    "ready_for_ui",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "backend_router_registered",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_phase1_request_fields",
    "request_adapter_fields",
    "request_schema_fields",
    "ui_route_registration",
    "backend_router_registration",
)


def build_turbocore_phase1_success_review(
    *,
    parity_report: Mapping[str, Any] | None = None,
    performance_report: Mapping[str, Any] | None = None,
    layout_probe: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Aggregate Phase 1 native-optimizer success evidence."""

    parity = _parity_summary(_as_dict(parity_report))
    performance = _performance_summary(_as_dict(performance_report))
    layout = _layout_summary(_as_dict(layout_probe), performance=performance)
    unsafe = _unsafe_claims(parity_report, performance_report, layout_probe)
    failure_events = _active_history(failure_history)
    blockers = _dedupe(
        parity["blocked_reasons"]
        + performance["blocked_reasons"]
        + layout["blocked_reasons"]
        + unsafe
        + [f"phase1_failure_history_not_clear:{event}" for event in failure_events]
    )
    ready = not blockers
    decision = SUCCESS_DECISION if ready else BLOCKED_DECISION
    return {
        "schema_version": 1,
        "package": "turbocore_phase1_success_review_v0",
        "gate": SCOPE,
        "ok": ready,
        "evidence_ready": ready,
        "ready_for_phase1_success_review": ready,
        "phase1_success_ready": ready,
        "phase1_native_optimizer_success_ready": ready,
        "manual_review_required": True,
        "decision": decision,
        "gate_decision": decision,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_phase1_request_fields": {},
        "criteria": {
            "update_parity_stable": parity["ok"],
            "clipping_and_finite_checks_predictable": parity["stateful_lifecycle_ok"],
            "representative_performance_gate_ready": performance["representative_performance_gate_ready"],
            "layout_cost_included_and_still_wins": layout["layout_cost_gate_ok"],
        },
        "parity_summary": parity,
        "performance_summary": performance,
        "layout_summary": layout,
        "failure_history_summary": {
            "clear": not failure_events,
            "count": len(failure_events),
            "events": failure_events,
        },
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(ready),
        "recommended_next_step": _recommended_next_step(ready),
        "notes": [
            "This is a Phase 1 evidence aggregator only.",
            "It does not enable native optimizer dispatch, training launch, request fields, UI/schema exposure, or backend routes.",
            "Passing this review means the roadmap evidence is coherent enough for owner review; product activation remains a separate decision.",
        ],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _parity_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    blocked: list[str] = []
    results = report.get("results") if isinstance(report.get("results"), list) else []
    by_name = {str(_as_dict(row).get("name") or ""): _as_dict(row) for row in results}
    stateful = by_name.get("native_optimizer_adamw_stateful", {})
    stateful_details = _as_dict(stateful.get("details"))
    if not report:
        blocked.append("phase1_parity_report_missing")
    elif _as_dict(report.get("summary")).get("ok") is not True:
        blocked.append("phase1_parity_summary_not_ok")
    for name in ("native_optimizer_adamw", "native_optimizer_adamw_stateful"):
        row = by_name.get(name)
        if not row:
            blocked.append(f"phase1_parity_result_missing:{name}")
        elif row.get("ok") is not True:
            blocked.append(f"phase1_parity_result_not_ok:{name}")
    lifecycle_ok = bool(
        stateful
        and stateful.get("ok") is True
        and stateful_details.get("restore_ok") is True
        and stateful_details.get("nonfinite_skip_ok") is True
        and stateful_details.get("nonfinite_params_unchanged") is True
        and float(stateful_details.get("max_grad_norm") or 0.0) > 0.0
    )
    if stateful and not lifecycle_ok:
        blocked.append("phase1_stateful_clipping_finite_lifecycle_not_proven")
    return {
        "ok": not blocked,
        "present": bool(report),
        "summary_ok": _as_dict(report.get("summary")).get("ok") is True if report else False,
        "required_results": ["native_optimizer_adamw", "native_optimizer_adamw_stateful"],
        "stateful_lifecycle_ok": lifecycle_ok,
        "max_abs_errors": {
            name: _float_or_none(row.get("max_abs_error")) for name, row in by_name.items() if name
        },
        "blocked_reasons": _dedupe(blocked),
    }


def _performance_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    gate = _performance_gate(report)
    blocked: list[str] = []
    if not gate:
        blocked.append("phase1_representative_performance_gate_missing")
    elif not bool(gate.get("representative_performance_gate_ready", False)):
        blocked.append("phase1_representative_performance_gate_not_ready")
    optimizer = _as_dict(_as_dict(gate.get("evidence")).get("optimizer_microbenchmark"))
    owner = _as_dict(_as_dict(gate.get("evidence")).get("owner_native_kernel"))
    training = _as_dict(_as_dict(gate.get("evidence")).get("training_matrix"))
    required_speedup = _float_or_none(gate.get("required_end_to_end_speedup")) or 1.03
    end_to_end_speedup = _float_or_none(training.get("end_to_end_speedup"))
    persistent_route_cost_ok = bool(
        gate
        and gate.get("representative_performance_gate_ready") is True
        and owner.get("ok") is True
        and training.get("ok") is True
        and training.get("native_dispatch_executed") is True
        and end_to_end_speedup is not None
        and end_to_end_speedup >= required_speedup
    )
    if gate and not bool(optimizer.get("ok", False)):
        blocked.append("phase1_optimizer_microbenchmark_not_ok")
    if gate and not bool(training.get("ok", False)):
        blocked.append("phase1_representative_training_matrix_not_ok")
    return {
        "ok": not blocked,
        "present": bool(gate),
        "representative_performance_gate_ready": bool(gate.get("representative_performance_gate_ready", False)) if gate else False,
        "promotion_gate_ok": bool(gate.get("promotion_gate_ok", False)) if gate else False,
        "optimizer_best_speedup_vs_baseline": optimizer.get("best_speedup_vs_baseline"),
        "owner_native_kernel_ok": bool(owner.get("ok", False)),
        "native_dispatch_executed": bool(training.get("native_dispatch_executed", False)),
        "end_to_end_speedup": end_to_end_speedup,
        "required_end_to_end_speedup": required_speedup,
        "persistent_buffer_route_cost_ok": persistent_route_cost_ok,
        "representative_steps": training.get("representative_steps"),
        "source_blocked_reasons": _string_list(gate.get("blocked_reasons")) if gate else [],
        "blocked_reasons": _dedupe(blocked),
    }


def _layout_summary(report: Mapping[str, Any], *, performance: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    blocked: list[str] = []
    gather_scatter_ok = bool(summary.get("layout_including_gather_scatter_gate_ok", False))
    persistent_route_ok = bool(performance.get("persistent_buffer_route_cost_ok", False))
    layout_cost_ok = gather_scatter_ok or persistent_route_ok
    if not report and not persistent_route_ok:
        blocked.append("phase1_layout_probe_missing")
    elif report.get("ok") is not True:
        blocked.append("phase1_layout_probe_not_ok")
    if report and not layout_cost_ok:
        blocked.append("phase1_layout_including_gather_scatter_gate_not_ok")
        blocked.append("phase1_layout_cost_not_proven_after_transfer_or_sync")
    return {
        "ok": not blocked,
        "present": bool(report),
        "layout_cost_gate_ok": layout_cost_ok,
        "integration_strategy": "gather_scatter" if gather_scatter_ok else ("persistent_buffer_route" if persistent_route_ok else "unproven"),
        "flat_kernel_gate_ok": bool(summary.get("flat_kernel_gate_ok", False)),
        "layout_including_gather_scatter_gate_ok": gather_scatter_ok,
        "persistent_buffer_route_cost_ok": persistent_route_ok,
        "flat_kernel_speedup": summary.get("flat_kernel_speedup"),
        "layout_including_gather_scatter_speedup": summary.get("layout_including_gather_scatter_speedup"),
        "persistent_buffer_route_speedup": performance.get("end_to_end_speedup"),
        "layout_tax_ms": summary.get("layout_tax_ms"),
        "recommendation": str(summary.get("recommendation") or ""),
        "blocked_reasons": _dedupe(blocked),
    }


def _performance_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("gate") == "turbocore_native_update_performance_gate_v0":
        return dict(report)
    for key in ("performance_gate", "native_update_performance_gate", "native_update_representative_performance_gate"):
        gate = _as_dict(report.get(key))
        if gate:
            return gate
    return {}


def _unsafe_claims(*reports: Mapping[str, Any] | None) -> list[str]:
    blocked: list[str] = []
    for index, report in enumerate(reports):
        value = _as_dict(report)
        if not value:
            continue
        for field in UNSAFE_TRUE_FIELDS:
            if value.get(field) is True:
                blocked.append(f"phase1_unsafe_true_field:{index}:{field}")
        for field in UNSAFE_NON_EMPTY_FIELDS:
            if bool(value.get(field)):
                blocked.append(f"phase1_unsafe_non_empty_field:{index}:{field}")
    return _dedupe(blocked)


def _active_history(values: Sequence[Any] | None) -> list[str]:
    out: list[str] = []
    for index, value in enumerate(values or []):
        if isinstance(value, Mapping):
            status = str(value.get("status") or "").lower()
            active = value.get("active") is True or status in {"open", "active", "blocked"}
            if not active:
                continue
            text = str(value.get("reason") or value.get("event") or f"event_{index}")
        else:
            text = str(value or "")
        if text:
            out.append(text)
    return _dedupe(out)


def _allowed_next_actions(ready: bool) -> list[str]:
    if ready:
        return [
            "record_phase1_success_review",
            "prepare_owner_review_package",
            "continue_default_off_optimizer_family_rollout",
        ]
    return ["repair_missing_or_failed_phase1_evidence", "rerun_phase1_review"]


def _recommended_next_step(ready: bool) -> str:
    if ready:
        return "record Phase 1 evidence as ready for owner review while keeping native update default-off"
    return "complete parity, representative performance, and layout-cost evidence before Phase 1 owner review"


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review TurboCore Phase 1 native optimizer success evidence")
    parser.add_argument("--parity-report", default="")
    parser.add_argument("--performance-report", default="")
    parser.add_argument("--layout-probe", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = build_turbocore_phase1_success_review(
        parity_report=load_json(args.parity_report) if args.parity_report else None,
        performance_report=load_json(args.performance_report) if args.performance_report else None,
        layout_probe=load_json(args.layout_probe) if args.layout_probe else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
