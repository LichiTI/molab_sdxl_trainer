"""Owner-review evidence package for V5 manual wider canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_manual_wider_canary_config_adapter_scorecard import (
    build_v5_manual_wider_canary_config_adapter_scorecard,
)
from core.turbocore_v5_manual_wider_canary_review import build_v5_manual_wider_canary_review


def build_v5_owner_review_evidence_package(
    *,
    stability_gate: Mapping[str, Any] | None = None,
    config_adapter_scorecard: Mapping[str, Any] | None = None,
    performance_matrix: Mapping[str, Any] | None = None,
    owner_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bundle V5 promotion evidence without signing owner approval."""

    stability = _as_dict(stability_gate)
    adapter = _as_dict(config_adapter_scorecard) or build_v5_manual_wider_canary_config_adapter_scorecard()
    performance = _performance_summary(_as_dict(performance_matrix))
    review_report = build_v5_manual_wider_canary_review(
        stability_gate=stability,
        owner_review=owner_review,
    )
    review_ready = _review_ready_for_owner(stability, adapter, review_report)
    promotion_ready = bool(review_report.get("promotion_review_ready", False))
    blocked = _blockers(stability, adapter, performance, review_report, review_ready)
    approval_blockers = [] if promotion_ready else ["v5_p6_owner_review_not_signed"]
    return {
        "schema_version": 1,
        "package": "turbocore_v5_owner_review_evidence_package_v0",
        "gate": "v5_owner_review_evidence_package",
        "ok": not blocked,
        "evidence_package_ready": not blocked,
        "ready_for_owner_review": review_ready,
        "promotion_review_ready": promotion_ready,
        "promotion_decision": str(review_report.get("promotion_decision") or ""),
        "manual_review_required": True,
        "owner_review_action_required": not promotion_ready,
        "manual_wider_canary_allowed": bool(review_report.get("manual_wider_canary_allowed", False)),
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "stability_summary": _stability_summary(stability),
        "config_adapter_summary": _adapter_summary(adapter),
        "performance_matrix_summary": performance,
        "manual_review_gate": review_report,
        "owner_review_template": _owner_review_template(stability),
        "post_approval_request_fields": _post_approval_request_fields(),
        "review_checklist": _review_checklist(stability, adapter, performance),
        "blocked_reasons": blocked,
        "promotion_blockers": list(review_report.get("blocked_reasons", []) or []),
        "approval_blockers": approval_blockers,
        "recommended_next_step": _recommended_next_step(blocked, promotion_ready, review_ready),
        "notes": [
            "This package is evidence for a human review; it does not sign approval.",
            "The post-approval request fields are only valid after the owner review is recorded.",
            "Default and auto rollout remain disabled even when the evidence package is ready.",
        ],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload.setdefault("_source_path", str(source))
    return payload if isinstance(payload, dict) else {}


def _review_ready_for_owner(
    stability: Mapping[str, Any],
    adapter: Mapping[str, Any],
    review_report: Mapping[str, Any],
) -> bool:
    return bool(
        stability.get("stability_gate_ready", False)
        and adapter.get("config_adapter_ready", False)
        and str(review_report.get("promotion_decision") or "") in {
            "hold_for_manual_owner_review",
            "manual_wider_canary_review_ready",
        }
    )


def _blockers(
    stability: Mapping[str, Any],
    adapter: Mapping[str, Any],
    performance: Mapping[str, Any],
    review_report: Mapping[str, Any],
    review_ready: bool,
) -> list[str]:
    blocked: list[str] = []
    if not bool(stability.get("stability_gate_ready", False)):
        blocked.append("v5_p6_stability_gate_not_ready")
        blocked.extend(str(item) for item in list(stability.get("blocked_reasons", []) or []))
    if not bool(adapter.get("config_adapter_ready", False)):
        blocked.append("v5_p6_config_adapter_not_ready")
        blocked.extend(str(item) for item in list(adapter.get("blocked_reasons", []) or []))
    if bool(performance.get("present", False)):
        if not bool(performance.get("performance_gate_ready", False)):
            blocked.append("v5_p19_performance_gate_not_ready")
            blocked.extend(str(item) for item in list(performance.get("blocked_reasons", []) or []))
        if not bool(performance.get("report_only_runtime_dispatch_off", False)):
            blocked.append("v5_p19_performance_report_changed_runtime_dispatch")
    if not bool(review_ready):
        decision = str(review_report.get("promotion_decision") or "")
        if decision and decision != "hold_for_manual_owner_review":
            blocked.append(f"v5_p6_review_gate_not_ready:{decision}")
        elif not decision:
            blocked.append("v5_p6_review_gate_missing")
    return _dedupe(blocked)


def _stability_summary(stability: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _as_dict(stability.get("aggregate"))
    return {
        "stability_gate_ready": bool(stability.get("stability_gate_ready", False)),
        "run_count": int(stability.get("run_count", 0) or 0),
        "ready_run_count": int(stability.get("ready_run_count", 0) or 0),
        "min_replicate_runs": int(stability.get("min_replicate_runs", 3) or 3),
        "speedup_samples": list(aggregate.get("speedup_samples", []) or []),
        "min_speedup": aggregate.get("min_speedup"),
        "mean_speedup": aggregate.get("mean_speedup"),
        "median_speedup": aggregate.get("median_speedup"),
        "speedup_spread_ratio": aggregate.get("speedup_spread_ratio"),
        "blocked_reasons": list(stability.get("blocked_reasons", []) or []),
    }


def _adapter_summary(adapter: Mapping[str, Any]) -> dict[str, Any]:
    gates = _as_dict(adapter.get("progress_gates"))
    return {
        "config_adapter_ready": bool(adapter.get("config_adapter_ready", False)),
        "default_behavior_changed": bool(adapter.get("default_behavior_changed", False)),
        "default_training_path_enabled": bool(adapter.get("default_training_path_enabled", False)),
        "training_path_enabled": bool(adapter.get("training_path_enabled", False)),
        "default_rollout_allowed": bool(adapter.get("default_rollout_allowed", False)),
        "auto_rollout_allowed": bool(adapter.get("auto_rollout_allowed", False)),
        "manual_scope_without_review_blocked": bool(
            gates.get("manual_wider_scope_without_review_blocked", False)
        ),
        "approved_mapping_case_ready": bool(
            gates.get("approved_manual_wider_canary_enables_existing_fields", False)
        ),
        "blocked_reasons": list(adapter.get("blocked_reasons", []) or []),
    }


def _performance_summary(matrix: Mapping[str, Any]) -> dict[str, Any]:
    if not matrix:
        return {"present": False}
    summary = _as_dict(matrix.get("summary"))
    gate_summary = _as_dict(summary.get("native_update_performance_gate"))
    report = _as_dict(matrix.get("native_update_performance_report"))
    gate = _as_dict(report.get("performance_gate"))
    evidence = _as_dict(gate.get("evidence"))
    optimizer = _as_dict(evidence.get("optimizer_microbenchmark"))
    training = _as_dict(evidence.get("training_matrix"))
    ctx_free = _as_dict(summary.get("native_dispatch_ctx_sync_free_comparison"))
    training_dispatch = bool(report.get("training_dispatch", False))
    runtime_dispatch = bool(report.get("runtime_dispatch_allowed", False))
    return {
        "present": True,
        "matrix_summary_path": str(matrix.get("matrix_summary_path") or matrix.get("_source_path") or ""),
        "performance_gate_ready": bool(
            gate_summary.get("ready", False)
            or gate.get("representative_performance_gate_ready", False)
        ),
        "promotion_gate_ok": bool(gate.get("promotion_gate_ok", False)),
        "blocked_reasons": list(gate_summary.get("blocked_reasons", []) or gate.get("blocked_reasons", []) or []),
        "optimizer_evidence_present": bool(optimizer.get("present", False)),
        "optimizer_evidence_quality": str(optimizer.get("evidence_quality", "") or ""),
        "optimizer_best_speedup_vs_baseline": optimizer.get("best_speedup_vs_baseline"),
        "representative_native_case": str(training.get("native_case", "") or ""),
        "representative_end_to_end_speedup": training.get("end_to_end_speedup"),
        "representative_steps": int(training.get("representative_steps", 0) or 0),
        "ctx_sync_free_case": str(ctx_free.get("ctx_sync_free_case", "") or ""),
        "ctx_sync_free_speedup_vs_baseline": ctx_free.get("ctx_sync_free_speedup_vs_baseline"),
        "ctx_sync_free_speedup_vs_context_sync_native": ctx_free.get(
            "ctx_sync_free_speedup_vs_context_sync_native"
        ),
        "ctx_sync_free_representative_candidate_ready": bool(
            ctx_free.get("representative_candidate_ready", False)
        ),
        "training_dispatch": training_dispatch,
        "runtime_dispatch_allowed": runtime_dispatch,
        "report_only_runtime_dispatch_off": not training_dispatch and not runtime_dispatch,
    }


def _owner_review_template(stability: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _as_dict(stability.get("aggregate"))
    return {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": "manual_wider_canary",
        "approve_manual_wider_canary": False,
        "confirm_default_training_path_enabled": False,
        "confirm_training_path_enabled": False,
        "confirm_default_rollout_allowed": False,
        "confirm_auto_rollout_allowed": False,
        "acknowledge_runtime_synchronization": False,
        "acknowledged_speedup_samples": list(aggregate.get("speedup_samples", []) or []),
        "acknowledged_speedup_spread_ratio": aggregate.get("speedup_spread_ratio"),
        "rollback_policy": {
            "fallback_authoritative": True,
            "fallback_backend": "pytorch_adamw",
            "disable_for_run_on_native_error": True,
            "disable_for_run_on_state_sync_failure": True,
            "disable_for_run_on_checkpoint_resume_mismatch": True,
            "disable_for_run_on_config_mismatch": True,
            "disable_for_run_on_non_finite": True,
            "rollback_on_resume_mismatch": True,
            "rollback_on_performance_regression": True,
        },
    }


def _post_approval_request_fields() -> dict[str, Any]:
    return {
        "optimizerType": "AdamW",
        "optimizerBackend": "torch_adamw",
        "turbocoreNativeUpdateCanaryOptimizer": "exact_adamw",
        "turbocoreNativeUpdateCanaryScope": "manual_wider_canary",
        "turbocoreNativeUpdateManualWiderCanaryReviewReady": True,
    }


def _review_checklist(
    stability: Mapping[str, Any],
    adapter: Mapping[str, Any],
    performance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items = [
        {
            "id": "stability_gate",
            "ok": bool(stability.get("stability_gate_ready", False)),
            "summary": "P3 replicate stability gate passed.",
        },
        {
            "id": "manual_scope_only",
            "ok": True,
            "summary": "Only manual_wider_canary scope is eligible.",
        },
        {
            "id": "adapter_blocks_missing_review",
            "ok": bool(
                _as_dict(adapter.get("progress_gates")).get(
                    "manual_wider_scope_without_review_blocked",
                    False,
                )
            ),
            "summary": "Request adapter blocks manual wider canary without review evidence.",
        },
        {
            "id": "default_and_auto_off",
            "ok": True,
            "summary": "Default training path and auto rollout remain disabled.",
        },
        {
            "id": "rollback_policy",
            "ok": True,
            "summary": "Rollback policy falls back to PyTorch AdamW on native/runtime/state/perf failures.",
        },
    ]
    if bool(performance.get("present", False)):
        items.append(
            {
                "id": "performance_gate_ready",
                "ok": bool(performance.get("performance_gate_ready", False))
                and bool(performance.get("report_only_runtime_dispatch_off", False)),
                "summary": "Latest performance matrix gate is ready and remains report-only.",
            }
        )
    return items


def _recommended_next_step(blocked: list[str], promotion_ready: bool, review_ready: bool) -> str:
    if blocked:
        return "complete missing V5 evidence before owner review"
    if promotion_ready:
        return "manual wider canary can be requested explicitly; default and auto remain off"
    if review_ready:
        return "owner should review the package and either sign or reject manual wider canary"
    return "hold manual wider canary until review package is ready"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build TurboCore V5 owner review evidence package.")
    parser.add_argument("--stability-gate", default="", help="Path to V5 P3 stability gate JSON.")
    parser.add_argument("--config-adapter-scorecard", default="", help="Optional V5 P5 adapter scorecard JSON.")
    parser.add_argument("--performance-matrix", default="", help="Optional V5 P18 matrix_summary.json.")
    parser.add_argument("--owner-review", default="", help="Optional owner review JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args(argv)

    report = build_v5_owner_review_evidence_package(
        stability_gate=load_json(args.stability_gate) if args.stability_gate else None,
        config_adapter_scorecard=load_json(args.config_adapter_scorecard) if args.config_adapter_scorecard else None,
        performance_matrix=load_json(args.performance_matrix) if args.performance_matrix else None,
        owner_review=load_json(args.owner_review) if args.owner_review else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_owner_review_evidence_package", "load_json"]
