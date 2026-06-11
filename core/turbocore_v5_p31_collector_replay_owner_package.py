"""Replay P31 collector evidence through P26/P27/P29 for TurboCore V5-P32.

This is a report-only contract. It consumes an existing P31 manual-run audit,
replays the embedded collector bundle through the existing default-off review
chain, and never launches training or emits request-adapter fields.
"""

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

from core.turbocore_v5_longer_replicate_failure_history_gate import (
    build_v5_longer_replicate_failure_history_gate,
)
from core.turbocore_v5_next_stage_review_decision import build_v5_next_stage_review_decision
from core.turbocore_v5_owner_next_stage_package import build_v5_owner_next_stage_package
from core.turbocore_v5_owner_review_evidence_package import load_json


P31_READY_DECISION = "longer_replicate_manual_run_audit_ready_default_off"
P32_READY_DECISION = "p31_collector_replay_owner_package_ready_default_off"
P32_PENDING_DECISION = "p31_collector_replay_hold_for_signed_next_stage_review_default_off"
P32_BLOCKED_DECISION = "p31_collector_replay_blocked_default_off"


def build_v5_p31_collector_replay_owner_package(
    *,
    p31_manual_run_audit: Mapping[str, Any] | None = None,
    owner_rollout_review_decision: Mapping[str, Any] | None = None,
    next_stage_review: Mapping[str, Any] | None = None,
    failure_history: Mapping[str, Any] | None = None,
    rollback_history: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a P32 replay package without running training."""

    p31 = _as_dict(p31_manual_run_audit)
    p31_summary = _p31_summary(p31)
    p28_bundle = _p31_collector_bundle(p31)
    thresholds = _thresholds(p31, p28_bundle)
    p26_gate = build_v5_longer_replicate_failure_history_gate(
        owner_rollout_review_decision=_as_dict(owner_rollout_review_decision),
        longer_replicate_evidence=p28_bundle,
        failure_history=failure_history,
        rollback_history=rollback_history,
        min_longer_replicate_runs=thresholds["min_runs"],
        min_end_to_end_speedup=thresholds["min_end_to_end_speedup"],
        max_speedup_spread_ratio=thresholds["max_speedup_spread_ratio"],
    )
    p27_decision = build_v5_next_stage_review_decision(
        p26_gate=p26_gate,
        next_stage_review=_as_dict(next_stage_review) if next_stage_review else None,
    )
    p29_package = build_v5_owner_next_stage_package(
        p28_evidence_bundle=p28_bundle,
        p26_gate=p26_gate,
        p27_decision=p27_decision,
    )

    blockers = _dedupe(
        _p31_blockers(p31_summary)
        + ([] if p26_gate.get("ok") else _prefixed("p26", p26_gate.get("blocked_reasons")))
        + ([] if p27_decision.get("ok") else _prefixed("p27", p27_decision.get("blocked_reasons")))
        + ([] if p29_package.get("ok") else _prefixed("p29", p29_package.get("blocked_reasons")))
    )
    waiting_for_review = (
        p31_summary["ready"]
        and bool(p26_gate.get("ok", False))
        and not bool(next_stage_review)
        and "p27:v5_p27_signed_next_stage_review_missing" in blockers
    )
    ready = not blockers
    decision = P32_READY_DECISION if ready else (P32_PENDING_DECISION if waiting_for_review else P32_BLOCKED_DECISION)
    return {
        "schema_version": 1,
        "package": "turbocore_v5_p31_collector_replay_owner_package_v0",
        "gate": "v5_p31_collector_replay_owner_package",
        "ok": ready,
        "p31_collector_replay_ready": ready,
        "owner_next_stage_package_ready": bool(p29_package.get("package_ready", False)),
        "ready_for_signed_next_stage_review": waiting_for_review,
        "decision": decision,
        "package_decision": decision,
        "gate_decision": decision,
        "manual_review_required": True,
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "runs_dispatched": False,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_replay_request_fields": {},
        "p31_manual_run_audit_summary": p31_summary,
        "collector_replay_invocation": {
            "p28_bundle_source": p31_summary["source_path"],
            "thresholds": thresholds,
            "p26_function": "build_v5_longer_replicate_failure_history_gate",
            "p27_function": "build_v5_next_stage_review_decision",
            "p29_function": "build_v5_owner_next_stage_package",
        },
        "p28_collector_bundle": p28_bundle,
        "p26_gate": p26_gate,
        "p27_decision": p27_decision,
        "p29_owner_next_stage_package": p29_package,
        "next_stage_review_template": p27_decision.get("next_stage_review_template", {}),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "recommended_next_step": _recommended_next_step(ready, waiting_for_review, blockers),
        "notes": [
            "P32 replays existing P31 evidence only; it does not launch longer replicate runs.",
            "P27 still requires an explicit signed next-stage review before P29 can be ready.",
            "Default rollout and request-adapter mapping remain disabled.",
        ],
    }


def _p31_summary(p31: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _p31_collector_bundle(p31)
    return {
        "present": bool(p31),
        "source_path": str(p31.get("_source_path") or p31.get("source_path") or ""),
        "ok": bool(p31.get("ok", False)),
        "manual_run_audit_ready": bool(p31.get("manual_run_audit_ready", False)),
        "collector_evidence_ready": bool(p31.get("collector_evidence_ready", False)),
        "decision": str(p31.get("decision") or p31.get("gate_decision") or ""),
        "default_off": _default_off_confirmed(p31),
        "request_adapter_off": _request_adapter_off(p31),
        "training_launch_allowed": bool(p31.get("training_launch_allowed", True)),
        "auto_launch_allowed": bool(p31.get("auto_launch_allowed", True)),
        "runs_dispatched": bool(p31.get("runs_dispatched", True)),
        "post_fields_empty": not bool(_as_dict(p31.get("post_audit_request_fields"))),
        "collector_bundle_present": bool(bundle),
        "collector_bundle_ready": bool(bundle.get("longer_replicate_evidence_ready", False)),
        "collector_bundle_ok": bool(bundle.get("ok", False)),
        "collector_run_count": int(bundle.get("run_count", 0) or 0),
        "collector_ready_run_count": int(bundle.get("ready_run_count", 0) or 0),
        "blocked_reasons": _string_list(p31.get("blocked_reasons")),
        "ready": _p31_ready(p31, bundle),
    }


def _p31_blockers(summary: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if not bool(summary.get("present", False)):
        blocked.append("v5_p32_p31_audit_missing")
    if not bool(summary.get("ready", False)):
        blocked.append("v5_p32_p31_audit_not_ready")
        blocked.extend(_string_list(summary.get("blocked_reasons")))
    if not bool(summary.get("collector_bundle_present", False)):
        blocked.append("v5_p32_p31_collector_bundle_missing")
    if not bool(summary.get("collector_bundle_ok", False)) or not bool(summary.get("collector_bundle_ready", False)):
        blocked.append("v5_p32_p31_collector_bundle_not_ready")
    if bool(summary.get("training_launch_allowed", True)):
        blocked.append("v5_p32_p31_training_launch_allowed_violation")
    if bool(summary.get("auto_launch_allowed", True)):
        blocked.append("v5_p32_p31_auto_launch_allowed_violation")
    if bool(summary.get("runs_dispatched", True)):
        blocked.append("v5_p32_p31_runs_dispatched_violation")
    if not bool(summary.get("default_off", False)):
        blocked.append("v5_p32_p31_default_off_violation")
    if not bool(summary.get("request_adapter_off", False)):
        blocked.append("v5_p32_p31_request_adapter_violation")
    if not bool(summary.get("post_fields_empty", False)):
        blocked.append("v5_p32_p31_post_fields_present")
    return _dedupe(blocked)


def _p31_ready(p31: Mapping[str, Any], bundle: Mapping[str, Any]) -> bool:
    return bool(
        p31
        and p31.get("ok") is True
        and p31.get("manual_run_audit_ready") is True
        and p31.get("collector_evidence_ready") is True
        and str(p31.get("decision") or p31.get("gate_decision") or "") == P31_READY_DECISION
        and _default_off_confirmed(p31)
        and _request_adapter_off(p31)
        and p31.get("training_launch_allowed") is False
        and p31.get("auto_launch_allowed") is False
        and p31.get("runs_dispatched") is False
        and not _as_dict(p31.get("post_audit_request_fields"))
        and bundle.get("ok") is True
        and bundle.get("longer_replicate_evidence_ready") is True
    )


def _p31_collector_bundle(p31: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _as_dict(p31.get("collector_bundle"))
    if bundle:
        return bundle
    return _as_dict(p31.get("p28_collector_bundle"))


def _thresholds(p31: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    invocation = _as_dict(p31.get("collector_invocation"))
    invocation_thresholds = _as_dict(invocation.get("thresholds"))
    return {
        "min_runs": _first_int(
            invocation_thresholds.get("min_runs"),
            bundle.get("min_longer_replicate_runs"),
            bundle.get("run_count"),
            5,
        ),
        "min_end_to_end_speedup": _first_float(
            invocation_thresholds.get("min_end_to_end_speedup"),
            bundle.get("min_end_to_end_speedup"),
            1.05,
        ),
        "max_speedup_spread_ratio": _first_float(
            invocation_thresholds.get("max_speedup_spread_ratio"),
            bundle.get("max_speedup_spread_ratio"),
            0.30,
        ),
    }


def _recommended_next_step(ready: bool, waiting_for_review: bool, blockers: list[str]) -> str:
    if ready:
        return "archive the P32 owner package; any next-stage run remains explicit and default-off"
    if waiting_for_review:
        return "collect a signed P27 next-stage owner review, then replay P32 again"
    if any(item.startswith("v5_p32_p31") for item in blockers):
        return "repair P31 manual-run audit evidence before replaying into P26"
    if any(item.startswith("p26:") for item in blockers):
        return "resolve P26 longer-replicate or history blockers before P29 packaging"
    if any(item.startswith("p27:") for item in blockers):
        return "complete P27 signed next-stage review evidence"
    if any(item.startswith("p29:") for item in blockers):
        return "repair P29 package inputs before owner archive"
    return "hold P32 until all replay blockers clear"


def _default_off_confirmed(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("default_training_path_enabled") is False
        and value.get("training_path_enabled") is False
        and value.get("default_rollout_allowed") is False
        and value.get("auto_rollout_allowed") is False
    )


def _request_adapter_off(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("request_adapter_mapping_allowed") is False
        and value.get("request_fields_emitted") is False
    )


def _prefixed(prefix: str, value: Any) -> list[str]:
    return [f"{prefix}:{item}" for item in _string_list(value)]


def _first_int(*values: Any) -> int:
    for value in values:
        try:
            out = int(value)
        except (TypeError, ValueError):
            continue
        if out > 0:
            return out
    return 0


def _first_float(*values: Any) -> float:
    for value in values:
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if out > 0.0:
            return out
    return 0.0


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P32 P31 collector replay owner package.")
    parser.add_argument("--p31-audit", default="", help="P31 manual longer-replicate audit JSON.")
    parser.add_argument("--owner-rollout-review-decision", default="", help="P25 owner rollout decision JSON.")
    parser.add_argument("--next-stage-review", default="", help="Optional signed P27 next-stage review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_p31_collector_replay_owner_package(
        p31_manual_run_audit=load_json(args.p31_audit) if args.p31_audit else None,
        owner_rollout_review_decision=(
            load_json(args.owner_rollout_review_decision) if args.owner_rollout_review_decision else None
        ),
        next_stage_review=load_json(args.next_stage_review) if args.next_stage_review else None,
        failure_history=load_json(args.failure_history) if args.failure_history else None,
        rollback_history=load_json(args.rollback_history) if args.rollback_history else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_p31_collector_replay_owner_package"]
