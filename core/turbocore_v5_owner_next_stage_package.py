"""Owner evidence package for TurboCore V5-P29 next-stage review material."""

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

from core.turbocore_v5_owner_review_evidence_package import load_json


P26_READY_DECISION = "longer_replicate_failure_history_review_ready"
P27_APPROVED_DECISION = "signed_next_stage_review_recorded_default_off"


def build_v5_owner_next_stage_package(
    *,
    p28_evidence_bundle: Mapping[str, Any] | None = None,
    p26_gate: Mapping[str, Any] | None = None,
    p27_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    p28 = _as_dict(p28_evidence_bundle)
    p26 = _as_dict(p26_gate)
    p27 = _as_dict(p27_decision)
    summaries = {
        "p28_evidence_bundle": _p28_summary(p28),
        "p26_gate": _p26_summary(p26),
        "p27_decision": _p27_summary(p27),
    }
    blocked = _blockers(summaries)
    ready = not blocked
    decision = "owner_next_stage_package_ready_default_off" if ready else "owner_next_stage_package_blocked_default_off"
    return {
        "schema_version": 1,
        "package": "turbocore_v5_owner_next_stage_package_v0",
        "gate": "v5_owner_next_stage_package",
        "ok": ready,
        "package_ready": ready,
        "decision": decision,
        "package_decision": decision,
        "gate_decision": decision,
        "ready_for_owner_archive": ready,
        "manual_review_required": True,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_package_request_fields": {},
        "evidence_summaries": summaries,
        "review_checklist": _review_checklist(summaries),
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(ready, summaries),
        "notes": [
            "This package only bundles P28/P26/P27 evidence for owner review material.",
            "It does not emit request-adapter fields or enable default rollout.",
            "P27 approval is recorded as default-off and remains manual-only.",
        ],
    }


def _p28_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _as_dict(bundle.get("aggregate"))
    return {
        "present": bool(bundle),
        "source_path": str(bundle.get("_source_path") or bundle.get("source_path") or ""),
        "ok": bool(bundle.get("ok", False)),
        "longer_replicate_evidence_ready": bool(bundle.get("longer_replicate_evidence_ready", False)),
        "run_count": int(bundle.get("run_count", 0) or 0),
        "ready_run_count": int(bundle.get("ready_run_count", 0) or 0),
        "min_speedup": aggregate.get("min_speedup"),
        "speedup_spread_ratio": aggregate.get("speedup_spread_ratio"),
        "default_off": _default_off_confirmed(bundle),
        "request_adapter_off": _request_adapter_off(bundle),
        "post_fields_empty": not bool(_as_dict(bundle.get("post_gate_request_fields"))),
        "blocked_reasons": _string_list(bundle.get("blocked_reasons")),
    }


def _p26_summary(gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(gate),
        "source_path": str(gate.get("_source_path") or gate.get("source_path") or ""),
        "ok": bool(gate.get("ok", False)),
        "decision": str(gate.get("decision") or gate.get("gate_decision") or gate.get("rollout_review_decision") or ""),
        "longer_replicate_failure_history_gate_ready": bool(
            gate.get("longer_replicate_failure_history_gate_ready", False)
        ),
        "manual_next_stage_review_allowed": bool(gate.get("manual_next_stage_review_allowed", False)),
        "default_off": _default_off_confirmed(gate),
        "request_adapter_off": _request_adapter_off(gate),
        "post_fields_empty": not bool(_as_dict(gate.get("post_gate_request_fields"))),
        "blocked_reasons": _string_list(gate.get("blocked_reasons")),
    }


def _p27_summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(decision),
        "source_path": str(decision.get("_source_path") or decision.get("source_path") or ""),
        "ok": bool(decision.get("ok", False)),
        "decision": str(
            decision.get("decision")
            or decision.get("gate_decision")
            or decision.get("next_stage_review_decision")
            or ""
        ),
        "decision_record_ready": bool(decision.get("decision_record_ready", False)),
        "signed_next_stage_review_recorded": bool(decision.get("signed_next_stage_review_recorded", False)),
        "signed_next_stage_review_signed": bool(decision.get("signed_next_stage_review_signed", False)),
        "approved_for_next_contract_stage": bool(decision.get("approved_for_next_contract_stage", False)),
        "rejected_for_default_off_hold": bool(decision.get("rejected_for_default_off_hold", False)),
        "rollback_required": bool(decision.get("rollback_required", False)),
        "default_off": _default_off_confirmed(decision),
        "request_adapter_off": _request_adapter_off(decision),
        "post_fields_empty": not bool(_as_dict(decision.get("post_review_request_fields"))),
        "blocked_reasons": _string_list(decision.get("blocked_reasons")),
    }


def _blockers(summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    p28 = summaries["p28_evidence_bundle"]
    p26 = summaries["p26_gate"]
    p27 = summaries["p27_decision"]
    blocked: list[str] = []
    if not bool(p28.get("present", False)):
        blocked.append("v5_p29_p28_evidence_missing")
    if not bool(p28.get("ok", False)) or not bool(p28.get("longer_replicate_evidence_ready", False)):
        blocked.append("v5_p29_p28_evidence_not_ready")
        blocked.extend(_string_list(p28.get("blocked_reasons")))
    if not bool(p26.get("present", False)):
        blocked.append("v5_p29_p26_gate_missing")
    if (
        not bool(p26.get("ok", False))
        or not bool(p26.get("longer_replicate_failure_history_gate_ready", False))
        or str(p26.get("decision") or "") != P26_READY_DECISION
    ):
        blocked.append("v5_p29_p26_gate_not_ready")
        blocked.extend(_string_list(p26.get("blocked_reasons")))
    if not bool(p26.get("manual_next_stage_review_allowed", False)):
        blocked.append("v5_p29_p26_manual_next_stage_not_allowed")
    if not bool(p27.get("present", False)):
        blocked.append("v5_p29_p27_decision_missing")
    if (
        not bool(p27.get("ok", False))
        or not bool(p27.get("decision_record_ready", False))
        or str(p27.get("decision") or "") != P27_APPROVED_DECISION
        or not bool(p27.get("approved_for_next_contract_stage", False))
    ):
        blocked.append("v5_p29_p27_decision_not_approved")
        blocked.extend(_string_list(p27.get("blocked_reasons")))
    if bool(p27.get("rejected_for_default_off_hold", False)):
        blocked.append("v5_p29_p27_rejected_for_default_off_hold")
    if bool(p27.get("rollback_required", False)):
        blocked.append("v5_p29_p27_rollback_required")
    for name, summary in summaries.items():
        if not bool(summary.get("default_off", False)):
            blocked.append(f"v5_p29_{name}_default_off_violation")
        if not bool(summary.get("request_adapter_off", False)):
            blocked.append(f"v5_p29_{name}_request_adapter_violation")
        if not bool(summary.get("post_fields_empty", False)):
            blocked.append(f"v5_p29_{name}_post_fields_present")
    return _dedupe(blocked)


def _review_checklist(summaries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": "p28_evidence_ready",
            "ok": bool(summaries["p28_evidence_bundle"].get("longer_replicate_evidence_ready", False)),
            "summary": "P28 longer replicate evidence bundle is ready.",
        },
        {
            "id": "p26_gate_ready",
            "ok": bool(summaries["p26_gate"].get("longer_replicate_failure_history_gate_ready", False)),
            "summary": "P26 gate accepts the longer replicate and history evidence.",
        },
        {
            "id": "p27_signed_approved",
            "ok": bool(summaries["p27_decision"].get("approved_for_next_contract_stage", False)),
            "summary": "P27 signed next-stage review was approved as default-off.",
        },
        {
            "id": "request_adapter_not_emitted",
            "ok": all(bool(summary.get("request_adapter_off", False)) for summary in summaries.values()),
            "summary": "No evidence source emits request-adapter fields.",
        },
    ]


def _recommended_next_step(ready: bool, summaries: Mapping[str, Mapping[str, Any]]) -> str:
    if ready:
        return "archive P29 owner package; any next experiment remains explicit and default-off"
    if not bool(summaries["p28_evidence_bundle"].get("longer_replicate_evidence_ready", False)):
        return "collect or repair P28 longer replicate evidence before packaging"
    if not bool(summaries["p26_gate"].get("longer_replicate_failure_history_gate_ready", False)):
        return "resolve P26 gate blockers before packaging"
    return "record an approved P27 signed next-stage decision before packaging"


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
    parser = argparse.ArgumentParser(description="Build V5 P29 owner next-stage evidence package.")
    parser.add_argument("--p28-evidence-bundle", default="", help="P28 longer replicate evidence bundle JSON.")
    parser.add_argument("--p26-gate", default="", help="P26 longer replicate failure-history gate JSON.")
    parser.add_argument("--p27-decision", default="", help="P27 signed next-stage review decision JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_owner_next_stage_package(
        p28_evidence_bundle=load_json(args.p28_evidence_bundle) if args.p28_evidence_bundle else None,
        p26_gate=load_json(args.p26_gate) if args.p26_gate else None,
        p27_decision=load_json(args.p27_decision) if args.p27_decision else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_owner_next_stage_package"]
