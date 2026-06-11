"""Archive integrity ledger for TurboCore V5-P34.

P34 records stable digests for the P33 signoff and the P32 owner package. It is
report-only: no training is launched, no request-adapter fields are emitted,
and no default rollout behavior changes.
"""

from __future__ import annotations

import argparse
import hashlib
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


P31_READY_DECISION = "longer_replicate_manual_run_audit_ready_default_off"
P26_READY_DECISION = "longer_replicate_failure_history_review_ready"
P27_APPROVED_DECISION = "signed_next_stage_review_recorded_default_off"
P29_READY_DECISION = "owner_next_stage_package_ready_default_off"
P32_READY_DECISION = "p31_collector_replay_owner_package_ready_default_off"
P33_READY_DECISION = "p32_final_owner_archive_signoff_ready_default_off"
P34_READY_DECISION = "p32_final_archive_integrity_ledger_ready_default_off"
P34_BLOCKED_DECISION = "p32_final_archive_integrity_ledger_blocked_default_off"


def build_v5_archive_integrity_ledger(
    *,
    p33_archive_signoff: Mapping[str, Any] | None = None,
    p32_owner_package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic digest ledger for archived V5 evidence."""

    p33 = _as_dict(p33_archive_signoff)
    p32 = _as_dict(p32_owner_package)
    p33_summary = _p33_summary(p33)
    p32_summary = _p32_summary(p32)
    evidence_chain = _evidence_chain(p33_summary, p32_summary, p32)
    artifacts = _artifact_digests(p33, p32)
    blockers = _blockers(p33_summary, p32_summary, artifacts)
    ready = not blockers
    decision = P34_READY_DECISION if ready else P34_BLOCKED_DECISION
    return {
        "schema_version": 1,
        "package": "turbocore_v5_archive_integrity_ledger_v0",
        "gate": "v5_archive_integrity_ledger",
        "ok": ready,
        "ledger_ready": ready,
        "archive_integrity_ledger_ready": ready,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "runs_dispatched": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_ledger_request_fields": {},
        "p33_archive_signoff_summary": p33_summary,
        "p32_owner_package_summary": p32_summary,
        "evidence_chain": evidence_chain,
        "artifact_digests": artifacts,
        "ledger_digest": _digest_value(
            {
                "artifact_digests": artifacts,
                "evidence_chain": evidence_chain,
                "p33_decision": p33_summary.get("decision"),
                "p32_decision": p32_summary.get("decision"),
            }
        ),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "recommended_next_step": _recommended_next_step(ready, blockers),
        "notes": [
            "P34 records deterministic evidence digests only; it does not approve rollout.",
            "Source paths are excluded from digests so archive hashes are portable.",
            "Any next run remains explicit, manual-only, and default-off.",
        ],
    }


def _p33_summary(signoff: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(signoff),
        "source_path": str(signoff.get("_source_path") or signoff.get("source_path") or ""),
        "ok": bool(signoff.get("ok", False)),
        "decision_record_ready": bool(signoff.get("decision_record_ready", False)),
        "owner_archive_package_ready": bool(signoff.get("owner_archive_package_ready", False)),
        "approved_for_owner_archive": bool(signoff.get("approved_for_owner_archive", False)),
        "rejected_for_default_off_hold": bool(signoff.get("rejected_for_default_off_hold", False)),
        "rollback_required": bool(signoff.get("rollback_required", False)),
        "decision": str(signoff.get("decision") or signoff.get("gate_decision") or signoff.get("package_decision") or ""),
        "default_off": _default_off_confirmed(signoff),
        "request_adapter_off": _request_adapter_off(signoff),
        "training_launch_allowed": bool(signoff.get("training_launch_allowed", True)),
        "auto_launch_allowed": bool(signoff.get("auto_launch_allowed", True)),
        "runs_dispatched": bool(signoff.get("runs_dispatched", True)),
        "post_fields_empty": not bool(_as_dict(signoff.get("post_archive_request_fields"))),
        "blocked_reasons": _string_list(signoff.get("blocked_reasons")),
        "ready": _p33_ready(signoff),
    }


def _p33_ready(signoff: Mapping[str, Any]) -> bool:
    return bool(
        signoff
        and signoff.get("ok") is True
        and signoff.get("decision_record_ready") is True
        and signoff.get("owner_archive_package_ready") is True
        and signoff.get("approved_for_owner_archive") is True
        and not bool(signoff.get("rejected_for_default_off_hold", False))
        and not bool(signoff.get("rollback_required", False))
        and str(signoff.get("decision") or signoff.get("gate_decision") or signoff.get("package_decision") or "")
        == P33_READY_DECISION
        and signoff.get("training_launch_allowed") is False
        and signoff.get("auto_launch_allowed") is False
        and signoff.get("runs_dispatched") is False
        and _default_off_confirmed(signoff)
        and _request_adapter_off(signoff)
        and not _as_dict(signoff.get("post_archive_request_fields"))
    )


def _p32_summary(package: Mapping[str, Any]) -> dict[str, Any]:
    p31 = _as_dict(package.get("p31_manual_run_audit_summary"))
    p29 = _as_dict(package.get("p29_owner_next_stage_package"))
    p27 = _as_dict(package.get("p27_decision"))
    p26 = _as_dict(package.get("p26_gate"))
    p28 = _as_dict(package.get("p28_collector_bundle"))
    return {
        "present": bool(package),
        "source_path": str(package.get("_source_path") or package.get("source_path") or ""),
        "ok": bool(package.get("ok", False)),
        "p31_collector_replay_ready": bool(package.get("p31_collector_replay_ready", False)),
        "owner_next_stage_package_ready": bool(package.get("owner_next_stage_package_ready", False)),
        "ready_for_signed_next_stage_review": bool(package.get("ready_for_signed_next_stage_review", False)),
        "decision": str(package.get("decision") or package.get("gate_decision") or package.get("package_decision") or ""),
        "default_off": _default_off_confirmed(package),
        "request_adapter_off": _request_adapter_off(package),
        "training_launch_allowed": bool(package.get("training_launch_allowed", True)),
        "auto_launch_allowed": bool(package.get("auto_launch_allowed", True)),
        "runs_dispatched": bool(package.get("runs_dispatched", True)),
        "post_fields_empty": not bool(_as_dict(package.get("post_replay_request_fields"))),
        "p31_decision": str(p31.get("decision") or ""),
        "p31_ready": bool(p31.get("manual_run_audit_ready", False))
        and bool(p31.get("collector_evidence_ready", False))
        and str(p31.get("decision") or "") == P31_READY_DECISION,
        "p28_ready": bool(p28.get("longer_replicate_evidence_ready", False)),
        "p26_decision": str(p26.get("decision") or p26.get("gate_decision") or p26.get("rollout_review_decision") or ""),
        "p26_ready": bool(p26.get("longer_replicate_failure_history_gate_ready", False))
        and str(p26.get("decision") or p26.get("gate_decision") or p26.get("rollout_review_decision") or "")
        == P26_READY_DECISION,
        "p27_decision": str(p27.get("decision") or p27.get("gate_decision") or p27.get("next_stage_review_decision") or ""),
        "p27_ready": bool(p27.get("signed_next_stage_review_signed", False))
        and bool(p27.get("signed_next_stage_review_recorded", False))
        and bool(p27.get("approved_for_next_contract_stage", False))
        and not bool(p27.get("rejected_for_default_off_hold", False))
        and not bool(p27.get("rollback_required", False))
        and str(p27.get("decision") or p27.get("gate_decision") or p27.get("next_stage_review_decision") or "")
        == P27_APPROVED_DECISION,
        "p29_decision": str(p29.get("decision") or p29.get("gate_decision") or p29.get("package_decision") or ""),
        "p29_ready": bool(p29.get("package_ready", False))
        and bool(p29.get("ready_for_owner_archive", False))
        and str(p29.get("decision") or p29.get("gate_decision") or p29.get("package_decision") or "")
        == P29_READY_DECISION,
        "run_count": int(p28.get("run_count", 0) or 0),
        "ready_run_count": int(p28.get("ready_run_count", 0) or 0),
        "blocked_reasons": _string_list(package.get("blocked_reasons")),
        "ready": _p32_ready(package, p31, p28, p26, p27, p29),
    }


def _p32_ready(
    package: Mapping[str, Any],
    p31: Mapping[str, Any],
    p28: Mapping[str, Any],
    p26: Mapping[str, Any],
    p27: Mapping[str, Any],
    p29: Mapping[str, Any],
) -> bool:
    return bool(
        package
        and package.get("ok") is True
        and package.get("p31_collector_replay_ready") is True
        and package.get("owner_next_stage_package_ready") is True
        and not bool(package.get("ready_for_signed_next_stage_review", False))
        and str(package.get("decision") or package.get("gate_decision") or package.get("package_decision") or "")
        == P32_READY_DECISION
        and package.get("training_launch_allowed") is False
        and package.get("auto_launch_allowed") is False
        and package.get("runs_dispatched") is False
        and _default_off_confirmed(package)
        and _request_adapter_off(package)
        and not _as_dict(package.get("post_replay_request_fields"))
        and bool(p31.get("manual_run_audit_ready", False))
        and bool(p31.get("collector_evidence_ready", False))
        and str(p31.get("decision") or "") == P31_READY_DECISION
        and bool(p28.get("longer_replicate_evidence_ready", False))
        and bool(p26.get("longer_replicate_failure_history_gate_ready", False))
        and str(p26.get("decision") or p26.get("gate_decision") or p26.get("rollout_review_decision") or "")
        == P26_READY_DECISION
        and bool(p27.get("signed_next_stage_review_signed", False))
        and bool(p27.get("signed_next_stage_review_recorded", False))
        and bool(p27.get("approved_for_next_contract_stage", False))
        and str(p27.get("decision") or p27.get("gate_decision") or p27.get("next_stage_review_decision") or "")
        == P27_APPROVED_DECISION
        and bool(p29.get("package_ready", False))
        and bool(p29.get("ready_for_owner_archive", False))
        and str(p29.get("decision") or p29.get("gate_decision") or p29.get("package_decision") or "")
        == P29_READY_DECISION
    )


def _evidence_chain(
    p33_summary: Mapping[str, Any],
    p32_summary: Mapping[str, Any],
    p32: Mapping[str, Any],
) -> list[dict[str, Any]]:
    p31 = _as_dict(p32.get("p31_manual_run_audit_summary"))
    p28 = _as_dict(p32.get("p28_collector_bundle"))
    p26 = _as_dict(p32.get("p26_gate"))
    p27 = _as_dict(p32.get("p27_decision"))
    p29 = _as_dict(p32.get("p29_owner_next_stage_package"))
    return [
        _chain_item("p33_archive_signoff", p33_summary.get("decision"), p33_summary.get("ready")),
        _chain_item("p32_owner_package", p32_summary.get("decision"), p32_summary.get("ready")),
        _chain_item("p31_manual_audit_summary", p31.get("decision"), p32_summary.get("p31_ready")),
        _chain_item("p28_collector_bundle", p28.get("decision") or p28.get("gate_decision"), p32_summary.get("p28_ready")),
        _chain_item("p26_failure_history_gate", p26.get("decision") or p26.get("gate_decision"), p32_summary.get("p26_ready")),
        _chain_item("p27_next_stage_decision", p27.get("decision") or p27.get("gate_decision"), p32_summary.get("p27_ready")),
        _chain_item("p29_owner_package", p29.get("decision") or p29.get("gate_decision"), p32_summary.get("p29_ready")),
    ]


def _chain_item(name: str, decision: Any, ready: Any) -> dict[str, Any]:
    return {"name": name, "decision": str(decision or ""), "ready": bool(ready)}


def _artifact_digests(p33: Mapping[str, Any], p32: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = [
        _digest_record("p33_archive_signoff", p33),
        _digest_record("p32_owner_package", p32),
    ]
    if p32:
        artifacts.extend(
            [
                _digest_record("p31_manual_audit_summary", _as_dict(p32.get("p31_manual_run_audit_summary"))),
                _digest_record("p28_collector_bundle", _as_dict(p32.get("p28_collector_bundle"))),
                _digest_record("p26_failure_history_gate", _as_dict(p32.get("p26_gate"))),
                _digest_record("p27_next_stage_decision", _as_dict(p32.get("p27_decision"))),
                _digest_record("p29_owner_package", _as_dict(p32.get("p29_owner_next_stage_package"))),
            ]
        )
    return artifacts


def _digest_record(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    canonical = _canonical_json(value)
    return {
        "name": name,
        "present": bool(value),
        "sha256": hashlib.sha256(canonical).hexdigest() if value else "",
        "canonical_size_bytes": len(canonical) if value else 0,
    }


def _blockers(
    p33_summary: Mapping[str, Any],
    p32_summary: Mapping[str, Any],
    artifacts: list[Mapping[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p33_summary.get("present", False)):
        blocked.append("v5_p34_p33_archive_signoff_missing")
    if not bool(p33_summary.get("ready", False)):
        blocked.append("v5_p34_p33_archive_signoff_not_ready")
        blocked.extend(_string_list(p33_summary.get("blocked_reasons")))
    if not bool(p32_summary.get("present", False)):
        blocked.append("v5_p34_p32_owner_package_missing")
    if not bool(p32_summary.get("ready", False)):
        blocked.append("v5_p34_p32_owner_package_not_ready")
        blocked.extend(_string_list(p32_summary.get("blocked_reasons")))
    for field, reason in (
        ("p31_ready", "v5_p34_p31_manual_audit_not_ready"),
        ("p28_ready", "v5_p34_p28_collector_bundle_not_ready"),
        ("p26_ready", "v5_p34_p26_gate_not_ready"),
        ("p27_ready", "v5_p34_p27_decision_not_ready"),
        ("p29_ready", "v5_p34_p29_package_not_ready"),
    ):
        if not bool(p32_summary.get(field, False)):
            blocked.append(reason)
    if not bool(p33_summary.get("post_fields_empty", False)):
        blocked.append("v5_p34_p33_post_fields_present")
    if not bool(p32_summary.get("post_fields_empty", False)):
        blocked.append("v5_p34_p32_post_fields_present")
    if any(not bool(item.get("present", False)) for item in artifacts):
        blocked.append("v5_p34_archive_artifact_missing")
    if any(item.get("present") and not item.get("sha256") for item in artifacts):
        blocked.append("v5_p34_archive_digest_missing")
    return _dedupe(blocked)


def _recommended_next_step(ready: bool, blockers: list[str]) -> str:
    if ready:
        return "archive the P34 ledger with P32/P33 evidence; any next run remains explicit and default-off"
    if any(item.startswith("v5_p34_p33") for item in blockers):
        return "repair or sign the P33 archive signoff before generating a ledger"
    if any(item.startswith("v5_p34_p32") for item in blockers):
        return "repair the P32 owner package before generating a ledger"
    return "hold P34 until the full archive evidence chain is present and ready"


def _digest_value(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    return value


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
    parser = argparse.ArgumentParser(description="Build V5 P34 archive integrity ledger.")
    parser.add_argument("--p33-archive-signoff", default="", help="P33 owner archive signoff JSON.")
    parser.add_argument("--p32-owner-package", default="", help="P32 P31 collector replay owner package JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_archive_integrity_ledger(
        p33_archive_signoff=load_json(args.p33_archive_signoff) if args.p33_archive_signoff else None,
        p32_owner_package=load_json(args.p32_owner_package) if args.p32_owner_package else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_archive_integrity_ledger"]
