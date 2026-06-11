"""Archive replay verifier for TurboCore V5-P35.

P35 replays the P34 archive integrity ledger from current P32/P33 evidence and
compares it with a previously archived ledger. It is report-only: no training is
launched, no request-adapter fields are emitted, and no default rollout changes.
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

from core.turbocore_v5_archive_integrity_ledger import build_v5_archive_integrity_ledger
from core.turbocore_v5_owner_review_evidence_package import load_json


P34_READY_DECISION = "p32_final_archive_integrity_ledger_ready_default_off"
P35_READY_DECISION = "p32_final_archive_replay_verification_ready_default_off"
P35_BLOCKED_DECISION = "p32_final_archive_replay_verification_blocked_default_off"


def build_v5_archive_replay_verifier(
    *,
    archived_archive_ledger: Mapping[str, Any] | None = None,
    p33_archive_signoff: Mapping[str, Any] | None = None,
    p32_owner_package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify an archived P34 ledger against a fresh P34 replay."""

    archived = _as_dict(archived_archive_ledger)
    replay = build_v5_archive_integrity_ledger(
        p33_archive_signoff=p33_archive_signoff,
        p32_owner_package=p32_owner_package,
    )
    archived_summary = _ledger_summary(archived)
    replay_summary = _ledger_summary(replay)
    digest_comparisons = _digest_comparisons(archived, replay)
    blockers = _blockers(archived_summary, replay_summary, digest_comparisons)
    ready = not blockers
    decision = P35_READY_DECISION if ready else P35_BLOCKED_DECISION
    return {
        "schema_version": 1,
        "package": "turbocore_v5_archive_replay_verifier_v0",
        "gate": "v5_archive_replay_verification",
        "ok": ready,
        "archive_replay_verification_ready": ready,
        "archived_ledger_matches_replay": ready,
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
        "post_replay_verification_request_fields": {},
        "archived_ledger_summary": archived_summary,
        "replay_ledger_summary": replay_summary,
        "digest_comparisons": digest_comparisons,
        "replay_ledger": replay,
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "recommended_next_step": _recommended_next_step(ready, blockers),
        "notes": [
            "P35 only verifies archived P34 digests against a fresh replay.",
            "Replay verification does not approve rollout, UI wiring, or request-adapter mapping.",
            "Any later real run remains explicit, manual-only, and default-off.",
        ],
    }


def build_v5_archive_replay_verification(
    *,
    archived_archive_ledger: Mapping[str, Any] | None = None,
    p33_archive_signoff: Mapping[str, Any] | None = None,
    p32_owner_package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility alias matching the P35 contract wording."""

    return build_v5_archive_replay_verifier(
        archived_archive_ledger=archived_archive_ledger,
        p33_archive_signoff=p33_archive_signoff,
        p32_owner_package=p32_owner_package,
    )


def _ledger_summary(ledger: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = _artifact_map(ledger.get("artifact_digests"))
    ready = _ledger_ready(ledger)
    return {
        "present": bool(ledger),
        "ok": bool(ledger.get("ok", False)),
        "ledger_ready": bool(ledger.get("ledger_ready", False)),
        "archive_integrity_ledger_ready": bool(ledger.get("archive_integrity_ledger_ready", False)),
        "decision": str(ledger.get("decision") or ledger.get("gate_decision") or ledger.get("package_decision") or ""),
        "ledger_digest": str(ledger.get("ledger_digest") or ""),
        "artifact_count": len(artifacts),
        "artifact_names": sorted(artifacts),
        "default_off": _default_off_confirmed(ledger),
        "request_adapter_off": _request_adapter_off(ledger),
        "training_launch_allowed": bool(ledger.get("training_launch_allowed", True)),
        "auto_launch_allowed": bool(ledger.get("auto_launch_allowed", True)),
        "runs_dispatched": bool(ledger.get("runs_dispatched", True)),
        "post_fields_empty": not bool(_as_dict(ledger.get("post_ledger_request_fields"))),
        "blocked_reasons": _string_list(ledger.get("blocked_reasons")),
        "ready": ready,
    }


def _ledger_ready(ledger: Mapping[str, Any]) -> bool:
    return bool(
        ledger
        and ledger.get("ok") is True
        and ledger.get("ledger_ready") is True
        and ledger.get("archive_integrity_ledger_ready") is True
        and str(ledger.get("decision") or ledger.get("gate_decision") or ledger.get("package_decision") or "")
        == P34_READY_DECISION
        and bool(ledger.get("ledger_digest"))
        and ledger.get("training_launch_allowed") is False
        and ledger.get("auto_launch_allowed") is False
        and ledger.get("runs_dispatched") is False
        and _default_off_confirmed(ledger)
        and _request_adapter_off(ledger)
        and not _as_dict(ledger.get("post_ledger_request_fields"))
    )


def _digest_comparisons(
    archived: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> list[dict[str, Any]]:
    comparisons = [_ledger_digest_comparison(archived, replay)]
    archived_artifacts = _artifact_map(archived.get("artifact_digests"))
    replay_artifacts = _artifact_map(replay.get("artifact_digests"))
    for name in sorted(set(archived_artifacts) | set(replay_artifacts)):
        comparisons.append(_artifact_comparison(name, archived_artifacts.get(name), replay_artifacts.get(name)))
    return comparisons


def _ledger_digest_comparison(archived: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    archived_digest = str(archived.get("ledger_digest") or "")
    replay_digest = str(replay.get("ledger_digest") or "")
    match = bool(archived_digest and replay_digest and archived_digest == replay_digest)
    return {
        "name": "ledger_digest",
        "kind": "ledger",
        "archived_present": bool(archived),
        "replay_present": bool(replay),
        "archived_sha256": archived_digest,
        "replay_sha256": replay_digest,
        "match": match,
        "reason": "" if match else "v5_p35_ledger_digest_mismatch",
    }


def _artifact_comparison(
    name: str,
    archived: Mapping[str, Any] | None,
    replay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    archived_item = _as_dict(archived)
    replay_item = _as_dict(replay)
    archived_sha = str(archived_item.get("sha256") or "")
    replay_sha = str(replay_item.get("sha256") or "")
    archived_size = int(archived_item.get("canonical_size_bytes", 0) or 0)
    replay_size = int(replay_item.get("canonical_size_bytes", 0) or 0)
    reason = ""
    if not archived_item:
        reason = f"v5_p35_archived_artifact_missing:{name}"
    elif not replay_item:
        reason = f"v5_p35_replay_artifact_missing:{name}"
    elif archived_sha != replay_sha:
        reason = f"v5_p35_artifact_digest_mismatch:{name}"
    elif archived_size != replay_size:
        reason = f"v5_p35_artifact_size_mismatch:{name}"
    return {
        "name": name,
        "kind": "artifact",
        "archived_present": bool(archived_item),
        "replay_present": bool(replay_item),
        "archived_sha256": archived_sha,
        "replay_sha256": replay_sha,
        "archived_size_bytes": archived_size,
        "replay_size_bytes": replay_size,
        "match": not bool(reason),
        "reason": reason,
    }


def _blockers(
    archived_summary: Mapping[str, Any],
    replay_summary: Mapping[str, Any],
    digest_comparisons: list[Mapping[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if not bool(archived_summary.get("present", False)):
        blocked.append("v5_p35_archived_ledger_missing")
    elif not bool(archived_summary.get("ready", False)):
        blocked.append("v5_p35_archived_ledger_not_ready")
        blocked.extend(_string_list(archived_summary.get("blocked_reasons")))
    if not bool(replay_summary.get("ready", False)):
        blocked.append("v5_p35_replay_ledger_not_ready")
        blocked.extend(_string_list(replay_summary.get("blocked_reasons")))
    if int(archived_summary.get("artifact_count", 0) or 0) != int(replay_summary.get("artifact_count", 0) or 0):
        blocked.append("v5_p35_artifact_count_mismatch")
    for item in digest_comparisons:
        if not bool(item.get("match", False)):
            blocked.append(str(item.get("reason") or "v5_p35_digest_comparison_mismatch"))
    return _dedupe(blocked)


def _artifact_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for item in value:
        record = _as_dict(item)
        name = str(record.get("name") or "")
        if name:
            mapped[name] = record
    return mapped


def _recommended_next_step(ready: bool, blockers: list[str]) -> str:
    if ready:
        return "archive the replay verification report with the P34 ledger; any next run remains explicit and default-off"
    if "v5_p35_archived_ledger_missing" in blockers:
        return "provide the archived P34 ledger before replay verification"
    if "v5_p35_archived_ledger_not_ready" in blockers:
        return "archive a ready P34 ledger before replay verification"
    if "v5_p35_replay_ledger_not_ready" in blockers:
        return "repair current P32/P33 evidence so the P34 replay is ready"
    return "hold P35 until archived and replayed digests match exactly"


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
    parser = argparse.ArgumentParser(description="Verify a V5 P34 archive ledger by replay.")
    parser.add_argument(
        "--archived-archive-ledger",
        "--archived-ledger",
        dest="archived_archive_ledger",
        default="",
        help="Archived P34 ledger JSON.",
    )
    parser.add_argument("--p33-archive-signoff", default="", help="Current P33 owner archive signoff JSON.")
    parser.add_argument("--p32-owner-package", default="", help="Current P32 owner package JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_archive_replay_verifier(
        archived_archive_ledger=load_json(args.archived_archive_ledger) if args.archived_archive_ledger else None,
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


__all__ = ["build_v5_archive_replay_verification", "build_v5_archive_replay_verifier"]
