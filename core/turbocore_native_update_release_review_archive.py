"""Archive gate for the TurboCore native-update release review package.

This gate is report-only.  It records whether a signed/default-off release
review package can be archived for owner release direction, while keeping
request, UI/schema, routing, training, runtime, and native dispatch closed.
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

from core.turbocore_native_update_release_review_package import READY_DECISION, SUPPLEMENTAL_GATES  # noqa: E402
from core.turbocore_v5_controlled_rollout_policy_evidence_gate_utils import (  # noqa: E402
    as_dict as _as_dict,
    dedupe as _dedupe,
    string_list as _string_list,
)


ARCHIVE_READY_DECISION = "native_update_release_review_archive_ready_default_off"
ARCHIVE_HOLD_DECISION = "native_update_release_review_archive_hold_for_recorded_review_default_off"
ARCHIVE_BLOCKED_DECISION = "native_update_release_review_archive_blocked_default_off"
GATE = "native_update_release_review_archive"
PACKAGE_NAME = "turbocore_native_update_release_review_archive_v0"
DEFAULT_PACKAGE_PATH = REPO_ROOT / "temp" / "turbocore_optimizer" / "native_update_release_review_package.json"
DEFAULT_OWNER_RELEASE_REVIEW_RECORD_PATH = (
    REPO_ROOT / "temp" / "turbocore_optimizer" / "native_update_owner_release_review_record.json"
)
DEFAULT_ARCHIVE_PATH = REPO_ROOT / "temp" / "turbocore_optimizer" / "native_update_release_review_archive.json"

UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "product_exposure_allowed",
    "release_gate_open",
    "training_launch_allowed",
    "training_launch_enabled",
    "training_launch_executed",
    "training_path_enabled",
    "training_dispatch",
    "runtime_dispatch_allowed",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_executed",
    "request_submitted",
    "job_created",
    "queue_enqueued",
    "run_record_written",
    "ready_for_ui",
    "ui_exposure_allowed",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "backend_router_registered",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_archive_request_fields",
    "post_release_request_fields",
    "post_native_update_request_fields",
    "post_product_exposure_request_fields",
    "request_adapter_fields",
    "request_schema_fields",
    "ui_route_registration",
    "backend_router_registration",
)


def build_native_update_release_review_archive(
    *,
    release_review_package: Mapping[str, Any] | None = None,
    owner_release_review_record: Mapping[str, Any] | None = None,
    load_owner_release_review_record: bool = True,
    write_artifact: bool = True,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    package = _as_dict(release_review_package) or _load_release_review_package()
    owner_record = (
        _as_dict(owner_release_review_record)
        if owner_release_review_record is not None
        else _load_owner_release_review_record()
        if load_owner_release_review_record
        else {}
    )
    summary = _package_summary(package)
    owner_record_summary = _owner_release_review_record_summary(owner_record)
    blockers = _archive_blockers(summary, owner_record_summary)
    archive_ready = not blockers
    waiting_for_recorded_review = (
        bool(summary.get("present"))
        and not archive_ready
        and _string_list(summary.get("blocked_reasons")) == ["native_update_release_owner_review_missing"]
        and "native_update_release_review_archive_review_not_recorded" in blockers
    )
    decision = (
        ARCHIVE_READY_DECISION
        if archive_ready
        else ARCHIVE_HOLD_DECISION
        if waiting_for_recorded_review
        else ARCHIVE_BLOCKED_DECISION
    )
    evidence_ready = archive_ready or waiting_for_recorded_review
    report = {
        "schema_version": 1,
        "package": PACKAGE_NAME,
        "gate": GATE,
        "ok": archive_ready,
        "evidence_ready": evidence_ready,
        "ready_for_review": evidence_ready,
        "archive_ready": archive_ready,
        "archive_recorded": archive_ready,
        "ready_for_owner_release_direction": archive_ready,
        "manual_review_required": True,
        "release_review_recorded": bool(summary.get("release_review_recorded", False)),
        "decision": decision,
        "source_decision": str(summary.get("decision") or ""),
        "default_behavior_changed": False,
        "product_exposure_allowed": False,
        "release_gate_open": False,
        "training_launch_allowed": False,
        "training_launch_enabled": False,
        "training_launch_executed": False,
        "training_path_enabled": False,
        "training_dispatch": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "native_dispatch_enabled": False,
        "native_dispatch_executed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "post_archive_request_fields": {},
        "post_release_request_fields": {},
        "release_review_package_summary": summary,
        "owner_release_review_record_summary": owner_record_summary,
        "allowed_next_actions": (
            ["await_owner_release_direction"] if archive_ready else ["record_owner_release_review_default_off"]
        ),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "recommended_next_step": (
            "archive release review package and await separate owner release direction"
            if archive_ready
            else "record owner release review while keeping product exposure disabled"
            if waiting_for_recorded_review
            else "repair native update release review package before archive"
        ),
        "notes": [
            "Archive readiness does not enable request, UI/schema, backend route, runtime, or native dispatch.",
            "A recorded release review still requires a separate owner release direction before product exposure.",
        ],
    }
    report["default_off"] = _default_off(report)
    if write_artifact:
        path = artifact_path or DEFAULT_ARCHIVE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_release_review_package() -> dict[str, Any]:
    if not DEFAULT_PACKAGE_PATH.exists():
        return {}
    try:
        return _as_dict(json.loads(DEFAULT_PACKAGE_PATH.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _load_owner_release_review_record() -> dict[str, Any]:
    if not DEFAULT_OWNER_RELEASE_REVIEW_RECORD_PATH.exists():
        return {}
    try:
        return _as_dict(json.loads(DEFAULT_OWNER_RELEASE_REVIEW_RECORD_PATH.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _package_summary(package: Mapping[str, Any]) -> dict[str, Any]:
    supplemental_summaries = _supplemental_summaries(package)
    supplemental = supplemental_summaries["optimizer_family_coverage"]
    multitensor = supplemental_summaries["native_update_optimizer_multitensor_release_hold"]
    optimizer_counts = _int_dict(supplemental.get("optimizer_family_counts"))
    handoff_counts = _optimizer_family_handoff_counts(package)
    source_summary = _optimizer_family_source_summary(supplemental)
    handoff_sources = _optimizer_family_handoff_sources(package)
    template = _as_dict(package.get("release_review_template"))
    handoff_template_digest = str(
        _as_dict(package.get("owner_release_review_handoff")).get("release_review_template_digest") or ""
    )
    computed_template_digest = _digest_payload(template)
    return {
        "present": bool(package),
        "ok": package.get("ok") is True,
        "evidence_ready": package.get("evidence_ready") is True,
        "ready_for_review": package.get("ready_for_review") is True,
        "ready_for_owner_release_review": package.get("ready_for_owner_release_review") is True,
        "release_review_recorded": package.get("release_review_recorded") is True,
        "decision": str(package.get("decision") or ""),
        "blocked_reasons": _string_list(package.get("blocked_reasons")),
        "promotion_blockers": _string_list(package.get("promotion_blockers")),
        "expected_gate_count": int(package.get("expected_gate_count", 0) or 0),
        "present_gate_count": int(package.get("present_gate_count", 0) or 0),
        "default_off_gate_count": int(package.get("default_off_gate_count", 0) or 0),
        "supplemental_gate_count": int(package.get("supplemental_gate_count", 0) or 0),
        "present_supplemental_gate_count": int(package.get("present_supplemental_gate_count", 0) or 0),
        "default_off_supplemental_gate_count": int(package.get("default_off_supplemental_gate_count", 0) or 0),
        "supplemental_gate_summaries": supplemental_summaries,
        "optimizer_family_coverage_ok": supplemental.get("ok") is True,
        "optimizer_family_coverage_evidence_ready": supplemental.get("evidence_ready") is True,
        "optimizer_family_coverage_ready_for_review": supplemental.get("ready_for_review") is True,
        "optimizer_family_coverage_default_off": supplemental.get("default_off") is True,
        "optimizer_family_source_count": source_summary["source_count"],
        "optimizer_family_source_names": source_summary["source_names"],
        "optimizer_family_source_payload_digest_match": source_summary["source_payload_digest_match"],
        "optimizer_family_handoff_sources": handoff_sources,
        "optimizer_family_handoff_sources_match": bool(source_summary and handoff_sources == source_summary),
        "optimizer_family_counts": optimizer_counts,
        "optimizer_family_handoff_counts": handoff_counts,
        "optimizer_family_handoff_counts_match": bool(optimizer_counts and handoff_counts == optimizer_counts),
        "native_update_optimizer_multitensor_release_hold_ok": multitensor.get("ok") is True,
        "native_update_optimizer_multitensor_release_hold_evidence_ready": multitensor.get("evidence_ready") is True,
        "native_update_optimizer_multitensor_release_hold_ready_for_review": multitensor.get("ready_for_review") is True,
        "native_update_optimizer_multitensor_release_hold_default_off": multitensor.get("default_off") is True,
        "release_review_template_digest": computed_template_digest,
        "handoff_release_review_template_digest": handoff_template_digest,
        "handoff_release_review_template_digest_match": bool(
            computed_template_digest and handoff_template_digest == computed_template_digest
        ),
        "reported_default_off": package.get("default_off") is True,
        "default_off": _default_off(package),
        "post_fields_empty": all(not _as_dict(package.get(field)) for field in UNSAFE_NON_EMPTY_FIELDS),
        "unsafe_claims": _unsafe_claims(package),
    }


def _owner_release_review_record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(record),
        "owner_packet_ready": record.get("owner_packet_ready") is True,
        "signed_review_present": record.get("signed_review_present") is True,
        "signed_review_valid": record.get("signed_review_valid") is True,
        "approval_recorded": record.get("approval_recorded") is True,
        "release_review_recorded": record.get("release_review_recorded") is True,
        "decision": str(record.get("decision") or ""),
        "source_release_review_template_digest": str(record.get("source_release_review_template_digest") or ""),
        "signed_review_template_digest": str(record.get("signed_review_template_digest") or ""),
        "signed_review_digest_match": record.get("signed_review_digest_match") is True,
        "release_package_decision": str(record.get("release_package_decision") or ""),
        "release_package_digest": str(record.get("release_package_digest") or ""),
        "blocked_reasons": _string_list(record.get("blocked_reasons")),
        "unsafe_claims": _unsafe_claims(record),
    }


def _archive_blockers(
    summary: Mapping[str, Any],
    owner_record_summary: Mapping[str, Any] | None = None,
) -> list[str]:
    blocked: list[str] = []
    if not summary.get("present"):
        blocked.append("native_update_release_review_archive_package_missing")
        return blocked
    if not summary.get("ok"):
        blocked.append("native_update_release_review_archive_package_not_ok")
    if not summary.get("evidence_ready"):
        blocked.append("native_update_release_review_archive_evidence_not_ready")
    if not summary.get("ready_for_review"):
        blocked.append("native_update_release_review_archive_not_ready_for_review")
    if not summary.get("ready_for_owner_release_review"):
        blocked.append("native_update_release_review_archive_not_ready_for_owner_release_review")
    if not summary.get("release_review_recorded"):
        blocked.append("native_update_release_review_archive_review_not_recorded")
    else:
        blocked.extend(_owner_release_review_record_blockers(_as_dict(owner_record_summary)))
    if str(summary.get("decision") or "") != READY_DECISION:
        blocked.append("native_update_release_review_archive_source_decision_not_recorded_default_off")
    if summary.get("present_gate_count") != summary.get("expected_gate_count"):
        blocked.append("native_update_release_review_archive_expected_gate_missing")
    if summary.get("default_off_gate_count") != summary.get("expected_gate_count"):
        blocked.append("native_update_release_review_archive_expected_gate_not_default_off")
    if summary.get("present_supplemental_gate_count") != summary.get("supplemental_gate_count"):
        blocked.append("native_update_release_review_archive_supplemental_gate_missing")
    if summary.get("default_off_supplemental_gate_count") != summary.get("supplemental_gate_count"):
        blocked.append("native_update_release_review_archive_supplemental_gate_not_default_off")
    for gate, item in _as_dict(summary.get("supplemental_gate_summaries")).items():
        if item.get("present") is not True:
            blocked.append(f"native_update_release_review_archive_supplemental_gate_missing:{gate}")
            continue
        for field in ("ok", "evidence_ready", "ready_for_review", "default_off"):
            if item.get(field) is not True:
                blocked.append(f"native_update_release_review_archive_supplemental_gate_{field}_failed:{gate}")
        for claim in _string_list(item.get("unsafe_claims")):
            blocked.append(f"native_update_release_review_archive_supplemental_gate_unsafe:{gate}:{claim}")
    for field in (
        "optimizer_family_coverage_ok",
        "optimizer_family_coverage_evidence_ready",
        "optimizer_family_coverage_ready_for_review",
        "optimizer_family_coverage_default_off",
        "optimizer_family_source_payload_digest_match",
        "optimizer_family_handoff_sources_match",
        "optimizer_family_handoff_counts_match",
        "native_update_optimizer_multitensor_release_hold_ok",
        "native_update_optimizer_multitensor_release_hold_evidence_ready",
        "native_update_optimizer_multitensor_release_hold_ready_for_review",
        "native_update_optimizer_multitensor_release_hold_default_off",
        "handoff_release_review_template_digest_match",
        "reported_default_off",
        "default_off",
        "post_fields_empty",
    ):
        if not summary.get(field):
            blocked.append(f"native_update_release_review_archive_{field}_failed")
    blocked.extend(_string_list(summary.get("unsafe_claims")))
    blocked.extend(_string_list(summary.get("promotion_blockers")))
    unexpected_blockers = [
        item for item in _string_list(summary.get("blocked_reasons"))
        if item != "native_update_release_owner_review_missing"
    ]
    blocked.extend(unexpected_blockers)
    return _dedupe(blocked)


def _owner_release_review_record_blockers(summary: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if not summary.get("present"):
        return ["native_update_release_review_archive_owner_record_missing"]
    for field in (
        "owner_packet_ready",
        "signed_review_present",
        "signed_review_valid",
        "approval_recorded",
        "release_review_recorded",
        "signed_review_digest_match",
    ):
        if summary.get(field) is not True:
            blocked.append(f"native_update_release_review_archive_owner_record_{field}_failed")
    if str(summary.get("release_package_decision") or "") != READY_DECISION:
        blocked.append("native_update_release_review_archive_owner_record_package_decision_not_recorded_default_off")
    for claim in _string_list(summary.get("unsafe_claims")):
        blocked.append(f"native_update_release_review_archive_owner_record_unsafe:{claim}")
    blocked.extend(
        f"native_update_release_review_archive_owner_record:{reason}"
        for reason in _string_list(summary.get("blocked_reasons"))
    )
    return blocked


def _default_off(package: Mapping[str, Any]) -> bool:
    return not _unsafe_claims(package) and all(not _as_dict(package.get(field)) for field in UNSAFE_NON_EMPTY_FIELDS)


def _int_dict(value: Any) -> dict[str, int]:
    return {str(key): int(item or 0) for key, item in _as_dict(value).items()}


def _supplemental_summaries(package: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    summaries = _as_dict(package.get("supplemental_gate_summaries"))
    return {
        gate: _compact_supplemental_summary(gate, _as_dict(summaries.get(gate)))
        for gate in SUPPLEMENTAL_GATES
    }


def _compact_supplemental_summary(gate: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        "present": summary.get("present") is True,
        "gate": str(summary.get("gate") or gate),
        "ok": summary.get("ok") is True,
        "evidence_ready": summary.get("evidence_ready") is True,
        "ready_for_review": summary.get("ready_for_review") is True,
        "default_off": summary.get("default_off") is True,
        "recommended_next_step": str(summary.get("recommended_next_step") or ""),
        "priority_next_gates": _string_list(summary.get("priority_next_gates")),
        "unsafe_claims": _string_list(summary.get("unsafe_claims")),
        "blocked_reasons": _string_list(summary.get("blocked_reasons")),
    }
    if gate == "optimizer_family_coverage":
        compact.update(
            {
                "source_count": int(summary.get("source_count", 0) or 0),
                "source_names": _string_list(summary.get("source_names")),
                "source_payload_digest_match": summary.get("source_payload_digest_match") is not False,
                "optimizer_family_counts": _int_dict(summary.get("optimizer_family_counts")),
            }
        )
    return compact


def _optimizer_family_handoff_counts(package: Mapping[str, Any]) -> dict[str, int]:
    handoff = _as_dict(package.get("owner_release_review_handoff"))
    counts = _as_dict(handoff.get("supplemental_acknowledgement_counts"))
    return _int_dict(counts.get("optimizer_family_coverage"))


def _optimizer_family_source_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_count": int(summary.get("source_count", 0) or 0),
        "source_names": _string_list(summary.get("source_names")),
        "source_payload_digest_match": summary.get("source_payload_digest_match") is not False,
    }


def _optimizer_family_handoff_sources(package: Mapping[str, Any]) -> dict[str, Any]:
    handoff = _as_dict(package.get("owner_release_review_handoff"))
    sources = _as_dict(handoff.get("supplemental_acknowledgement_sources"))
    return _optimizer_family_source_summary(_as_dict(sources.get("optimizer_family_coverage")))


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {str(k): v for k, v in value.items() if not str(k).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _unsafe_claims(package: Mapping[str, Any]) -> list[str]:
    claims = [field for field in UNSAFE_TRUE_FIELDS if package.get(field) is True]
    claims.extend(field for field in UNSAFE_NON_EMPTY_FIELDS if _as_dict(package.get(field)))
    return _dedupe(claims)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-review-package", type=Path, default=DEFAULT_PACKAGE_PATH)
    parser.add_argument("--owner-release-review-record", type=Path, default=DEFAULT_OWNER_RELEASE_REVIEW_RECORD_PATH)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARCHIVE_PATH)
    args = parser.parse_args(argv)
    package = {}
    if args.release_review_package.exists():
        package = _as_dict(json.loads(args.release_review_package.read_text(encoding="utf-8")))
    owner_record = {}
    if args.owner_release_review_record.exists():
        owner_record = _as_dict(json.loads(args.owner_release_review_record.read_text(encoding="utf-8")))
    report = build_native_update_release_review_archive(
        release_review_package=package,
        owner_release_review_record=owner_record,
        artifact_path=args.artifact_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_native_update_release_review_archive"]
