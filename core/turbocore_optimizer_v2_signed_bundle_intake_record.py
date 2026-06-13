"""Validate a signed v2 TurboCore optimizer signature bundle.

The default artifact records intake readiness only.  Real approvals are recorded
only when a caller supplies a signed bundle; smoke tests validate synthetic
bundles in memory without writing approval artifacts.
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

from core.turbocore_native_update_owner_release_direction_record import (
    build_native_update_owner_release_direction_record,
)
from core.turbocore_native_update_owner_release_review_record import (
    build_native_update_owner_release_review_record,
)
from core.turbocore_native_update_product_exposure_decision import (
    READY_DECISION as PRODUCT_EXPOSURE_READY_DECISION,
    build_native_update_product_exposure_decision,
)


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle_intake_record.json"
SIGNATURE_BUNDLE_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_signature_bundle_packet.json"
OWNER_REVIEW_PACKET_ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_review_packet.json"
PRODUCT_EXPOSURE_DECISION_ARTIFACT = ARTIFACT_DIR / "native_update_product_exposure_decision.json"
OWNER_DIRECTION_PACKET_ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_direction_packet.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"
APPROVAL_FIELDS_BY_SIGNATURE_ID = {
    "owner_release_review": "approve_native_update_release_review_package",
    "product_exposure_review": "approve_native_update_product_exposure_decision",
    "owner_release_direction": "approve_native_update_owner_release_direction",
}
SIGNATURE_ID_ORDER = ("owner_release_review", "product_exposure_review", "owner_release_direction")
PHASE1_SIGNATURE_IDS = ("owner_release_review", "product_exposure_review")
PHASE2_SIGNATURE_IDS = ("owner_release_direction",)


def build_optimizer_v2_signed_bundle_intake_record(
    *,
    signature_bundle: Mapping[str, Any] | None = None,
    signed_bundle: Mapping[str, Any] | None = None,
    owner_review_packet: Mapping[str, Any] | None = None,
    product_exposure_decision: Mapping[str, Any] | None = None,
    owner_direction_packet: Mapping[str, Any] | None = None,
    allow_unsigned_template: bool = False,
    write_artifact: bool = True,
    write_approval_artifacts: bool = False,
) -> dict[str, Any]:
    bundle = _as_dict(signature_bundle) or _read_json(SIGNATURE_BUNDLE_ARTIFACT)
    signed = _as_dict(signed_bundle)
    owner_packet = _as_dict(owner_review_packet) or _read_json(OWNER_REVIEW_PACKET_ARTIFACT)
    exposure = _as_dict(product_exposure_decision) or _read_json(PRODUCT_EXPOSURE_DECISION_ARTIFACT)
    direction_packet = _as_dict(owner_direction_packet) or _read_json(OWNER_DIRECTION_PACKET_ARTIFACT)

    signatures = _signed_entries(signed)
    expected_signature_ids = set(_expected_template_digests(bundle))
    missing_signed_signature_ids = _missing_signed_signature_ids(expected_signature_ids, signatures)
    unknown_signature_ids = sorted(signature_id for signature_id in signatures if signature_id not in expected_signature_ids)
    signed_present = bool(signed)
    unsigned_template_marker = signed.get("unsigned_template") is True
    current_bundle_digest = _digest_payload(bundle)
    signed_bundle_digest = str(signed.get("source_signature_bundle_digest") or "")
    source_digest_match = bool(signed_present and current_bundle_digest and signed_bundle_digest == current_bundle_digest)
    stale_signed_bundle = bool(signed_present and signed_bundle_digest != current_bundle_digest)
    owner_review_record = _owner_review_record(owner_packet, signatures, write_approval_artifacts)
    product_exposure_record = _product_exposure_record(exposure, signatures)
    direction_record = _owner_direction_record(direction_packet, signatures, write_approval_artifacts)
    records = [owner_review_record, product_exposure_record, direction_record]
    records_by_id = {str(record["signature_id"]): record for record in records}
    ready_records = sum(1 for record in records if record["recorded"])
    unsafe = _unsafe_claims(records)
    template_digest_blockers = _template_digest_blockers(bundle, signatures)
    unknown_entry_blockers = [f"unknown_signed_entry:{signature_id}" for signature_id in unknown_signature_ids]
    not_ready_signature_ids = _not_ready_signature_ids(bundle, signatures)
    not_ready_entry_blockers = [f"signed_entry_not_ready_for_signature:{signature_id}" for signature_id in not_ready_signature_ids]
    manual_field_checks = _manual_field_checks(bundle, signatures)
    missing_manual_fields_by_signature_id = _missing_manual_fields_by_signature_id(manual_field_checks)
    manual_field_blockers = _manual_field_blockers(manual_field_checks)
    stale_blockers = ["signed_bundle_source_digest_stale_or_missing"] if stale_signed_bundle else []
    unsigned_blockers = (
        ["signed_bundle_unsigned_template_marker_present"]
        if unsigned_template_marker and not allow_unsigned_template
        else []
    )
    unsigned_template_allowed = bool(unsigned_template_marker and allow_unsigned_template)
    unsigned_template_blocked = bool(unsigned_template_marker and not allow_unsigned_template)
    template_digest_mismatch = bool(template_digest_blockers)
    unknown_entry_present = bool(unknown_signature_ids)
    not_ready_entry_present = bool(not_ready_signature_ids)
    manual_field_blockers_for_ok = [] if unsigned_template_allowed else manual_field_blockers
    valid_for_record = bool(
        signed_present
        and source_digest_match
        and not unsigned_template_marker
        and not template_digest_blockers
        and not unknown_entry_present
        and not not_ready_entry_present
        and not manual_field_blockers
        and ready_records == 3
        and not unsafe
    )
    valid_record_context = bool(
        signed_present
        and source_digest_match
        and not unsigned_template_marker
        and not template_digest_mismatch
        and not unknown_entry_present
        and not not_ready_entry_present
        and not manual_field_blockers
        and not unsafe
    )
    phase1_valid_records = _valid_record_count(records_by_id, PHASE1_SIGNATURE_IDS) if valid_record_context else 0
    phase2_valid_records = _valid_record_count(records_by_id, PHASE2_SIGNATURE_IDS) if valid_record_context else 0
    phase1_manual_ready = _manual_field_ready_count(manual_field_checks, PHASE1_SIGNATURE_IDS)
    phase2_manual_ready = _manual_field_ready_count(manual_field_checks, PHASE2_SIGNATURE_IDS)
    phase1_missing_signature_count = sum(
        1 for signature_id in missing_signed_signature_ids if signature_id in PHASE1_SIGNATURE_IDS
    )
    phase2_missing_signature_count = sum(
        1 for signature_id in missing_signed_signature_ids if signature_id in PHASE2_SIGNATURE_IDS
    )
    missing_signature_count = len(missing_signed_signature_ids)
    approval_artifact_written = bool(write_approval_artifacts and valid_for_record)
    payload = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_signed_bundle_intake_record_v0",
        "gate": "optimizer_v2_signed_bundle_intake_record",
        "roadmap": ROADMAP,
        "ok": not unsafe
        and not stale_signed_bundle
        and not unsigned_template_blocked
        and not template_digest_mismatch
        and not unknown_entry_present
        and not not_ready_entry_present
        and not manual_field_blockers_for_ok,
        "signed_bundle_present": signed_present,
        "signed_bundle_source_digest_match": source_digest_match,
        "signed_bundle_source_digest_stale": stale_signed_bundle,
        "signed_bundle_unsigned_template_marker": unsigned_template_marker,
        "missing_signed_signature_ids": missing_signed_signature_ids,
        "unknown_signed_signature_ids": unknown_signature_ids,
        "allow_unsigned_template": bool(allow_unsigned_template),
        "signed_bundle_valid": valid_for_record,
        "records_all_valid": valid_for_record,
        "approval_recorded": False,
        "approval_artifact_written": approval_artifact_written,
        "roadmap_complete": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "source_artifacts": {
            "signature_bundle": str(SIGNATURE_BUNDLE_ARTIFACT),
            "owner_release_review_packet": str(OWNER_REVIEW_PACKET_ARTIFACT),
            "product_exposure_decision": str(PRODUCT_EXPOSURE_DECISION_ARTIFACT),
            "owner_release_direction_packet": str(OWNER_DIRECTION_PACKET_ARTIFACT),
        },
        "current_signature_bundle_digest": current_bundle_digest,
        "signed_bundle_source_signature_bundle_digest": signed_bundle_digest,
        "record_checks": records,
        "manual_field_checks": manual_field_checks,
        "missing_manual_fields_by_signature_id": missing_manual_fields_by_signature_id,
        "summary": {
            "v2_signed_bundle_entry_count": len(SIGNATURE_ID_ORDER),
            "v2_signed_bundle_present_count": 1 if signed_present else 0,
            "v2_signed_bundle_valid_record_count": ready_records
            if valid_record_context
            else 0,
            "v2_signed_bundle_phase1_valid_record_count": phase1_valid_records,
            "v2_signed_bundle_phase1_ready_count": 1 if phase1_valid_records == len(PHASE1_SIGNATURE_IDS) else 0,
            "v2_signed_bundle_phase2_valid_record_count": phase2_valid_records,
            "v2_signed_bundle_phase2_ready_count": 1 if phase2_valid_records == len(PHASE2_SIGNATURE_IDS) else 0,
            "v2_signed_bundle_full_ready_count": 1 if valid_for_record else 0,
            "v2_signed_bundle_missing_signature_count": missing_signature_count,
            "v2_signed_bundle_phase1_missing_signature_count": phase1_missing_signature_count,
            "v2_signed_bundle_phase2_missing_signature_count": phase2_missing_signature_count,
            "v2_signed_bundle_source_digest_match_count": 1 if source_digest_match else 0,
            "v2_signed_bundle_source_digest_stale_count": 1 if stale_signed_bundle else 0,
            "v2_signed_bundle_unsigned_template_count": 1 if unsigned_template_marker else 0,
            "v2_signed_bundle_template_digest_mismatch_count": len(template_digest_blockers),
            "v2_signed_bundle_unknown_entry_count": len(unknown_signature_ids),
            "v2_signed_bundle_not_ready_entry_count": len(not_ready_signature_ids),
            "v2_signed_bundle_manual_field_check_count": len(manual_field_checks),
            "v2_signed_bundle_manual_field_ready_count": sum(
                1 for check in manual_field_checks if check["manual_fields_ready"]
            ),
            "v2_signed_bundle_manual_field_missing_count": sum(
                len(check["missing_manual_fields"]) for check in manual_field_checks
            ),
            "v2_signed_bundle_manual_field_missing_signature_count": len(missing_manual_fields_by_signature_id),
            "v2_signed_bundle_manual_field_shape_ready_count": (
                1 if signed_present and manual_field_checks and not manual_field_blockers else 0
            ),
            "v2_signed_bundle_phase1_manual_field_ready_count": phase1_manual_ready,
            "v2_signed_bundle_phase2_manual_field_ready_count": phase2_manual_ready,
            "v2_signed_bundle_owner_review_recorded_count": 1 if owner_review_record["recorded"] else 0,
            "v2_signed_bundle_product_exposure_recorded_count": 1 if product_exposure_record["recorded"] else 0,
            "v2_signed_bundle_owner_direction_recorded_count": 1 if direction_record["recorded"] else 0,
            "v2_signed_bundle_approval_artifact_written_count": 1 if approval_artifact_written else 0,
            "v2_signed_bundle_runtime_dispatch_ready_count": 0,
            "v2_signed_bundle_native_dispatch_allowed_count": 0,
            "v2_signed_bundle_training_path_enabled_count": 0,
            "v2_signed_bundle_product_native_ready_count": 0,
            "v2_signed_bundle_default_behavior_changed_count": 0,
            "v2_signed_bundle_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(
            ([] if signed_present else ["signed_bundle_missing"])
            + stale_blockers
            + unsigned_blockers
            + template_digest_blockers
            + unknown_entry_blockers
            + not_ready_entry_blockers
            + manual_field_blockers
            + [reason for record in records for reason in record["blocked_reasons"]]
            + unsafe
        ),
        "promotion_blockers": _dedupe(
            [
                "v2_signed_bundle_not_recorded_as_real_approval",
                "product_dispatch_still_requires_explicit_route_binding",
            ]
            + ([] if signed_present else ["signed_bundle_missing"])
            + stale_blockers
            + (["signed_bundle_is_still_unsigned_template"] if unsigned_template_marker else [])
            + template_digest_blockers
            + unknown_entry_blockers
            + not_ready_entry_blockers
            + manual_field_blockers
            + [reason for record in records for reason in record["blocked_reasons"]]
            + unsafe
        ),
        "recommended_next_step": (
            "supply a real signed bundle generated from the current v2 signature bundle packet"
            if not signed_present
            else "remove unsigned_template marker before submitting reviewer-returned approvals"
            if unsigned_template_blocked
            else "return a signed bundle generated from the current signature-entry templates"
            if template_digest_mismatch
            else "remove unknown signed entries before submitting reviewer-returned approvals"
            if unknown_entry_present
            else "remove signed entries that are not ready for the current approval phase"
            if not_ready_entry_present
            else "use --allow-unsigned-template only for reviewer handoff template shape checks"
            if unsigned_template_allowed
            else "after external approval, rerun the individual record validators with write_approval_artifacts enabled"
        ),
        "notes": [
            "This intake validates signed content but does not write approval artifacts by default.",
            "Synthetic signed bundles in smoke tests are not counted as real approval.",
            "Validated approvals still do not enable product route binding or native dispatch.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _owner_review_record(
    packet: Mapping[str, Any],
    signatures: Mapping[str, Mapping[str, Any]],
    write_artifact: bool,
) -> dict[str, Any]:
    signed = _as_dict(signatures.get("owner_release_review"))
    result = build_native_update_owner_release_review_record(
        signed_review=signed or None,
        owner_packet=packet,
        approval_preflight=_phase1_preflight_ready("owner_release_review", signed) if signed else None,
        write_artifact=write_artifact,
    )
    return _record_check("owner_release_review", result.get("release_review_recorded") is True, result)


def _product_exposure_record(
    decision: Mapping[str, Any],
    signatures: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    signed = _as_dict(signatures.get("product_exposure_review"))
    result = build_native_update_product_exposure_decision(
        training_launch_contract=_as_dict(decision.get("training_launch_contract_summary")),
        product_exposure_evidence=_product_exposure_evidence_from_summary(decision),
        product_exposure_review=signed or None,
        approval_preflight=_phase1_preflight_ready("product_exposure_review", signed) if signed else None,
    )
    return _record_check(
        "product_exposure_review",
        result.get("product_exposure_decision_recorded") is True
        and result.get("decision") == PRODUCT_EXPOSURE_READY_DECISION,
        result,
    )


def _owner_direction_record(
    packet: Mapping[str, Any],
    signatures: Mapping[str, Mapping[str, Any]],
    write_artifact: bool,
) -> dict[str, Any]:
    signed = _as_dict(signatures.get("owner_release_direction"))
    result = build_native_update_owner_release_direction_record(
        signed_direction=signed or None,
        owner_direction_packet=packet,
        approval_preflight=_phase2_preflight_ready("owner_release_direction", signed) if signed else None,
        write_artifact=write_artifact,
    )
    return _record_check("owner_release_direction", result.get("owner_release_direction_recorded") is True, result)


def _record_check(signature_id: str, recorded: bool, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "signature_id": signature_id,
        "recorded": recorded,
        "decision": str(result.get("decision") or ""),
        "blocked_reasons": _strings(result.get("blocked_reasons")),
    }


def _phase1_preflight_ready(signature_id: str, signed_payload: Mapping[str, Any]) -> dict[str, Any]:
    return _approval_preflight(
        phase1_ready=True,
        phase2_ready=False,
        signature_id=signature_id,
        signed_payload=signed_payload,
    )


def _phase2_preflight_ready(signature_id: str, signed_payload: Mapping[str, Any]) -> dict[str, Any]:
    return _approval_preflight(
        phase1_ready=True,
        phase2_ready=True,
        signature_id=signature_id,
        signed_payload=signed_payload,
    )


def _approval_preflight(
    *,
    phase1_ready: bool,
    phase2_ready: bool,
    signature_id: str,
    signed_payload: Mapping[str, Any],
) -> dict[str, Any]:
    digest = _digest_payload(signed_payload)
    return {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_approval_execution_preflight_v0",
        "gate": "optimizer_v2_approval_execution_preflight",
        "ok": True,
        "phase1_record_inputs_ready": bool(phase1_ready),
        "phase2_direction_inputs_ready": bool(phase2_ready),
        "approval_recorded": False,
        "approval_artifact_written": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "signed_checks": [
            {
                "schema_version": 1,
                "id": signature_id,
                "valid": True,
                "signed_payload_digest": digest,
                "signed_bundle_entry_digest": digest,
                "extracted_entry_digest_match": True,
                "extracted_entry_digest_mismatch": False,
                "extracted_entry_source_missing": False,
            }
        ],
        "summary": {
            "v2_approval_preflight_phase1_ready_count": 1 if phase1_ready else 0,
            "v2_approval_preflight_phase2_ready_count": 1 if phase2_ready else 0,
        },
    }


def _valid_record_count(records_by_id: Mapping[str, Mapping[str, Any]], signature_ids: tuple[str, ...]) -> int:
    return sum(1 for signature_id in signature_ids if records_by_id.get(signature_id, {}).get("recorded") is True)


def _missing_signed_signature_ids(
    expected_signature_ids: set[str],
    signatures: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    ordered_ids = [
        signature_id
        for signature_id in SIGNATURE_ID_ORDER
        if not expected_signature_ids or signature_id in expected_signature_ids
    ]
    ordered_ids.extend(sorted(signature_id for signature_id in expected_signature_ids if signature_id not in ordered_ids))
    return [signature_id for signature_id in ordered_ids if signature_id not in signatures]


def _manual_field_checks(
    signature_bundle: Mapping[str, Any],
    signatures: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    entries = _signature_entries_by_id(signature_bundle)
    checks: list[dict[str, Any]] = []
    for signature_id, signed in signatures.items():
        entry = entries.get(signature_id, {})
        required_acks = _strings(entry.get("required_acknowledgement_fields"))
        missing: list[str] = []
        if not str(signed.get("reviewer") or "").strip():
            missing.append("reviewer")
        if not str(signed.get("reviewed_at") or "").strip():
            missing.append("reviewed_at")
        approval_field = APPROVAL_FIELDS_BY_SIGNATURE_ID.get(signature_id, "")
        if approval_field and signed.get(approval_field) is not True:
            missing.append(approval_field)
        for field in required_acks:
            if signed.get(field) is not True:
                missing.append(field)
        checks.append(
            {
                "schema_version": 1,
                "signature_id": signature_id,
                "manual_fields_ready": not missing,
                "reviewer_present": bool(str(signed.get("reviewer") or "").strip()),
                "reviewed_at_present": bool(str(signed.get("reviewed_at") or "").strip()),
                "approval_field": approval_field,
                "approval_field_ready": bool(approval_field and signed.get(approval_field) is True),
                "required_acknowledgement_fields": required_acks,
                "missing_manual_fields": missing,
            }
        )
    return checks


def _manual_field_blockers(checks: list[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for check in checks:
        signature_id = str(check.get("signature_id") or "")
        for field in _strings(check.get("missing_manual_fields")):
            blockers.append(f"{signature_id}_manual_field_missing:{field}")
    return blockers


def _missing_manual_fields_by_signature_id(checks: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {
        signature_id: missing
        for check in checks
        for signature_id, missing in [
            (
                str(check.get("signature_id") or ""),
                _strings(check.get("missing_manual_fields")),
            )
        ]
        if signature_id and missing
    }


def _manual_field_ready_count(checks: list[Mapping[str, Any]], signature_ids: tuple[str, ...]) -> int:
    by_id = {str(check.get("signature_id") or ""): check for check in checks}
    return sum(1 for signature_id in signature_ids if by_id.get(signature_id, {}).get("manual_fields_ready") is True)


def _not_ready_signature_ids(
    signature_bundle: Mapping[str, Any],
    signatures: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    entries = _signature_entries_by_id(signature_bundle)
    return sorted(
        signature_id
        for signature_id in signatures
        if signature_id in entries and entries[signature_id].get("ready_for_signature") is not True
    )


def _signature_entries_by_id(signature_bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = signature_bundle.get("signature_entries")
    if not isinstance(entries, (list, tuple)):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for raw in entries:
        entry = _as_dict(raw)
        signature_id = str(entry.get("signature_id") or "")
        if signature_id:
            out[signature_id] = entry
    return out


def _template_digest_blockers(
    signature_bundle: Mapping[str, Any],
    signatures: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if not signatures:
        return []
    expected = _expected_template_digests(signature_bundle)
    blockers: list[str] = []
    for signature_id, signed in signatures.items():
        expected_digest = str(expected.get(signature_id) or "")
        signed_digest = _signed_template_digest(signature_id, signed)
        if not expected_digest:
            blockers.append(f"{signature_id}_source_v2_signature_template_digest_expected_missing")
        elif not signed_digest:
            blockers.append(f"{signature_id}_source_v2_signature_template_digest_missing")
        elif signed_digest != expected_digest:
            blockers.append(f"{signature_id}_source_v2_signature_template_digest_mismatch")
    return _dedupe(blockers)


def _expected_template_digests(signature_bundle: Mapping[str, Any]) -> dict[str, str]:
    entries = signature_bundle.get("signature_entries")
    if not isinstance(entries, (list, tuple)):
        return {}
    out: dict[str, str] = {}
    for raw in entries:
        entry = _as_dict(raw)
        signature_id = str(entry.get("signature_id") or "")
        if signature_id:
            out[signature_id] = str(entry.get("source_template_digest") or "")
    return out


def _signed_template_digest(signature_id: str, signed: Mapping[str, Any]) -> str:
    generic = str(signed.get("source_v2_signature_template_digest") or "")
    if generic:
        return generic
    if signature_id == "owner_release_review":
        return str(signed.get("source_release_review_template_digest") or "")
    if signature_id == "owner_release_direction":
        return str(signed.get("source_owner_release_direction_template_digest") or "")
    return ""


def _signed_entries(signed_bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = signed_bundle.get("signed_entries")
    if isinstance(entries, Mapping):
        return {str(key): _as_dict(value) for key, value in entries.items()}
    if isinstance(entries, list):
        out = {}
        for entry in entries:
            item = _as_dict(entry)
            signature_id = str(item.get("signature_id") or "")
            signature = _as_dict(item.get("signature")) or item
            if signature_id:
                out[signature_id] = signature
        return out
    return {}


def _product_exposure_evidence_from_summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(decision.get("product_exposure_evidence_summary"))
    source = str(summary.get("source") or "bundle-intake://product-exposure-evidence")
    return {
        "schema_version": 1,
        "evidence": "native_update_product_exposure_evidence_v0",
        "ok": summary.get("ok") is True,
        "product_exposure_decision_ready": summary.get("ready") is True,
        "report_only": True,
        "contract_only": True,
        "product_exposure_decision_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "default_off": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "source": source,
        "artifact_digest": str(summary.get("digest") or ""),
        "sections": [
            "training_launch_contract_reference",
            "owner_exposure_decision_boundary",
            "request_adapter_boundary",
            "request_schema_boundary",
            "backend_router_boundary",
            "launcher_ui_boundary",
            "webui_boundary",
            "release_gate_boundary",
            "no_training_launch_boundary",
            "no_request_submission_boundary",
            "no_request_ui_schema_boundary",
            "rollback_policy",
            "observability_policy",
        ],
        "owner_exposure_decision_boundary": [_default_off_row("owner_exposure_decision_boundary", source)],
        "request_adapter_boundary": [_default_off_row("request_adapter_boundary", source)],
        "request_schema_boundary": [_default_off_row("request_schema_boundary", source)],
        "backend_router_boundary": [_default_off_row("backend_router_boundary", source)],
        "launcher_ui_boundary": [_default_off_row("launcher_ui_boundary", source)],
        "webui_boundary": [_default_off_row("webui_boundary", source)],
        "release_gate_boundary": [_default_off_row("release_gate_boundary", source)],
        "rollback_policy": [{"id": "rollback_policy", "ready": True, "source": source}],
        "observability_policy": [{"id": "observability_policy", "ready": True, "source": source}],
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
    }


def _default_off_row(row_id: str, source: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "ready": True,
        "source": source,
        "product_exposure_allowed": False,
        "product_exposure_enabled": False,
        "product_exposure_approved": False,
        "release_gate_open": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "request_submitted": False,
        "job_created": False,
        "queue_enqueued": False,
        "run_record_written": False,
        "ready_for_ui": False,
        "ui_exposure_allowed": False,
        "launcher_exposure_allowed": False,
        "webui_exposure_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "backend_router_registered": False,
    }


def _unsafe_claims(records: list[Mapping[str, Any]]) -> list[str]:
    claims: list[str] = []
    for record in records:
        if record.get("recorded") is True:
            continue
        for reason in _strings(record.get("blocked_reasons")):
            if "unsafe" in reason:
                claims.append(reason)
    return _dedupe(claims)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _read_json_if_supplied(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return _read_json(Path(path))


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signed-bundle", default="", help="Path to a reviewer-signed v2 bundle JSON.")
    parser.add_argument("--signature-bundle", default="", help="Optional v2 signature bundle packet JSON path.")
    parser.add_argument("--owner-review-packet", default="", help="Optional owner release review packet JSON path.")
    parser.add_argument(
        "--product-exposure-decision",
        default="",
        help="Optional product exposure decision JSON path.",
    )
    parser.add_argument(
        "--owner-direction-packet",
        default="",
        help="Optional owner release direction packet JSON path.",
    )
    parser.add_argument(
        "--allow-unsigned-template",
        action="store_true",
        help="Allow reviewer handoff template shape checks without treating it as a valid approval.",
    )
    parser.add_argument("--no-artifact", action="store_true", help="Print validation without writing intake artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_signed_bundle_intake_record(
        signature_bundle=_read_json_if_supplied(args.signature_bundle),
        signed_bundle=_read_json_if_supplied(args.signed_bundle),
        owner_review_packet=_read_json_if_supplied(args.owner_review_packet),
        product_exposure_decision=_read_json_if_supplied(args.product_exposure_decision),
        owner_direction_packet=_read_json_if_supplied(args.owner_direction_packet),
        allow_unsigned_template=bool(args.allow_unsigned_template),
        write_artifact=not bool(args.no_artifact),
        write_approval_artifacts=False,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_optimizer_v2_signed_bundle_intake_record"]


if __name__ == "__main__":
    raise SystemExit(main())
