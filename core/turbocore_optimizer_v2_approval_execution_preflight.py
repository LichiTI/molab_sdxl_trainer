"""Preflight inputs for executing real TurboCore optimizer v2 approvals.

The preflight checks the files that later approval-record validators consume.
It does not write approval records and never enables native optimizer dispatch.
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

from core.turbocore_optimizer_v2_support_artifact_guard import (  # noqa: E402
    post_record_request_field_emissions as _post_record_request_field_emissions,
    support_artifact_blockers as _support_artifact_blockers,
    support_artifact_source_binding_blockers as _support_artifact_source_binding_blockers,
    unsafe_fields as _unsafe_fields,
)


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_approval_execution_preflight.json"
DEFAULT_PATHS = {
    "signature_bundle": ARTIFACT_DIR / "turbocore_optimizer_v2_signature_bundle_packet.json",
    "signed_bundle": ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle.reviewed.json",
    "owner_release_review": ARTIFACT_DIR / "signed_owner_release_review.json",
    "product_exposure_review": ARTIFACT_DIR / "signed_product_exposure_review.json",
    "owner_release_direction": ARTIFACT_DIR / "signed_owner_release_direction.json",
    "training_launch_contract": ARTIFACT_DIR / "native_update_training_launch_contract.json",
    "product_exposure_evidence": ARTIFACT_DIR / "native_update_product_exposure_evidence.json",
    "owner_release_direction_packet": ARTIFACT_DIR / "native_update_owner_release_direction_packet.json",
}
PHASE_INPUT_IDS = {
    "shared": ("signature_bundle", "signed_bundle"),
    "phase1": (
        "owner_release_review",
        "product_exposure_review",
        "training_launch_contract",
        "product_exposure_evidence",
    ),
    "phase2": ("owner_release_direction", "owner_release_direction_packet"),
}
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def build_optimizer_v2_approval_execution_preflight(
    *,
    paths: Mapping[str, str | Path] | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    resolved = _resolve_paths(paths)
    file_checks = {name: _file_check(name, path) for name, path in resolved.items()}
    signature_bundle_ok = _signature_bundle_check(file_checks["signature_bundle"])
    signed_bundle_ok = _signed_bundle_check(file_checks["signed_bundle"], signature_bundle_ok)
    signed_checks = {
        "owner_release_review": _signed_check(
            "owner_release_review",
            file_checks["owner_release_review"],
            signed_bundle_ok,
        ),
        "product_exposure_review": _signed_check(
            "product_exposure_review",
            file_checks["product_exposure_review"],
            signed_bundle_ok,
        ),
        "owner_release_direction": _signed_check(
            "owner_release_direction",
            file_checks["owner_release_direction"],
            signed_bundle_ok,
        ),
    }
    support_checks = {
        "training_launch_contract": _support_check("training_launch_contract", file_checks["training_launch_contract"]),
        "product_exposure_evidence": _support_check("product_exposure_evidence", file_checks["product_exposure_evidence"]),
        "owner_release_direction_packet": _support_check(
            "owner_release_direction_packet",
            file_checks["owner_release_direction_packet"],
        ),
    }
    phase1_ready = bool(
        signed_bundle_ok["valid"]
        and signed_checks["owner_release_review"]["valid"]
        and signed_checks["product_exposure_review"]["valid"]
        and support_checks["training_launch_contract"]["valid"]
        and support_checks["product_exposure_evidence"]["valid"]
    )
    phase2_ready = bool(
        signed_checks["owner_release_direction"]["valid"]
        and support_checks["owner_release_direction_packet"]["valid"]
    )
    unsafe = _unsafe_claims(file_checks, signed_checks)
    valid_count = sum(1 for item in file_checks.values() if item["valid_json"])
    present_count = sum(1 for item in file_checks.values() if item["present"])
    parse_error_count = sum(1 for item in file_checks.values() if item["parse_error"])
    missing_input_ids_by_phase = _missing_input_ids_by_phase(file_checks)
    extracted_hard_fail = any(item.get("hard_fail") is True for item in signed_checks.values())
    support_hard_fail = any(item.get("hard_fail") is True for item in support_checks.values())
    preflight_hard_fail = bool(signed_bundle_ok.get("hard_fail") or extracted_hard_fail or support_hard_fail)
    signed_payload_digest_present_count = sum(
        1 for item in signed_checks.values() if str(item.get("signed_payload_digest") or "").strip()
    )
    signed_bundle_entry_digest_present_count = sum(
        1 for item in signed_checks.values() if str(item.get("signed_bundle_entry_digest") or "").strip()
    )
    signed_payload_bundle_digest_match_count = sum(
        1
        for item in signed_checks.values()
        if str(item.get("signed_payload_digest") or "").strip()
        and item.get("signed_payload_digest") == item.get("signed_bundle_entry_digest")
    )
    post_record_request_field_emission_count = sum(
        int(item.get("post_record_request_field_emission_count", 0) or 0)
        for item in support_checks.values()
    )
    support_source_binding_ready_count = sum(
        1 for item in support_checks.values() if item.get("source_binding_ready") is True
    )
    support_source_binding_blocker_count = sum(
        len(_strings(item.get("source_binding_blockers"))) for item in support_checks.values()
    )
    payload = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_approval_execution_preflight_v0",
        "gate": "optimizer_v2_approval_execution_preflight",
        "roadmap": ROADMAP,
        "ok": not unsafe and not preflight_hard_fail,
        "phase1_record_inputs_ready": phase1_ready and not unsafe and not preflight_hard_fail,
        "phase2_direction_inputs_ready": phase2_ready and not unsafe and not preflight_hard_fail,
        "full_record_execution_ready": phase1_ready and phase2_ready and not unsafe and not preflight_hard_fail,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "roadmap_complete": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "input_paths": {name: str(path) for name, path in resolved.items()},
        "missing_input_ids_by_phase": missing_input_ids_by_phase,
        "file_checks": list(file_checks.values()),
        "signed_checks": list(signed_checks.values()),
        "signature_bundle_check": signature_bundle_ok,
        "support_checks": list(support_checks.values()),
        "summary": {
            "v2_approval_preflight_file_count": len(file_checks),
            "v2_approval_preflight_present_file_count": present_count,
            "v2_approval_preflight_valid_json_file_count": valid_count,
            "v2_approval_preflight_missing_file_count": len(file_checks) - present_count,
            "v2_approval_preflight_missing_shared_input_count": len(missing_input_ids_by_phase["shared"]),
            "v2_approval_preflight_missing_phase1_input_count": len(missing_input_ids_by_phase["phase1"]),
            "v2_approval_preflight_missing_phase2_input_count": len(missing_input_ids_by_phase["phase2"]),
            "v2_approval_preflight_parse_error_count": parse_error_count,
            "v2_approval_preflight_signature_bundle_valid_count": 1 if signature_bundle_ok["valid"] else 0,
            "v2_approval_preflight_signed_bundle_valid_count": 1 if signed_bundle_ok["valid"] else 0,
            "v2_approval_preflight_source_digest_match_count": 1 if signed_bundle_ok["source_digest_match"] else 0,
            "v2_approval_preflight_source_digest_stale_count": 1 if signed_bundle_ok["source_digest_stale"] else 0,
            "v2_approval_preflight_template_digest_mismatch_count": int(
                signed_bundle_ok["template_digest_mismatch_count"]
            ),
            "v2_approval_preflight_unknown_entry_count": int(signed_bundle_ok["unknown_entry_count"]),
            "v2_approval_preflight_not_ready_entry_count": int(signed_bundle_ok["not_ready_entry_count"]),
            "v2_approval_preflight_unsigned_template_count": 1 if signed_bundle_ok["unsigned_template"] else 0,
            "v2_approval_preflight_signed_payload_digest_present_count": signed_payload_digest_present_count,
            "v2_approval_preflight_signed_payload_digest_missing_count": (
                len(signed_checks) - signed_payload_digest_present_count
            ),
            "v2_approval_preflight_signed_bundle_entry_digest_present_count": (
                signed_bundle_entry_digest_present_count
            ),
            "v2_approval_preflight_signed_payload_bundle_digest_match_count": (
                signed_payload_bundle_digest_match_count
            ),
            "v2_approval_preflight_extracted_entry_digest_match_count": sum(
                1 for item in signed_checks.values() if item["extracted_entry_digest_match"]
            ),
            "v2_approval_preflight_extracted_entry_digest_mismatch_count": sum(
                1 for item in signed_checks.values() if item["extracted_entry_digest_mismatch"]
            ),
            "v2_approval_preflight_extracted_entry_source_missing_count": sum(
                1 for item in signed_checks.values() if item["extracted_entry_source_missing"]
            ),
            "v2_approval_preflight_support_ready_count": sum(
                1 for item in support_checks.values() if item["valid"]
            ),
            "v2_approval_preflight_support_invalid_count": sum(
                1 for item in support_checks.values() if item["present"] and not item["valid"]
            ),
            "v2_approval_preflight_support_source_binding_ready_count": (
                support_source_binding_ready_count
            ),
            "v2_approval_preflight_support_source_binding_blocker_count": (
                support_source_binding_blocker_count
            ),
            "v2_approval_preflight_post_record_request_field_emission_count": (
                post_record_request_field_emission_count
            ),
            "v2_approval_preflight_owner_review_ready_count": 1 if signed_checks["owner_release_review"]["valid"] else 0,
            "v2_approval_preflight_product_exposure_ready_count": 1
            if signed_checks["product_exposure_review"]["valid"]
            else 0,
            "v2_approval_preflight_owner_direction_ready_count": 1
            if signed_checks["owner_release_direction"]["valid"]
            else 0,
            "v2_approval_preflight_hard_fail_count": 1 if preflight_hard_fail else 0,
            "v2_approval_preflight_phase1_ready_count": 1 if phase1_ready and not unsafe and not preflight_hard_fail else 0,
            "v2_approval_preflight_phase2_ready_count": 1 if phase2_ready and not unsafe and not preflight_hard_fail else 0,
            "v2_approval_preflight_full_ready_count": 1 if phase1_ready and phase2_ready and not unsafe and not preflight_hard_fail else 0,
            "v2_approval_preflight_approval_recorded_count": 0,
            "v2_approval_preflight_runtime_dispatch_ready_count": 0,
            "v2_approval_preflight_native_dispatch_allowed_count": 0,
            "v2_approval_preflight_training_path_enabled_count": 0,
            "v2_approval_preflight_product_native_ready_count": 0,
            "v2_approval_preflight_default_behavior_changed_count": 0,
            "v2_approval_preflight_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(
            [reason for item in file_checks.values() for reason in _strings(item.get("blocked_reasons"))]
            + [reason for item in signed_checks.values() for reason in _strings(item.get("blocked_reasons"))]
            + [reason for item in support_checks.values() for reason in _strings(item.get("blocked_reasons"))]
            + _strings(signature_bundle_ok.get("blocked_reasons"))
            + _strings(signed_bundle_ok.get("blocked_reasons"))
            + ([] if signed_bundle_ok["valid"] else ["signed_bundle_not_valid_for_preflight"])
            + unsafe
        ),
        "promotion_blockers": _dedupe(
            [
                "v2_approval_preflight_not_approval_record",
                "product_dispatch_still_requires_explicit_route_binding",
            ]
            + ([] if phase1_ready else ["phase1_record_inputs_not_ready"])
            + ([] if phase2_ready else ["phase2_direction_inputs_not_ready"])
            + unsafe
        ),
        "recommended_next_step": (
            "run owner release review and product exposure record validators"
            if phase1_ready and not unsafe and not preflight_hard_fail
            else "supply signed bundle, extracted signed reviews, and default-off support artifacts"
        ),
        "notes": [
            "Phase 1 covers owner-release review and product-exposure record inputs.",
            "Phase 2 covers owner-release direction inputs after archive/direction packet rebuild.",
            "Preflight is read-only and never enables product/native dispatch.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _resolve_paths(paths: Mapping[str, str | Path] | None) -> dict[str, Path]:
    overrides = paths or {}
    return {
        name: Path(overrides.get(name) or default)
        for name, default in DEFAULT_PATHS.items()
    }


def _missing_input_ids_by_phase(file_checks: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    return {
        phase: [
            input_id
            for input_id in input_ids
            if _as_dict(file_checks.get(input_id)).get("present") is not True
        ]
        for phase, input_ids in PHASE_INPUT_IDS.items()
    }


def _file_check(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "id": name,
            "path": str(path),
            "present": False,
            "valid_json": False,
            "parse_error": False,
            "payload": {},
            "blocked_reasons": [f"{name}_missing"],
        }
    try:
        payload = _as_dict(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        return {
            "schema_version": 1,
            "id": name,
            "path": str(path),
            "present": True,
            "valid_json": False,
            "parse_error": True,
            "payload": {},
            "blocked_reasons": [f"{name}_json_parse_error:{exc.lineno}:{exc.colno}"],
        }
    return {
        "schema_version": 1,
        "id": name,
        "path": str(path),
        "present": True,
        "valid_json": True,
        "parse_error": False,
        "payload": payload,
        "blocked_reasons": [],
    }


def _signature_bundle_check(file_check: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(file_check.get("payload"))
    entries = payload.get("signature_entries")
    valid = file_check.get("valid_json") is True and isinstance(entries, list)
    return {
        "schema_version": 1,
        "id": "signature_bundle",
        "valid": valid,
        "digest": _digest_payload(payload) if valid else "",
        "expected_template_digests": _expected_template_digests(payload),
        "ready_by_signature_id": _ready_by_signature_id(payload),
        "blocked_reasons": [] if valid else ["signature_bundle_missing_signature_entries"],
    }


def _signed_bundle_check(
    file_check: Mapping[str, Any],
    signature_bundle_check: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _as_dict(file_check.get("payload"))
    entries = _signed_entries(payload)
    expected = _as_dict(signature_bundle_check.get("expected_template_digests"))
    ready_by_signature_id = _as_dict(signature_bundle_check.get("ready_by_signature_id"))
    expected_ids = set(expected)
    unknown_ids = sorted(signature_id for signature_id in entries if signature_id not in expected_ids)
    not_ready_ids = sorted(
        signature_id
        for signature_id in entries
        if signature_id in ready_by_signature_id and ready_by_signature_id.get(signature_id) is not True
    )
    source_digest = str(payload.get("source_signature_bundle_digest") or "")
    current_digest = str(signature_bundle_check.get("digest") or "")
    source_digest_match = bool(source_digest and current_digest and source_digest == current_digest)
    source_digest_stale = bool(file_check.get("present") is True and (source_digest or current_digest)) and not source_digest_match
    unsigned_template = payload.get("unsigned_template") is True
    template_blockers = _template_digest_blockers(expected, entries)
    base_valid = file_check.get("valid_json") is True and isinstance(payload.get("signed_entries"), (Mapping, list))
    valid = bool(
        base_valid
        and signature_bundle_check.get("valid") is True
        and source_digest_match
        and not unsigned_template
        and not unknown_ids
        and not not_ready_ids
        and not template_blockers
    )
    return {
        "schema_version": 1,
        "id": "signed_bundle",
        "valid": valid,
        "source_digest_match": source_digest_match,
        "source_digest_stale": source_digest_stale,
        "unsigned_template": unsigned_template,
        "unknown_signed_signature_ids": unknown_ids,
        "not_ready_signed_signature_ids": not_ready_ids,
        "signed_entries": entries,
        "template_digest_mismatch_count": len(template_blockers),
        "unknown_entry_count": len(unknown_ids),
        "not_ready_entry_count": len(not_ready_ids),
        "hard_fail": bool(file_check.get("present") is True and not valid),
        "blocked_reasons": _dedupe(
            ([] if base_valid else ["signed_bundle_missing_signed_entries"])
            + ([] if signature_bundle_check.get("valid") is True else ["signature_bundle_not_valid_for_preflight"])
            + ([] if source_digest_match else ["signed_bundle_source_digest_stale_or_missing"])
            + (["signed_bundle_unsigned_template_marker_present"] if unsigned_template else [])
            + [f"unknown_signed_entry:{signature_id}" for signature_id in unknown_ids]
            + [f"signed_entry_not_ready_for_signature:{signature_id}" for signature_id in not_ready_ids]
            + template_blockers
        ),
    }


def _signed_check(
    signature_id: str,
    file_check: Mapping[str, Any],
    signed_bundle_check: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _as_dict(file_check.get("payload"))
    blockers = list(_strings(file_check.get("blocked_reasons")))
    signed_entries = _as_dict(signed_bundle_check.get("signed_entries"))
    source_entry = _as_dict(signed_entries.get(signature_id))
    extracted_digest_match = False
    extracted_digest_mismatch = False
    extracted_source_missing = False
    if file_check.get("valid_json") is True:
        if not str(payload.get("reviewer") or "").strip():
            blockers.append(f"{signature_id}_reviewer_missing")
        if not str(payload.get("reviewed_at") or "").strip():
            blockers.append(f"{signature_id}_reviewed_at_missing")
        approve_fields = [key for key in payload if key.startswith("approve_")]
        if not approve_fields:
            blockers.append(f"{signature_id}_approve_field_missing")
        blockers.extend(f"{key}_not_true" for key in approve_fields if payload.get(key) is not True)
        blockers.extend(
            f"{key}_not_true"
            for key in payload
            if key.startswith("acknowledge_") and payload.get(key) is not True
        )
        if source_entry:
            extracted_digest_match = _digest_payload(payload) == _digest_payload(source_entry)
            if not extracted_digest_match:
                extracted_digest_mismatch = True
                blockers.append(f"{signature_id}_extracted_signed_entry_digest_mismatch")
        elif signed_bundle_check.get("valid") is True:
            extracted_source_missing = True
            blockers.append(f"{signature_id}_signed_bundle_entry_missing_for_extracted_file")
    return {
        "schema_version": 1,
        "id": signature_id,
        "valid": file_check.get("valid_json") is True and not blockers,
        "signed_payload_digest": _digest_payload(payload) if file_check.get("valid_json") is True else "",
        "signed_bundle_entry_digest": _digest_payload(source_entry) if source_entry else "",
        "extracted_entry_digest_match": extracted_digest_match,
        "extracted_entry_digest_mismatch": extracted_digest_mismatch,
        "extracted_entry_source_missing": extracted_source_missing,
        "hard_fail": bool(extracted_digest_mismatch or extracted_source_missing),
        "blocked_reasons": _dedupe(blockers),
    }


def _support_check(check_id: str, file_check: Mapping[str, Any]) -> dict[str, Any]:
    blockers = list(_strings(file_check.get("blocked_reasons")))
    payload = _as_dict(file_check.get("payload"))
    post_record_request_field_emissions: list[str] = []
    source_binding_blockers: list[str] = []
    if file_check.get("valid_json") is True:
        post_record_request_field_emissions = _post_record_request_field_emissions(payload)
        source_binding_blockers = _support_artifact_source_binding_blockers(check_id, payload)
        blockers.extend(_support_artifact_blockers(check_id, payload))
        blockers.extend(
            f"{check_id}_post_record_request_field_emitted:{path}"
            for path in post_record_request_field_emissions
        )
        blockers.extend(_unsafe_fields(payload, prefix=f"{check_id}_unsafe"))
    return {
        "schema_version": 1,
        "id": check_id,
        "present": file_check.get("present") is True,
        "valid": file_check.get("valid_json") is True and not blockers,
        "hard_fail": bool(file_check.get("present") is True and blockers),
        "source_binding_ready": (
            file_check.get("valid_json") is True and not blockers and not source_binding_blockers
        ),
        "source_binding_blockers": _dedupe(source_binding_blockers),
        "post_record_request_field_emission_count": len(post_record_request_field_emissions),
        "post_record_request_field_emission_paths": post_record_request_field_emissions,
        "blocked_reasons": _dedupe(blockers),
    }


def _expected_template_digests(signature_bundle: Mapping[str, Any]) -> dict[str, str]:
    entries = signature_bundle.get("signature_entries")
    if not isinstance(entries, list):
        return {}
    out: dict[str, str] = {}
    for raw in entries:
        entry = _as_dict(raw)
        signature_id = str(entry.get("signature_id") or "")
        if signature_id:
            out[signature_id] = str(entry.get("source_template_digest") or "")
    return out


def _ready_by_signature_id(signature_bundle: Mapping[str, Any]) -> dict[str, bool]:
    entries = signature_bundle.get("signature_entries")
    if not isinstance(entries, list):
        return {}
    out: dict[str, bool] = {}
    for raw in entries:
        entry = _as_dict(raw)
        signature_id = str(entry.get("signature_id") or "")
        if signature_id:
            out[signature_id] = entry.get("ready_for_signature") is True
    return out


def _template_digest_blockers(
    expected: Mapping[str, Any],
    signatures: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    for signature_id, signed in signatures.items():
        if signature_id not in expected:
            continue
        expected_digest = str(expected.get(signature_id) or "")
        signed_digest = _signed_template_digest(signature_id, signed)
        if not expected_digest:
            blockers.append(f"{signature_id}_source_v2_signature_template_digest_expected_missing")
        elif not signed_digest:
            blockers.append(f"{signature_id}_source_v2_signature_template_digest_missing")
        elif signed_digest != expected_digest:
            blockers.append(f"{signature_id}_source_v2_signature_template_digest_mismatch")
    return _dedupe(blockers)


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
        out: dict[str, dict[str, Any]] = {}
        for raw in entries:
            item = _as_dict(raw)
            signature_id = str(item.get("signature_id") or "")
            signature = _as_dict(item.get("signature")) or item
            if signature_id:
                out[signature_id] = signature
        return out
    return {}


def _unsafe_claims(
    file_checks: Mapping[str, Mapping[str, Any]],
    signed_checks: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    claims: list[str] = []
    for item in file_checks.values():
        claims.extend(_unsafe_fields(_as_dict(item.get("payload")), prefix=f"{item.get('id')}_unsafe"))
    for item in signed_checks.values():
        if item.get("valid") is not True:
            continue
        payload = _as_dict(file_checks[str(item.get("id"))].get("payload"))
        claims.extend(_unsafe_fields(payload, prefix=f"{item.get('id')}_unsafe"))
    return _dedupe(claims)


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


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-bundle", default="", help="Current v2 signature bundle packet JSON.")
    parser.add_argument("--signed-bundle", default="", help="Reviewer-returned signed bundle JSON.")
    parser.add_argument("--owner-review", default="", help="Extracted signed owner release review JSON.")
    parser.add_argument("--product-exposure-review", default="", help="Extracted signed product exposure review JSON.")
    parser.add_argument("--owner-direction", default="", help="Extracted signed owner release direction JSON.")
    parser.add_argument("--training-launch-contract", default="", help="Training launch contract JSON.")
    parser.add_argument("--product-exposure-evidence", default="", help="Product exposure evidence JSON.")
    parser.add_argument("--owner-direction-packet", default="", help="Owner release direction packet JSON.")
    parser.add_argument("--no-artifact", action="store_true", help="Print preflight without writing artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = {
        "signature_bundle": args.signature_bundle,
        "signed_bundle": args.signed_bundle,
        "owner_release_review": args.owner_review,
        "product_exposure_review": args.product_exposure_review,
        "owner_release_direction": args.owner_direction,
        "training_launch_contract": args.training_launch_contract,
        "product_exposure_evidence": args.product_exposure_evidence,
        "owner_release_direction_packet": args.owner_direction_packet,
    }
    payload = build_optimizer_v2_approval_execution_preflight(
        paths={key: value for key, value in paths.items() if value},
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_optimizer_v2_approval_execution_preflight"]


if __name__ == "__main__":
    raise SystemExit(main())
