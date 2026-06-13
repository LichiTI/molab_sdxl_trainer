"""Shared default-off guards for TurboCore v2 approval preflight records."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Sequence


UnsafeBlockers = Callable[[Mapping[str, Any]], Sequence[str]]


def approval_preflight_record_blockers(
    preflight: Mapping[str, Any] | None,
    *,
    phase: str,
    signature_id: str,
    signed_payload: Mapping[str, Any],
    unsafe_blockers: UnsafeBlockers | None = None,
) -> list[str]:
    if not preflight:
        return ["approval_execution_preflight_missing"]
    if preflight.get("gate") != "optimizer_v2_approval_execution_preflight":
        return ["approval_execution_preflight_gate_mismatch"]
    blockers: list[str] = []
    if preflight.get("ok") is not True:
        blockers.append("approval_execution_preflight_not_ok")
    if not approval_preflight_phase_ready(preflight, phase=phase):
        blockers.append(f"approval_execution_preflight_{phase}_not_ready")
    if preflight.get("approval_recorded") is True or preflight.get("approval_artifact_written") is True:
        blockers.append("approval_execution_preflight_unexpected_approval_artifact")
    if unsafe_blockers is not None:
        blockers.extend(unsafe_blockers(preflight))
    blockers.extend(_signed_digest_blockers(preflight, signature_id, signed_payload))
    return _dedupe(blockers)


def approval_preflight_phase_ready(preflight: Mapping[str, Any] | None, *, phase: str) -> bool:
    if not preflight:
        return False
    summary = _as_dict(preflight.get("summary"))
    if phase == "phase1":
        return bool(
            preflight.get("phase1_record_inputs_ready") is True
            and int(summary.get("v2_approval_preflight_phase1_ready_count", 0) or 0) == 1
        )
    return bool(
        preflight.get("phase2_direction_inputs_ready") is True
        and int(summary.get("v2_approval_preflight_phase2_ready_count", 0) or 0) == 1
    )


def approval_preflight_signed_digest_match(
    preflight: Mapping[str, Any] | None,
    signature_id: str,
    signed_payload: Mapping[str, Any],
) -> bool:
    if not preflight or not signed_payload:
        return False
    expected = _signed_digest(preflight, signature_id)
    return bool(expected and expected == digest_payload(signed_payload))


def approval_preflight_record_binding(
    preflight: Mapping[str, Any] | None,
    signature_id: str,
    signed_payload: Mapping[str, Any],
) -> dict[str, Any]:
    preflight_payload = _as_dict(preflight)
    signed_check = _signed_check(preflight_payload, signature_id)
    signed_payload_digest = digest_payload(signed_payload)
    preflight_signed_payload_digest = str(signed_check.get("signed_payload_digest") or "")
    preflight_bundle_entry_digest = str(signed_check.get("signed_bundle_entry_digest") or "")
    signed_payload_match = bool(
        signed_payload_digest
        and preflight_signed_payload_digest
        and preflight_signed_payload_digest == signed_payload_digest
    )
    bundle_entry_match = bool(
        signed_payload_digest
        and preflight_bundle_entry_digest
        and preflight_bundle_entry_digest == signed_payload_digest
    )
    preflight_digest = digest_payload(preflight_payload)
    return {
        "approval_preflight_digest": preflight_digest,
        "record_signed_payload_digest": signed_payload_digest,
        "approval_preflight_signed_payload_digest": preflight_signed_payload_digest,
        "approval_preflight_signed_bundle_entry_digest": preflight_bundle_entry_digest,
        "approval_preflight_signed_payload_digest_match": signed_payload_match,
        "approval_preflight_signed_bundle_entry_digest_match": bundle_entry_match,
        "approval_preflight_binding_ready": bool(preflight_digest and signed_payload_match and bundle_entry_match),
    }


def digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {str(key): item for key, item in value.items() if not str(key).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _signed_digest_blockers(
    preflight: Mapping[str, Any],
    signature_id: str,
    signed_payload: Mapping[str, Any],
) -> list[str]:
    signed_check = _signed_check(preflight, signature_id)
    if not signed_check:
        return [f"approval_execution_preflight_{signature_id}_signed_check_missing"]
    blockers: list[str] = []
    if signed_check.get("valid") is not True:
        blockers.append(f"approval_execution_preflight_{signature_id}_signed_check_not_valid")
    if signed_check.get("extracted_entry_digest_match") is not True:
        blockers.append(f"approval_execution_preflight_{signature_id}_extracted_entry_digest_not_matched")
    expected = str(signed_check.get("signed_payload_digest") or "")
    bundle_digest = str(signed_check.get("signed_bundle_entry_digest") or "")
    if not expected:
        blockers.append(f"approval_execution_preflight_{signature_id}_signed_payload_digest_missing")
    elif expected != digest_payload(signed_payload):
        blockers.append(f"approval_execution_preflight_{signature_id}_signed_payload_digest_mismatch")
    if bundle_digest and expected and bundle_digest != expected:
        blockers.append(f"approval_execution_preflight_{signature_id}_signed_bundle_digest_mismatch")
    return _dedupe(blockers)


def _signed_digest(preflight: Mapping[str, Any], signature_id: str) -> str:
    return str(_signed_check(preflight, signature_id).get("signed_payload_digest") or "")


def _signed_check(preflight: Mapping[str, Any], signature_id: str) -> dict[str, Any]:
    for raw in preflight.get("signed_checks") or []:
        item = _as_dict(raw)
        if item.get("id") == signature_id:
            return item
    return {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = [
    "approval_preflight_record_binding",
    "approval_preflight_phase_ready",
    "approval_preflight_record_blockers",
    "approval_preflight_signed_digest_match",
    "digest_payload",
]
