"""Shared support-artifact guards for TurboCore v2 approval inputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def support_artifact_blockers(check_id: str, payload: Mapping[str, Any]) -> list[str]:
    blockers = _support_artifact_base_blockers(check_id, payload)
    if not blockers:
        blockers.extend(support_artifact_source_binding_blockers(check_id, payload))
    return _dedupe(blockers)


def _support_artifact_base_blockers(check_id: str, payload: Mapping[str, Any]) -> list[str]:
    if check_id == "training_launch_contract":
        return _training_launch_contract_blockers(payload)
    if check_id == "product_exposure_evidence":
        return _product_exposure_evidence_blockers(payload)
    if check_id == "owner_release_direction_packet":
        return _owner_direction_packet_blockers(payload)
    return []


def support_artifact_source_binding_blockers(check_id: str, payload: Mapping[str, Any]) -> list[str]:
    if _support_artifact_base_blockers(check_id, payload):
        return []
    if check_id == "training_launch_contract":
        return _training_launch_contract_source_blockers(payload)
    if check_id == "product_exposure_evidence":
        return _product_exposure_evidence_source_blockers(payload)
    if check_id == "owner_release_direction_packet":
        return _owner_direction_packet_source_blockers(payload)
    return []


def unsafe_fields(payload: Mapping[str, Any], *, prefix: str) -> list[str]:
    claims: list[str] = []
    for field in (
        "approval_recorded",
        "approval_artifact_written",
        "default_behavior_changed",
        "runtime_dispatch_ready",
        "runtime_dispatch_allowed",
        "native_dispatch_allowed",
        "training_path_enabled",
        "product_native_ready",
        "product_exposure_allowed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "backend_router_registered",
    ):
        if payload.get(field) is True:
            claims.append(f"{prefix}:{field}")
    return claims


def post_record_request_field_emissions(value: Any, *, path: str = "") -> list[str]:
    emissions: list[str] = []
    if isinstance(value, Mapping):
        for key, raw in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if (
                key_text.startswith("post_")
                and key_text.endswith("_request_fields")
                and _has_request_field_emission(raw)
            ):
                emissions.append(child_path)
            emissions.extend(post_record_request_field_emissions(raw, path=child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            emissions.extend(post_record_request_field_emissions(item, path=child_path))
    return _dedupe(emissions)


def _training_launch_contract_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if payload.get("package") != "turbocore_native_update_training_launch_contract_v0":
        blockers.append("training_launch_contract_package_mismatch")
    if payload.get("gate") != "native_update_training_launch_contract":
        blockers.append("training_launch_contract_gate_mismatch")
    if payload.get("ok") is not True:
        blockers.append("training_launch_contract_not_ok")
    if payload.get("evidence_ready") is not True:
        blockers.append("training_launch_contract_evidence_not_ready")
    if payload.get("ready_for_training_launch_review") is not True:
        blockers.append("training_launch_contract_not_ready_for_review")
    if payload.get("post_training_launch_request_fields", {}) != {}:
        blockers.append("training_launch_contract_post_request_fields_not_empty")
    return blockers


def _product_exposure_evidence_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if payload.get("evidence") != "native_update_product_exposure_evidence_v0":
        blockers.append("product_exposure_evidence_id_mismatch")
    for field in (
        "ok",
        "default_off",
        "report_only",
        "contract_only",
        "product_exposure_decision_only",
        "records_evidence_only",
        "manual_only",
        "internal_only",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
        "product_exposure_decision_ready",
    ):
        if payload.get(field) is not True:
            blockers.append(f"product_exposure_evidence_{field}_missing")
    return blockers


def _owner_direction_packet_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if payload.get("package") != "turbocore_native_update_owner_release_direction_packet_v0":
        blockers.append("owner_direction_packet_package_mismatch")
    if payload.get("gate") != "native_update_owner_release_direction":
        blockers.append("owner_direction_packet_gate_mismatch")
    if payload.get("ok") is not True:
        blockers.append("owner_direction_packet_not_ok")
    if payload.get("ready_for_owner_direction_signature") is not True:
        blockers.append("owner_direction_packet_not_ready_for_signature")
    if payload.get("post_owner_release_request_fields", {}) != {}:
        blockers.append("owner_direction_packet_post_request_fields_not_empty")
    return blockers


def _training_launch_contract_source_blockers(payload: Mapping[str, Any]) -> list[str]:
    step = _as_dict(payload.get("training_step_execution_contract_summary"))
    launch = _as_dict(payload.get("training_launch_evidence_summary"))
    blockers: list[str] = []
    if not str(step.get("source") or ""):
        blockers.append("training_launch_contract_training_step_source_missing")
    if not str(step.get("digest") or ""):
        blockers.append("training_launch_contract_training_step_digest_missing")
    if not str(launch.get("source") or ""):
        blockers.append("training_launch_contract_launch_evidence_source_missing")
    if not str(launch.get("digest") or ""):
        blockers.append("training_launch_contract_launch_evidence_digest_missing")
    return blockers


def _product_exposure_evidence_source_blockers(payload: Mapping[str, Any]) -> list[str]:
    source = str(payload.get("source") or payload.get("_source_path") or "")
    digest = str(payload.get("sha256") or payload.get("artifact_digest") or _digest_payload(payload))
    blockers: list[str] = []
    if not source:
        blockers.append("product_exposure_evidence_source_missing")
    if not digest:
        blockers.append("product_exposure_evidence_digest_missing")
    return blockers


def _owner_direction_packet_source_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for field in (
        "source_release_review_archive_digest",
        "source_product_exposure_decision_digest",
        "source_stable_first_release_scope_digest",
        "source_owner_release_direction_template_digest",
    ):
        if not str(payload.get(field) or ""):
            blockers.append(f"owner_direction_packet_{field}_missing")
    return blockers


def _has_request_field_emission(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return value not in (None, "", False)


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {str(key): item for key, item in value.items() if not str(key).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "post_record_request_field_emissions",
    "support_artifact_blockers",
    "support_artifact_source_binding_blockers",
    "unsafe_fields",
]
