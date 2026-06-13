"""Validate a signed TurboCore native-update owner release direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_native_update_owner_release_direction_packet import (
    REQUIRED_ACKS,
    SCOPE,
    build_native_update_owner_release_direction_packet,
)
from core.turbocore_optimizer_v2_approval_preflight_guard import (
    approval_preflight_phase_ready as _approval_preflight_phase_ready,
    approval_preflight_record_binding as _approval_preflight_record_binding,
    approval_preflight_record_blockers as _approval_preflight_record_blockers,
    approval_preflight_signed_digest_match as _approval_preflight_signed_digest_match,
    digest_payload as _digest_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
PACKET_ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_direction_packet.json"
ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_direction_record.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"

UNSAFE_TRUE_FIELDS = (
    "product_exposure_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "ui_exposure_allowed",
    "backend_router_registered",
    "runtime_dispatch_allowed",
    "native_dispatch_allowed",
    "training_path_enabled",
    "training_launch_executed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_owner_release_request_fields",
    "post_product_exposure_request_fields",
    "post_training_route_request_fields",
    "request_adapter_fields",
    "request_schema_fields",
)


def build_native_update_owner_release_direction_record(
    *,
    signed_direction: Mapping[str, Any] | None = None,
    owner_direction_packet: Mapping[str, Any] | None = None,
    approval_preflight: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    packet = _owner_direction_packet(owner_direction_packet, directory)
    direction = _as_dict(signed_direction)
    direction_present = bool(direction)
    blockers = _packet_blockers(packet)
    if not direction_present:
        blockers.append("signed_owner_release_direction_missing")
    else:
        blockers.extend(
            _approval_preflight_blockers(
                approval_preflight,
                phase="phase2",
                signature_id="owner_release_direction",
                signed_payload=direction,
            )
        )
        blockers.extend(_signed_direction_blockers(direction, packet))

    signed_ready = bool(direction_present and not blockers)
    preflight_binding = _approval_preflight_record_binding(
        approval_preflight,
        "owner_release_direction",
        direction,
    )
    payload = {
        "schema_version": 1,
        "package": "turbocore_native_update_owner_release_direction_record_v0",
        "gate": "native_update_owner_release_direction_record",
        "ok": not _unsafe_claims(packet, "owner_release_direction_packet")
        and (not direction_present or signed_ready),
        "roadmap": ROADMAP,
        "owner_direction_packet_ready": packet.get("ready_for_owner_direction_signature") is True,
        "signed_direction_present": direction_present,
        "signed_direction_valid": signed_ready,
        "owner_release_direction_recorded": signed_ready,
        "owner_release_approval_recorded": signed_ready,
        "decision": (
            "native_update_owner_release_direction_recorded_default_off"
            if signed_ready
            else "native_update_owner_release_direction_waiting_for_signed_owner_direction"
        ),
        "source_owner_direction_packet_digest": _digest_payload(packet),
        "source_owner_release_direction_template_digest": str(
            packet.get("source_owner_release_direction_template_digest") or ""
        ),
        "approval_preflight_present": bool(approval_preflight),
        "approval_preflight_phase2_ready": _approval_preflight_phase_ready(approval_preflight, phase="phase2"),
        "approval_preflight_signed_direction_digest_match": _approval_preflight_signed_digest_match(
            approval_preflight,
            "owner_release_direction",
            direction,
        ),
        **preflight_binding,
        "signed_owner_release_direction_template_digest": str(
            direction.get("source_owner_release_direction_template_digest") or ""
        ),
        "signed_owner_release_direction_digest_match": bool(
            direction_present
            and direction.get("source_owner_release_direction_template_digest")
            == packet.get("source_owner_release_direction_template_digest")
        ),
        "blocked_reasons": _dedupe(blockers),
        "owner_action_required": not signed_ready,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "post_owner_release_request_fields": {},
        "summary": {
            "owner_release_direction_ready_for_signature_count": 1
            if packet.get("ready_for_owner_direction_signature") is True
            else 0,
            "owner_release_direction_recorded_count": 1 if signed_ready else 0,
            "owner_release_direction_approval_recorded_count": 1 if signed_ready else 0,
            "owner_release_approval_recorded_count": 1 if signed_ready else 0,
            "approval_preflight_phase2_ready_count": 1
            if _approval_preflight_phase_ready(approval_preflight, phase="phase2")
            else 0,
            "approval_preflight_signed_direction_digest_match_count": 1
            if _approval_preflight_signed_digest_match(approval_preflight, "owner_release_direction", direction)
            else 0,
            "approval_preflight_digest_present_count": 1
            if preflight_binding["approval_preflight_digest"]
            else 0,
            "approval_preflight_signed_payload_digest_present_count": 1
            if preflight_binding["approval_preflight_signed_payload_digest"]
            else 0,
            "approval_preflight_signed_bundle_entry_digest_present_count": 1
            if preflight_binding["approval_preflight_signed_bundle_entry_digest"]
            else 0,
            "approval_preflight_binding_ready_count": 1
            if preflight_binding["approval_preflight_binding_ready"]
            else 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
        },
        "recommended_next_step": (
            "continue to product training-route binding preflight with this recorded owner direction"
            if signed_ready
            else "provide a signed owner release direction generated from the current direction template"
        ),
        "notes": [
            "This validator records owner direction only when a matching signed direction is supplied.",
            "A recorded owner direction still keeps request/UI/schema exposure and native dispatch disabled.",
            "Product training-route binding remains a separate post-direction step.",
        ],
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _owner_direction_packet(owner_direction_packet: Mapping[str, Any] | None, directory: Path) -> dict[str, Any]:
    if owner_direction_packet is not None:
        return _as_dict(owner_direction_packet)
    source = directory / PACKET_ARTIFACT.name
    if source.exists():
        return _read_json(source)
    return build_native_update_owner_release_direction_packet(artifact_dir=directory, write_artifact=True)


def _packet_blockers(packet: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not packet:
        blockers.append("owner_release_direction_packet_missing")
    elif packet.get("ready_for_owner_direction_signature") is not True:
        blockers.append("owner_release_direction_packet_not_ready_for_signature")
    if packet and packet.get("owner_release_direction_recorded") is True:
        blockers.append("owner_release_direction_packet_unexpected_recorded_direction")
    blockers.extend(_unsafe_claims(packet, "owner_release_direction_packet"))
    return _dedupe(blockers)


def _signed_direction_blockers(direction: Mapping[str, Any], packet: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    expected_digest = str(packet.get("source_owner_release_direction_template_digest") or "")
    if not direction.get("source_owner_release_direction_template_digest"):
        blockers.append("signed_owner_release_direction_template_digest_missing")
    elif str(direction.get("source_owner_release_direction_template_digest")) != expected_digest:
        blockers.append("signed_owner_release_direction_template_digest_mismatch")
    if str(direction.get("requested_scope") or "") != SCOPE:
        blockers.append("signed_owner_release_direction_requested_scope_mismatch")
    if direction.get("approve_native_update_owner_release_direction") is not True:
        blockers.append("signed_owner_release_direction_not_approved")
    for field in REQUIRED_ACKS:
        if direction.get(field) is not True:
            blockers.append(f"signed_owner_release_direction_ack_missing:{field}")
    blockers.extend(_unsafe_claims(direction, "signed_owner_release_direction"))
    return _dedupe(blockers)


def _approval_preflight_blockers(
    preflight: Mapping[str, Any] | None,
    *,
    phase: str,
    signature_id: str,
    signed_payload: Mapping[str, Any],
) -> list[str]:
    return _approval_preflight_record_blockers(
        preflight,
        phase=phase,
        signature_id=signature_id,
        signed_payload=signed_payload,
        unsafe_blockers=lambda value: _unsafe_claims(value, "approval_execution_preflight"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _read_json_if_supplied(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return _read_json(Path(path))


def _unsafe_claims(value: Mapping[str, Any], label: str) -> list[str]:
    if not value:
        return []
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"{label}_unsafe:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        field_value = value.get(field)
        if field_value not in (None, {}, [], "", ()):
            blocked.append(f"{label}_unsafe_non_empty:{field}")
    return blocked


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signed-direction", default="", help="Path to owner-signed release direction JSON.")
    parser.add_argument("--owner-direction-packet", default="", help="Optional owner direction packet JSON path.")
    parser.add_argument("--approval-preflight", default="", help="Required v2 approval execution preflight JSON for signed records.")
    parser.add_argument("--artifact-dir", default="", help="Directory containing native-update release artifacts.")
    parser.add_argument("--no-artifact", action="store_true", help="Print validation without writing the record artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_native_update_owner_release_direction_record(
        signed_direction=_read_json_if_supplied(args.signed_direction),
        owner_direction_packet=_read_json_if_supplied(args.owner_direction_packet),
        approval_preflight=_read_json_if_supplied(args.approval_preflight),
        artifact_dir=args.artifact_dir or None,
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_native_update_owner_release_direction_record"]


if __name__ == "__main__":
    raise SystemExit(main())
