"""Validate a signed TurboCore native-update release review record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_native_update_owner_release_review_packet import (
    build_native_update_owner_release_review_packet,
)
from core.turbocore_native_update_release_review_package import (
    READY_DECISION,
    build_native_update_release_review_package,
    load_gate_artifacts,
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
PACKET_ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_review_packet.json"
ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_review_record.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def build_native_update_owner_release_review_record(
    *,
    signed_review: Mapping[str, Any] | None = None,
    owner_packet: Mapping[str, Any] | None = None,
    approval_preflight: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    packet = _owner_packet(owner_packet, directory)
    review = _as_dict(signed_review)
    packet_ready = bool(packet.get("ready_for_owner_signature", False))
    signed_review_present = bool(review)
    blockers = _packet_blockers(packet)
    if not signed_review_present:
        blockers.append("signed_owner_release_review_missing")
        package = {}
    else:
        blockers.extend(
            _approval_preflight_blockers(
                approval_preflight,
                phase="phase1",
                signature_id="owner_release_review",
                signed_payload=review,
            )
        )
        blockers.extend(_signed_review_blockers(review, packet))
        package = build_native_update_release_review_package(
            gate_artifacts=load_gate_artifacts(directory),
            release_review=review,
        )
        blockers.extend(_package_blockers(package))

    signed_ready = bool(
        signed_review_present
        and not blockers
        and package.get("release_review_recorded") is True
        and package.get("decision") == READY_DECISION
        and not _unsafe_top_level_enabled(package)
    )
    preflight_binding = _approval_preflight_record_binding(
        approval_preflight,
        "owner_release_review",
        review,
    )
    payload = {
        "schema_version": 1,
        "package": "turbocore_native_update_owner_release_review_record_v0",
        "gate": "native_update_owner_release_review_record",
        "ok": packet_ready and (not signed_review_present or signed_ready),
        "roadmap": ROADMAP,
        "owner_packet_ready": packet_ready,
        "signed_review_present": signed_review_present,
        "signed_review_valid": signed_ready,
        "approval_recorded": signed_ready,
        "release_review_recorded": signed_ready,
        "decision": (
            "native_update_release_review_recorded_default_off"
            if signed_ready
            else "native_update_release_review_waiting_for_signed_owner_record"
        ),
        "source_owner_packet_digest": _digest_payload(packet),
        "source_release_review_template_digest": str(packet.get("source_release_review_template_digest", "") or ""),
        "signed_review_template_digest": str(review.get("source_release_review_template_digest", "") or ""),
        "approval_preflight_present": bool(approval_preflight),
        "approval_preflight_phase1_ready": _approval_preflight_phase_ready(approval_preflight, phase="phase1"),
        "approval_preflight_signed_review_digest_match": _approval_preflight_signed_digest_match(
            approval_preflight,
            "owner_release_review",
            review,
        ),
        **preflight_binding,
        "signed_review_digest_match": bool(
            signed_review_present
            and review.get("source_release_review_template_digest")
            == packet.get("source_release_review_template_digest")
        ),
        "release_package_decision": str(package.get("decision", "") or ""),
        "release_package_digest": _digest_payload(package),
        "blocked_reasons": _dedupe(blockers),
        "owner_action_required": not signed_ready,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "summary": {
            "expected_gate_count": int(package.get("expected_gate_count", 0) or 0)
            if package
            else int(_as_dict(packet.get("compact_evidence")).get("expected_gate_count", 0) or 0),
            "supplemental_gate_count": int(package.get("supplemental_gate_count", 0) or 0)
            if package
            else int(_as_dict(packet.get("compact_evidence")).get("supplemental_gate_count", 0) or 0),
            "release_review_recorded_count": 1 if signed_ready else 0,
            "approval_preflight_phase1_ready_count": 1
            if _approval_preflight_phase_ready(approval_preflight, phase="phase1")
            else 0,
            "approval_preflight_signed_review_digest_match_count": 1
            if _approval_preflight_signed_digest_match(approval_preflight, "owner_release_review", review)
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
            "product_exposure_allowed_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
        },
        "recommended_next_step": (
            "archive signed default-off release review and wait for separate owner release direction"
            if signed_ready
            else "provide a signed owner review record generated from the current packet template"
        ),
        "notes": [
            "This validator records approval only when a signed owner review is supplied.",
            "A recorded default-off review still keeps product exposure and native training dispatch disabled.",
            "Stale or tampered review digests are blocked before release review is recorded.",
        ],
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _owner_packet(owner_packet: Mapping[str, Any] | None, directory: Path) -> dict[str, Any]:
    if owner_packet is not None:
        return _as_dict(owner_packet)
    source = directory / PACKET_ARTIFACT.name
    if source.exists():
        return _read_json(source)
    return build_native_update_owner_release_review_packet(artifact_dir=directory, write_artifact=True)


def _packet_blockers(packet: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not packet:
        blockers.append("owner_release_review_packet_missing")
    elif packet.get("ready_for_owner_signature") is not True:
        blockers.append("owner_release_review_packet_not_ready")
    if packet and packet.get("approval_recorded") is True:
        blockers.append("owner_release_review_packet_unexpected_approval_recorded")
    if packet and _unsafe_top_level_enabled(packet):
        blockers.append("owner_release_review_packet_unsafe_top_level_claim")
    return blockers


def _signed_review_blockers(review: Mapping[str, Any], packet: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    expected_digest = str(packet.get("source_release_review_template_digest", "") or "")
    if not review.get("source_release_review_template_digest"):
        blockers.append("signed_owner_release_review_template_digest_missing")
    elif str(review.get("source_release_review_template_digest")) != expected_digest:
        blockers.append("signed_owner_release_review_template_digest_mismatch")
    if str(review.get("requested_scope", "") or "") != str(packet.get("required_requested_scope", "") or ""):
        blockers.append("signed_owner_release_review_requested_scope_mismatch")
    return blockers


def _package_blockers(package: Mapping[str, Any]) -> list[str]:
    if not package:
        return ["signed_owner_release_review_package_missing"]
    if package.get("release_review_recorded") is True and package.get("decision") == READY_DECISION:
        return []
    return [f"release_package:{item}" for item in _strings(package.get("blocked_reasons"))] or [
        "release_package_signed_review_not_recorded"
    ]


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
        unsafe_blockers=lambda value: ["approval_execution_preflight_unsafe_top_level_claim"]
        if _unsafe_top_level_enabled(value)
        else [],
    )


def _read_json(path: Path) -> dict[str, Any]:
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _read_json_if_supplied(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return _read_json(Path(path))


def _unsafe_top_level_enabled(value: Mapping[str, Any]) -> bool:
    for key in (
        "approval_artifact_written",
        "default_behavior_changed",
        "product_exposure_allowed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "runtime_dispatch_allowed",
        "native_dispatch_allowed",
        "training_path_enabled",
        "training_launch_executed",
    ):
        if value.get(key) is True:
            return True
    return bool(value.get("post_release_request_fields"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signed-review", default="", help="Path to owner-signed release review JSON.")
    parser.add_argument("--owner-packet", default="", help="Optional owner review packet JSON path.")
    parser.add_argument("--approval-preflight", default="", help="Required v2 approval execution preflight JSON for signed records.")
    parser.add_argument("--artifact-dir", default="", help="Directory containing native-update release artifacts.")
    parser.add_argument("--no-artifact", action="store_true", help="Print validation without writing the record artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_native_update_owner_release_review_record(
        signed_review=_read_json_if_supplied(args.signed_review),
        owner_packet=_read_json_if_supplied(args.owner_packet),
        approval_preflight=_read_json_if_supplied(args.approval_preflight),
        artifact_dir=args.artifact_dir or None,
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_native_update_owner_release_review_record"]


if __name__ == "__main__":
    raise SystemExit(main())
