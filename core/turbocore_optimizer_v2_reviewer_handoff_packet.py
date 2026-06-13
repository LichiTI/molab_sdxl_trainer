"""Reviewer handoff packet for TurboCore optimizer v2 signatures.

This packet turns the current signable templates into a reviewer-facing JSON
template.  It is not an approval record and never enables native dispatch.
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

from core.turbocore_optimizer_v2_signature_bundle_packet import (  # noqa: E402
    ARTIFACT as SIGNATURE_BUNDLE_ARTIFACT,
    build_optimizer_v2_signature_bundle_packet,
)


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_reviewer_handoff_packet.json"
SIGNED_BUNDLE_TEMPLATE_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle_template.json"
REVIEWER_RETURNED_SIGNED_BUNDLE_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle.reviewed.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"
SIGNATURE_BUNDLE_PLACEHOLDER = "temp\\turbocore_optimizer\\turbocore_optimizer_v2_signature_bundle_packet.json"
SIGNED_BUNDLE_TEMPLATE_PLACEHOLDER = "temp\\turbocore_optimizer\\turbocore_optimizer_v2_signed_bundle_template.json"
REVIEWER_RETURNED_SIGNED_BUNDLE_PLACEHOLDER = "temp\\turbocore_optimizer\\turbocore_optimizer_v2_signed_bundle.reviewed.json"
EXTRACTION_OUTPUT_DIR = "temp\\turbocore_optimizer"
OWNER_REVIEW_PLACEHOLDER = "temp\\turbocore_optimizer\\signed_owner_release_review.json"
PRODUCT_EXPOSURE_REVIEW_PLACEHOLDER = "temp\\turbocore_optimizer\\signed_product_exposure_review.json"
OWNER_DIRECTION_PLACEHOLDER = "temp\\turbocore_optimizer\\signed_owner_release_direction.json"
TRAINING_LAUNCH_CONTRACT_PLACEHOLDER = "temp\\turbocore_optimizer\\native_update_training_launch_contract.json"
PRODUCT_EXPOSURE_EVIDENCE_PLACEHOLDER = "temp\\turbocore_optimizer\\native_update_product_exposure_evidence.json"
OWNER_DIRECTION_PACKET_PLACEHOLDER = "temp\\turbocore_optimizer\\native_update_owner_release_direction_packet.json"
PHASE1_SIGNATURE_IDS = ("owner_release_review", "product_exposure_review")
PHASE2_SIGNATURE_IDS = ("owner_release_direction",)
APPROVAL_FIELDS_BY_SIGNATURE_ID = {
    "owner_release_review": "approve_native_update_release_review_package",
    "product_exposure_review": "approve_native_update_product_exposure_decision",
    "owner_release_direction": "approve_native_update_owner_release_direction",
}


def build_optimizer_v2_reviewer_handoff_packet(
    *,
    signature_bundle: Mapping[str, Any] | None = None,
    write_artifact: bool = True,
    write_signed_bundle_template: bool = True,
) -> dict[str, Any]:
    bundle = _as_dict(signature_bundle) or _read_json(SIGNATURE_BUNDLE_ARTIFACT)
    if not bundle:
        bundle = build_optimizer_v2_signature_bundle_packet(write_artifact=True)
    entries = [_as_dict(entry) for entry in _as_list(bundle.get("signature_entries"))]
    ready_entries = [entry for entry in entries if entry.get("ready_for_signature") is True]
    blocked_entries = [entry for entry in entries if entry.get("ready_for_signature") is not True]
    unsafe = _unsafe_claims(bundle)
    signed_template = _signed_bundle_template(bundle, ready_entries)
    required_manual_fields = _required_manual_fields_by_signature_id(ready_entries)
    phase1_manifest = _phase1_handoff_manifest(required_manual_fields)
    phase1_post_return_commands = _phase1_post_return_operator_commands()
    signed_template_entries = signed_template["signed_entries"]
    payload = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_reviewer_handoff_packet_v0",
        "gate": "optimizer_v2_reviewer_handoff_packet",
        "roadmap": ROADMAP,
        "ok": bool(bundle.get("signature_bundle_ready") is True and ready_entries and not unsafe),
        "reviewer_handoff_ready": bool(bundle.get("signature_bundle_ready") is True and ready_entries and not unsafe),
        "approval_recorded": False,
        "approval_artifact_written": False,
        "roadmap_complete": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "source_signature_bundle": str(SIGNATURE_BUNDLE_ARTIFACT),
        "source_signature_bundle_digest": _digest_payload(bundle),
        "signed_bundle_template_artifact": str(SIGNED_BUNDLE_TEMPLATE_ARTIFACT),
        "reviewer_returned_signed_bundle_artifact": str(REVIEWER_RETURNED_SIGNED_BUNDLE_ARTIFACT),
        "ready_signature_entries": [_handoff_entry(entry) for entry in ready_entries],
        "blocked_signature_entries": [_blocked_entry(entry) for entry in blocked_entries],
        "phase1_handoff_manifest": phase1_manifest,
        "phase1_post_return_operator_commands": phase1_post_return_commands,
        "signed_bundle_template": signed_template,
        "validation_commands": [
            _template_dry_run_command()
        ],
        "real_return_validation_commands": [
            _real_return_validation_command()
        ],
        "summary": {
            "v2_reviewer_handoff_entry_count": len(entries),
            "v2_reviewer_handoff_ready_entry_count": len(ready_entries),
            "v2_reviewer_handoff_blocked_entry_count": len(blocked_entries),
            "v2_reviewer_handoff_signed_template_entry_count": len(signed_template_entries),
            "v2_reviewer_handoff_phase_metadata_ready_count": 1
            if _phase_metadata_ready(ready_entries, blocked_entries, signed_template_entries, phase1_manifest)
            else 0,
            "v2_reviewer_handoff_phase1_template_entry_count": sum(
                1
                for entry in signed_template_entries.values()
                if _as_dict(entry).get("phase") == "phase1"
            ),
            "v2_reviewer_handoff_phase2_deferred_signature_count": len(
                phase1_manifest["phase2_deferred_signature_ids"]
            ),
            "v2_reviewer_handoff_required_manual_field_signature_count": len(required_manual_fields),
            "v2_reviewer_handoff_required_manual_field_count": sum(
                len(fields) for fields in required_manual_fields.values()
            ),
            "v2_reviewer_handoff_phase1_signature_count": len(phase1_manifest["required_signature_ids"]),
            "v2_reviewer_handoff_phase2_blocked_signature_count": len(
                phase1_manifest["deferred_signature_ids"]
            ),
            "v2_reviewer_handoff_phase1_required_manual_field_count": sum(
                len(fields) for fields in phase1_manifest["required_manual_fields_by_signature_id"].values()
            ),
            "v2_reviewer_handoff_template_dry_run_command_count": 1,
            "v2_reviewer_handoff_real_return_validation_command_count": 1,
            "v2_reviewer_handoff_phase1_post_return_command_count": len(phase1_post_return_commands),
            "v2_reviewer_handoff_phase1_post_return_pre_record_command_count": sum(
                1 for command in phase1_post_return_commands if command["writes_approval_record"] is False
            ),
            "v2_reviewer_handoff_phase1_post_return_approval_record_command_count": sum(
                1 for command in phase1_post_return_commands if command["writes_approval_record"] is True
            ),
            "v2_reviewer_handoff_packet_ready_count": 1
            if bundle.get("signature_bundle_ready") is True and ready_entries and not unsafe
            else 0,
            "v2_reviewer_handoff_approval_recorded_count": 0,
            "v2_reviewer_handoff_runtime_dispatch_ready_count": 0,
            "v2_reviewer_handoff_native_dispatch_allowed_count": 0,
            "v2_reviewer_handoff_training_path_enabled_count": 0,
            "v2_reviewer_handoff_product_native_ready_count": 0,
            "v2_reviewer_handoff_default_behavior_changed_count": 0,
            "v2_reviewer_handoff_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(
            _strings(bundle.get("blocked_reasons")) + [reason for entry in blocked_entries for reason in _strings(entry.get("blocked_reasons"))] + unsafe
        ),
        "promotion_blockers": _dedupe(
            [
                "v2_reviewer_handoff_waiting_for_real_signatures",
                "v2_signed_bundle_template_is_unsigned",
                "product_dispatch_still_requires_explicit_route_binding",
            ]
            + [reason for entry in blocked_entries for reason in _strings(entry.get("blocked_reasons"))]
            + unsafe
        ),
        "recommended_next_step": (
            "fill reviewer/reviewed_at/approve/ack fields in the signed bundle template, then validate it with the intake CLI"
        ),
        "notes": [
            "The signed bundle template is intentionally unsigned.",
            "Blocked entries remain listed for sequencing but are not placed in the current signed template.",
            "Phase 1 expects reviewer-returned JSON at the reviewed signed-bundle path before any record validator runs.",
            "This handoff does not write approval artifacts or enable native optimizer dispatch.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_signed_bundle_template:
        SIGNED_BUNDLE_TEMPLATE_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        SIGNED_BUNDLE_TEMPLATE_ARTIFACT.write_text(
            json.dumps(signed_template, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _signed_bundle_template(bundle: Mapping[str, Any], ready_entries: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_signed_bundle_template_v0",
        "source_signature_bundle_digest": _digest_payload(bundle),
        "unsigned_template": True,
        "approval_recorded": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "required_manual_fields_by_signature_id": _required_manual_fields_by_signature_id(ready_entries),
        "signed_entries": {
            str(entry.get("signature_id") or ""): _signed_template_entry(entry)
            for entry in ready_entries
        },
    }


def _phase1_handoff_manifest(required_manual_fields: Mapping[str, list[str]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "phase1",
        "required_signature_ids": list(PHASE1_SIGNATURE_IDS),
        "deferred_signature_ids": list(PHASE2_SIGNATURE_IDS),
        "phase1_required_signature_ids": list(PHASE1_SIGNATURE_IDS),
        "phase2_deferred_signature_ids": list(PHASE2_SIGNATURE_IDS),
        "signed_bundle_template_artifact": str(SIGNED_BUNDLE_TEMPLATE_ARTIFACT),
        "reviewer_returned_signed_bundle_artifact": str(REVIEWER_RETURNED_SIGNED_BUNDLE_ARTIFACT),
        "required_manual_fields_by_signature_id": {
            signature_id: list(required_manual_fields.get(signature_id, []))
            for signature_id in PHASE1_SIGNATURE_IDS
        },
        "template_dry_run_command": (
            _template_dry_run_command()
        ),
        "real_return_validation_command": _real_return_validation_command(),
    }


def _phase1_post_return_operator_commands() -> list[dict[str, Any]]:
    return [
        _operator_command(
            1,
            "validate_phase1_signed_bundle",
            _real_return_validation_command(),
            "Validate reviewer-returned phase1 signed bundle without writing approval artifacts.",
        ),
        _operator_command(
            2,
            "extract_phase1_signed_reviews",
            (
                "backend\\env\\python-flashattention\\python.exe "
                "backend\\core\\turbocore_optimizer_v2_signed_bundle_extractor.py "
                f"--signature-bundle {SIGNATURE_BUNDLE_PLACEHOLDER} "
                f"--signed-bundle {REVIEWER_RETURNED_SIGNED_BUNDLE_PLACEHOLDER} "
                f"--output-dir {EXTRACTION_OUTPUT_DIR} "
                "--write-extracted-artifacts"
            ),
            "Extract owner/product signed review JSON files for record validators.",
        ),
        _operator_command(
            3,
            "preflight_phase1_record_inputs",
            (
                "backend\\env\\python-flashattention\\python.exe "
                "backend\\core\\turbocore_optimizer_v2_approval_execution_preflight.py "
                f"--signature-bundle {SIGNATURE_BUNDLE_PLACEHOLDER} "
                f"--signed-bundle {REVIEWER_RETURNED_SIGNED_BUNDLE_PLACEHOLDER} "
                f"--owner-review {OWNER_REVIEW_PLACEHOLDER} "
                f"--product-exposure-review {PRODUCT_EXPOSURE_REVIEW_PLACEHOLDER} "
                f"--owner-direction {OWNER_DIRECTION_PLACEHOLDER} "
                f"--training-launch-contract {TRAINING_LAUNCH_CONTRACT_PLACEHOLDER} "
                f"--product-exposure-evidence {PRODUCT_EXPOSURE_EVIDENCE_PLACEHOLDER} "
                f"--owner-direction-packet {OWNER_DIRECTION_PACKET_PLACEHOLDER}"
            ),
            "Preflight extracted phase1 inputs and default-off support artifacts before record validators.",
        ),
    ]


def _operator_command(order: int, command_id: str, command: str, description: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "order": order,
        "id": command_id,
        "phase": "phase1",
        "command": command,
        "description": description,
        "writes_approval_record": False,
        "approval_artifact_written": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
    }


def _template_dry_run_command() -> str:
    return (
        "backend\\env\\python-flashattention\\python.exe "
        "backend\\core\\turbocore_optimizer_v2_signed_bundle_intake_record.py "
        f"--signature-bundle {SIGNATURE_BUNDLE_PLACEHOLDER} "
        f"--signed-bundle {SIGNED_BUNDLE_TEMPLATE_PLACEHOLDER} "
        "--allow-unsigned-template "
        "--no-artifact"
    )


def _real_return_validation_command() -> str:
    return (
        "backend\\env\\python-flashattention\\python.exe "
        "backend\\core\\turbocore_optimizer_v2_signed_bundle_intake_record.py "
        f"--signature-bundle {SIGNATURE_BUNDLE_PLACEHOLDER} "
        f"--signed-bundle {REVIEWER_RETURNED_SIGNED_BUNDLE_PLACEHOLDER} "
        "--no-artifact"
    )


def _handoff_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    signature_id = str(entry.get("signature_id") or "")
    return {
        "schema_version": 1,
        "signature_id": signature_id,
        "phase": _signature_phase(signature_id),
        "target_gate": str(entry.get("target_gate") or ""),
        "requested_scope": str(entry.get("requested_scope") or ""),
        "source_template_digest": str(entry.get("source_template_digest") or ""),
        "required_acknowledgement_fields": _strings(entry.get("required_acknowledgement_fields")),
        "required_manual_fields": _required_manual_fields(entry),
        "ready_for_signature": True,
    }


def _blocked_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    signature_id = str(entry.get("signature_id") or "")
    return {
        "schema_version": 1,
        "signature_id": signature_id,
        "phase": _signature_phase(signature_id),
        "target_gate": str(entry.get("target_gate") or ""),
        "requested_scope": str(entry.get("requested_scope") or ""),
        "ready_for_signature": False,
        "blocked_reasons": _strings(entry.get("blocked_reasons")),
    }


def _signed_template_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    signature_id = str(entry.get("signature_id") or "")
    template = _as_dict(entry.get("template"))
    template.setdefault("signature_id", signature_id)
    template.setdefault("phase", _signature_phase(signature_id))
    return template


def _signature_phase(signature_id: str) -> str:
    if signature_id in PHASE1_SIGNATURE_IDS:
        return "phase1"
    if signature_id in PHASE2_SIGNATURE_IDS:
        return "phase2"
    return ""


def _phase_metadata_ready(
    ready_entries: list[Mapping[str, Any]],
    blocked_entries: list[Mapping[str, Any]],
    signed_entries: Mapping[str, Any],
    phase1_manifest: Mapping[str, Any],
) -> bool:
    phase1_ids = list(phase1_manifest.get("phase1_required_signature_ids") or [])
    phase2_ids = list(phase1_manifest.get("phase2_deferred_signature_ids") or [])
    return bool(
        phase1_manifest.get("phase") == "phase1"
        and phase1_ids == list(PHASE1_SIGNATURE_IDS)
        and phase2_ids == list(PHASE2_SIGNATURE_IDS)
        and all(_signature_phase(str(entry.get("signature_id") or "")) for entry in ready_entries)
        and all(_signature_phase(str(entry.get("signature_id") or "")) for entry in blocked_entries)
        and all(
            _as_dict(entry).get("phase") == "phase1"
            for entry in signed_entries.values()
        )
    )


def _unsafe_claims(report: Mapping[str, Any]) -> list[str]:
    claims: list[str] = []
    for field in (
        "default_behavior_changed",
        "runtime_dispatch_ready",
        "native_dispatch_allowed",
        "training_path_enabled",
        "product_native_ready",
        "approval_recorded",
    ):
        if report.get(field) is True:
            claims.append(f"signature_bundle_unsafe:{field}")
    return claims


def _required_manual_fields_by_signature_id(entries: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {
        signature_id: fields
        for entry in entries
        for signature_id, fields in [(str(entry.get("signature_id") or ""), _required_manual_fields(entry))]
        if signature_id
    }


def _required_manual_fields(entry: Mapping[str, Any]) -> list[str]:
    signature_id = str(entry.get("signature_id") or "")
    approval_field = APPROVAL_FIELDS_BY_SIGNATURE_ID.get(signature_id, "")
    return _dedupe(
        [
            "reviewer",
            "reviewed_at",
            approval_field,
            *_strings(entry.get("required_acknowledgement_fields")),
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


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
    parser.add_argument("--signature-bundle", default="", help="Optional v2 signature bundle packet JSON path.")
    parser.add_argument("--no-artifact", action="store_true", help="Print validation without writing packet artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_reviewer_handoff_packet(
        signature_bundle=_read_json(Path(args.signature_bundle)) if args.signature_bundle else None,
        write_artifact=not bool(args.no_artifact),
        write_signed_bundle_template=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


__all__ = ["build_optimizer_v2_reviewer_handoff_packet"]


if __name__ == "__main__":
    raise SystemExit(main())
