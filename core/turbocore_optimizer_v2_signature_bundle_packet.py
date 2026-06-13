"""Signable bundle for the remaining TurboCore optimizer v2 gates.

The bundle groups existing signable templates and their prerequisites.  It is
not an approval record and never enables native optimizer dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_signature_bundle_packet.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"

SOURCE_ARTIFACTS = {
    "remaining_gate_handoff": ARTIFACT_DIR / "turbocore_optimizer_v2_remaining_gate_handoff_scorecard.json",
    "owner_release_review_packet": ARTIFACT_DIR / "native_update_owner_release_review_packet.json",
    "product_exposure_decision": ARTIFACT_DIR / "native_update_product_exposure_decision.json",
    "owner_release_direction_packet": ARTIFACT_DIR / "native_update_owner_release_direction_packet.json",
}


def build_optimizer_v2_signature_bundle_packet(
    *,
    source_reports: Mapping[str, Mapping[str, Any]] | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    sources = _load_sources(source_reports)
    handoff = sources["remaining_gate_handoff"]
    owner_review = sources["owner_release_review_packet"]
    exposure = sources["product_exposure_decision"]
    direction = sources["owner_release_direction_packet"]

    entries = [
        _owner_release_review_entry(owner_review),
        _product_exposure_entry(exposure),
        _owner_release_direction_entry(direction),
    ]
    ready_now_count = sum(1 for entry in entries if entry["ready_for_signature"])
    blocked = _dedupe(reason for entry in entries for reason in entry["blocked_reasons"])
    unsafe = _unsafe_claims(handoff, owner_review, exposure, direction)
    payload = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_signature_bundle_packet_v0",
        "gate": "optimizer_v2_signature_bundle_packet",
        "roadmap": ROADMAP,
        "ok": not unsafe and _handoff_ready(handoff),
        "signature_bundle_ready": not unsafe and _handoff_ready(handoff),
        "roadmap_complete": False,
        "approval_recorded": False,
        "product_exposure_decision_recorded": False,
        "owner_release_direction_recorded": False,
        "promotion_ready": False,
        "manual_review_required": True,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "source_artifacts": {name: str(path) for name, path in SOURCE_ARTIFACTS.items()},
        "signature_entries": entries,
        "summary": {
            "v2_signature_bundle_entry_count": len(entries),
            "v2_signature_bundle_ready_for_signature_count": ready_now_count,
            "v2_signature_bundle_blocked_entry_count": len(entries) - ready_now_count,
            "v2_signature_bundle_owner_review_ready_count": 1 if entries[0]["ready_for_signature"] else 0,
            "v2_signature_bundle_product_exposure_ready_count": 1 if entries[1]["ready_for_signature"] else 0,
            "v2_signature_bundle_owner_direction_ready_count": 1 if entries[2]["ready_for_signature"] else 0,
            "v2_signature_bundle_approval_recorded_count": 0,
            "v2_signature_bundle_runtime_dispatch_ready_count": 0,
            "v2_signature_bundle_native_dispatch_allowed_count": 0,
            "v2_signature_bundle_training_path_enabled_count": 0,
            "v2_signature_bundle_product_native_ready_count": 0,
            "v2_signature_bundle_default_behavior_changed_count": 0,
            "v2_signature_bundle_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(blocked + unsafe),
        "promotion_blockers": _dedupe(
            [
                "v2_signature_bundle_waiting_for_real_signatures",
                "v2_owner_release_review_not_recorded",
                "v2_product_exposure_decision_not_recorded",
                "v2_owner_release_direction_not_recorded",
            ]
            + blocked
            + unsafe
        ),
        "recommended_next_step": (
            "sign ready owner-review and product-exposure templates, then rebuild archive and owner-direction packet"
        ),
        "notes": [
            "Ready-for-signature entries are templates only; no approval is recorded by this packet.",
            "Owner release direction remains blocked until release review archive and product exposure decision are recorded.",
            "Default product training remains PyTorch authoritative.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _owner_release_review_entry(packet: Mapping[str, Any]) -> dict[str, Any]:
    template = _as_dict(packet.get("signable_review_record_template"))
    source_digest = str(packet.get("source_release_review_template_digest") or "")
    ready = packet.get("ready_for_owner_signature") is True
    return {
        "schema_version": 1,
        "signature_id": "owner_release_review",
        "target_gate": "native_update_owner_release_review_record",
        "requested_scope": str(packet.get("required_requested_scope") or template.get("requested_scope") or ""),
        "ready_for_signature": ready,
        "records_approval_when_signed": True,
        "approval_recorded": False,
        "template_digest": _digest_payload(template),
        "source_template_digest": source_digest,
        "required_acknowledgement_fields": _strings(packet.get("required_acknowledgement_fields")),
        "template": _with_v2_template_digest(_clear_review_identity(template), source_digest),
        "blocked_reasons": [] if ready else ["owner_release_review_packet_not_ready"],
    }


def _product_exposure_entry(decision: Mapping[str, Any]) -> dict[str, Any]:
    template = _as_dict(decision.get("product_exposure_review_template"))
    source_digest = _digest_payload(template)
    ready = decision.get("ready_for_product_exposure_review") is True
    return {
        "schema_version": 1,
        "signature_id": "product_exposure_review",
        "target_gate": "native_update_product_exposure_decision",
        "requested_scope": str(template.get("requested_scope") or "native_update_product_exposure_decision"),
        "ready_for_signature": ready,
        "records_approval_when_signed": True,
        "approval_recorded": False,
        "template_digest": _digest_payload(template),
        "source_template_digest": source_digest,
        "required_acknowledgement_fields": [
            key for key in template if key.startswith("acknowledge_")
        ],
        "template": _with_v2_template_digest(_clear_review_identity(template), source_digest),
        "blocked_reasons": [] if ready else ["product_exposure_decision_not_ready_for_review"],
    }


def _owner_release_direction_entry(packet: Mapping[str, Any]) -> dict[str, Any]:
    template = _as_dict(packet.get("signable_owner_release_direction_template"))
    source_digest = str(packet.get("source_owner_release_direction_template_digest") or "")
    ready = packet.get("ready_for_owner_direction_signature") is True
    blockers = _strings(packet.get("blocked_reasons"))
    if not ready and not blockers:
        blockers = ["owner_release_direction_prerequisites_not_ready"]
    return {
        "schema_version": 1,
        "signature_id": "owner_release_direction",
        "target_gate": "native_update_owner_release_direction_record",
        "requested_scope": str(template.get("requested_scope") or "native_update_owner_release_direction"),
        "ready_for_signature": ready,
        "records_approval_when_signed": True,
        "approval_recorded": False,
        "template_digest": _digest_payload(template),
        "source_template_digest": source_digest,
        "required_acknowledgement_fields": _strings(packet.get("required_acknowledgement_fields")),
        "template": _with_v2_template_digest(_clear_review_identity(template), source_digest),
        "blocked_reasons": blockers,
    }


def _load_sources(source_reports: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    overrides = source_reports or {}
    return {
        name: _as_dict(overrides.get(name)) or _read_json(path)
        for name, path in SOURCE_ARTIFACTS.items()
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _handoff_ready(handoff: Mapping[str, Any]) -> bool:
    return bool(
        handoff.get("handoff_ready") is True
        and _summary_int(handoff, "v2_remaining_gate_total_count") == 6
        and _summary_int(handoff, "v2_remaining_gate_default_off_guard_count") == 1
    )


def _unsafe_claims(*reports: Mapping[str, Any]) -> list[str]:
    claims: list[str] = []
    for index, report in enumerate(reports):
        for field in (
            "default_behavior_changed",
            "runtime_dispatch_ready",
            "native_dispatch_allowed",
            "runtime_dispatch_allowed",
            "training_path_enabled",
            "product_native_ready",
            "product_exposure_allowed",
            "request_fields_emitted",
            "schema_exposure_allowed",
            "ui_exposure_allowed",
            "backend_router_registered",
        ):
            if report.get(field) is True:
                claims.append(f"source_{index}_unsafe:{field}")
    return _dedupe(claims)


def _clear_review_identity(template: Mapping[str, Any]) -> dict[str, Any]:
    out = _as_dict(template)
    out["reviewer"] = ""
    out["reviewed_at"] = ""
    for key in list(out):
        if key.startswith("approve_"):
            out[key] = False
        if key.startswith("acknowledge_"):
            out[key] = False
    return out


def _with_v2_template_digest(template: Mapping[str, Any], digest: str) -> dict[str, Any]:
    out = _as_dict(template)
    out["source_v2_signature_template_digest"] = str(digest or "")
    return out


def _summary_int(report: Mapping[str, Any], key: str) -> int:
    summary = _as_dict(report.get("summary"))
    try:
        return int(summary.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


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
    parser.add_argument("--no-artifact", action="store_true", help="Print validation without writing packet artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_signature_bundle_packet(write_artifact=not bool(args.no_artifact))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


__all__ = ["build_optimizer_v2_signature_bundle_packet"]


if __name__ == "__main__":
    raise SystemExit(main())
