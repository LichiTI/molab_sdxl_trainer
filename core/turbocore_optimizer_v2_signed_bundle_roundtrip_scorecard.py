"""Dry-run reviewer-returned bundle roundtrip for TurboCore optimizer v2.

This scorecard exercises the current signed-bundle handoff path without
recording approvals: reviewer handoff template -> synthetic signed bundle ->
intake validator -> extraction -> execution preflight.  It is an operator
readiness check, not a release approval.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_v2_approval_execution_preflight import (  # noqa: E402
    build_optimizer_v2_approval_execution_preflight,
)
from core.turbocore_optimizer_v2_reviewer_handoff_packet import (  # noqa: E402
    build_optimizer_v2_reviewer_handoff_packet,
)
from core.turbocore_optimizer_v2_signature_bundle_packet import (  # noqa: E402
    ARTIFACT as SIGNATURE_BUNDLE_ARTIFACT,
    SOURCE_ARTIFACTS as SIGNATURE_BUNDLE_SOURCE_ARTIFACTS,
    build_optimizer_v2_signature_bundle_packet,
)
from core.turbocore_optimizer_v2_signed_bundle_extractor import (  # noqa: E402
    build_optimizer_v2_signed_bundle_extraction_record,
)
from core.turbocore_optimizer_v2_signed_bundle_intake_record import (  # noqa: E402
    build_optimizer_v2_signed_bundle_intake_record,
)


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle_roundtrip_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def build_optimizer_v2_signed_bundle_roundtrip_scorecard(
    *,
    reviewer_handoff: Mapping[str, Any] | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    signature_bundle = (
        _read_json(SIGNATURE_BUNDLE_ARTIFACT)
        if reviewer_handoff is not None
        else build_optimizer_v2_signature_bundle_packet(write_artifact=False)
    )
    handoff = _as_dict(reviewer_handoff) or build_optimizer_v2_reviewer_handoff_packet(
        signature_bundle=signature_bundle,
        write_artifact=False,
        write_signed_bundle_template=False,
    )
    ready_templates = _ready_templates(handoff)
    signed_bundle = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_signed_bundle_roundtrip_dry_run_v0",
        "source_signature_bundle_digest": str(handoff.get("source_signature_bundle_digest") or ""),
        "unsigned_template": False,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "signed_entries": {
            signature_id: _sign_template(signature_id, template)
            for signature_id, template in ready_templates.items()
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        paths = _preflight_paths(temp_root)
        _write_json(paths["signature_bundle"], signature_bundle)
        _write_json(paths["signed_bundle"], signed_bundle)
        extraction = build_optimizer_v2_signed_bundle_extraction_record(
            signature_bundle=signature_bundle,
            signed_bundle=signed_bundle,
            output_dir=temp_root,
            write_artifact=False,
            write_extracted_artifacts=True,
        )
        _write_json(paths["training_launch_contract"], _training_launch_contract_artifact())
        _write_json(paths["product_exposure_evidence"], _product_exposure_evidence_artifact())
        intake = build_optimizer_v2_signed_bundle_intake_record(
            signature_bundle=signature_bundle,
            signed_bundle=signed_bundle,
            owner_review_packet=_read_json(SIGNATURE_BUNDLE_SOURCE_ARTIFACTS["owner_release_review_packet"]),
            product_exposure_decision=_read_json(SIGNATURE_BUNDLE_SOURCE_ARTIFACTS["product_exposure_decision"]),
            owner_direction_packet=_read_json(SIGNATURE_BUNDLE_SOURCE_ARTIFACTS["owner_release_direction_packet"]),
            write_artifact=False,
            write_approval_artifacts=False,
        )
        preflight = build_optimizer_v2_approval_execution_preflight(paths=paths, write_artifact=False)

    extraction_summary = _as_dict(extraction.get("summary"))
    intake_summary = _as_dict(intake.get("summary"))
    preflight_summary = _as_dict(preflight.get("summary"))
    unsafe = _unsafe_claims(handoff, signed_bundle, intake, extraction, preflight)
    ready_template_count = len(ready_templates)
    owner_direction_ready = "owner_release_direction" in ready_templates
    phase1_ready = preflight_summary.get("v2_approval_preflight_phase1_ready_count") == 1
    phase2_ready = preflight_summary.get("v2_approval_preflight_phase2_ready_count") == 1
    extraction_digest_shape_ready = bool(
        _int(extraction_summary.get("v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count"))
        >= 2
    )
    preflight_digest_shape_ready = bool(
        _int(preflight_summary.get("v2_approval_preflight_signed_payload_digest_present_count")) >= 2
        and _int(preflight_summary.get("v2_approval_preflight_signed_bundle_entry_digest_present_count")) >= 2
        and _int(preflight_summary.get("v2_approval_preflight_signed_payload_bundle_digest_match_count")) >= 2
    )
    roundtrip_integrity_ready = bool(extraction_digest_shape_ready and preflight_digest_shape_ready)
    payload = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_signed_bundle_roundtrip_scorecard_v0",
        "gate": "optimizer_v2_signed_bundle_roundtrip_dry_run",
        "roadmap": ROADMAP,
        "ok": bool(ready_template_count >= 2 and phase1_ready and roundtrip_integrity_ready and not unsafe),
        "roundtrip_dry_run_ready": bool(
            ready_template_count >= 2 and phase1_ready and roundtrip_integrity_ready and not unsafe
        ),
        "approval_recorded": False,
        "approval_artifact_written": False,
        "roadmap_complete": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "input_artifacts_are_temporary": True,
        "signed_template_signature_ids": list(ready_templates),
        "owner_direction_signature_ready": owner_direction_ready,
        "intake_record_checks": intake.get("record_checks", []),
        "extraction_entry_checks": extraction.get("entry_checks", []),
        "preflight_signed_checks": preflight.get("signed_checks", []),
        "summary": {
            "v2_signed_bundle_roundtrip_ready_template_count": ready_template_count,
            "v2_signed_bundle_roundtrip_signed_bundle_present_count": 1,
            "v2_signed_bundle_roundtrip_intake_valid_record_count": _int(
                intake_summary.get("v2_signed_bundle_valid_record_count")
            ),
            "v2_signed_bundle_roundtrip_extracted_entry_count": _int(
                extraction_summary.get("v2_signed_bundle_extraction_extractable_entry_count")
            ),
            "v2_signed_bundle_roundtrip_extraction_signed_entry_digest_present_count": _int(
                extraction_summary.get("v2_signed_bundle_extraction_signed_entry_digest_present_count")
            ),
            "v2_signed_bundle_roundtrip_extraction_extractable_signed_entry_digest_present_count": _int(
                extraction_summary.get("v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count")
            ),
            "v2_signed_bundle_roundtrip_extraction_digest_shape_ready_count": (
                1 if extraction_digest_shape_ready else 0
            ),
            "v2_signed_bundle_roundtrip_extraction_artifact_written_count": _int(
                extraction_summary.get("v2_signed_bundle_extraction_artifact_written_count")
            ),
            "v2_signed_bundle_roundtrip_preflight_signed_payload_digest_present_count": _int(
                preflight_summary.get("v2_approval_preflight_signed_payload_digest_present_count")
            ),
            "v2_signed_bundle_roundtrip_preflight_signed_payload_digest_missing_count": _int(
                preflight_summary.get("v2_approval_preflight_signed_payload_digest_missing_count")
            ),
            "v2_signed_bundle_roundtrip_preflight_signed_bundle_entry_digest_present_count": _int(
                preflight_summary.get("v2_approval_preflight_signed_bundle_entry_digest_present_count")
            ),
            "v2_signed_bundle_roundtrip_preflight_signed_payload_bundle_digest_match_count": _int(
                preflight_summary.get("v2_approval_preflight_signed_payload_bundle_digest_match_count")
            ),
            "v2_signed_bundle_roundtrip_preflight_digest_shape_ready_count": 1
            if preflight_digest_shape_ready
            else 0,
            "v2_signed_bundle_roundtrip_phase1_ready_count": 1 if phase1_ready else 0,
            "v2_signed_bundle_roundtrip_phase2_ready_count": 1 if phase2_ready else 0,
            "v2_signed_bundle_roundtrip_full_ready_count": 1 if phase1_ready and phase2_ready else 0,
            "v2_signed_bundle_roundtrip_owner_direction_blocked_count": 0 if owner_direction_ready else 1,
            "v2_signed_bundle_roundtrip_approval_recorded_count": 0,
            "v2_signed_bundle_roundtrip_approval_artifact_written_count": 0,
            "v2_signed_bundle_roundtrip_runtime_dispatch_ready_count": 0,
            "v2_signed_bundle_roundtrip_native_dispatch_allowed_count": 0,
            "v2_signed_bundle_roundtrip_training_path_enabled_count": 0,
            "v2_signed_bundle_roundtrip_product_native_ready_count": 0,
            "v2_signed_bundle_roundtrip_default_behavior_changed_count": 0,
            "v2_signed_bundle_roundtrip_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(
            ([] if ready_template_count >= 2 else ["roundtrip_ready_template_count_below_two"])
            + ([] if phase1_ready else ["roundtrip_phase1_preflight_not_ready"])
            + ([] if extraction_digest_shape_ready else ["roundtrip_extraction_digest_shape_not_ready"])
            + ([] if preflight_digest_shape_ready else ["roundtrip_preflight_digest_shape_not_ready"])
            + ([] if owner_direction_ready else ["owner_release_direction_not_ready_for_signature"])
            + unsafe
        ),
        "promotion_blockers": _dedupe(
            [
                "v2_signed_bundle_roundtrip_is_dry_run_only",
                "v2_signed_bundle_roundtrip_not_real_approval",
                "product_dispatch_still_requires_explicit_route_binding",
            ]
            + ([] if owner_direction_ready else ["owner_release_direction_not_ready_for_signature"])
            + unsafe
        ),
        "recommended_next_step": (
            "send the reviewer handoff template to real reviewers; use this dry-run only to verify returned bundle shape"
        ),
        "notes": [
            "The signed bundle in this scorecard is synthetic and lives only in a temporary directory.",
            "The dry-run proves the current owner-review/product-exposure returned-bundle path reaches phase 1 preflight.",
            "Owner release direction remains blocked until its real prerequisites are recorded.",
            "No approval artifact is written and native optimizer dispatch remains default-off.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _ready_templates(handoff: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    template = _as_dict(handoff.get("signed_bundle_template"))
    entries = template.get("signed_entries")
    if not isinstance(entries, Mapping):
        return {}
    return {str(key): _as_dict(value) for key, value in entries.items()}


def _sign_template(signature_id: str, template: Mapping[str, Any]) -> dict[str, Any]:
    signed = _as_dict(template)
    signed["reviewer"] = f"synthetic_{signature_id}_roundtrip_dry_run"
    signed["reviewed_at"] = "2026-06-09"
    for key in list(signed):
        if key.startswith("approve_") or key.startswith("acknowledge_"):
            signed[key] = True
    return signed


def _preflight_paths(root: Path) -> dict[str, Path]:
    return {
        "signature_bundle": root / "turbocore_optimizer_v2_signature_bundle_packet.json",
        "signed_bundle": root / "turbocore_optimizer_v2_signed_bundle.reviewed.json",
        "owner_release_review": root / "signed_owner_release_review.json",
        "product_exposure_review": root / "signed_product_exposure_review.json",
        "owner_release_direction": root / "signed_owner_release_direction.json",
        "training_launch_contract": root / "native_update_training_launch_contract.json",
        "product_exposure_evidence": root / "native_update_product_exposure_evidence.json",
        "owner_release_direction_packet": root / "native_update_owner_release_direction_packet.json",
    }


def _training_launch_contract_artifact() -> dict[str, Any]:
    return _default_off_support_fields() | {
        "schema_version": 1,
        "package": "turbocore_native_update_training_launch_contract_v0",
        "gate": "native_update_training_launch_contract",
        "ok": True,
        "evidence_ready": True,
        "ready_for_training_launch_review": True,
        "post_training_launch_request_fields": {},
        "training_step_execution_contract_summary": {
            "source": "roundtrip://training_step_execution_contract",
            "digest": "training-step-digest",
        },
        "training_launch_evidence_summary": {
            "source": "roundtrip://training_launch_evidence",
            "digest": "training-launch-digest",
        },
    }


def _product_exposure_evidence_artifact() -> dict[str, Any]:
    return _default_off_support_fields() | {
        "schema_version": 1,
        "evidence": "native_update_product_exposure_evidence_v0",
        "source": "roundtrip://native_update_product_exposure_evidence",
        "ok": True,
        "default_off": True,
        "report_only": True,
        "contract_only": True,
        "product_exposure_decision_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "product_exposure_decision_ready": True,
    }


def _default_off_support_fields() -> dict[str, Any]:
    return {
        "approval_recorded": False,
        "approval_artifact_written": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _unsafe_claims(*reports: Mapping[str, Any]) -> list[str]:
    claims: list[str] = []
    for index, report in enumerate(reports):
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
            if report.get(field) is True:
                claims.append(f"roundtrip_source_{index}_unsafe:{field}")
        summary = _as_dict(report.get("summary"))
        for key, value in summary.items():
            if key.endswith(("runtime_dispatch_ready_count", "native_dispatch_allowed_count", "training_path_enabled_count")):
                if _int(value):
                    claims.append(f"roundtrip_source_{index}_unsafe:{key}")
            if key.endswith(("product_native_ready_count", "default_behavior_changed_count", "unsafe_claim_count")):
                if _int(value):
                    claims.append(f"roundtrip_source_{index}_unsafe:{key}")
    return _dedupe(claims)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-artifact", action="store_true", help="Print scorecard without writing artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_signed_bundle_roundtrip_scorecard(write_artifact=not bool(args.no_artifact))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


__all__ = ["build_optimizer_v2_signed_bundle_roundtrip_scorecard"]


if __name__ == "__main__":
    raise SystemExit(main())
