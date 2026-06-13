"""Extract signed TurboCore optimizer v2 bundle entries for record validators.

This utility only splits a reviewer-returned bundle into the JSON files that
the existing record validators already expect.  It does not record approvals
and never enables native optimizer dispatch.
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


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle_extraction_record.json"
DEFAULT_SIGNED_BUNDLE = ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle.reviewed.json"
SIGNATURE_BUNDLE_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_signature_bundle_packet.json"
OUTPUT_ARTIFACTS = {
    "owner_release_review": ARTIFACT_DIR / "signed_owner_release_review.json",
    "product_exposure_review": ARTIFACT_DIR / "signed_product_exposure_review.json",
    "owner_release_direction": ARTIFACT_DIR / "signed_owner_release_direction.json",
}
PHASE1_SIGNATURE_IDS = ("owner_release_review", "product_exposure_review")
PHASE2_SIGNATURE_IDS = ("owner_release_direction",)
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def build_optimizer_v2_signed_bundle_extraction_record(
    *,
    signature_bundle: Mapping[str, Any] | None = None,
    signed_bundle: Mapping[str, Any] | None = None,
    output_dir: Path | str | None = None,
    write_artifact: bool = True,
    write_extracted_artifacts: bool = False,
) -> dict[str, Any]:
    bundle = _as_dict(signature_bundle) or _read_json(SIGNATURE_BUNDLE_ARTIFACT)
    signed = _as_dict(signed_bundle) if signed_bundle is not None else _read_json(DEFAULT_SIGNED_BUNDLE)
    output_root = Path(output_dir) if output_dir else ARTIFACT_DIR
    output_artifacts = {
        signature_id: output_root / path.name for signature_id, path in OUTPUT_ARTIFACTS.items()
    }
    entries = _signed_entries(signed)
    expected_template_digests = _expected_template_digests(bundle)
    ready_by_signature_id = _ready_by_signature_id(bundle)
    checks = [
        _entry_check(
            signature_id,
            entries.get(signature_id),
            output_artifacts[signature_id],
            expected_template_digests.get(signature_id, ""),
            ready_by_signature_id.get(signature_id) is True,
        )
        for signature_id in OUTPUT_ARTIFACTS
    ]
    unsafe = _unsafe_claims(signed, checks)
    expected_signature_ids = set(expected_template_digests)
    unknown_signature_ids = sorted(signature_id for signature_id in entries if signature_id not in expected_signature_ids)
    current_digest = _digest_payload(bundle)
    signed_digest = str(signed.get("source_signature_bundle_digest") or "")
    signed_present = bool(signed)
    unsigned_template_marker = signed.get("unsigned_template") is True
    source_digest_match = bool(signed_present and current_digest and signed_digest == current_digest)
    source_digest_stale = bool(signed_present and signed_digest != current_digest)
    source_blockers = ["signed_bundle_source_digest_stale_or_missing"] if source_digest_stale else []
    unsigned_blockers = ["signed_bundle_unsigned_template_marker_present"] if unsigned_template_marker else []
    missing_signed_signature_ids = _missing_signed_signature_ids(entries)
    extractable = [check for check in checks if check["extractable"]]
    phase1_extractable_count = _extractable_count(checks, PHASE1_SIGNATURE_IDS)
    phase2_extractable_count = _extractable_count(checks, PHASE2_SIGNATURE_IDS)
    phase1_missing_signature_count = sum(
        1 for signature_id in missing_signed_signature_ids if signature_id in PHASE1_SIGNATURE_IDS
    )
    phase2_missing_signature_count = sum(
        1 for signature_id in missing_signed_signature_ids if signature_id in PHASE2_SIGNATURE_IDS
    )
    template_digest_mismatch_count = _template_digest_mismatch_count(checks)
    template_digest_blockers = ["signed_bundle_template_digest_mismatch"] if template_digest_mismatch_count else []
    unknown_entry_blockers = [f"unknown_signed_entry:{signature_id}" for signature_id in unknown_signature_ids]
    not_ready_signature_ids = _not_ready_signature_ids(checks)
    not_ready_entry_blockers = [f"signed_entry_not_ready_for_signature:{signature_id}" for signature_id in not_ready_signature_ids]
    signed_entry_digest_present_count = _signed_entry_digest_present_count(checks)
    extractable_signed_entry_digest_present_count = _extractable_signed_entry_digest_present_count(checks)
    written = 0
    if (
        write_extracted_artifacts
        and not unsafe
        and not source_digest_stale
        and not unsigned_template_marker
        and not template_digest_mismatch_count
        and not unknown_signature_ids
        and not not_ready_signature_ids
    ):
        output_root.mkdir(parents=True, exist_ok=True)
        for check in extractable:
            path = Path(str(check["output_path"]))
            path.write_text(
                json.dumps(check["signed_entry"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written += 1
    all_extractable = (
        signed_present
        and source_digest_match
        and not unsigned_template_marker
        and not template_digest_mismatch_count
        and not unknown_signature_ids
        and not not_ready_signature_ids
        and len(extractable) == len(OUTPUT_ARTIFACTS)
        and not unsafe
    )
    valid_extraction_context = bool(
        signed_present
        and source_digest_match
        and not unsigned_template_marker
        and not template_digest_mismatch_count
        and not unknown_signature_ids
        and not not_ready_signature_ids
        and not unsafe
    )
    phase1_ready = bool(valid_extraction_context and phase1_extractable_count == len(PHASE1_SIGNATURE_IDS))
    phase2_ready = bool(valid_extraction_context and phase2_extractable_count == len(PHASE2_SIGNATURE_IDS))
    payload = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_signed_bundle_extraction_record_v0",
        "gate": "optimizer_v2_signed_bundle_extraction_record",
        "roadmap": ROADMAP,
        "ok": not unsafe
        and not source_digest_stale
        and not unsigned_template_marker
        and not template_digest_mismatch_count
        and not unknown_signature_ids
        and not not_ready_signature_ids,
        "signed_bundle_present": signed_present,
        "signed_bundle_source_digest_match": source_digest_match,
        "signed_bundle_source_digest_stale": source_digest_stale,
        "signed_bundle_unsigned_template_marker": unsigned_template_marker,
        "unknown_signed_signature_ids": unknown_signature_ids,
        "missing_signed_signature_ids": missing_signed_signature_ids,
        "extraction_ready": all_extractable,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "roadmap_complete": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "source_signed_bundle": str(DEFAULT_SIGNED_BUNDLE),
        "source_signature_bundle": str(SIGNATURE_BUNDLE_ARTIFACT),
        "current_signature_bundle_digest": current_digest,
        "signed_bundle_source_signature_bundle_digest": signed_digest,
        "write_extracted_artifacts": bool(write_extracted_artifacts),
        "entry_checks": checks,
        "output_artifacts": {signature_id: str(path) for signature_id, path in output_artifacts.items()},
        "summary": {
            "v2_signed_bundle_extraction_entry_count": len(OUTPUT_ARTIFACTS),
            "v2_signed_bundle_extraction_present_count": 1 if signed_present else 0,
            "v2_signed_bundle_extraction_source_digest_match_count": 1 if source_digest_match else 0,
            "v2_signed_bundle_extraction_source_digest_stale_count": 1 if source_digest_stale else 0,
            "v2_signed_bundle_extraction_unsigned_template_count": 1 if unsigned_template_marker else 0,
            "v2_signed_bundle_extraction_template_digest_mismatch_count": template_digest_mismatch_count,
            "v2_signed_bundle_extraction_unknown_entry_count": len(unknown_signature_ids),
            "v2_signed_bundle_extraction_missing_signature_count": len(missing_signed_signature_ids),
            "v2_signed_bundle_extraction_phase1_missing_signature_count": phase1_missing_signature_count,
            "v2_signed_bundle_extraction_phase2_missing_signature_count": phase2_missing_signature_count,
            "v2_signed_bundle_extraction_not_ready_entry_count": len(not_ready_signature_ids),
            "v2_signed_bundle_extraction_signed_entry_digest_present_count": signed_entry_digest_present_count,
            "v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count": (
                extractable_signed_entry_digest_present_count
                if signed_present
                and not source_digest_stale
                and not unsigned_template_marker
                and not template_digest_mismatch_count
                and not unknown_signature_ids
                and not not_ready_signature_ids
                else 0
            ),
            "v2_signed_bundle_extraction_extractable_entry_count": len(extractable)
            if signed_present
            and not source_digest_stale
            and not unsigned_template_marker
            and not template_digest_mismatch_count
            and not unknown_signature_ids
            and not not_ready_signature_ids
            else 0,
            "v2_signed_bundle_extraction_phase1_extractable_entry_count": phase1_extractable_count
            if valid_extraction_context
            else 0,
            "v2_signed_bundle_extraction_phase1_ready_for_record_count": 1 if phase1_ready else 0,
            "v2_signed_bundle_extraction_phase2_extractable_entry_count": phase2_extractable_count
            if valid_extraction_context
            else 0,
            "v2_signed_bundle_extraction_phase2_ready_for_record_count": 1 if phase2_ready else 0,
            "v2_signed_bundle_extraction_full_ready_for_record_count": 1 if all_extractable else 0,
            "v2_signed_bundle_extraction_missing_entry_count": len(OUTPUT_ARTIFACTS) - len(extractable)
            if signed_present
            and not source_digest_stale
            and not unsigned_template_marker
            and not template_digest_mismatch_count
            and not unknown_signature_ids
            and not not_ready_signature_ids
            else len(OUTPUT_ARTIFACTS),
            "v2_signed_bundle_extraction_owner_review_extracted_count": _count_extractable(checks, "owner_release_review")
            if not source_digest_stale
            else 0,
            "v2_signed_bundle_extraction_product_exposure_extracted_count": _count_extractable(checks, "product_exposure_review")
            if not source_digest_stale
            else 0,
            "v2_signed_bundle_extraction_owner_direction_extracted_count": _count_extractable(checks, "owner_release_direction")
            if not source_digest_stale
            else 0,
            "v2_signed_bundle_extraction_artifact_written_count": written,
            "v2_signed_bundle_extraction_ready_for_record_count": 1 if all_extractable else 0,
            "v2_signed_bundle_extraction_approval_recorded_count": 0,
            "v2_signed_bundle_extraction_runtime_dispatch_ready_count": 0,
            "v2_signed_bundle_extraction_native_dispatch_allowed_count": 0,
            "v2_signed_bundle_extraction_training_path_enabled_count": 0,
            "v2_signed_bundle_extraction_product_native_ready_count": 0,
            "v2_signed_bundle_extraction_default_behavior_changed_count": 0,
            "v2_signed_bundle_extraction_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(
            ([] if signed_present else ["signed_bundle_missing"])
            + source_blockers
            + unsigned_blockers
            + template_digest_blockers
            + unknown_entry_blockers
            + not_ready_entry_blockers
            + [reason for check in checks for reason in _strings(check.get("blocked_reasons"))]
            + unsafe
        ),
        "promotion_blockers": _dedupe(
            [
                "v2_signed_bundle_extraction_not_approval_record",
                "product_dispatch_still_requires_explicit_route_binding",
            ]
            + ([] if all_extractable else ["signed_bundle_extraction_incomplete"])
            + source_blockers
            + unsigned_blockers
            + template_digest_blockers
            + unknown_entry_blockers
            + not_ready_entry_blockers
            + unsafe
        ),
        "recommended_next_step": (
            "extract signed entries, then run the individual approval record validators"
            if all_extractable
            else "supply a reviewer-returned signed bundle with all required signature entries"
        ),
        "notes": [
            "Extraction writes only signed review JSON inputs for record validators.",
            "Extraction is not an approval record and does not enable product/native dispatch.",
            "Partial extraction is allowed for review handoff checks but does not complete the roadmap.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _entry_check(
    signature_id: str,
    entry: Mapping[str, Any] | None,
    output_path: Path,
    expected_template_digest: str,
    ready_for_signature: bool,
) -> dict[str, Any]:
    signed = _as_dict(entry)
    blockers = _entry_blockers(signature_id, signed, expected_template_digest, ready_for_signature)
    signed_entry_digest = _digest_payload(signed) if signed else ""
    return {
        "schema_version": 1,
        "signature_id": signature_id,
        "extractable": bool(signed and not blockers),
        "output_path": str(output_path),
        "ready_for_signature": ready_for_signature,
        "signed_entry": signed,
        "signed_entry_digest": signed_entry_digest,
        "blocked_reasons": blockers,
    }


def _entry_blockers(
    signature_id: str,
    signed: Mapping[str, Any],
    expected_template_digest: str,
    ready_for_signature: bool,
) -> list[str]:
    if not signed:
        return [f"{signature_id}_missing"]
    blockers: list[str] = []
    if not ready_for_signature:
        blockers.append(f"{signature_id}_not_ready_for_signature")
    signed_template_digest = _signed_template_digest(signature_id, signed)
    if not expected_template_digest:
        blockers.append(f"{signature_id}_source_v2_signature_template_digest_expected_missing")
    elif not signed_template_digest:
        blockers.append(f"{signature_id}_source_v2_signature_template_digest_missing")
    elif signed_template_digest != expected_template_digest:
        blockers.append(f"{signature_id}_source_v2_signature_template_digest_mismatch")
    if not str(signed.get("reviewer") or "").strip():
        blockers.append(f"{signature_id}_reviewer_missing")
    if not str(signed.get("reviewed_at") or "").strip():
        blockers.append(f"{signature_id}_reviewed_at_missing")
    approve_fields = [key for key in signed if key.startswith("approve_")]
    if not approve_fields:
        blockers.append(f"{signature_id}_approve_field_missing")
    blockers.extend(f"{key}_not_true" for key in approve_fields if signed.get(key) is not True)
    blockers.extend(
        f"{key}_not_true"
        for key in signed
        if key.startswith("acknowledge_") and signed.get(key) is not True
    )
    return blockers


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


def _ready_by_signature_id(signature_bundle: Mapping[str, Any]) -> dict[str, bool]:
    entries = signature_bundle.get("signature_entries")
    if not isinstance(entries, (list, tuple)):
        return {}
    out: dict[str, bool] = {}
    for raw in entries:
        entry = _as_dict(raw)
        signature_id = str(entry.get("signature_id") or "")
        if signature_id:
            out[signature_id] = entry.get("ready_for_signature") is True
    return out


def _not_ready_signature_ids(checks: list[Mapping[str, Any]]) -> list[str]:
    return sorted(
        str(check.get("signature_id") or "")
        for check in checks
        if _as_dict(check.get("signed_entry")) and check.get("ready_for_signature") is not True
    )


def _missing_signed_signature_ids(entries: Mapping[str, Any]) -> list[str]:
    return [signature_id for signature_id in OUTPUT_ARTIFACTS if signature_id not in entries]


def _signed_template_digest(signature_id: str, signed: Mapping[str, Any]) -> str:
    generic = str(signed.get("source_v2_signature_template_digest") or "")
    if generic:
        return generic
    if signature_id == "owner_release_review":
        return str(signed.get("source_release_review_template_digest") or "")
    if signature_id == "owner_release_direction":
        return str(signed.get("source_owner_release_direction_template_digest") or "")
    return ""


def _template_digest_mismatch_count(checks: list[Mapping[str, Any]]) -> int:
    return sum(
        1
        for check in checks
        for reason in _strings(check.get("blocked_reasons"))
        if "source_v2_signature_template_digest" in reason
    )


def _signed_entry_digest_present_count(checks: list[Mapping[str, Any]]) -> int:
    return sum(1 for check in checks if str(check.get("signed_entry_digest") or "").strip())


def _extractable_signed_entry_digest_present_count(checks: list[Mapping[str, Any]]) -> int:
    return sum(
        1
        for check in checks
        if check.get("extractable") is True and str(check.get("signed_entry_digest") or "").strip()
    )


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


def _unsafe_claims(signed_bundle: Mapping[str, Any], checks: list[Mapping[str, Any]]) -> list[str]:
    claims: list[str] = []
    for field in (
        "approval_recorded",
        "approval_artifact_written",
        "default_behavior_changed",
        "runtime_dispatch_ready",
        "native_dispatch_allowed",
        "training_path_enabled",
        "product_native_ready",
    ):
        if signed_bundle.get(field) is True:
            claims.append(f"signed_bundle_unsafe:{field}")
    for check in checks:
        entry = _as_dict(check.get("signed_entry"))
        for field in (
            "default_behavior_changed",
            "runtime_dispatch_ready",
            "native_dispatch_allowed",
            "training_path_enabled",
            "product_native_ready",
        ):
            if entry.get(field) is True:
                claims.append(f"{check.get('signature_id')}_unsafe:{field}")
    return _dedupe(claims)


def _count_extractable(checks: list[Mapping[str, Any]], signature_id: str) -> int:
    return 1 if any(check.get("signature_id") == signature_id and check.get("extractable") is True for check in checks) else 0


def _extractable_count(checks: list[Mapping[str, Any]], signature_ids: tuple[str, ...]) -> int:
    return sum(_count_extractable(checks, signature_id) for signature_id in signature_ids)


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
    parser.add_argument("--signature-bundle", default="", help="Path to the current v2 signature bundle packet JSON.")
    parser.add_argument("--signed-bundle", default="", help="Path to a reviewer-returned signed v2 bundle JSON.")
    parser.add_argument("--output-dir", default="", help="Directory for extracted signed review JSON files.")
    parser.add_argument("--write-extracted-artifacts", action="store_true", help="Write extracted signed review JSON files.")
    parser.add_argument("--no-artifact", action="store_true", help="Print extraction report without writing report artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_signed_bundle_extraction_record(
        signature_bundle=_read_json_if_supplied(args.signature_bundle),
        signed_bundle=_read_json_if_supplied(args.signed_bundle),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        write_artifact=not bool(args.no_artifact),
        write_extracted_artifacts=bool(args.write_extracted_artifacts),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_optimizer_v2_signed_bundle_extraction_record"]


if __name__ == "__main__":
    raise SystemExit(main())
