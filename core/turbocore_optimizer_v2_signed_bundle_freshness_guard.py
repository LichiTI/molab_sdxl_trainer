"""Freshness guard for TurboCore optimizer v2 signed bundles.

The guard checks that reviewer-facing templates and reviewer-returned bundles
refer to the current v2 signature bundle digest.  It is read-only: stale
bundles are blocked, but no approval artifacts are recorded and no native
dispatch path is enabled.
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

from core.turbocore_optimizer_v2_reviewer_handoff_packet import (  # noqa: E402
    ARTIFACT as REVIEWER_HANDOFF_ARTIFACT,
    SIGNED_BUNDLE_TEMPLATE_ARTIFACT,
    build_optimizer_v2_reviewer_handoff_packet,
)
from core.turbocore_optimizer_v2_signature_bundle_packet import (  # noqa: E402
    ARTIFACT as SIGNATURE_BUNDLE_ARTIFACT,
    build_optimizer_v2_signature_bundle_packet,
)


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle_freshness_guard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def build_optimizer_v2_signed_bundle_freshness_guard(
    *,
    signature_bundle: Mapping[str, Any] | None = None,
    reviewer_handoff: Mapping[str, Any] | None = None,
    signed_bundle: Mapping[str, Any] | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    bundle = _as_dict(signature_bundle) or _read_json(SIGNATURE_BUNDLE_ARTIFACT)
    if not bundle:
        bundle = build_optimizer_v2_signature_bundle_packet(write_artifact=True)
    handoff = _as_dict(reviewer_handoff) or _read_json(REVIEWER_HANDOFF_ARTIFACT)
    if not handoff:
        handoff = build_optimizer_v2_reviewer_handoff_packet(
            signature_bundle=bundle,
            write_artifact=True,
            write_signed_bundle_template=True,
        )
    signed = _as_dict(signed_bundle)
    template = _as_dict(handoff.get("signed_bundle_template")) or _read_json(SIGNED_BUNDLE_TEMPLATE_ARTIFACT)

    current_digest = _digest_payload(bundle)
    handoff_digest = str(handoff.get("source_signature_bundle_digest") or "")
    template_digest = str(template.get("source_signature_bundle_digest") or "")
    signed_digest = str(signed.get("source_signature_bundle_digest") or "")
    template_fresh = bool(current_digest and handoff_digest == current_digest and template_digest == current_digest)
    signed_present = bool(signed)
    signed_fresh = bool(signed_present and signed_digest == current_digest)
    signed_stale = bool(signed_present and signed_digest != current_digest)
    unsafe = _unsafe_claims(bundle, handoff, template, signed)
    ready_entries = _ready_signature_ids(handoff)
    signed_entries = _signed_signature_ids(signed)
    unknown_signed_entries = [signature_id for signature_id in signed_entries if signature_id not in ready_entries]
    payload = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_signed_bundle_freshness_guard_v0",
        "gate": "optimizer_v2_signed_bundle_freshness_guard",
        "roadmap": ROADMAP,
        "ok": bool(template_fresh and not signed_stale and not unknown_signed_entries and not unsafe),
        "freshness_guard_ready": bool(template_fresh and not signed_stale and not unknown_signed_entries and not unsafe),
        "approval_recorded": False,
        "approval_artifact_written": False,
        "roadmap_complete": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "current_signature_bundle_digest": current_digest,
        "handoff_source_signature_bundle_digest": handoff_digest,
        "template_source_signature_bundle_digest": template_digest,
        "signed_bundle_source_signature_bundle_digest": signed_digest,
        "ready_signature_ids": ready_entries,
        "signed_signature_ids": signed_entries,
        "unknown_signed_signature_ids": unknown_signed_entries,
        "source_artifacts": {
            "signature_bundle": str(SIGNATURE_BUNDLE_ARTIFACT),
            "reviewer_handoff": str(REVIEWER_HANDOFF_ARTIFACT),
            "signed_bundle_template": str(SIGNED_BUNDLE_TEMPLATE_ARTIFACT),
        },
        "summary": {
            "v2_signed_bundle_freshness_guard_ready_count": 1
            if template_fresh and not signed_stale and not unknown_signed_entries and not unsafe
            else 0,
            "v2_signed_bundle_freshness_current_digest_present_count": 1 if current_digest else 0,
            "v2_signed_bundle_freshness_template_digest_match_count": 1 if template_fresh else 0,
            "v2_signed_bundle_freshness_signed_bundle_present_count": 1 if signed_present else 0,
            "v2_signed_bundle_freshness_signed_bundle_digest_match_count": 1 if signed_fresh else 0,
            "v2_signed_bundle_freshness_stale_signed_bundle_count": 1 if signed_stale else 0,
            "v2_signed_bundle_freshness_unknown_signed_entry_count": len(unknown_signed_entries),
            "v2_signed_bundle_freshness_approval_recorded_count": 0,
            "v2_signed_bundle_freshness_approval_artifact_written_count": 0,
            "v2_signed_bundle_freshness_runtime_dispatch_ready_count": 0,
            "v2_signed_bundle_freshness_native_dispatch_allowed_count": 0,
            "v2_signed_bundle_freshness_training_path_enabled_count": 0,
            "v2_signed_bundle_freshness_product_native_ready_count": 0,
            "v2_signed_bundle_freshness_default_behavior_changed_count": 0,
            "v2_signed_bundle_freshness_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(
            ([] if current_digest else ["current_signature_bundle_digest_missing"])
            + ([] if template_fresh else ["signed_bundle_template_digest_stale_or_missing"])
            + (["signed_bundle_digest_stale"] if signed_stale else [])
            + [f"unknown_signed_entry:{signature_id}" for signature_id in unknown_signed_entries]
            + unsafe
        ),
        "promotion_blockers": _dedupe(
            [
                "v2_signed_bundle_freshness_guard_is_not_approval_record",
                "product_dispatch_still_requires_explicit_route_binding",
            ]
            + (["signed_bundle_digest_stale"] if signed_stale else [])
            + unsafe
        ),
        "recommended_next_step": (
            "send only the current signed-bundle template; reject reviewer bundles whose source digest differs"
        ),
        "notes": [
            "The guard checks freshness only; it does not validate reviewer identity or record approval.",
            "A reviewer-returned bundle must carry the current source_signature_bundle_digest.",
            "Default product training remains PyTorch authoritative.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _ready_signature_ids(handoff: Mapping[str, Any]) -> list[str]:
    entries = handoff.get("ready_signature_entries")
    if not isinstance(entries, (list, tuple)):
        return []
    return [
        str(_as_dict(entry).get("signature_id") or "")
        for entry in entries
        if str(_as_dict(entry).get("signature_id") or "")
    ]


def _signed_signature_ids(signed_bundle: Mapping[str, Any]) -> list[str]:
    entries = signed_bundle.get("signed_entries")
    if isinstance(entries, Mapping):
        return [str(key) for key in entries]
    if isinstance(entries, (list, tuple)):
        ids: list[str] = []
        for entry in entries:
            signature_id = str(_as_dict(entry).get("signature_id") or "")
            if signature_id:
                ids.append(signature_id)
        return ids
    return []


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
                claims.append(f"freshness_source_{index}_unsafe:{field}")
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


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-bundle", default="", help="Optional current v2 signature bundle packet JSON path.")
    parser.add_argument("--reviewer-handoff", default="", help="Optional v2 reviewer handoff packet JSON path.")
    parser.add_argument("--signed-bundle", default="", help="Optional reviewer-returned signed bundle JSON path.")
    parser.add_argument("--no-artifact", action="store_true", help="Print scorecard without writing artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_signed_bundle_freshness_guard(
        signature_bundle=_read_json_if_supplied(args.signature_bundle),
        reviewer_handoff=_read_json_if_supplied(args.reviewer_handoff),
        signed_bundle=_read_json_if_supplied(args.signed_bundle),
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


__all__ = ["build_optimizer_v2_signed_bundle_freshness_guard"]


if __name__ == "__main__":
    raise SystemExit(main())
