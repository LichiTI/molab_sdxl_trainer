"""Owner action packet for TurboCore native-update release review.

The packet makes the remaining human review step explicit without recording or
simulating approval.  It is a signable template plus evidence digest summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_native_update_owner_release_handoff_summary import (
    build_native_update_owner_release_handoff_summary,
)
from core.turbocore_native_update_release_review_package import (
    HOLD_DECISION,
    build_native_update_release_review_package,
    load_gate_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
RELEASE_PACKAGE_ARTIFACT = ARTIFACT_DIR / "native_update_release_review_package.json"
OWNER_HANDOFF_ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_handoff_summary.json"
ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_review_packet.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def build_native_update_owner_release_review_packet(
    *,
    release_package: Mapping[str, Any] | None = None,
    handoff_summary: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    package = _release_package(release_package, directory)
    handoff = _handoff_summary(handoff_summary, package, directory)
    review_template = _as_dict(package.get("release_review_template"))
    handoff_template = _as_dict(handoff.get("review_template_for_owner"))
    blocked_reasons = _strings(package.get("blocked_reasons"))
    required_ack_fields = [
        "acknowledge_all_expected_gates_present",
        "acknowledge_all_gates_default_off",
        "acknowledge_no_request_ui_schema_exposure",
        "acknowledge_no_training_launch_or_native_execution",
        "acknowledge_product_exposure_requires_separate_owner_direction",
    ]
    signable_template = dict(review_template)
    source_template_digest = _digest_payload(review_template)
    source_package_digest = _digest_payload(package)
    signable_template["approve_native_update_release_review_package"] = False
    signable_template["reviewer"] = ""
    signable_template["reviewed_at"] = ""
    signable_template["source_release_review_template_digest"] = source_template_digest
    signable_template["source_release_package_digest"] = source_package_digest
    for field in required_ack_fields:
        signable_template[field] = False

    ready = bool(
        package.get("ok") is True
        and package.get("evidence_ready") is True
        and package.get("ready_for_owner_release_review") is True
        and package.get("decision") == HOLD_DECISION
        and blocked_reasons == ["native_update_release_owner_review_missing"]
        and handoff.get("technical_evidence_ready") is True
        and not _unsafe_top_level_enabled(package)
    )
    payload = {
        "schema_version": 1,
        "package": "turbocore_native_update_owner_release_review_packet_v0",
        "gate": "native_update_owner_release_review_packet",
        "ok": ready,
        "roadmap": ROADMAP,
        "ready_for_owner_signature": ready,
        "approval_recorded": False,
        "release_review_recorded": False,
        "decision": "native_update_owner_signature_required_default_off",
        "source_release_package_decision": str(package.get("decision", "") or ""),
        "source_release_package_digest": source_package_digest,
        "source_release_review_template_digest": source_template_digest,
        "source_handoff_digest": _digest_payload(handoff),
        "source_handoff_template_digest": str(handoff.get("source_release_package_digest", "") or ""),
        "digest_match": _digest_payload(review_template) == str(
            _as_dict(package.get("owner_release_review_handoff")).get("release_review_template_digest", "")
        ),
        "blocked_reasons": blocked_reasons,
        "owner_action_required": True,
        "required_owner_actions": [
            "fill reviewer",
            "fill reviewed_at",
            "set required acknowledgements true after review",
            "set approve_native_update_release_review_package true only if approving this default-off package",
            "rerun native_update_release_review_package with the signed review record",
        ],
        "required_review_fields": _strings(handoff.get("required_review_fields")),
        "required_requested_scope": str(handoff.get("required_requested_scope", "") or ""),
        "required_acknowledgement_fields": required_ack_fields,
        "required_gate_acknowledgement_count": int(handoff.get("required_gate_acknowledgement_count", 0) or 0),
        "required_supplemental_acknowledgements": _strings(
            handoff.get("required_supplemental_acknowledgements")
        ),
        "signable_review_record_template": signable_template,
        "owner_display_template": handoff_template,
        "compact_evidence": {
            "expected_gate_count": int(package.get("expected_gate_count", 0) or 0),
            "present_gate_count": int(package.get("present_gate_count", 0) or 0),
            "default_off_gate_count": int(package.get("default_off_gate_count", 0) or 0),
            "supplemental_gate_count": int(package.get("supplemental_gate_count", 0) or 0),
            "present_supplemental_gate_count": int(package.get("present_supplemental_gate_count", 0) or 0),
            "default_off_supplemental_gate_count": int(
                package.get("default_off_supplemental_gate_count", 0) or 0
            ),
            **_as_dict(handoff.get("summary")),
        },
        "must_remain_false": _strings(handoff.get("must_remain_false")),
        "must_remain_empty": _strings(handoff.get("must_remain_empty")),
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "recommended_next_step": "owner should sign or reject the default-off release review packet",
        "notes": [
            "This packet does not record owner approval.",
            "The signable template keeps approve_native_update_release_review_package=false by default.",
            "A signed release review still does not enable product exposure or native training dispatch.",
        ],
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _release_package(release_package: Mapping[str, Any] | None, directory: Path) -> dict[str, Any]:
    if release_package is not None:
        return _as_dict(release_package)
    source = directory / RELEASE_PACKAGE_ARTIFACT.name
    if source.exists():
        return _read_json(source)
    return build_native_update_release_review_package(gate_artifacts=load_gate_artifacts(directory))


def _handoff_summary(handoff_summary: Mapping[str, Any] | None, package: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    if handoff_summary is not None:
        return _as_dict(handoff_summary)
    source = directory / OWNER_HANDOFF_ARTIFACT.name
    if source.exists():
        return _read_json(source)
    return build_native_update_owner_release_handoff_summary(
        release_package=package,
        artifact_dir=directory,
        write_artifact=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _read_json_if_supplied(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return _read_json(Path(path))


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {str(key): item for key, item in value.items() if not str(key).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _unsafe_top_level_enabled(package: Mapping[str, Any]) -> bool:
    for key in (
        "product_exposure_allowed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "runtime_dispatch_allowed",
        "native_dispatch_allowed",
        "training_path_enabled",
        "training_launch_executed",
    ):
        if package.get(key) is True:
            return True
    return bool(package.get("post_release_request_fields"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-package", default="", help="Optional release package JSON path.")
    parser.add_argument("--handoff-summary", default="", help="Optional owner handoff summary JSON path.")
    parser.add_argument("--artifact-dir", default="", help="Directory containing native-update release artifacts.")
    parser.add_argument("--no-artifact", action="store_true", help="Print packet without writing the packet artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_native_update_owner_release_review_packet(
        release_package=_read_json_if_supplied(args.release_package),
        handoff_summary=_read_json_if_supplied(args.handoff_summary),
        artifact_dir=args.artifact_dir or None,
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_native_update_owner_release_review_packet"]


if __name__ == "__main__":
    raise SystemExit(main())
