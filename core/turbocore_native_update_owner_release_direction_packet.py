"""Owner release-direction packet for TurboCore native optimizer exposure.

This artifact sits after the release-review archive and product-exposure
decision.  It creates a signable owner-direction surface but does not record
approval unless a matching signed direction is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_direction_packet.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"

ARCHIVE_ARTIFACT = ARTIFACT_DIR / "native_update_release_review_archive.json"
PRODUCT_EXPOSURE_ARTIFACT = ARTIFACT_DIR / "native_update_product_exposure_decision.json"
STABLE_SCOPE_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_stable_first_release_scope.json"

SCOPE = "native_update_owner_release_direction"
RECORDED_DECISION = "native_update_owner_release_direction_recorded_default_off"
WAITING_DECISION = "native_update_owner_release_direction_waiting_for_signed_owner_direction"

REQUIRED_ACKS = (
    "acknowledge_release_review_archive_ready",
    "acknowledge_product_exposure_decision_recorded_default_off",
    "acknowledge_stable_first_release_default_off_scope",
    "acknowledge_no_request_ui_schema_or_training_launch",
    "acknowledge_route_binding_requires_separate_step",
    "acknowledge_runtime_dispatch_remains_default_off_until_route_binding",
)
UNSAFE_TRUE_FIELDS = (
    "product_exposure_allowed",
    "product_exposure_enabled",
    "release_gate_open",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "ui_exposure_allowed",
    "backend_router_registered",
    "training_launch_allowed",
    "training_launch_enabled",
    "training_launch_executed",
    "training_path_enabled",
    "runtime_dispatch_allowed",
    "native_dispatch_allowed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_owner_release_request_fields",
    "post_product_exposure_request_fields",
    "post_training_route_request_fields",
    "request_adapter_fields",
    "request_schema_fields",
    "backend_router_registration",
    "ui_route_registration",
)


def build_native_update_owner_release_direction_packet(
    *,
    release_review_archive: Mapping[str, Any] | None = None,
    product_exposure_decision: Mapping[str, Any] | None = None,
    stable_first_release_scope: Mapping[str, Any] | None = None,
    signed_direction: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    archive = _source(release_review_archive, directory / ARCHIVE_ARTIFACT.name)
    exposure = _source(product_exposure_decision, directory / PRODUCT_EXPOSURE_ARTIFACT.name)
    stable = _source(stable_first_release_scope, directory / STABLE_SCOPE_ARTIFACT.name)
    direction = _as_dict(signed_direction)

    archive_ready = _archive_ready(archive)
    exposure_recorded = _product_exposure_recorded(exposure)
    stable_ready = _stable_scope_ready(stable)
    unsafe = _unsafe_claims(archive, "release_review_archive")
    unsafe += _unsafe_claims(exposure, "product_exposure_decision")
    unsafe += _unsafe_claims(stable, "stable_first_release_scope")
    prereq_blockers = _dedupe(
        ([] if archive_ready else ["release_review_archive_not_ready"])
        + ([] if exposure_recorded else ["product_exposure_decision_not_recorded"])
        + ([] if stable_ready else ["stable_first_release_scope_not_ready"])
        + unsafe
    )
    template = _signable_template(archive, exposure, stable, prereq_blockers)
    direction_blockers = _signed_direction_blockers(direction, template, prereq_blockers)
    direction_recorded = bool(direction and not prereq_blockers and not direction_blockers)
    blocked = _dedupe(prereq_blockers + direction_blockers)
    payload = {
        "schema_version": 1,
        "package": "turbocore_native_update_owner_release_direction_packet_v0",
        "gate": SCOPE,
        "ok": not unsafe,
        "roadmap": ROADMAP,
        "ready_for_owner_direction_signature": not prereq_blockers,
        "owner_direction_present": bool(direction),
        "owner_direction_valid": direction_recorded,
        "owner_release_direction_recorded": direction_recorded,
        "owner_release_approval_recorded": direction_recorded,
        "decision": RECORDED_DECISION if direction_recorded else WAITING_DECISION,
        "source_release_review_archive_digest": _digest_payload(archive),
        "source_product_exposure_decision_digest": _digest_payload(exposure),
        "source_stable_first_release_scope_digest": _digest_payload(stable),
        "source_owner_release_direction_template_digest": _digest_payload(template),
        "signed_owner_release_direction_template_digest": str(
            direction.get("source_owner_release_direction_template_digest") or ""
        ),
        "signed_owner_release_direction_digest_match": bool(
            direction
            and direction.get("source_owner_release_direction_template_digest")
            == _digest_payload(template)
        ),
        "signable_owner_release_direction_template": template,
        "required_acknowledgement_fields": list(REQUIRED_ACKS),
        "release_review_archive_summary": _archive_summary(archive),
        "product_exposure_decision_summary": _product_exposure_summary(exposure),
        "stable_first_release_scope_summary": _stable_summary(stable),
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
        "blocked_reasons": blocked,
        "promotion_blockers": _dedupe(
            blocked + ([] if direction_recorded else ["owner_release_direction_not_recorded"])
        ),
        "summary": {
            "release_review_archive_ready_count": 1 if archive_ready else 0,
            "product_exposure_decision_recorded_count": 1 if exposure_recorded else 0,
            "stable_first_release_turbocore_optimizer_blocker_count": _stable_blocker_count(stable),
            "turbocore_optimizer_default_off_release_scope_ready_count": 1 if stable_ready else 0,
            "owner_release_direction_ready_for_signature_count": 1 if not prereq_blockers else 0,
            "owner_release_direction_recorded_count": 1 if direction_recorded else 0,
            "owner_release_direction_approval_recorded_count": 1 if direction_recorded else 0,
            "owner_release_approval_recorded_count": 1 if direction_recorded else 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
        },
        "recommended_next_step": (
            "proceed to product training-route binding preflight in a separate step"
            if direction_recorded
            else "record product exposure and signed owner release direction before route binding"
            if not prereq_blockers
            else "finish archive/product-exposure/stable-scope prerequisites before owner release direction"
        ),
        "notes": [
            "This packet is report-only unless a matching signed owner direction is supplied.",
            "Recording owner release direction does not emit request fields, expose UI, or launch training.",
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


def _source(value: Mapping[str, Any] | None, path: Path) -> dict[str, Any]:
    if value is not None:
        return _as_dict(value)
    if not path.exists():
        return {}
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _archive_ready(archive: Mapping[str, Any]) -> bool:
    return bool(
        archive.get("archive_ready") is True
        and archive.get("archive_recorded") is True
        and archive.get("ready_for_owner_release_direction") is True
        and archive.get("default_off") is True
    )


def _product_exposure_recorded(exposure: Mapping[str, Any]) -> bool:
    return bool(
        exposure.get("product_exposure_decision_recorded") is True
        and exposure.get("decision") == "native_update_product_exposure_decision_recorded_default_off"
    )


def _stable_scope_ready(stable: Mapping[str, Any]) -> bool:
    summary = _as_dict(stable.get("summary"))
    return bool(
        stable.get("ok") is True
        and stable.get("turbocore_optimizer_default_off_release_scope_ready") is True
        and int(summary.get("stable_first_release_turbocore_optimizer_blocker_count", 0) or 0) == 0
        and int(summary.get("turbocore_optimizer_default_off_release_scope_ready_count", 0) or 0) == 1
    )


def _stable_blocker_count(stable: Mapping[str, Any]) -> int:
    return int(_as_dict(stable.get("summary")).get("stable_first_release_turbocore_optimizer_blocker_count", 0) or 0)


def _signed_direction_blockers(
    direction: Mapping[str, Any],
    template: Mapping[str, Any],
    prereq_blockers: list[str],
) -> list[str]:
    if not direction:
        return []
    blocked: list[str] = []
    if prereq_blockers:
        blocked.append("owner_release_direction_prerequisites_not_ready")
    expected_digest = _digest_payload(template)
    if direction.get("source_owner_release_direction_template_digest") != expected_digest:
        blocked.append("owner_release_direction_template_digest_mismatch")
    if direction.get("requested_scope") != SCOPE:
        blocked.append("owner_release_direction_requested_scope_mismatch")
    if direction.get("approve_native_update_owner_release_direction") is not True:
        blocked.append("owner_release_direction_not_approved")
    for field in REQUIRED_ACKS:
        if direction.get(field) is not True:
            blocked.append(f"owner_release_direction_ack_missing:{field}")
    blocked.extend(_unsafe_claims(direction, "owner_release_direction"))
    return _dedupe(blocked)


def _signable_template(
    archive: Mapping[str, Any],
    exposure: Mapping[str, Any],
    stable: Mapping[str, Any],
    prereq_blockers: list[str],
) -> dict[str, Any]:
    template: dict[str, Any] = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": SCOPE,
        "approve_native_update_owner_release_direction": False,
        "prerequisites_ready": not prereq_blockers,
        "acknowledged_evidence": {
            "release_review_archive_digest": _digest_payload(archive),
            "product_exposure_decision_digest": _digest_payload(exposure),
            "stable_first_release_scope_digest": _digest_payload(stable),
        },
    }
    for field in REQUIRED_ACKS:
        template[field] = False
    return template


def _archive_summary(archive: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(archive),
        "archive_ready": archive.get("archive_ready") is True,
        "archive_recorded": archive.get("archive_recorded") is True,
        "ready_for_owner_release_direction": archive.get("ready_for_owner_release_direction") is True,
        "default_off": archive.get("default_off") is True,
        "blocked_reasons": _strings(archive.get("blocked_reasons")),
    }


def _product_exposure_summary(exposure: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(exposure),
        "ok": exposure.get("ok") is True,
        "product_exposure_decision_recorded": exposure.get("product_exposure_decision_recorded") is True,
        "decision": str(exposure.get("decision") or ""),
        "blocked_reasons": _strings(exposure.get("blocked_reasons")),
    }


def _stable_summary(stable: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(stable),
        "ok": stable.get("ok") is True,
        "turbocore_optimizer_default_off_release_scope_ready": (
            stable.get("turbocore_optimizer_default_off_release_scope_ready") is True
        ),
        "stable_first_release_turbocore_optimizer_blocker_count": _stable_blocker_count(stable),
        "blocked_reasons": _strings(stable.get("blocked_reasons")),
    }


def _unsafe_claims(report: Mapping[str, Any], label: str) -> list[str]:
    if not report:
        return [f"{label}_missing"]
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if report.get(field) is True:
            blocked.append(f"{label}_unsafe:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        value = report.get(field)
        if value not in (None, {}, [], "", ()):
            blocked.append(f"{label}_unsafe_non_empty:{field}")
    return blocked


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {str(key): item for key, item in value.items() if not str(key).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


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
    parser.add_argument("--artifact-dir", default="", help="Directory containing native-update release artifacts.")
    parser.add_argument("--no-artifact", action="store_true", help="Print packet without writing the packet artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_native_update_owner_release_direction_packet(
        artifact_dir=args.artifact_dir or None,
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_native_update_owner_release_direction_packet"]


if __name__ == "__main__":
    raise SystemExit(main())
