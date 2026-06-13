"""V2 remaining-gate handoff for TurboCore optimizer release work.

This scorecard gathers the still-open v2 approval gates into one review
surface.  It does not record owner approval, product exposure approval, or
native dispatch readiness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_remaining_gate_handoff_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"

SOURCE_ARTIFACTS = {
    "owner_release_hold_package": ARTIFACT_DIR / "turbocore_optimizer_owner_release_hold_package_scorecard.json",
    "adaptive_lr_chain": ARTIFACT_DIR / "turbocore_adaptive_lr_chain_scorecard.json",
    "product_exposure_decision": ARTIFACT_DIR / "native_update_product_exposure_decision.json",
    "owner_release_review_packet": ARTIFACT_DIR / "native_update_owner_release_review_packet.json",
    "owner_release_review_record": ARTIFACT_DIR / "native_update_owner_release_review_record.json",
    "owner_release_direction_packet": ARTIFACT_DIR / "native_update_owner_release_direction_packet.json",
    "owner_release_direction_record": ARTIFACT_DIR / "native_update_owner_release_direction_record.json",
    "release_artifact_first_validation": ARTIFACT_DIR
    / "turbocore_optimizer_release_artifact_first_validation_scorecard.json",
    "native_readiness_gap": ARTIFACT_DIR / "turbocore_optimizer_native_readiness_gap_scorecard.json",
}

DEFAULT_OFF_COUNT_FIELDS = (
    "runtime_dispatch_ready_count",
    "native_dispatch_allowed_count",
    "training_path_enabled_count",
    "product_native_ready_count",
    "default_behavior_changed_count",
)


def build_optimizer_v2_remaining_gate_handoff_scorecard(
    *,
    source_reports: Mapping[str, Mapping[str, Any]] | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    sources = _load_sources(source_reports)
    owner_hold = sources["owner_release_hold_package"]
    adaptive_lr = sources["adaptive_lr_chain"]
    exposure = sources["product_exposure_decision"]
    review_packet = sources["owner_release_review_packet"]
    review_record = sources["owner_release_review_record"]
    direction_packet = sources["owner_release_direction_packet"]
    direction_record = sources["owner_release_direction_record"]
    release_artifact = sources["release_artifact_first_validation"]
    native_readiness = sources["native_readiness_gap"]

    rows = [
        *_owner_release_rows(owner_hold, review_packet, review_record, direction_packet, direction_record),
        _adaptive_lr_product_exposure_row(adaptive_lr, exposure),
    ]
    release_ready = _release_artifact_ready(release_artifact)
    native_default_off_ok = _native_readiness_default_off(native_readiness)
    unsafe_claims = _unsafe_claims(sources)
    open_rows = [row for row in rows if row["gate_closed"] is not True]
    payload = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_v2_remaining_gate_handoff_scorecard_v0",
        "gate": "optimizer_v2_remaining_gate_handoff",
        "roadmap": ROADMAP,
        "ok": release_ready and native_default_off_ok and not unsafe_claims,
        "handoff_ready": release_ready and native_default_off_ok and not unsafe_claims,
        "roadmap_complete": False,
        "promotion_ready": False,
        "manual_review_required": True,
        "approval_recorded": False,
        "product_exposure_decision_recorded": exposure.get("product_exposure_decision_recorded") is True,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "source_artifacts": {name: str(path) for name, path in SOURCE_ARTIFACTS.items()},
        "rows": rows,
        "summary": {
            "v2_remaining_gate_total_count": len(rows),
            "v2_remaining_gate_open_count": len(open_rows),
            "v2_remaining_gate_closed_count": len(rows) - len(open_rows),
            "v2_remaining_gate_owner_release_open_count": sum(
                1 for row in rows if row["gate_family"] == "owner_release" and not row["gate_closed"]
            ),
            "v2_remaining_gate_product_exposure_open_count": sum(
                1 for row in rows if row["gate_family"] == "product_exposure" and not row["gate_closed"]
            ),
            "v2_remaining_gate_handoff_ready_count": 1 if release_ready and native_default_off_ok else 0,
            "v2_remaining_gate_release_artifact_ready_count": 1 if release_ready else 0,
            "v2_remaining_gate_default_off_guard_count": 1 if native_default_off_ok and not unsafe_claims else 0,
            "v2_remaining_gate_owner_review_packet_ready_count": _bool_count(
                review_packet.get("ready_for_owner_signature")
                or review_packet.get("ready_for_owner_release_review")
            ),
            "v2_remaining_gate_owner_review_recorded_count": _bool_count(
                review_record.get("release_review_recorded")
            ),
            "v2_remaining_gate_owner_direction_ready_for_signature_count": _summary_int(
                direction_packet,
                "owner_release_direction_ready_for_signature_count",
            ),
            "v2_remaining_gate_owner_direction_recorded_count": _summary_int(
                direction_record,
                "owner_release_direction_recorded_count",
            ),
            "v2_remaining_gate_product_exposure_ready_for_review_count": _bool_count(
                exposure.get("ready_for_product_exposure_review")
            ),
            "v2_remaining_gate_product_exposure_decision_recorded_count": _bool_count(
                exposure.get("product_exposure_decision_recorded")
            ),
            "v2_remaining_gate_runtime_dispatch_ready_count": 0,
            "v2_remaining_gate_native_dispatch_allowed_count": 0,
            "v2_remaining_gate_training_path_enabled_count": 0,
            "v2_remaining_gate_product_native_ready_count": 0,
            "v2_remaining_gate_default_behavior_changed_count": 0,
            "v2_remaining_gate_unsafe_claim_count": len(unsafe_claims),
        },
        "blocked_reasons": _dedupe(
            [reason for row in open_rows for reason in row["blocked_reasons"]] + unsafe_claims
        ),
        "promotion_blockers": _dedupe(
            [
                "v2_owner_release_approval_missing",
                "v2_product_exposure_decision_not_recorded",
                "v2_product_dispatch_not_approved",
            ]
            + [reason for row in open_rows for reason in row["blocked_reasons"]]
            + unsafe_claims
        ),
        "recommended_next_step": (
            "collect signed owner/release review and product exposure decision records while keeping default dispatch off"
            if release_ready and native_default_off_ok
            else "refresh release artifacts and default-off guards before collecting signatures"
        ),
        "notes": [
            "This handoff is review-ready evidence, not an approval record.",
            "Synthetic smoke signatures are not counted as owner/release approval.",
            "Default product training remains PyTorch authoritative and native optimizer dispatch remains closed.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _owner_release_rows(
    owner_hold: Mapping[str, Any],
    review_packet: Mapping[str, Any],
    review_record: Mapping[str, Any],
    direction_packet: Mapping[str, Any],
    direction_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    hold_rows = _as_list(owner_hold.get("rows"))
    if not hold_rows:
        hold_rows = [
            {"roadmap_item": item, "family_id": item}
            for item in ("O2-1", "O2-2", "O2-3", "O2-4", "O2-5")
        ]
    review_ready = bool(
        review_packet.get("ready_for_owner_signature") or review_packet.get("ready_for_owner_release_review")
    )
    review_recorded = review_record.get("release_review_recorded") is True
    direction_ready = _summary_int(direction_packet, "owner_release_direction_ready_for_signature_count") > 0
    direction_recorded = _summary_int(direction_record, "owner_release_direction_recorded_count") > 0
    rows = []
    for row in hold_rows:
        family_id = str(row.get("family_id") or row.get("roadmap_item") or "")
        roadmap_item = str(row.get("roadmap_item") or family_id)
        gate_closed = bool(review_recorded and direction_recorded)
        blockers = []
        if not row.get("owner_release_hold_ready", True):
            blockers.append(f"{family_id}_hold_package_not_ready")
        if not review_ready:
            blockers.append("owner_release_review_packet_not_ready")
        if not review_recorded:
            blockers.append("owner_release_review_not_recorded")
        if not direction_ready:
            blockers.append("owner_release_direction_not_ready_for_signature")
        if not direction_recorded:
            blockers.append("owner_release_direction_not_recorded")
        blockers.extend(_row_default_off_blockers(row, family_id))
        rows.append(
            {
                "schema_version": 1,
                "roadmap_item": roadmap_item,
                "gate_family": "owner_release",
                "family_id": family_id,
                "gate_closed": gate_closed,
                "handoff_ready": bool(row.get("owner_release_hold_ready", True) and review_ready),
                "manual_record_required": not gate_closed,
                "required_record": "signed owner release review and signed owner release direction",
                "source_gate": str(row.get("source_gate") or owner_hold.get("gate") or ""),
                "blocked_reasons": _dedupe(blockers),
            }
        )
    return rows


def _adaptive_lr_product_exposure_row(
    adaptive_lr: Mapping[str, Any],
    exposure: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(adaptive_lr.get("summary"))
    ready_for_review = exposure.get("ready_for_product_exposure_review") is True
    decision_recorded = exposure.get("product_exposure_decision_recorded") is True
    blockers = []
    if int(summary.get("adaptive_lr_chain_ready_stage_count", 0) or 0) < 8:
        blockers.append("adaptive_lr_chain_prerequisites_not_ready")
    if int(summary.get("adaptive_lr_chain_product_exposure_gate_ready_count", 0) or 0) != 0:
        blockers.append("adaptive_lr_product_exposure_gate_unexpectedly_ready_without_record")
    if not ready_for_review:
        blockers.append("product_exposure_evidence_not_ready_for_review")
    if not decision_recorded:
        blockers.append("product_exposure_decision_not_recorded")
    blockers.extend(_row_default_off_blockers(adaptive_lr, "adaptive_lr_chain"))
    blockers.extend(_row_default_off_blockers(exposure, "product_exposure_decision"))
    return {
        "schema_version": 1,
        "roadmap_item": "O3-9",
        "gate_family": "product_exposure",
        "family_id": "adaptive_lr",
        "gate_closed": decision_recorded,
        "handoff_ready": ready_for_review,
        "manual_record_required": not decision_recorded,
        "required_record": "signed native_update_product_exposure_decision review",
        "source_gate": str(exposure.get("gate") or adaptive_lr.get("gate") or ""),
        "blocked_reasons": _dedupe(blockers),
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _as_dict(payload)


def _release_artifact_ready(report: Mapping[str, Any]) -> bool:
    summary = _as_dict(report.get("summary"))
    return bool(
        report.get("release_artifact_first_validation_ready")
        or int(summary.get("release_artifact_first_validation_ready_count", 0) or 0) == 1
    )


def _native_readiness_default_off(report: Mapping[str, Any]) -> bool:
    summary = _as_dict(report.get("summary"))
    keys = (
        "default_off_product_runtime_dispatch_ready_optimizer_count",
        "default_off_product_native_dispatch_allowed_optimizer_count",
        "default_off_product_training_path_enabled_optimizer_count",
        "default_off_product_product_native_ready_optimizer_count",
    )
    return bool(report) and all(int(summary.get(key, 0) or 0) == 0 for key in keys)


def _unsafe_claims(sources: Mapping[str, Mapping[str, Any]]) -> list[str]:
    claims: list[str] = []
    for source_name, report in sources.items():
        summary = _as_dict(report.get("summary"))
        for field in DEFAULT_OFF_COUNT_FIELDS:
            if int(summary.get(field, 0) or 0) != 0:
                claims.append(f"{source_name}_unsafe_count:{field}")
        for field in (
            "default_behavior_changed",
            "runtime_dispatch_ready",
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
                claims.append(f"{source_name}_unsafe_flag:{field}")
    return _dedupe(claims)


def _row_default_off_blockers(report: Mapping[str, Any], label: str) -> list[str]:
    blockers: list[str] = []
    for field in (
        "runtime_dispatch_ready",
        "native_dispatch_allowed",
        "training_path_enabled",
        "default_behavior_changed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
    ):
        if report.get(field) is True:
            blockers.append(f"{label}_unsafe:{field}")
    return blockers


def _summary_int(report: Mapping[str, Any], key: str) -> int:
    summary = _as_dict(report.get("summary"))
    try:
        return int(summary.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _bool_count(value: Any) -> int:
    return 1 if value is True else 0


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_as_dict(item) for item in value if isinstance(item, Mapping)]


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_optimizer_v2_remaining_gate_handoff_scorecard"]
