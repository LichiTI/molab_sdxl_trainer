"""Record-input checklist for TurboCore optimizer v2 approvals.

The checklist is a read-only release handoff artifact.  It lists the concrete
files that must exist before phase1 owner/product records and phase2 owner
direction records are executed.  It never writes approval records and never
enables native optimizer dispatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_v2_support_artifact_guard import (  # noqa: E402
    support_artifact_blockers as _support_artifact_blockers,
    support_artifact_source_binding_blockers as _support_artifact_source_binding_blockers,
)


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_record_input_checklist.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def _summary_at_least(key: str, minimum: int) -> Callable[[Mapping[str, Any]], bool]:
    return lambda report: _summary_int(report, key) >= minimum


def _field_true(key: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda report: report.get(key) is True


def _json_ready(report: Mapping[str, Any]) -> bool:
    return bool(report)


def _support_ready(check_id: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda report: not _support_artifact_blockers(check_id, report)


CHECKLIST_INPUTS: tuple[dict[str, Any], ...] = (
    {
        "id": "signature_bundle",
        "phase": "phase1_records",
        "path": "turbocore_optimizer_v2_signature_bundle_packet.json",
        "ready": _summary_at_least("v2_signature_bundle_ready_for_signature_count", 2),
        "blocker": "signature_bundle_not_ready_for_phase1",
    },
    {
        "id": "signed_bundle",
        "phase": "phase1_records",
        "path": "turbocore_optimizer_v2_signed_bundle.reviewed.json",
        "ready": _json_ready,
        "blocker": "signed_bundle_missing",
    },
    {
        "id": "signed_owner_release_review",
        "phase": "phase1_records",
        "path": "signed_owner_release_review.json",
        "ready": _json_ready,
        "blocker": "signed_owner_release_review_missing",
    },
    {
        "id": "signed_product_exposure_review",
        "phase": "phase1_records",
        "path": "signed_product_exposure_review.json",
        "ready": _json_ready,
        "blocker": "signed_product_exposure_review_missing",
    },
    {
        "id": "training_launch_contract",
        "phase": "phase1_records",
        "path": "native_update_training_launch_contract.json",
        "ready": _support_ready("training_launch_contract"),
        "blocker": "training_launch_contract_not_ready",
        "support_check": "training_launch_contract",
    },
    {
        "id": "product_exposure_evidence",
        "phase": "phase1_records",
        "path": "native_update_product_exposure_evidence.json",
        "ready": _support_ready("product_exposure_evidence"),
        "blocker": "product_exposure_evidence_not_ready",
        "support_check": "product_exposure_evidence",
    },
    {
        "id": "owner_release_review_record",
        "phase": "phase2_direction",
        "path": "native_update_owner_release_review_record.json",
        "ready": _field_true("release_review_recorded"),
        "blocker": "owner_release_review_not_recorded",
    },
    {
        "id": "product_exposure_decision",
        "phase": "phase2_direction",
        "path": "native_update_product_exposure_decision.json",
        "ready": _field_true("product_exposure_decision_recorded"),
        "blocker": "product_exposure_decision_not_recorded",
    },
    {
        "id": "release_review_archive",
        "phase": "phase2_direction",
        "path": "native_update_release_review_archive.json",
        "ready": _field_true("archive_ready"),
        "blocker": "release_review_archive_not_ready",
    },
    {
        "id": "owner_release_direction_packet",
        "phase": "phase2_direction",
        "path": "native_update_owner_release_direction_packet.json",
        "ready": _support_ready("owner_release_direction_packet"),
        "blocker": "owner_release_direction_packet_not_ready",
        "support_check": "owner_release_direction_packet",
    },
    {
        "id": "phase2_signature_bundle",
        "phase": "phase2_direction",
        "path": "turbocore_optimizer_v2_signature_bundle_packet.json",
        "ready": _summary_at_least("v2_signature_bundle_owner_direction_ready_count", 1),
        "blocker": "phase2_signature_bundle_owner_direction_not_ready",
    },
    {
        "id": "phase2_reviewer_handoff",
        "phase": "phase2_direction",
        "path": "turbocore_optimizer_v2_reviewer_handoff_packet.json",
        "ready": _summary_at_least("v2_reviewer_handoff_signed_template_entry_count", 3),
        "blocker": "phase2_reviewer_handoff_owner_direction_template_missing",
    },
    {
        "id": "signed_owner_release_direction",
        "phase": "phase2_direction",
        "path": "signed_owner_release_direction.json",
        "ready": _json_ready,
        "blocker": "signed_owner_release_direction_missing",
    },
)


def build_optimizer_v2_record_input_checklist(
    *,
    source_reports: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    items = [_check_item(spec, directory, source_reports or {}) for spec in CHECKLIST_INPUTS]
    unsafe = _unsafe_claims(items)
    support_items = [item for item in items if item.get("support_check")]
    phase1 = [item for item in items if item["phase"] == "phase1_records"]
    phase2 = [item for item in items if item["phase"] == "phase2_direction"]
    phase1_ready = all(item["ready"] for item in phase1)
    phase2_ready = all(item["ready"] for item in phase2)
    missing_item_ids_by_phase = _missing_item_ids_by_phase(items)
    blockers = _dedupe([reason for item in items for reason in _strings(item.get("blocked_reasons"))] + unsafe)
    payload = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_record_input_checklist_v0",
        "gate": "optimizer_v2_record_input_checklist",
        "roadmap": ROADMAP,
        "ok": not unsafe,
        "record_input_checklist_artifact_ready": not unsafe,
        "record_input_checklist_full_ready": phase1_ready and phase2_ready and not unsafe,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "roadmap_complete": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "missing_item_ids_by_phase": missing_item_ids_by_phase,
        "items": items,
        "summary": {
            "v2_record_input_checklist_item_count": len(items),
            "v2_record_input_checklist_present_item_count": sum(1 for item in items if item["present"]),
            "v2_record_input_checklist_valid_json_item_count": sum(1 for item in items if item["valid_json"]),
            "v2_record_input_checklist_missing_item_count": sum(1 for item in items if not item["present"]),
            "v2_record_input_checklist_ready_item_count": sum(1 for item in items if item["ready"]),
            "v2_record_input_checklist_phase1_item_count": len(phase1),
            "v2_record_input_checklist_phase1_ready_item_count": sum(1 for item in phase1 if item["ready"]),
            "v2_record_input_checklist_phase1_missing_item_count": len(
                missing_item_ids_by_phase["phase1_records"]
            ),
            "v2_record_input_checklist_phase1_ready_count": 1 if phase1_ready else 0,
            "v2_record_input_checklist_phase2_item_count": len(phase2),
            "v2_record_input_checklist_phase2_ready_item_count": sum(1 for item in phase2 if item["ready"]),
            "v2_record_input_checklist_phase2_missing_item_count": len(
                missing_item_ids_by_phase["phase2_direction"]
            ),
            "v2_record_input_checklist_phase2_ready_count": 1 if phase2_ready else 0,
            "v2_record_input_checklist_full_ready_count": 1 if phase1_ready and phase2_ready else 0,
            "v2_record_input_checklist_support_check_count": len(support_items),
            "v2_record_input_checklist_support_ready_count": sum(
                1 for item in support_items if item["support_ready"]
            ),
            "v2_record_input_checklist_support_blocked_item_count": sum(
                1 for item in support_items if item["support_blockers"]
            ),
            "v2_record_input_checklist_support_blocker_count": sum(
                len(_strings(item.get("support_blockers"))) for item in support_items
            ),
            "v2_record_input_checklist_support_source_binding_ready_count": sum(
                1 for item in support_items if item["support_source_binding_ready"]
            ),
            "v2_record_input_checklist_support_source_binding_blocker_count": sum(
                len(_strings(item.get("support_source_binding_blockers"))) for item in support_items
            ),
            "v2_record_input_checklist_artifact_ready_count": 1 if not unsafe else 0,
            "v2_record_input_checklist_approval_recorded_count": 0,
            "v2_record_input_checklist_runtime_dispatch_ready_count": 0,
            "v2_record_input_checklist_native_dispatch_allowed_count": 0,
            "v2_record_input_checklist_training_path_enabled_count": 0,
            "v2_record_input_checklist_product_native_ready_count": 0,
            "v2_record_input_checklist_default_behavior_changed_count": 0,
            "v2_record_input_checklist_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": blockers,
        "promotion_blockers": _dedupe(
            [
                "record_input_checklist_is_not_approval_record",
                "real_owner_release_review_signature_missing",
                "real_product_exposure_review_signature_missing",
                "real_owner_release_direction_signature_missing",
            ]
            + blockers
        ),
        "recommended_next_step": _recommended_next_step(phase1_ready, phase2_ready),
        "notes": [
            "This checklist inventories real record-validator inputs only.",
            "Phase1 requires current signature bundle, returned signed bundle, extracted owner/product reviews, and default-off support artifacts.",
            "Phase2 requires recorded phase1 decisions, rebuilt archive/owner-direction packet, refreshed signature bundle/reviewer handoff, and a real signed owner direction.",
            "Native optimizer dispatch remains default-off.",
        ],
    }
    if write_artifact:
        output = directory / ARTIFACT.name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _missing_item_ids_by_phase(items: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {
        phase: [
            str(item.get("id") or "")
            for item in items
            if item.get("phase") == phase and item.get("present") is not True
        ]
        for phase in ("phase1_records", "phase2_direction")
    }


def _check_item(
    spec: Mapping[str, Any],
    directory: Path,
    source_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    item_id = str(spec["id"])
    path = directory / str(spec["path"])
    report = _as_dict(source_reports.get(item_id)) if source_reports else _read_json(path)
    present = bool(report) if source_reports else path.exists()
    valid_json = bool(report) if present else False
    support_blockers = (
        _support_artifact_blockers(str(spec["support_check"]), report)
        if valid_json and spec.get("support_check")
        else []
    )
    support_source_binding_blockers = (
        _support_artifact_source_binding_blockers(str(spec["support_check"]), report)
        if valid_json and spec.get("support_check")
        else []
    )
    support_check = str(spec.get("support_check") or "")
    ready = bool(present and valid_json and spec["ready"](report) and not support_blockers)
    unsafe = _unsafe_claims_for_report(item_id, report)
    blockers = ([] if ready else [str(spec["blocker"])]) + support_blockers + unsafe
    return {
        "schema_version": 1,
        "id": item_id,
        "phase": str(spec["phase"]),
        "path": str(path),
        "present": present,
        "valid_json": valid_json,
        "ready": ready,
        "support_check": support_check,
        "support_ready": bool(support_check and present and valid_json and not support_blockers),
        "support_blockers": _dedupe(support_blockers),
        "support_source_binding_ready": bool(
            support_check and present and valid_json and not support_blockers and not support_source_binding_blockers
        ),
        "support_source_binding_blockers": _dedupe(support_source_binding_blockers),
        "approval_recorded": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "blocked_reasons": _dedupe(blockers),
    }


def _recommended_next_step(phase1_ready: bool, phase2_ready: bool) -> str:
    if not phase1_ready:
        return "collect real phase1 signed bundle, extract owner/product reviews, then run approval execution preflight"
    if not phase2_ready:
        return "record phase1 decisions, rebuild release archive, owner direction packet, signature bundle, and reviewer handoff, then collect phase2 signed direction"
    return "run approval execution preflight immediately before the explicit record validators"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _as_dict(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return {}


def _unsafe_claims(items: list[Mapping[str, Any]]) -> list[str]:
    return _dedupe([reason for item in items for reason in _strings(item.get("blocked_reasons")) if "_unsafe:" in reason])


def _unsafe_claims_for_report(name: str, report: Mapping[str, Any]) -> list[str]:
    claims: list[str] = []
    for field in (
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
        "launcher_exposure_allowed",
        "job_created",
    ):
        if report.get(field) is True:
            claims.append(f"{name}_unsafe:{field}")
    return _dedupe(claims)


def _summary_int(report: Mapping[str, Any], key: str) -> int:
    summary = _as_dict(report.get("summary"))
    try:
        return int(summary.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


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
    parser.add_argument("--artifact-dir", default="", help="Directory containing v2 approval-chain artifacts.")
    parser.add_argument("--no-artifact", action="store_true", help="Print checklist without writing artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_record_input_checklist(
        artifact_dir=args.artifact_dir or None,
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_optimizer_v2_record_input_checklist"]


if __name__ == "__main__":
    raise SystemExit(main())
