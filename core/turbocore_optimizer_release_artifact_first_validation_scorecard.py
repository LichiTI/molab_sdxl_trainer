"""V2 O5 artifact-first release validation for TurboCore optimizer evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_release_artifact_first_validation_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"
EXPECTED_RECORD_INPUT_ITEM_COUNT = 13
EXPECTED_PHASE1_RECORD_INPUT_ITEM_COUNT = 6
EXPECTED_PHASE2_RECORD_INPUT_ITEM_COUNT = 7
EXPECTED_RECORD_INPUT_SUPPORT_CHECK_COUNT = 3
EXPECTED_PHASE1_HANDOFF_POST_RETURN_COMMAND_COUNT = 3
PHASE2_REFRESH_INPUT_IDS = frozenset(
    {
        "phase2_signature_bundle",
        "phase2_reviewer_handoff",
    }
)

REQUIRED_ARTIFACTS = (
    ("native_readiness_gap", "turbocore_optimizer_native_readiness_gap_scorecard.json"),
    ("owner_release_hold_package", "turbocore_optimizer_owner_release_hold_package_scorecard.json"),
    ("adaptive_lr_chain", "turbocore_adaptive_lr_chain_scorecard.json"),
    ("product_route_binding_chain", "turbocore_optimizer_product_route_binding_chain_scorecard.json"),
    ("coverage_suite", "turbocore_optimizer_smoke_suite_coverage.json"),
    ("release_suite", "turbocore_optimizer_smoke_suite_release.json"),
    ("actual_training_coverage", "turbocore_plugin_actual_training_coverage_scorecard.json"),
    ("product_exposure_decision", "native_update_product_exposure_decision.json"),
    ("release_review_archive", "native_update_release_review_archive.json"),
    ("v2_reviewer_handoff", "turbocore_optimizer_v2_reviewer_handoff_packet.json"),
    ("v2_approval_command_audit", "turbocore_optimizer_v2_approval_command_audit.json"),
    ("v2_record_input_checklist", "turbocore_optimizer_v2_record_input_checklist.json"),
)


def build_optimizer_release_artifact_first_validation_scorecard(
    *,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    rows = [_artifact_row(directory, artifact_id, filename) for artifact_id, filename in REQUIRED_ARTIFACTS]
    checks = _cross_artifact_checks(rows)
    command_audit, _ = _read_json(directory / "turbocore_optimizer_v2_approval_command_audit.json")
    checklist, _ = _read_json(directory / "turbocore_optimizer_v2_record_input_checklist.json")
    reviewer_handoff, _ = _read_json(directory / "turbocore_optimizer_v2_reviewer_handoff_packet.json")
    ready = all(row["artifact_ready"] for row in rows) and all(check["ok"] for check in checks)
    payload = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_release_artifact_first_validation_scorecard_v0",
        "gate": "optimizer_release_artifact_first_validation",
        "roadmap": ROADMAP,
        "roadmap_section": "O5-2",
        "ok": ready,
        "artifact_first_release_validation_ready": ready,
        "promotion_ready": False,
        "report_only": True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "rows": rows,
        "checks": checks,
        "summary": {
            "release_artifact_first_required_artifact_count": len(rows),
            "release_artifact_first_ready_artifact_count": sum(1 for row in rows if row["artifact_ready"]),
            "release_artifact_first_missing_artifact_count": sum(1 for row in rows if not row["exists"]),
            "release_artifact_first_parse_error_count": sum(1 for row in rows if row["parse_error"]),
            "release_artifact_first_cross_check_count": len(checks),
            "release_artifact_first_cross_check_ready_count": sum(1 for check in checks if check["ok"]),
            "release_artifact_first_v2_record_input_item_count": _summary_int(
                checklist,
                "v2_record_input_checklist_item_count",
            ),
            "release_artifact_first_v2_record_input_phase1_item_count": _summary_int(
                checklist,
                "v2_record_input_checklist_phase1_item_count",
            ),
            "release_artifact_first_v2_record_input_phase2_item_count": _summary_int(
                checklist,
                "v2_record_input_checklist_phase2_item_count",
            ),
            "release_artifact_first_v2_phase2_refresh_inputs_tracked_count": _phase2_refresh_inputs_tracked_count(
                checklist,
            ),
            "release_artifact_first_v2_record_input_checklist_shape_ready_count": (
                1 if _record_input_checklist_shape_ready(checklist) else 0
            ),
            "release_artifact_first_v2_record_input_support_check_count": _summary_int(
                checklist,
                "v2_record_input_checklist_support_check_count",
            ),
            "release_artifact_first_v2_record_input_support_ready_count": _summary_int(
                checklist,
                "v2_record_input_checklist_support_ready_count",
            ),
            "release_artifact_first_v2_record_input_support_blocked_item_count": _summary_int(
                checklist,
                "v2_record_input_checklist_support_blocked_item_count",
            ),
            "release_artifact_first_v2_record_input_support_blocker_count": _summary_int(
                checklist,
                "v2_record_input_checklist_support_blocker_count",
            ),
            "release_artifact_first_v2_record_input_support_source_binding_ready_count": _summary_int(
                checklist,
                "v2_record_input_checklist_support_source_binding_ready_count",
            ),
            "release_artifact_first_v2_record_input_support_source_binding_blocker_count": _summary_int(
                checklist,
                "v2_record_input_checklist_support_source_binding_blocker_count",
            ),
            "release_artifact_first_v2_record_input_support_shape_ready_count": (
                1 if _record_input_support_shape_ready(checklist) else 0
            ),
            "release_artifact_first_v2_reviewer_handoff_phase_metadata_ready_count": _summary_int(
                reviewer_handoff,
                "v2_reviewer_handoff_phase_metadata_ready_count",
            ),
            "release_artifact_first_v2_reviewer_handoff_phase1_template_entry_count": _summary_int(
                reviewer_handoff,
                "v2_reviewer_handoff_phase1_template_entry_count",
            ),
            "release_artifact_first_v2_reviewer_handoff_phase2_deferred_signature_count": _summary_int(
                reviewer_handoff,
                "v2_reviewer_handoff_phase2_deferred_signature_count",
            ),
            "release_artifact_first_v2_reviewer_handoff_phase_shape_ready_count": (
                1 if _reviewer_handoff_phase_shape_ready(reviewer_handoff) else 0
            ),
            "release_artifact_first_v2_command_audit_expected_path_binding_ready_count": _summary_int(
                command_audit,
                "v2_approval_command_audit_expected_path_binding_ready_count",
            ),
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_count": _summary_int(
                command_audit,
                "v2_approval_command_audit_phase1_handoff_post_return_command_count",
            ),
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_pre_record_command_count": _summary_int(
                command_audit,
                "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count",
            ),
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_approval_record_command_count": _summary_int(
                command_audit,
                "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count",
            ),
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_match_count": _summary_int(
                command_audit,
                "v2_approval_command_audit_phase1_handoff_post_return_command_match_count",
            ),
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_mismatch_count": _summary_int(
                command_audit,
                "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count",
            ),
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_ready_count": _summary_int(
                command_audit,
                "v2_approval_command_audit_phase1_handoff_post_return_ready_count",
            ),
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_shape_ready_count": (
                1 if _command_audit_phase1_handoff_ready(command_audit) else 0
            ),
            "release_artifact_first_validation_ready_count": 1 if ready else 0,
            "release_artifact_first_runtime_dispatch_ready_count": 0,
            "release_artifact_first_native_dispatch_allowed_count": 0,
            "release_artifact_first_training_path_enabled_count": 0,
            "release_artifact_first_default_behavior_changed_count": 0,
            "release_artifact_first_product_native_ready_count": 0,
        },
        "blocked_reasons": _dedupe(
            [reason for row in rows for reason in row["blocked_reasons"]]
            + [reason for check in checks for reason in check["blocked_reasons"]]
        ),
        "promotion_blockers": [
            "owner_release_approval_missing",
            "product_training_route_not_bound",
            "product_exposure_gate_open",
        ],
        "recommended_next_step": "continue owner/release approval and product exposure gates while keeping default product dispatch off",
        "notes": [
            "This validation reads existing artifacts only.",
            "It does not rebuild evidence, record approval, bind routes, or launch training.",
            "Default product counters must remain zero.",
        ],
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _artifact_row(directory: Path, artifact_id: str, filename: str) -> dict[str, Any]:
    path = directory / filename
    payload, parse_error = _read_json(path)
    validators = _validators_for(artifact_id, payload)
    blocked = [] if path.exists() else [f"{artifact_id}_artifact_missing"]
    if parse_error:
        blocked.append(f"{artifact_id}_artifact_parse_error")
    blocked.extend(reason for ok, reason in validators if not ok)
    return {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "filename": filename,
        "path": str(path),
        "exists": path.exists(),
        "parse_error": parse_error,
        "artifact_ready": path.exists() and not parse_error and not blocked,
        "artifact_ok": payload.get("ok") is True,
        "roadmap": str(payload.get("roadmap") or ""),
        "gate": str(payload.get("gate") or ""),
        "profile": str(payload.get("profile") or ""),
        "runtime_dispatch_ready": payload.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": payload.get("native_dispatch_allowed") is True,
        "training_path_enabled": payload.get("training_path_enabled") is True,
        "default_behavior_changed": payload.get("default_behavior_changed") is True,
        "product_native_ready": payload.get("product_native_ready") is True,
        "blocked_reasons": _dedupe(blocked),
    }


def _validators_for(artifact_id: str, payload: Mapping[str, Any]) -> list[tuple[bool, str]]:
    summary = _as_dict(payload.get("summary"))
    status_summary = _as_dict(payload.get("status_summary"))
    if artifact_id == "native_readiness_gap":
        return [
            (payload.get("ok") is True, "native_readiness_gap_not_ok"),
            (payload.get("artifact_first") is True, "native_readiness_gap_not_artifact_first"),
            (payload.get("roadmap") == ROADMAP, "native_readiness_gap_wrong_roadmap"),
            (summary.get("default_off_product_runtime_dispatch_ready_optimizer_count") == 0, "default_off_runtime_dispatch_not_zero"),
            (summary.get("signed_post_approval_preview_runtime_dispatch_ready_optimizer_count") == 124, "signed_preview_runtime_dispatch_not_124"),
            (summary.get("roadmap_v2_open_work_open_item_count") == 6, "roadmap_v2_open_item_count_not_6"),
        ]
    if artifact_id == "owner_release_hold_package":
        return [
            (payload.get("ok") is True, "owner_release_hold_package_not_ok"),
            (_as_dict(payload.get("summary")).get("owner_release_hold_package_ready_family_count") == 5, "owner_release_hold_package_ready_family_count_not_5"),
            (_as_dict(payload.get("summary")).get("owner_release_hold_package_training_path_enabled_count") == 0, "owner_release_hold_package_training_path_enabled_not_zero"),
        ]
    if artifact_id == "adaptive_lr_chain":
        return [
            (payload.get("ok") is True, "adaptive_lr_chain_not_ok"),
            (summary.get("adaptive_lr_chain_ready_stage_count") == 8, "adaptive_lr_chain_ready_stage_count_not_8"),
            (summary.get("adaptive_lr_chain_open_stage_count") == 1, "adaptive_lr_chain_open_stage_count_not_1"),
            (summary.get("adaptive_lr_chain_training_path_enabled_count") == 0, "adaptive_lr_chain_training_path_enabled_not_zero"),
        ]
    if artifact_id == "product_route_binding_chain":
        return [
            (payload.get("ok") is True, "product_route_binding_chain_not_ok"),
            (summary.get("product_route_binding_chain_ready_stage_count") == 7, "product_route_binding_chain_ready_stage_count_not_7"),
            (summary.get("product_route_binding_chain_product_training_route_bound_count") == 0, "product_route_binding_bound_not_zero"),
            (summary.get("product_route_binding_chain_training_path_enabled_count") == 0, "product_route_binding_training_path_not_zero"),
        ]
    if artifact_id == "coverage_suite":
        return [
            (payload.get("ok") is True, "coverage_suite_not_ok"),
            (payload.get("profile") == "coverage", "coverage_suite_wrong_profile"),
            ("quick/coverage/release are artifact-first by default." in payload.get("artifact_policy", []), "coverage_suite_artifact_policy_missing"),
        ]
    if artifact_id == "release_suite":
        return [
            (payload.get("ok") is True, "release_suite_not_ok"),
            (payload.get("profile") == "release", "release_suite_wrong_profile"),
            ("quick/coverage/release are artifact-first by default." in payload.get("artifact_policy", []), "release_suite_artifact_policy_missing"),
            (status_summary.get("product_route_binding_chain_training_path_enabled_count") == 0, "release_suite_route_binding_training_path_not_zero"),
        ]
    if artifact_id == "actual_training_coverage":
        return [
            (payload.get("ok") is True, "actual_training_coverage_not_ok"),
            (payload.get("actual_training_complete") is True, "actual_training_coverage_not_complete"),
            (payload.get("roadmap") == ROADMAP, "actual_training_coverage_wrong_roadmap"),
            (summary.get("selected_plugin_optimizer_count") == 124, "actual_training_selected_optimizer_count_not_124"),
            (summary.get("trainer_resume_parity_proven_count") == 124, "actual_training_resume_parity_not_124"),
            (summary.get("per_optimizer_native_training_count") == 124, "actual_training_native_training_not_124"),
            (summary.get("actual_training_gap_count") == 0, "actual_training_gap_not_zero"),
            (summary.get("training_path_enabled_count") == 0, "actual_training_training_path_enabled_not_zero"),
            (summary.get("native_dispatch_allowed_count") == 0, "actual_training_native_dispatch_allowed_not_zero"),
            (summary.get("product_native_ready_count") == 0, "actual_training_product_native_ready_not_zero"),
        ]
    if artifact_id == "product_exposure_decision":
        return [
            (payload.get("ok") is True, "product_exposure_decision_not_ok"),
            (payload.get("ready_for_product_exposure_review") is True, "product_exposure_not_ready_for_review"),
            (payload.get("product_exposure_decision_recorded") is False, "product_exposure_decision_unexpectedly_recorded"),
            (payload.get("training_path_enabled") is False, "product_exposure_training_path_not_false"),
        ]
    if artifact_id == "release_review_archive":
        return [
            (payload.get("evidence_ready") is True, "release_review_archive_evidence_not_ready"),
            (payload.get("ready_for_review") is True, "release_review_archive_not_ready_for_review"),
            (payload.get("archive_ready") is False, "release_review_archive_unexpectedly_ready"),
            (payload.get("training_path_enabled") is False, "release_review_archive_training_path_not_false"),
        ]
    if artifact_id == "v2_approval_command_audit":
        return [
            (payload.get("ok") is True, "v2_approval_command_audit_not_ok"),
            (payload.get("approval_command_audit_ready") is True, "v2_approval_command_audit_not_ready"),
            (
                summary.get("v2_approval_command_audit_expected_path_binding_ready_count") == 1,
                "v2_approval_command_audit_path_binding_not_ready",
            ),
            (
                summary.get("v2_approval_command_audit_preflight_artifact_write_count") == 2,
                "v2_approval_command_audit_preflight_artifact_write_not_2",
            ),
            (
                summary.get("v2_approval_command_audit_record_command_preflight_arg_count") == 3,
                "v2_approval_command_audit_record_preflight_arg_not_3",
            ),
            (
                _command_audit_phase1_handoff_ready(payload),
                "v2_approval_command_audit_phase1_handoff_post_return_not_ready",
            ),
            (
                summary.get("v2_approval_command_audit_approval_recorded_count") == 0,
                "v2_approval_command_audit_approval_recorded_not_zero",
            ),
            (
                summary.get("v2_approval_command_audit_training_path_enabled_count") == 0,
                "v2_approval_command_audit_training_path_not_zero",
            ),
            (
                summary.get("v2_approval_command_audit_native_dispatch_allowed_count") == 0,
                "v2_approval_command_audit_native_dispatch_not_zero",
            ),
        ]
    if artifact_id == "v2_reviewer_handoff":
        return [
            (payload.get("ok") is True, "v2_reviewer_handoff_not_ok"),
            (payload.get("reviewer_handoff_ready") is True, "v2_reviewer_handoff_not_ready"),
            (_reviewer_handoff_phase_shape_ready(payload), "v2_reviewer_handoff_phase_shape_not_ready"),
            (
                summary.get("v2_reviewer_handoff_approval_recorded_count") == 0,
                "v2_reviewer_handoff_approval_recorded_not_zero",
            ),
            (
                summary.get("v2_reviewer_handoff_training_path_enabled_count") == 0,
                "v2_reviewer_handoff_training_path_not_zero",
            ),
            (
                summary.get("v2_reviewer_handoff_native_dispatch_allowed_count") == 0,
                "v2_reviewer_handoff_native_dispatch_not_zero",
            ),
        ]
    if artifact_id == "v2_record_input_checklist":
        return [
            (payload.get("ok") is True, "v2_record_input_checklist_not_ok"),
            (payload.get("record_input_checklist_artifact_ready") is True, "v2_record_input_checklist_not_ready"),
            (
                summary.get("v2_record_input_checklist_item_count") == EXPECTED_RECORD_INPUT_ITEM_COUNT,
                "v2_record_input_checklist_item_count_not_13",
            ),
            (
                summary.get("v2_record_input_checklist_phase1_item_count") == EXPECTED_PHASE1_RECORD_INPUT_ITEM_COUNT,
                "v2_record_input_checklist_phase1_item_count_not_6",
            ),
            (
                summary.get("v2_record_input_checklist_phase2_item_count") == EXPECTED_PHASE2_RECORD_INPUT_ITEM_COUNT,
                "v2_record_input_checklist_phase2_item_count_not_7",
            ),
            (_phase2_refresh_inputs_tracked_count(payload) == 2, "v2_record_input_checklist_phase2_refresh_inputs_missing"),
            (_record_input_support_shape_ready(payload), "v2_record_input_checklist_support_shape_not_ready"),
            (summary.get("v2_record_input_checklist_artifact_ready_count") == 1, "v2_record_input_checklist_artifact_ready_not_1"),
            (summary.get("v2_record_input_checklist_full_ready_count") == 0, "v2_record_input_checklist_unexpectedly_full_ready"),
            (summary.get("v2_record_input_checklist_training_path_enabled_count") == 0, "v2_record_input_checklist_training_path_not_zero"),
            (summary.get("v2_record_input_checklist_native_dispatch_allowed_count") == 0, "v2_record_input_checklist_native_dispatch_not_zero"),
        ]
    return [(False, f"{artifact_id}_validator_missing")]


def _record_input_checklist_shape_ready(checklist: Mapping[str, Any]) -> bool:
    return (
        _summary_int(checklist, "v2_record_input_checklist_item_count") == EXPECTED_RECORD_INPUT_ITEM_COUNT
        and _summary_int(checklist, "v2_record_input_checklist_phase1_item_count")
        == EXPECTED_PHASE1_RECORD_INPUT_ITEM_COUNT
        and _summary_int(checklist, "v2_record_input_checklist_phase2_item_count")
        == EXPECTED_PHASE2_RECORD_INPUT_ITEM_COUNT
        and _phase2_refresh_inputs_tracked_count(checklist) == len(PHASE2_REFRESH_INPUT_IDS)
    )


def _record_input_support_shape_ready(checklist: Mapping[str, Any]) -> bool:
    return (
        _summary_int(checklist, "v2_record_input_checklist_support_check_count")
        >= EXPECTED_RECORD_INPUT_SUPPORT_CHECK_COUNT
        and _summary_int(checklist, "v2_record_input_checklist_support_ready_count") >= 2
        and _summary_int(checklist, "v2_record_input_checklist_support_source_binding_ready_count") >= 2
        and _summary_int(checklist, "v2_record_input_checklist_support_source_binding_blocker_count") <= 1
    )


def _command_audit_phase1_handoff_ready(command_audit: Mapping[str, Any]) -> bool:
    return (
        _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_command_count")
        == EXPECTED_PHASE1_HANDOFF_POST_RETURN_COMMAND_COUNT
        and _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count")
        == EXPECTED_PHASE1_HANDOFF_POST_RETURN_COMMAND_COUNT
        and _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count")
        == 0
        and _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_command_match_count")
        == EXPECTED_PHASE1_HANDOFF_POST_RETURN_COMMAND_COUNT
        and _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count")
        == 0
        and _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_ready_count") == 1
    )


def _reviewer_handoff_phase_shape_ready(reviewer_handoff: Mapping[str, Any]) -> bool:
    return (
        _summary_int(reviewer_handoff, "v2_reviewer_handoff_phase_metadata_ready_count") == 1
        and _summary_int(reviewer_handoff, "v2_reviewer_handoff_phase1_template_entry_count") == 2
        and _summary_int(reviewer_handoff, "v2_reviewer_handoff_phase2_deferred_signature_count") == 1
        and _summary_int(reviewer_handoff, "v2_reviewer_handoff_phase1_signature_count") == 2
        and _summary_int(reviewer_handoff, "v2_reviewer_handoff_phase2_blocked_signature_count") == 1
        and _summary_int(reviewer_handoff, "v2_reviewer_handoff_phase1_required_manual_field_count") == 18
        and _summary_int(reviewer_handoff, "v2_reviewer_handoff_phase1_post_return_command_count") == 3
        and _summary_int(reviewer_handoff, "v2_reviewer_handoff_phase1_post_return_pre_record_command_count") == 3
        and _summary_int(reviewer_handoff, "v2_reviewer_handoff_phase1_post_return_approval_record_command_count")
        == 0
    )


def _phase2_refresh_inputs_tracked_count(checklist: Mapping[str, Any]) -> int:
    items = checklist.get("items")
    if isinstance(items, list):
        seen = {
            str(item.get("id"))
            for item in items
            if isinstance(item, Mapping) and str(item.get("phase")) == "phase2_direction"
        }
        return len(PHASE2_REFRESH_INPUT_IDS.intersection(seen))
    if (
        _summary_int(checklist, "v2_record_input_checklist_item_count") == EXPECTED_RECORD_INPUT_ITEM_COUNT
        and _summary_int(checklist, "v2_record_input_checklist_phase2_item_count")
        == EXPECTED_PHASE2_RECORD_INPUT_ITEM_COUNT
    ):
        return len(PHASE2_REFRESH_INPUT_IDS)
    return 0


def _cross_artifact_checks(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["artifact_id"]: row for row in rows}
    return [
        _check(
            "all_required_artifacts_ready",
            all(row.get("artifact_ready") is True for row in rows),
            "release_artifact_first_required_artifact_not_ready",
        ),
        _check(
            "default_product_dispatch_stays_zero",
            all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows)
            and all(row.get("training_path_enabled") is False for row in rows)
            and all(row.get("default_behavior_changed") is False for row in rows)
            and all(row.get("product_native_ready") is False for row in rows),
            "release_artifact_first_default_product_counter_not_zero",
        ),
        _check(
            "v2_core_artifacts_present",
            all(
                by_id.get(artifact_id, {}).get("artifact_ready") is True
                for artifact_id in (
                    "native_readiness_gap",
                    "owner_release_hold_package",
                    "adaptive_lr_chain",
                    "product_route_binding_chain",
                    "actual_training_coverage",
                    "v2_reviewer_handoff",
                    "v2_approval_command_audit",
                    "v2_record_input_checklist",
                )
            ),
            "release_artifact_first_v2_core_artifact_missing",
        ),
    ]


def _check(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "check": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _read_json(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, True
    return (dict(payload), False) if isinstance(payload, Mapping) else ({}, True)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _summary_int(report: Mapping[str, Any], key: str) -> int:
    summary = _as_dict(report.get("summary"))
    try:
        return int(summary.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["build_optimizer_release_artifact_first_validation_scorecard"]
