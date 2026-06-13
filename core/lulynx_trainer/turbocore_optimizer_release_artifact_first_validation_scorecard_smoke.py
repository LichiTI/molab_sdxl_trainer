"""Smoke checks for the v2 O5 artifact-first release validation."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_release_artifact_first_validation_scorecard import (  # noqa: E402
    build_optimizer_release_artifact_first_validation_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_optimizer_release_artifact_first_validation_scorecard(write_artifact=True)
    summary = report["summary"]
    artifact_path = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_optimizer_release_artifact_first_validation_scorecard.json"
    )

    assert report["scorecard"] == "turbocore_optimizer_release_artifact_first_validation_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["artifact_first_release_validation_ready"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["product_native_ready"] is False, report
    assert summary["release_artifact_first_required_artifact_count"] == 12, report
    assert summary["release_artifact_first_ready_artifact_count"] == 12, report
    assert summary["release_artifact_first_missing_artifact_count"] == 0, report
    assert summary["release_artifact_first_parse_error_count"] == 0, report
    assert summary["release_artifact_first_cross_check_count"] == 3, report
    assert summary["release_artifact_first_cross_check_ready_count"] == 3, report
    assert summary["release_artifact_first_v2_record_input_item_count"] == 13, report
    assert summary["release_artifact_first_v2_record_input_phase1_item_count"] == 6, report
    assert summary["release_artifact_first_v2_record_input_phase2_item_count"] == 7, report
    assert summary["release_artifact_first_v2_phase2_refresh_inputs_tracked_count"] == 2, report
    assert summary["release_artifact_first_v2_record_input_checklist_shape_ready_count"] == 1, report
    assert summary["release_artifact_first_v2_record_input_support_check_count"] == 3, report
    assert summary["release_artifact_first_v2_record_input_support_ready_count"] == 2, report
    assert summary["release_artifact_first_v2_record_input_support_blocked_item_count"] == 1, report
    assert summary["release_artifact_first_v2_record_input_support_blocker_count"] == 1, report
    assert summary["release_artifact_first_v2_record_input_support_source_binding_ready_count"] == 2, report
    assert summary["release_artifact_first_v2_record_input_support_source_binding_blocker_count"] == 0, report
    assert summary["release_artifact_first_v2_record_input_support_shape_ready_count"] == 1, report
    assert summary["release_artifact_first_v2_reviewer_handoff_phase_metadata_ready_count"] == 1, report
    assert summary["release_artifact_first_v2_reviewer_handoff_phase1_template_entry_count"] == 2, report
    assert summary["release_artifact_first_v2_reviewer_handoff_phase2_deferred_signature_count"] == 1, report
    assert summary["release_artifact_first_v2_reviewer_handoff_phase_shape_ready_count"] == 1, report
    assert summary["release_artifact_first_v2_command_audit_expected_path_binding_ready_count"] == 1, report
    assert summary["release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_count"] == 3, report
    assert (
        summary["release_artifact_first_v2_command_audit_phase1_handoff_post_return_pre_record_command_count"] == 3
    ), report
    assert (
        summary["release_artifact_first_v2_command_audit_phase1_handoff_post_return_approval_record_command_count"]
        == 0
    ), report
    assert summary["release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_match_count"] == 3, report
    assert (
        summary["release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_mismatch_count"] == 0
    ), report
    assert summary["release_artifact_first_v2_command_audit_phase1_handoff_post_return_ready_count"] == 1, report
    assert summary["release_artifact_first_v2_command_audit_phase1_handoff_post_return_shape_ready_count"] == 1, report
    assert summary["release_artifact_first_validation_ready_count"] == 1, report
    assert summary["release_artifact_first_runtime_dispatch_ready_count"] == 0, report
    assert summary["release_artifact_first_native_dispatch_allowed_count"] == 0, report
    assert summary["release_artifact_first_training_path_enabled_count"] == 0, report
    assert summary["release_artifact_first_default_behavior_changed_count"] == 0, report
    assert summary["release_artifact_first_product_native_ready_count"] == 0, report
    assert report["blocked_reasons"] == [], report
    assert "product_exposure_gate_open" in report["promotion_blockers"], report
    assert artifact_path.exists(), artifact_path

    with tempfile.TemporaryDirectory(prefix="lulynx_turbocore_artifact_first_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
        for source in source_dir.glob("*.json"):
            shutil.copy2(source, temp_dir / source.name)
        checklist_path = temp_dir / "turbocore_optimizer_v2_record_input_checklist.json"
        stale_checklist = json.loads(checklist_path.read_text(encoding="utf-8"))
        stale_summary = dict(stale_checklist.get("summary") or {})
        for key in (
            "v2_record_input_checklist_support_check_count",
            "v2_record_input_checklist_support_ready_count",
            "v2_record_input_checklist_support_blocked_item_count",
            "v2_record_input_checklist_support_blocker_count",
            "v2_record_input_checklist_support_source_binding_ready_count",
            "v2_record_input_checklist_support_source_binding_blocker_count",
        ):
            stale_summary.pop(key, None)
        stale_checklist["summary"] = stale_summary
        checklist_path.write_text(
            json.dumps(stale_checklist, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stale_report = build_optimizer_release_artifact_first_validation_scorecard(
            artifact_dir=temp_dir,
            write_artifact=False,
        )
    assert stale_report["ok"] is False, stale_report
    assert stale_report["summary"]["release_artifact_first_v2_record_input_checklist_shape_ready_count"] == 1, stale_report
    assert stale_report["summary"]["release_artifact_first_v2_record_input_support_shape_ready_count"] == 0, stale_report
    assert "v2_record_input_checklist_support_shape_not_ready" in stale_report["blocked_reasons"], stale_report

    with tempfile.TemporaryDirectory(prefix="lulynx_turbocore_artifact_first_reviewer_handoff_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
        for source in source_dir.glob("*.json"):
            shutil.copy2(source, temp_dir / source.name)
        handoff_path = temp_dir / "turbocore_optimizer_v2_reviewer_handoff_packet.json"
        stale_handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        stale_handoff_summary = dict(stale_handoff.get("summary") or {})
        for key in (
            "v2_reviewer_handoff_phase_metadata_ready_count",
            "v2_reviewer_handoff_phase1_template_entry_count",
            "v2_reviewer_handoff_phase2_deferred_signature_count",
        ):
            stale_handoff_summary.pop(key, None)
        stale_handoff["summary"] = stale_handoff_summary
        handoff_path.write_text(
            json.dumps(stale_handoff, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stale_handoff_report = build_optimizer_release_artifact_first_validation_scorecard(
            artifact_dir=temp_dir,
            write_artifact=False,
        )
    assert stale_handoff_report["ok"] is False, stale_handoff_report
    assert (
        stale_handoff_report["summary"]["release_artifact_first_v2_reviewer_handoff_phase_shape_ready_count"] == 0
    ), stale_handoff_report
    assert "v2_reviewer_handoff_phase_shape_not_ready" in stale_handoff_report["blocked_reasons"], stale_handoff_report

    with tempfile.TemporaryDirectory(prefix="lulynx_turbocore_artifact_first_command_audit_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
        for source in source_dir.glob("*.json"):
            shutil.copy2(source, temp_dir / source.name)
        command_audit_path = temp_dir / "turbocore_optimizer_v2_approval_command_audit.json"
        stale_command_audit = json.loads(command_audit_path.read_text(encoding="utf-8"))
        stale_command_summary = dict(stale_command_audit.get("summary") or {})
        for key in (
            "v2_approval_command_audit_phase1_handoff_post_return_command_count",
            "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count",
            "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count",
            "v2_approval_command_audit_phase1_handoff_post_return_command_match_count",
            "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count",
            "v2_approval_command_audit_phase1_handoff_post_return_ready_count",
        ):
            stale_command_summary.pop(key, None)
        stale_command_audit["summary"] = stale_command_summary
        command_audit_path.write_text(
            json.dumps(stale_command_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stale_command_report = build_optimizer_release_artifact_first_validation_scorecard(
            artifact_dir=temp_dir,
            write_artifact=False,
        )
    assert stale_command_report["ok"] is False, stale_command_report
    assert (
        stale_command_report["summary"][
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_shape_ready_count"
        ]
        == 0
    ), stale_command_report
    assert (
        "v2_approval_command_audit_phase1_handoff_post_return_not_ready"
        in stale_command_report["blocked_reasons"]
    ), stale_command_report

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_release_artifact_first_validation_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "stale_record_input_support_shape_guard_checked": True,
        "stale_reviewer_handoff_phase_shape_guard_checked": True,
        "stale_command_audit_handoff_guard_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
