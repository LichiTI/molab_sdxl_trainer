"""Smoke checks for the v2 approval command-chain audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_v2_approval_command_audit import (  # noqa: E402
    ARTIFACT,
    build_optimizer_v2_approval_command_audit,
)
from core.turbocore_optimizer_v2_approval_execution_plan import (  # noqa: E402
    build_optimizer_v2_approval_execution_plan,
)
from core.turbocore_optimizer_v2_reviewer_handoff_packet import (  # noqa: E402
    build_optimizer_v2_reviewer_handoff_packet,
)


def run_smoke() -> dict[str, Any]:
    handoff = build_optimizer_v2_reviewer_handoff_packet(write_artifact=True)
    plan = build_optimizer_v2_approval_execution_plan(reviewer_handoff=handoff, write_artifact=True)
    audit = build_optimizer_v2_approval_command_audit(approval_execution_plan=plan, write_artifact=True)
    summary = audit["summary"]

    assert audit["ok"] is True, audit
    assert audit["approval_command_audit_ready"] is True, audit
    assert audit["approval_recorded"] is False, audit
    assert audit["approval_artifact_written"] is False, audit
    assert audit["runtime_dispatch_ready"] is False, audit
    assert audit["native_dispatch_allowed"] is False, audit
    assert audit["training_path_enabled"] is False, audit
    assert audit["product_native_ready"] is False, audit
    assert summary["v2_approval_command_audit_step_count"] == 16, audit
    assert summary["v2_approval_command_audit_command_step_count"] == 14, audit
    assert summary["v2_approval_command_audit_entrypoint_exists_count"] == 14, audit
    assert summary["v2_approval_command_audit_missing_entrypoint_count"] == 0, audit
    assert summary["v2_approval_command_audit_order_valid_count"] == 1, audit
    assert summary["v2_approval_command_audit_record_after_preflight_count"] == 3, audit
    assert summary["v2_approval_command_audit_record_before_preflight_count"] == 0, audit
    assert summary["v2_approval_command_audit_record_after_real_signature_count"] == 3, audit
    assert summary["v2_approval_command_audit_record_before_real_signature_count"] == 0, audit
    assert summary["v2_approval_command_audit_phase2_order_valid_count"] == 1, audit
    assert summary["v2_approval_command_audit_phase2_blocker_count"] == 0, audit
    assert summary["v2_approval_command_audit_signature_order_valid_count"] == 1, audit
    assert summary["v2_approval_command_audit_signature_blocker_count"] == 0, audit
    assert summary["v2_approval_command_audit_phase_marker_valid_count"] == 1, audit
    assert summary["v2_approval_command_audit_phase_marker_blocker_count"] == 0, audit
    assert summary["v2_approval_command_audit_required_marker_missing_count"] == 0, audit
    assert summary["v2_approval_command_audit_preflight_artifact_write_count"] == 2, audit
    assert summary["v2_approval_command_audit_preflight_no_artifact_blocker_count"] == 0, audit
    assert summary["v2_approval_command_audit_record_command_preflight_arg_count"] == 3, audit
    assert summary["v2_approval_command_audit_unsigned_template_allowed_count"] == 0, audit
    assert summary["v2_approval_command_audit_expected_arg_binding_count"] == 35, audit
    assert summary["v2_approval_command_audit_expected_arg_mismatch_count"] == 0, audit
    assert summary["v2_approval_command_audit_expected_path_binding_ready_count"] == 1, audit
    assert summary["v2_approval_command_audit_phase1_handoff_post_return_command_count"] == 3, audit
    assert summary["v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count"] == 3, audit
    assert summary["v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count"] == 0, audit
    assert summary["v2_approval_command_audit_phase1_handoff_post_return_command_match_count"] == 3, audit
    assert summary["v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count"] == 0, audit
    assert summary["v2_approval_command_audit_phase1_handoff_post_return_ready_count"] == 1, audit
    assert summary["v2_approval_command_audit_ready_count"] == 1, audit
    assert summary["v2_approval_command_audit_approval_recorded_count"] == 0, audit
    assert summary["v2_approval_command_audit_runtime_dispatch_ready_count"] == 0, audit
    assert summary["v2_approval_command_audit_native_dispatch_allowed_count"] == 0, audit
    assert summary["v2_approval_command_audit_training_path_enabled_count"] == 0, audit
    assert summary["v2_approval_command_audit_product_native_ready_count"] == 0, audit
    assert summary["v2_approval_command_audit_default_behavior_changed_count"] == 0, audit
    assert summary["v2_approval_command_audit_unsafe_claim_count"] == 0, audit
    assert ARTIFACT.exists(), ARTIFACT
    phase2_broken_plan = json.loads(json.dumps(plan))
    for step in phase2_broken_plan["steps"]:
        if step["id"] == "collect_real_owner_direction_signature":
            step["order"] = 6
        elif step["id"] == "record_owner_release_review":
            step["order"] = 7
    phase2_broken_audit = build_optimizer_v2_approval_command_audit(
        approval_execution_plan=phase2_broken_plan,
        write_artifact=False,
    )
    phase2_broken_summary = phase2_broken_audit["summary"]
    assert phase2_broken_audit["ok"] is False, phase2_broken_audit
    assert phase2_broken_summary["v2_approval_command_audit_phase2_order_valid_count"] == 0, phase2_broken_audit
    assert phase2_broken_summary["v2_approval_command_audit_phase2_blocker_count"] >= 1, phase2_broken_audit

    phase2_freshness_broken_plan = json.loads(json.dumps(plan))
    for step in phase2_freshness_broken_plan["steps"]:
        if step["id"] == "collect_real_owner_direction_signature":
            step["order"] = 10
        elif step["id"] == "rebuild_phase2_signature_bundle":
            step["order"] = 12
    phase2_freshness_broken_audit = build_optimizer_v2_approval_command_audit(
        approval_execution_plan=phase2_freshness_broken_plan,
        write_artifact=False,
    )
    phase2_freshness_summary = phase2_freshness_broken_audit["summary"]
    assert phase2_freshness_broken_audit["ok"] is False, phase2_freshness_broken_audit
    assert phase2_freshness_summary["v2_approval_command_audit_phase2_order_valid_count"] == 0, phase2_freshness_broken_audit
    assert phase2_freshness_summary["v2_approval_command_audit_phase2_blocker_count"] >= 1, phase2_freshness_broken_audit

    signature_broken_plan = json.loads(json.dumps(plan))
    for step in signature_broken_plan["steps"]:
        if step["id"] == "record_owner_release_direction":
            step["order"] = 11
        elif step["id"] == "rebuild_owner_direction_packet":
            step["order"] = 13
    signature_broken_audit = build_optimizer_v2_approval_command_audit(
        approval_execution_plan=signature_broken_plan,
        write_artifact=False,
    )
    signature_broken_summary = signature_broken_audit["summary"]
    assert signature_broken_audit["ok"] is False, signature_broken_audit
    assert signature_broken_summary["v2_approval_command_audit_signature_order_valid_count"] == 0, signature_broken_audit
    assert signature_broken_summary["v2_approval_command_audit_signature_blocker_count"] >= 1, signature_broken_audit
    assert signature_broken_summary["v2_approval_command_audit_record_before_real_signature_count"] == 1, signature_broken_audit

    phase_broken_plan = json.loads(json.dumps(plan))
    for step in phase_broken_plan["steps"]:
        if step["id"] == "record_owner_release_direction":
            step["phase"] = "phase1"
    phase_broken_audit = build_optimizer_v2_approval_command_audit(
        approval_execution_plan=phase_broken_plan,
        write_artifact=False,
    )
    phase_broken_summary = phase_broken_audit["summary"]
    assert phase_broken_audit["ok"] is False, phase_broken_audit
    assert phase_broken_summary["v2_approval_command_audit_phase_marker_valid_count"] == 0, phase_broken_audit
    assert phase_broken_summary["v2_approval_command_audit_phase_marker_blocker_count"] >= 1, phase_broken_audit

    preflight_no_artifact_plan = json.loads(json.dumps(plan))
    for step in preflight_no_artifact_plan["steps"]:
        if step["id"] == "preflight_phase1_record_inputs":
            step["command"] = f"{step['command']} --no-artifact"
    preflight_no_artifact_audit = build_optimizer_v2_approval_command_audit(
        approval_execution_plan=preflight_no_artifact_plan,
        write_artifact=False,
    )
    preflight_no_artifact_summary = preflight_no_artifact_audit["summary"]
    assert preflight_no_artifact_audit["ok"] is False, preflight_no_artifact_audit
    assert preflight_no_artifact_summary["v2_approval_command_audit_preflight_artifact_write_count"] == 1, (
        preflight_no_artifact_audit
    )
    assert preflight_no_artifact_summary["v2_approval_command_audit_preflight_no_artifact_blocker_count"] == 1, (
        preflight_no_artifact_audit
    )

    wrong_preflight_path_plan = json.loads(json.dumps(plan))
    for step in wrong_preflight_path_plan["steps"]:
        if step["id"] == "record_owner_release_review":
            step["command"] = step["command"].replace(
                "temp\\turbocore_optimizer\\turbocore_optimizer_v2_approval_execution_preflight.json",
                "temp\\turbocore_optimizer\\old_approval_execution_preflight.json",
            )
    wrong_preflight_path_audit = build_optimizer_v2_approval_command_audit(
        approval_execution_plan=wrong_preflight_path_plan,
        write_artifact=False,
    )
    wrong_preflight_path_summary = wrong_preflight_path_audit["summary"]
    assert wrong_preflight_path_audit["ok"] is False, wrong_preflight_path_audit
    assert wrong_preflight_path_summary["v2_approval_command_audit_expected_arg_mismatch_count"] == 1, (
        wrong_preflight_path_audit
    )
    assert wrong_preflight_path_summary["v2_approval_command_audit_expected_path_binding_ready_count"] == 0, (
        wrong_preflight_path_audit
    )

    wrong_signed_bundle_path_plan = json.loads(json.dumps(plan))
    for step in wrong_signed_bundle_path_plan["steps"]:
        if step["id"] == "validate_phase1_signed_bundle":
            step["command"] = step["command"].replace(
                "temp\\turbocore_optimizer\\turbocore_optimizer_v2_signed_bundle.reviewed.json",
                "temp\\turbocore_optimizer\\stale_signed_bundle.reviewed.json",
            )
    wrong_signed_bundle_path_audit = build_optimizer_v2_approval_command_audit(
        approval_execution_plan=wrong_signed_bundle_path_plan,
        write_artifact=False,
    )
    wrong_signed_bundle_path_summary = wrong_signed_bundle_path_audit["summary"]
    assert wrong_signed_bundle_path_audit["ok"] is False, wrong_signed_bundle_path_audit
    assert wrong_signed_bundle_path_summary["v2_approval_command_audit_expected_arg_mismatch_count"] == 1, (
        wrong_signed_bundle_path_audit
    )

    handoff_alignment_broken_plan = json.loads(json.dumps(plan))
    handoff_alignment_broken_plan["phase1_handoff_post_return_alignment"]["handoff_commands"][0] = (
        "backend\\env\\python-flashattention\\python.exe backend\\core\\stale_intake.py"
    )
    handoff_alignment_broken_audit = build_optimizer_v2_approval_command_audit(
        approval_execution_plan=handoff_alignment_broken_plan,
        write_artifact=False,
    )
    handoff_alignment_broken_summary = handoff_alignment_broken_audit["summary"]
    assert handoff_alignment_broken_audit["ok"] is False, handoff_alignment_broken_audit
    assert (
        handoff_alignment_broken_summary[
            "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count"
        ]
        == 1
    ), handoff_alignment_broken_audit
    assert handoff_alignment_broken_summary["v2_approval_command_audit_phase1_handoff_post_return_ready_count"] == 0, (
        handoff_alignment_broken_audit
    )

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_approval_command_audit_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": audit["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
