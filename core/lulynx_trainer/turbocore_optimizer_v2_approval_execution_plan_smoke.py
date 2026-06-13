"""Smoke checks for the v2 approval execution plan."""

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

from core.turbocore_optimizer_v2_approval_execution_plan import (  # noqa: E402
    build_optimizer_v2_approval_execution_plan,
)
from core.turbocore_optimizer_v2_reviewer_handoff_packet import (  # noqa: E402
    build_optimizer_v2_reviewer_handoff_packet,
)


ARTIFACT = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_optimizer_v2_approval_execution_plan.json"


def run_smoke() -> dict[str, Any]:
    handoff = build_optimizer_v2_reviewer_handoff_packet(write_artifact=True)
    plan = build_optimizer_v2_approval_execution_plan(reviewer_handoff=handoff, write_artifact=True)
    summary = plan["summary"]
    steps = plan["steps"]

    assert plan["ok"] is True, plan
    assert plan["execution_plan_ready"] is True, plan
    assert plan["approval_recorded"] is False, plan
    assert plan["approval_artifact_written"] is False, plan
    assert plan["runtime_dispatch_ready"] is False, plan
    assert plan["native_dispatch_allowed"] is False, plan
    assert plan["training_path_enabled"] is False, plan
    assert plan["product_native_ready"] is False, plan
    assert plan["expected_signature_bundle_path"].endswith("turbocore_optimizer_v2_signature_bundle_packet.json"), plan
    assert [step["order"] for step in steps] == list(range(1, 17)), plan
    assert [step["id"] for step in steps] == [
        "generate_unsigned_reviewer_template",
        "collect_real_owner_and_product_reviews",
        "validate_phase1_signed_bundle",
        "extract_phase1_signed_bundle_records",
        "preflight_phase1_record_inputs",
        "record_owner_release_review",
        "record_product_exposure_decision",
        "rebuild_release_review_archive",
        "rebuild_owner_direction_packet",
        "rebuild_phase2_signature_bundle",
        "regenerate_phase2_reviewer_template",
        "collect_real_owner_direction_signature",
        "validate_phase2_signed_bundle",
        "extract_phase2_signed_bundle_records",
        "preflight_phase2_record_inputs",
        "record_owner_release_direction",
    ], plan
    assert summary["v2_approval_execution_step_count"] == 16, plan
    assert summary["v2_approval_execution_phase1_step_count"] == 6, plan
    assert summary["v2_approval_execution_phase2_step_count"] == 9, plan
    assert summary["v2_approval_execution_shared_step_count"] == 1, plan
    assert summary["v2_approval_execution_ready_step_count"] == 3, plan
    assert summary["v2_approval_execution_manual_signature_step_count"] == 2, plan
    assert summary["v2_approval_execution_record_step_count"] == 3, plan
    assert summary["v2_approval_execution_extraction_step_count"] == 2, plan
    assert summary["v2_approval_execution_preflight_step_count"] == 2, plan
    assert summary["v2_approval_execution_phase1_record_chain_step_count"] == 5, plan
    assert summary["v2_approval_execution_phase2_record_chain_step_count"] == 4, plan
    assert summary["v2_approval_execution_reviewer_template_entry_count"] == 2, plan
    assert summary["v2_approval_execution_blocked_signature_entry_count"] == 1, plan
    assert summary["v2_approval_execution_phase1_handoff_post_return_command_count"] == 3, plan
    assert summary["v2_approval_execution_phase1_handoff_post_return_pre_record_command_count"] == 3, plan
    assert summary["v2_approval_execution_phase1_handoff_post_return_approval_record_command_count"] == 0, plan
    assert summary["v2_approval_execution_phase1_handoff_post_return_command_match_count"] == 3, plan
    assert summary["v2_approval_execution_phase1_handoff_post_return_command_mismatch_count"] == 0, plan
    assert summary["v2_approval_execution_phase1_handoff_post_return_ready_count"] == 1, plan
    assert summary["v2_approval_execution_plan_ready_count"] == 1, plan
    assert summary["v2_approval_execution_approval_recorded_count"] == 0, plan
    assert summary["v2_approval_execution_runtime_dispatch_ready_count"] == 0, plan
    assert summary["v2_approval_execution_native_dispatch_allowed_count"] == 0, plan
    assert summary["v2_approval_execution_training_path_enabled_count"] == 0, plan
    assert summary["v2_approval_execution_product_native_ready_count"] == 0, plan
    assert summary["v2_approval_execution_default_behavior_changed_count"] == 0, plan
    assert summary["v2_approval_execution_unsafe_claim_count"] == 0, plan
    phase_by_id = {step["id"]: step["phase"] for step in steps}
    assert phase_by_id["generate_unsigned_reviewer_template"] == "shared", plan
    assert phase_by_id["record_owner_release_review"] == "phase1", plan
    assert phase_by_id["record_owner_release_direction"] == "phase2", plan
    assert "--signature-bundle" in steps[2]["command"], plan
    assert "--signature-bundle" in steps[3]["command"], plan
    assert "--training-launch-contract" in steps[4]["command"], plan
    assert "--product-exposure-evidence" in steps[4]["command"], plan
    assert "--owner-direction-packet" in steps[4]["command"], plan
    assert "turbocore_optimizer_v2_signature_bundle_packet.py" in steps[9]["command"], plan
    assert "turbocore_optimizer_v2_reviewer_handoff_packet.py" in steps[10]["command"], plan
    assert "--signature-bundle" in steps[12]["command"], plan
    assert "--write-extracted-artifacts" in steps[13]["command"], plan
    assert "--owner-direction-packet" in steps[14]["command"], plan
    assert ARTIFACT.exists(), ARTIFACT

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_approval_execution_plan_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": plan["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
