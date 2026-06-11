"""Smoke checks for Native Training Performance P6 audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from devtools.audit_native_training_performance_p6 import build_p6_performance_audit  # noqa: E402
from turbocore_native_update_release_review_package_smoke import (  # noqa: E402
    _assert_optimizer_family_counts,
    _assert_summary_counts_preserved,
)


def run_smoke() -> dict[str, Any]:
    audit = build_p6_performance_audit(quick=True)
    assert audit["audit"] == "native_training_performance_p6_audit_v0", audit
    assert audit["ok"] is True, audit
    assert "attention_native_route" in audit["sections"], audit
    assert "checkpoint_streaming_route" in audit["sections"], audit
    assert "checkpoint_artifact_streaming_io" in audit["sections"], audit
    assert "optimizer_family_coverage" in audit["sections"], audit
    assert "cuda_graph_route" in audit["sections"], audit
    assert "cuda_graph_observe_manifest" in audit["sections"], audit
    assert "cuda_graph_training_integration_review" in audit["sections"], audit
    assert "async_checkpoint_writer" in audit["sections"], audit
    assert "async_checkpoint_writer_observe_manifest" in audit["sections"], audit
    assert "async_checkpoint_writer_training_integration_review" in audit["sections"], audit
    assert "native_data_pipeline_observe" in audit["sections"], audit
    assert "native_data_pipeline_adapter_shadow" in audit["sections"], audit
    assert "native_data_pipeline_semantic_h2d" in audit["sections"], audit
    assert "native_data_pipeline_e2e_shadow" in audit["sections"], audit
    assert "native_data_pipeline_canary_rollout_policy" in audit["sections"], audit
    assert "native_data_pipeline_training_integration_review" in audit["sections"], audit
    assert "native_update_release_review_package" in audit["sections"], audit
    assert "native_update_release_review_archive" in audit["sections"], audit
    section = audit["sections"]["attention_native_route"]
    assert section["ok"] is True, section
    assert section["parity"]["parity_ok"] is True, section
    assert section["backward"]["backward_parity_ok"] is True, section
    checkpoint_section = audit["sections"]["checkpoint_streaming_route"]
    assert checkpoint_section["ok"] is True, checkpoint_section
    assert checkpoint_section["parity"]["parity_ok"] is True, checkpoint_section
    assert checkpoint_section["parity"]["finite_gradients"] is True, checkpoint_section
    artifact_section = audit["sections"]["checkpoint_artifact_streaming_io"]
    assert artifact_section["ok"] is True, artifact_section
    if artifact_section["capabilities"]["native_entrypoint"]:
        assert artifact_section["parity"]["parity_ok"] is True, artifact_section
        assert artifact_section["benchmark"]["ok"] is True, artifact_section
    optimizer_section = audit["sections"]["optimizer_family_coverage"]
    assert optimizer_section["ok"] is True, optimizer_section
    assert optimizer_section["training_path_enabled"] is False, optimizer_section
    assert optimizer_section["default_behavior_changed"] is False, optimizer_section
    optimizer_summary = optimizer_section["summary"]
    _assert_optimizer_family_counts(optimizer_summary, allow_extra=True)
    assert optimizer_section["recommended_next_step"] == (
        "keep native dispatch unwired until explicit owner/release approval is recorded"
    ), optimizer_section
    assert optimizer_section["priority_groups"], optimizer_section
    for group in optimizer_section["priority_groups"]:
        next_gate = str(group.get("next_gate", "")).lower()
        assert (
            "await explicit owner" in next_gate
            or "until explicit owner" in next_gate
            or "record explicit owner" in next_gate
            or "explicit owner/release approval" in next_gate
        ), group
    native_release_review = audit["sections"]["native_update_release_review_package"]
    assert native_release_review["ok"] is True, native_release_review
    assert native_release_review["evidence_ready"] is True, native_release_review
    assert native_release_review["ready_for_owner_release_review"] is True, native_release_review
    assert native_release_review["p6_release_review_evidence_ready"] is True, native_release_review
    assert native_release_review["release_review_recorded"] is False, native_release_review
    assert native_release_review["decision"] == (
        "native_update_release_review_hold_for_owner_review_default_off"
    ), native_release_review
    assert native_release_review["present_gate_count"] == native_release_review["expected_gate_count"], native_release_review
    assert native_release_review["default_off_gate_count"] == native_release_review["expected_gate_count"], native_release_review
    assert native_release_review["present_supplemental_gate_count"] == native_release_review["supplemental_gate_count"], native_release_review
    assert native_release_review["default_off_supplemental_gate_count"] == native_release_review["supplemental_gate_count"], native_release_review
    optimizer_supplemental = native_release_review["supplemental_gate_summaries"]["optimizer_family_coverage"]
    multitensor_supplemental = native_release_review["supplemental_gate_summaries"][
        "native_update_optimizer_multitensor_release_hold"
    ]
    assert optimizer_supplemental["ok"] is True, native_release_review
    assert optimizer_supplemental["evidence_ready"] is True, native_release_review
    assert optimizer_supplemental["ready_for_review"] is True, native_release_review
    assert optimizer_supplemental["default_off"] is True, native_release_review
    assert multitensor_supplemental["ok"] is True, native_release_review
    assert multitensor_supplemental["evidence_ready"] is True, native_release_review
    assert multitensor_supplemental["ready_for_review"] is True, native_release_review
    assert multitensor_supplemental["default_off"] is True, native_release_review
    assert multitensor_supplemental["unsafe_claims"] == [], native_release_review
    assert optimizer_supplemental["source_count"] >= 1, native_release_review
    assert optimizer_supplemental["source_payload_digest_match"] is True, native_release_review
    _assert_optimizer_family_counts(optimizer_supplemental["optimizer_family_counts"], allow_extra=True)
    _assert_summary_counts_preserved(
        {"summary": optimizer_summary},
        optimizer_supplemental["optimizer_family_counts"],
    )
    handoff_sources = native_release_review["owner_release_review_handoff"]["supplemental_acknowledgement_sources"][
        "optimizer_family_coverage"
    ]
    assert handoff_sources["source_count"] == optimizer_supplemental["source_count"], native_release_review
    assert handoff_sources["source_names"] == optimizer_supplemental["source_names"], native_release_review
    assert (
        handoff_sources["source_payload_digest_match"] == optimizer_supplemental["source_payload_digest_match"]
    ), native_release_review
    assert native_release_review["training_path_enabled"] is False, native_release_review
    assert native_release_review["runtime_dispatch_allowed"] is False, native_release_review
    assert native_release_review["native_dispatch_allowed"] is False, native_release_review
    assert native_release_review["product_exposure_allowed"] is False, native_release_review
    assert native_release_review["post_release_request_fields"] == {}, native_release_review
    assert "native_update_release_owner_review_missing" in native_release_review["review_hold_reasons"], native_release_review
    assert native_release_review["blocked_reasons"] == [], native_release_review
    release_archive = audit["sections"]["native_update_release_review_archive"]
    assert release_archive["ok"] is True, release_archive
    assert release_archive["p6_release_review_archive_evidence_ready"] is True, release_archive
    assert release_archive["archive_ready"] is False, release_archive
    assert release_archive["archive_recorded"] is False, release_archive
    assert release_archive["ready_for_owner_release_direction"] is False, release_archive
    assert release_archive["decision"] == (
        "native_update_release_review_archive_hold_for_recorded_review_default_off"
    ), release_archive
    assert release_archive["runtime_dispatch_allowed"] is False, release_archive
    assert release_archive["native_dispatch_allowed"] is False, release_archive
    assert release_archive["training_path_enabled"] is False, release_archive
    assert release_archive["training_dispatch"] is False, release_archive
    assert release_archive["post_archive_request_fields"] == {}, release_archive
    assert release_archive["post_release_request_fields"] == {}, release_archive
    archive_package_summary = release_archive["release_review_package_summary"]
    assert archive_package_summary["optimizer_family_source_payload_digest_match"] is True, release_archive
    assert archive_package_summary["optimizer_family_handoff_sources_match"] is True, release_archive
    archive_multitensor = archive_package_summary["supplemental_gate_summaries"][
        "native_update_optimizer_multitensor_release_hold"
    ]
    assert archive_multitensor["present"] is True, release_archive
    assert archive_multitensor["evidence_ready"] is True, release_archive
    assert archive_multitensor["ready_for_review"] is True, release_archive
    assert archive_multitensor["default_off"] is True, release_archive
    assert "native_update_release_review_archive_review_not_recorded" in release_archive[
        "archive_hold_reasons"
    ], release_archive
    assert release_archive["blocked_reasons"] == [], release_archive
    cuda_graph_section = audit["sections"]["cuda_graph_route"]
    assert cuda_graph_section["ok"] is True, cuda_graph_section
    assert cuda_graph_section["training_path_enabled"] is False, cuda_graph_section
    assert cuda_graph_section["runtime_dispatch_ready"] is False, cuda_graph_section
    assert cuda_graph_section["default_behavior_changed"] is False, cuda_graph_section
    assert cuda_graph_section["static_contract"]["static_contract_ready"] is True, cuda_graph_section
    cuda_graph_manifest = audit["sections"]["cuda_graph_observe_manifest"]
    assert cuda_graph_manifest["ok"] is True, cuda_graph_manifest
    assert cuda_graph_manifest["observe_manifest_ready"] is True, cuda_graph_manifest
    assert cuda_graph_manifest["training_path_enabled"] is False, cuda_graph_manifest
    assert cuda_graph_manifest["route_decision"]["decision"] == "would_select_cuda_graph_observe_but_dispatch_disabled", cuda_graph_manifest
    cuda_graph_review = audit["sections"]["cuda_graph_training_integration_review"]
    assert cuda_graph_review["ok"] is True, cuda_graph_review
    assert cuda_graph_review["promotion_ready"] is True, cuda_graph_review
    assert cuda_graph_review["review_gate_ready"] is True, cuda_graph_review
    assert cuda_graph_review["training_path_enabled"] is False, cuda_graph_review
    assert cuda_graph_review["runtime_dispatch_ready"] is False, cuda_graph_review
    assert cuda_graph_review["native_dispatch_allowed"] is False, cuda_graph_review
    assert cuda_graph_review["review_package"]["manual_review_required"] is True, cuda_graph_review
    assert cuda_graph_review["review_package"]["allowed_initial_modes"] == ["off", "observe"], cuda_graph_review
    assert cuda_graph_review["review_package"]["rollback_policy"]["fallback_authoritative"] is True, cuda_graph_review
    async_writer = audit["sections"]["async_checkpoint_writer"]
    assert async_writer["ok"] is True, async_writer
    assert async_writer["promotion_ready"] is True, async_writer
    assert async_writer["training_path_enabled"] is False, async_writer
    assert async_writer["proof"]["atomic_commit_ok"] is True, async_writer
    assert async_writer["proof"]["parity_ok"] is True, async_writer
    async_writer_manifest = audit["sections"]["async_checkpoint_writer_observe_manifest"]
    assert async_writer_manifest["ok"] is True, async_writer_manifest
    assert async_writer_manifest["promotion_ready"] is True, async_writer_manifest
    assert async_writer_manifest["observe_manifest_ready"] is True, async_writer_manifest
    assert async_writer_manifest["training_path_enabled"] is False, async_writer_manifest
    assert async_writer_manifest["runtime_dispatch_ready"] is False, async_writer_manifest
    assert async_writer_manifest["native_dispatch_allowed"] is False, async_writer_manifest
    assert async_writer_manifest["route_decision"]["decision"] == (
        "would_select_async_checkpoint_writer_observe_but_dispatch_disabled"
    ), async_writer_manifest
    assert async_writer_manifest["manifest"]["rollback_policy"]["fallback_authoritative"] is True, async_writer_manifest
    async_writer_review = audit["sections"]["async_checkpoint_writer_training_integration_review"]
    assert async_writer_review["ok"] is True, async_writer_review
    assert async_writer_review["promotion_ready"] is True, async_writer_review
    assert async_writer_review["review_gate_ready"] is True, async_writer_review
    assert async_writer_review["training_path_enabled"] is False, async_writer_review
    assert async_writer_review["runtime_dispatch_ready"] is False, async_writer_review
    assert async_writer_review["native_dispatch_allowed"] is False, async_writer_review
    assert async_writer_review["review_package"]["manual_review_required"] is True, async_writer_review
    assert async_writer_review["review_package"]["allowed_initial_modes"] == ["off", "observe"], async_writer_review
    assert async_writer_review["review_package"]["rollback_policy"]["fallback_authoritative"] is True, async_writer_review
    native_data_pipeline = audit["sections"]["native_data_pipeline_observe"]
    assert native_data_pipeline["ok"] is True, native_data_pipeline
    assert native_data_pipeline["promotion_ready"] is True, native_data_pipeline
    assert native_data_pipeline["observe_manifest_ready"] is True, native_data_pipeline
    assert native_data_pipeline["training_path_enabled"] is False, native_data_pipeline
    assert native_data_pipeline["runtime_dispatch_ready"] is False, native_data_pipeline
    assert native_data_pipeline["route_decision"]["decision"] == (
        "would_select_native_data_pipeline_observe_but_dispatch_disabled"
    ), native_data_pipeline
    assert native_data_pipeline["probes"]["workspace_lifecycle"]["native_runtime"] is True, native_data_pipeline
    assert native_data_pipeline["probes"]["shuffled_plan"]["native_runtime"] is True, native_data_pipeline
    assert native_data_pipeline["probes"]["descriptor_probe"]["descriptor_parity_ok"] is True, native_data_pipeline
    native_data_adapter = audit["sections"]["native_data_pipeline_adapter_shadow"]
    assert native_data_adapter["ok"] is True, native_data_adapter
    assert native_data_adapter["promotion_ready"] is True, native_data_adapter
    assert native_data_adapter["adapter_shadow_ready"] is True, native_data_adapter
    assert native_data_adapter["training_path_enabled"] is False, native_data_adapter
    assert native_data_adapter["runtime_dispatch_ready"] is False, native_data_adapter
    assert native_data_adapter["adapter_route"]["decision"] == (
        "shadow_adapter_prepared_fallback_authoritative"
    ), native_data_adapter
    assert native_data_adapter["adapter_envelope"]["native_data_authority"] == "none", native_data_adapter
    native_data_semantic = audit["sections"]["native_data_pipeline_semantic_h2d"]
    assert native_data_semantic["ok"] is True, native_data_semantic
    assert native_data_semantic["promotion_ready"] is True, native_data_semantic
    assert native_data_semantic["semantic_h2d_matrix_ready"] is True, native_data_semantic
    assert native_data_semantic["training_path_enabled"] is False, native_data_semantic
    assert native_data_semantic["runtime_dispatch_ready"] is False, native_data_semantic
    assert native_data_semantic["semantic_matrix"]["failed_case_count"] == 0, native_data_semantic
    assert native_data_semantic["descriptor_parity"]["descriptor_parity_ok"] is True, native_data_semantic
    assert native_data_semantic["h2d_ownership_contract"]["copy_independent"] is True, native_data_semantic
    assert native_data_semantic["h2d_ownership_contract"]["native_pipeline_owns_device_tensor"] is False, native_data_semantic
    native_data_e2e = audit["sections"]["native_data_pipeline_e2e_shadow"]
    assert native_data_e2e["ok"] is True, native_data_e2e
    assert native_data_e2e["promotion_ready"] is True, native_data_e2e
    assert native_data_e2e["e2e_shadow_ready"] is True, native_data_e2e
    assert native_data_e2e["training_path_enabled"] is False, native_data_e2e
    assert native_data_e2e["runtime_dispatch_ready"] is False, native_data_e2e
    assert native_data_e2e["shadow_case"]["loss_parity_ok"] is True, native_data_e2e
    assert native_data_e2e["shadow_case"]["native_shadow_updates_original"] is False, native_data_e2e
    native_data_policy = audit["sections"]["native_data_pipeline_canary_rollout_policy"]
    assert native_data_policy["ok"] is True, native_data_policy
    assert native_data_policy["promotion_ready"] is True, native_data_policy
    assert native_data_policy["canary_rollout_policy_ready"] is True, native_data_policy
    assert native_data_policy["policy"]["canary_enabled_by_default"] is False, native_data_policy
    assert native_data_policy["policy"]["explicit_opt_in_required"] is True, native_data_policy
    assert native_data_policy["policy"]["rollback_policy"]["fallback_authoritative"] is True, native_data_policy
    assert native_data_policy["training_path_enabled"] is False, native_data_policy
    native_data_review = audit["sections"]["native_data_pipeline_training_integration_review"]
    assert native_data_review["ok"] is True, native_data_review
    assert native_data_review["promotion_ready"] is True, native_data_review
    assert native_data_review["review_gate_ready"] is True, native_data_review
    assert native_data_review["training_path_enabled"] is False, native_data_review
    assert native_data_review["runtime_dispatch_ready"] is False, native_data_review
    assert native_data_review["native_dispatch_allowed"] is False, native_data_review
    assert native_data_review["review_package"]["manual_review_required"] is True, native_data_review
    assert native_data_review["review_package"]["allowed_initial_modes"] == ["off", "observe"], native_data_review
    assert native_data_review["review_package"]["rollback_policy"]["fallback_authoritative"] is True, native_data_review
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p6_audit_smoke",
        "ok": True,
        "milestone_completed": audit["milestone_completed"],
        "recommended_next_step": audit["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
