"""Smoke checks for TurboCore native update promotion scorecard."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_update_promotion_scorecard import build_native_update_promotion_scorecard  # noqa: E402
from turbocore_native_update_release_review_package_smoke import (  # noqa: E402
    EXPECTED_OPTIMIZER_FAMILY_COUNTS,
    _artifact_map,
    _assert_optimizer_family_counts,
    _assert_summary_counts_preserved,
    _write_real_artifact_case,
)


def _shadow_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "training_path_enabled": False,
        "direct_grad_lifecycle_integrated": True,
        "checkpoint_metadata_integrated": True,
        "checkpoint_owner_state_enabled": True,
        "after_optimizer": {
            "compared": True,
            "parity_ok_loose": True,
            "max_abs_param_diff": 1.0e-8,
            "mean_abs_param_diff": 1.0e-9,
        },
        "copyback_probe": {
            "scratch_copyback_validated": True,
            "real_parameters_mutated": False,
            "elapsed_ms": 0.2,
        },
        "copyback_dispatch_probe": {
            "copyback_dispatch_enabled": True,
            "copyback_dispatch_validated": True,
            "copyback_dispatch_target": "training_parameters",
            "real_parameters_mutated": True,
            "real_parameters_restored": True,
            "elapsed_ms": 0.1,
        },
        "native_binding_probe": {
            "request_shape_ready": True,
            "tensor_object_binding_ready": True,
            "launch_plan_ready": True,
            "stream_lifetime_bound": True,
            "stream_guard_ready": True,
            "event_chain_verified": True,
        },
        "owner_native_launch_probe": {
            "ok": True,
            "attempted": True,
            "kernel_executed": True,
            "parity_ok": True,
            "persistent_owner_mutated": False,
            "event_chain_probe_requested": True,
            "event_chain_probe_attempted": True,
            "event_chain_verified": True,
            "pre_launch_ordering_verified": True,
            "post_launch_ordering_verified": True,
            "stream_wait_event_verified": True,
            "elapsed_ms": 0.5,
        },
        "optimizer_performance_gate": {
            "gate": "turbocore_optimizer_performance_gate",
            "ok": True,
            "promotion_gate_ok": True,
            "runtime_dispatch_allowed": False,
            "evidence_quality": "promotion_benchmark",
            "best_candidate": {
                "optimizer": "turbocore_cuda_adamw_v0",
                "speedup_vs_baseline": 1.35,
            },
        },
    }


def _performance_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "report": "synthetic_native_update_performance_report_for_scorecard_smoke",
        "training_dispatch": False,
        "training_path_enabled": False,
        "optimizer_performance_gate": {
            "gate": "turbocore_optimizer_performance_gate",
            "ok": True,
            "promotion_gate_ok": True,
            "runtime_dispatch_allowed": False,
            "evidence_quality": "promotion_benchmark",
            "best_candidate": {
                "optimizer": "turbocore_cuda_adamw_v0",
                "speedup_vs_baseline": 1.35,
            },
        },
        "benchmark_matrix": {
            "schema_version": 1,
            "matrix": "turbocore_update_benchmark_matrix_v0",
            "run": True,
            "summary": {
                "case_count": 2,
                "executed_count": 2,
                "all_success": True,
                "mean_step_ms_by_case": {
                    "baseline_phase": 1000.0,
                    "native_update_dispatch": 950.0,
                },
            },
            "cases": [
                {
                    "case": {"name": "baseline_phase"},
                    "summary": {"steps_completed": 24, "mean_step_ms": 1000.0},
                },
                {
                    "case": {"name": "native_update_dispatch"},
                    "summary": {
                        "steps_completed": 24,
                        "mean_step_ms": 950.0,
                        "native_dispatch_executed": True,
                    },
                },
            ],
        },
    }


def _promotion_readiness_ok() -> dict[str, object]:
    return {
        "schema_version": 1,
        "report": "synthetic_native_update_readiness_for_scorecard_smoke",
        "ok": True,
        "training_path_enabled": False,
        "native_kernel_present": True,
        "training_dispatch_kernel_present": True,
        "diagnostic_runtime_available": True,
        "performance_test_ready": True,
        "stream_lifetime_bound": True,
        "stream_lifetime_ownership_bound": True,
        "stream_ordering_verified": True,
        "event_chain_verified": True,
        "static_checks": {
            "ok": True,
            "optimizer": "AdamW",
            "parameter_tensors": 1,
            "blocked_reasons": [],
        },
        "owner_checks": {
            "ok": True,
            "direct_gradient_write_boundary_ready": True,
            "direct_gradient_write_native_supported": True,
            "direct_gradient_write_training_integrated": True,
            "owner_gradient_sync_boundary_ready": True,
            "owner_gradient_sync_supported": True,
            "owner_gradient_sync_training_integrated": True,
            "blocked_reasons": [],
        },
        "native_checks": {
            "ok": True,
            "native_optimizer_schema_ok": True,
            "flat_owner_contract_ready": True,
            "reference_flat_owner_ready": True,
            "training_flat_owner_promoted": True,
            "training_dispatch_kernel_contract_ready": True,
            "training_dispatch_kernel_present": True,
            "training_dispatch_kernel_promoted": True,
            "diagnostic_runtime_available": True,
            "stream_lifetime_bound": True,
            "stream_lifetime_ownership_bound": True,
            "stream_ordering_verified": True,
            "event_chain_verified": True,
            "performance_test_ready": True,
            "blocked_reasons": [],
        },
        "blocked_reasons": [],
    }


def _product_exposure_decision_ready() -> dict[str, object]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_product_exposure_decision_v0",
        "gate": "native_update_product_exposure_decision",
        "ok": True,
        "evidence_ready": True,
        "ready_for_product_exposure_review": True,
        "product_exposure_decision_recorded": True,
        "manual_review_required": True,
        "decision": "native_update_product_exposure_decision_recorded_default_off",
        "default_behavior_changed": False,
        "product_exposure_allowed": False,
        "training_launch_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ready_for_ui": False,
        "backend_router_registered": False,
        "post_product_exposure_request_fields": {},
        "blocked_reasons": [],
        "promotion_blockers": [],
    }


def _optimizer_family_coverage_summary() -> dict[str, object]:
    priority_next_gates = [
        "keep AdamW variant canary dispatch unwired until explicit owner approval is recorded",
        "record explicit owner/release approval artifacts for fp32, quantized, and schedule-free simple variants",
        "keep adaptive-LR native dispatch unwired until explicit owner/release approval is recorded",
        "keep factored/custom native dispatch unwired until explicit owner/release approval is recorded",
        "keep plugin selected-family native dispatch unwired until explicit owner/release approval is recorded",
    ]
    return {
        "gate": "optimizer_family_coverage",
        "expected_gate": "optimizer_family_coverage",
        "present": True,
        "ok": True,
        "evidence_ready": True,
        "ready_for_review": True,
        "decision": "",
        "default_off": True,
        "request_fields_empty": True,
        "unsafe_claims": [],
        "blocked_reasons": [],
        "source_count": 2,
        "source_names": [
            "turbocore_optimizer_family_coverage_scorecard.json",
            "turbocore_optimizer_coverage_scorecard.json",
        ],
        "source_payload_digest_match": True,
        "digest": "synthetic_optimizer_family_coverage_digest_for_promotion_smoke",
        "recommended_next_step": "keep native dispatch unwired until explicit owner/release approval is recorded",
        "priority_group_count": len(priority_next_gates),
        "priority_next_gates": priority_next_gates,
        "optimizer_family_counts": _optimizer_family_counts(),
    }


def _release_review_package_ready() -> dict[str, object]:
    optimizer_family_coverage = _optimizer_family_coverage_summary()
    multitensor_release_hold = _multitensor_release_hold_summary()
    release_review_template = {
        "acknowledged_supplemental_gates": {
            "optimizer_family_coverage": {
                "digest": optimizer_family_coverage["digest"],
                "decision": optimizer_family_coverage["decision"],
                "evidence_ready": optimizer_family_coverage["evidence_ready"],
                "ready_for_review": optimizer_family_coverage["ready_for_review"],
                "default_off": optimizer_family_coverage["default_off"],
                "optimizer_family_counts": optimizer_family_coverage["optimizer_family_counts"],
                "recommended_next_step": optimizer_family_coverage["recommended_next_step"],
                "priority_next_gates": optimizer_family_coverage["priority_next_gates"],
            },
            "native_update_optimizer_multitensor_release_hold": {
                "digest": multitensor_release_hold["digest"],
                "decision": multitensor_release_hold["decision"],
                "evidence_ready": multitensor_release_hold["evidence_ready"],
                "ready_for_review": multitensor_release_hold["ready_for_review"],
                "default_off": multitensor_release_hold["default_off"],
                "recommended_next_step": multitensor_release_hold["recommended_next_step"],
                "priority_next_gates": multitensor_release_hold["priority_next_gates"],
            },
        },
    }
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_release_review_package_v0",
        "gate": "native_update_release_review_package",
        "ok": True,
        "evidence_ready": True,
        "ready_for_review": True,
        "ready_for_owner_release_review": True,
        "release_review_recorded": True,
        "manual_review_required": True,
        "decision": "native_update_release_review_recorded_default_off",
        "default_off": True,
        "expected_gate_count": 12,
        "present_gate_count": 12,
        "default_off_gate_count": 12,
        "supplemental_gate_count": 2,
        "present_supplemental_gate_count": 2,
        "default_off_supplemental_gate_count": 2,
        "supplemental_gate_summaries": {
            "optimizer_family_coverage": optimizer_family_coverage,
            "native_update_optimizer_multitensor_release_hold": multitensor_release_hold,
        },
        "release_review_template": release_review_template,
        "owner_release_review_handoff": {
            "handoff": "native_update_release_owner_review_handoff_v0",
            "release_review_template_digest": _digest_payload(release_review_template),
            "supplemental_acknowledgement_counts": {
                "optimizer_family_coverage": dict(optimizer_family_coverage["optimizer_family_counts"]),
            },
            "supplemental_acknowledgement_sources": {
                "optimizer_family_coverage": {
                    "source_count": optimizer_family_coverage["source_count"],
                    "source_names": list(optimizer_family_coverage["source_names"]),
                    "source_payload_digest_match": optimizer_family_coverage["source_payload_digest_match"],
                },
            },
        },
        "default_behavior_changed": False,
        "release_gate_open": False,
        "training_launch_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ready_for_ui": False,
        "backend_router_registered": False,
        "post_release_request_fields": {},
        "blocked_reasons": [],
        "promotion_blockers": [],
    }


def _owner_release_review_record_ready() -> dict[str, object]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_owner_release_review_record_v0",
        "gate": "native_update_owner_release_review_record",
        "ok": True,
        "owner_packet_ready": True,
        "signed_review_present": True,
        "signed_review_valid": True,
        "approval_recorded": True,
        "release_review_recorded": True,
        "decision": "native_update_release_review_recorded_default_off",
        "signed_review_digest_match": True,
        "release_package_decision": "native_update_release_review_recorded_default_off",
        "blocked_reasons": [],
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
    }


def _multitensor_release_hold_summary() -> dict[str, object]:
    return {
        "gate": "native_update_optimizer_multitensor_release_hold",
        "expected_gate": "native_update_optimizer_multitensor_release_hold",
        "present": True,
        "ok": True,
        "evidence_ready": True,
        "ready_for_review": True,
        "decision": "native_update_optimizer_multitensor_hold_for_owner_review_default_off",
        "default_off": True,
        "request_fields_empty": True,
        "unsafe_claims": [],
        "blocked_reasons": [],
        "digest": "synthetic_optimizer_multitensor_release_hold_digest_for_promotion_smoke",
        "recommended_next_step": "record explicit owner/release approval for optimizer multi-tensor native update",
        "priority_group_count": 0,
        "priority_next_gates": [],
    }


def _real_release_review_package_default_off() -> dict[str, object]:
    return _write_real_artifact_case(_artifact_map())


def _assert_dispatch_closed(report: dict[str, object]) -> None:
    assert report["training_dispatch"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_step_executed"] is False, report
    assert report["pytorch_optimizer_authoritative"] is True, report
    assert report["fallback_to_pytorch_required"] is True, report
    assert report["should_call_pytorch_optimizer_step"] is True, report
    assert report["native_mutation_allowed"] is False, report
    assert report["training_parameter_mutation_allowed"] is False, report


def _optimizer_family_counts() -> dict[str, int]:
    return dict(EXPECTED_OPTIMIZER_FAMILY_COUNTS)


def _digest_payload(value: dict[str, object]) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def test_scorecard_is_coherent_but_not_promotion_ready() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        runtime_context={},
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    reasons = set(report["blocked_reasons"])
    blockers = set(report["promotion_blockers"])
    primary_blockers = set(report["primary_promotion_blockers"])
    derived_blockers = set(report["derived_promotion_blockers"])
    assert report["scorecard"] == "turbocore_native_update_promotion_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["checks"]["scorecard_evidence_coherent"] is True, report
    assert report["promotion_ready"] is False, report
    _assert_dispatch_closed(report)
    assert "native_dispatch_runtime_not_implemented" in blockers, report
    assert "native_dispatch_training_path_disabled" in blockers, report
    assert "representative_performance_gate_missing" in blockers, report
    assert "native_update_product_exposure_decision_missing" in blockers, report
    assert "native_update_release_review_package_missing" in blockers, report
    assert "native_runtime_recovery_training_dispatch_disabled" in blockers, report
    assert "owner_gradient_sync_default_off" in primary_blockers, report
    assert "direct_gradient_write_default_off" not in primary_blockers, report
    assert "stream_lifetime_ownership_default_off" in primary_blockers, report
    assert "native_dispatch_training_runtime_executor_default_off" in primary_blockers, report
    assert "native_update_release_review_package_missing" in primary_blockers, report
    assert "native_dispatch_runtime_not_implemented" in derived_blockers, report
    assert "native_dispatch_training_path_disabled" in derived_blockers, report
    assert "native_dispatch_diagnostic_executor_call_disabled" in derived_blockers, report
    assert "stream_lifetime_unbound" not in reasons, report
    assert "stream_lifetime_ownership_default_off" in reasons, report
    assert "owner_gradient_sync_default_off" in reasons, report
    assert "direct_gradient_write_default_off" not in reasons, report
    assert "native_dispatch_runtime_not_implemented" in reasons, report
    assert report["readiness"]["training_path_enabled"] is False, report
    assert report["dispatch_preflight"]["would_allow_native_dispatch"] is False, report
    assert report["dispatch_contract"]["would_allow_native_dispatch"] is False, report
    assert report["dispatch_request"]["dispatch_allowed"] is False, report
    assert report["kernel_launch_plan"]["launch_allowed"] is False, report
    assert report["dispatch_arming"]["armed_for_native_dispatch"] is False, report
    assert report["dispatch_arming"]["call_pytorch_optimizer_step"] is True, report
    assert report["dispatch_runtime"]["native_step_executed"] is False, report
    assert report["dispatch_runtime"]["should_call_pytorch_optimizer_step"] is True, report
    assert report["dispatch_execution_plan"]["execution_allowed"] is False, report
    assert report["dispatch_execution_plan"]["should_call_pytorch_optimizer_step"] is True, report
    assert "native_dispatch_training_runtime_executor_default_off" in report["dispatch_execution_plan"]["blocked_reasons"], report
    assert "native_dispatch_runtime_executor_missing" not in report["dispatch_execution_plan"]["blocked_reasons"], report
    assert report["dispatch_executor_probe"]["called"] is False, report
    assert report["dispatch_executor_probe"]["should_call_pytorch_optimizer_step"] is True, report
    assert "native_dispatch_diagnostic_executor_call_disabled" in report["dispatch_executor_probe"]["blocked_reasons"], report
    assert report["dispatch_runtime_diagnostic_replay"]["native_step_executed"] is False, report
    assert "native_dispatch_diagnostic_executor_replay_disabled" in report["dispatch_runtime_diagnostic_replay"]["blocked_reasons"], report


def test_scorecard_accepts_representative_performance_gate_without_dispatch() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={},
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    reasons = set(report["blocked_reasons"])
    blockers = set(report["promotion_blockers"])
    primary_blockers = set(report["primary_promotion_blockers"])
    derived_blockers = set(report["derived_promotion_blockers"])
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    _assert_dispatch_closed(report)
    assert report["performance_gate"]["representative_performance_gate_ready"] is True, report
    assert report["performance_gate"]["runtime_dispatch_allowed"] is False, report
    assert report["checks"]["representative_performance_gate_ready"] is True, report
    assert "representative_performance_gate_missing" not in reasons, report
    assert "representative_performance_gate_missing" not in blockers, report
    assert "native_dispatch_runtime_not_implemented" in blockers, report
    assert "native_dispatch_training_path_disabled" in blockers, report
    assert "native_runtime_recovery_training_dispatch_disabled" in blockers, report
    assert "stream_lifetime_unbound" not in blockers, report
    assert "stream_lifetime_ownership_default_off" in blockers, report
    assert "owner_gradient_sync_default_off" in blockers, report
    assert "direct_gradient_write_default_off" not in blockers, report
    assert "stream_lifetime_ownership_default_off" in primary_blockers, report
    assert "owner_gradient_sync_default_off" in primary_blockers, report
    assert "direct_gradient_write_default_off" not in primary_blockers, report
    assert "native_dispatch_runtime_not_implemented" in derived_blockers, report
    assert "native_dispatch_training_path_disabled" in derived_blockers, report
    assert "native_update_gate_not_enabled" in derived_blockers, report
    assert report["dispatch_preflight"]["performance_test_ready"] is True, report
    assert report["dispatch_contract"]["would_allow_native_dispatch"] is False, report


def test_scorecard_can_replay_diagnostic_executor_without_training_dispatch() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={},
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
        diagnostic_executor_replay=True,
    )
    _assert_dispatch_closed(report)
    replay = report["dispatch_runtime_diagnostic_replay"]
    plan = report["dispatch_diagnostic_execution_plan"]
    probe = report["dispatch_diagnostic_executor_probe"]
    result = probe["result"]
    assert replay["training_dispatch"] is False, report
    assert replay["training_path_enabled"] is False, report
    assert replay["native_step_executed"] is False, report
    assert replay["should_call_pytorch_optimizer_step"] is True, report
    assert plan["diagnostic_executor_preconditions_ready"] is True, report
    assert plan["training_executor_preconditions_ready"] is False, report
    assert probe["called"] is True, report
    assert probe["ok"] is True, report
    assert probe["native_step_executed"] is False, report
    assert result["diagnostic_replay"] is True, report
    assert result["shadow_owner_native_kernel_evidence"] is True, report


def test_scorecard_exercises_explicit_training_dispatch_without_promotion() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.0)
    param.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_executor_present": True,
            "native_update_runtime_execution_guard_enabled": True,
            "native_update_training_mutation_guard_enabled": True,
            "native_update_training_dispatch_enabled": True,
            "native_update_runtime_dispatch_available": True,
            "training_path_enabled": True,
            "native_update_owner_gradient_sync_guard_enabled": True,
            "native_update_owner_gradient_sync_bound": True,
            "native_update_training_executor_config": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "max_grad_norm": 0.0,
                "prefer_triton": False,
            },
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    primary_blockers = set(report["primary_promotion_blockers"])
    derived_blockers = set(report["derived_promotion_blockers"])
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_dispatch"] is True, report
    assert report["training_path_enabled"] is True, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_step_executed"] is False, report
    assert report["should_call_pytorch_optimizer_step"] is True, report
    assert report["fallback_to_pytorch_required"] is True, report
    assert report["dispatch_request"]["dispatch_allowed"] is False, report
    assert report["kernel_launch_plan"]["launch_allowed"] is False, report
    assert report["dispatch_arming"]["execute_native_step"] is False, report
    assert report["dispatch_execution_plan"]["execution_allowed"] is False, report
    assert report["dispatch_training_executor"]["called"] is False, report
    assert report["dispatch_training_executor"]["native_step_executed"] is False, report
    assert "native_dispatch_native_kernel_not_promoted" in blockers, report
    assert "native_dispatch_native_kernel_not_promoted" in primary_blockers, report
    assert "native_dispatch_runtime_not_implemented" in primary_blockers, report
    assert "native_dispatch_training_path_disabled" in primary_blockers, report
    assert "native_runtime_recovery_training_dispatch_disabled" not in blockers, report
    assert "native_recovery_keeps_dispatch_disabled" not in blockers, report
    assert optimizer.state[param] == {}, optimizer.state[param]


def test_scorecard_requires_recorded_release_review_package() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    release = _release_review_package_ready()
    release["release_review_recorded"] = False
    release["decision"] = "native_update_release_review_rejected_default_off"
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": release,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    primary_blockers = set(report["primary_promotion_blockers"])
    assert report["promotion_ready"] is False, report
    assert report["release_review_package"]["release_review_recorded"] is False, report
    assert "native_update_release_review_not_recorded" in blockers, report
    assert "native_update_release_review_not_recorded" in primary_blockers, report


def test_scorecard_blocks_invalid_owner_release_review_record_when_supplied() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    invalid_record = _owner_release_review_record_ready()
    invalid_record["signed_review_valid"] = False
    invalid_record["signed_review_digest_match"] = False
    invalid_record["blocked_reasons"] = ["signed_owner_release_review_template_digest_mismatch"]
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": _release_review_package_ready(),
            "native_update_owner_release_review_record": invalid_record,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    owner_record = report["release_review_package"]["owner_release_review_record"]
    assert report["promotion_ready"] is False, report
    assert owner_record["present"] is True, report
    assert owner_record["signed_review_valid"] is False, report
    assert owner_record["signed_review_digest_match"] is False, report
    assert "native_update_release_review_owner_record_signed_review_valid_failed" in blockers, report
    assert "native_update_release_review_owner_record_signed_review_digest_match_failed" in blockers, report
    assert (
        "native_update_release_review_owner_record:signed_owner_release_review_template_digest_mismatch"
        in blockers
    ), report


def test_scorecard_requires_release_review_package_ready_default_off_flags() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    release = _release_review_package_ready()
    release["ready_for_review"] = False
    release["default_off"] = False
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": release,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    release_report = report["release_review_package"]
    assert report["promotion_ready"] is False, report
    assert release_report["ready_for_review"] is False, report
    assert release_report["default_off"] is False, report
    assert "native_update_release_review_not_ready_for_review" in blockers, report
    assert "native_update_release_review_not_default_off" in blockers, report


def test_scorecard_requires_optimizer_family_coverage_supplemental_gate() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    release = _release_review_package_ready()
    release["present_supplemental_gate_count"] = 0
    release["default_off_supplemental_gate_count"] = 0
    release["supplemental_gate_summaries"] = {}
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": release,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    primary_blockers = set(report["primary_promotion_blockers"])
    release_report = report["release_review_package"]
    assert report["promotion_ready"] is False, report
    assert release_report["supplemental_gate_count"] == 2, report
    assert release_report["present_supplemental_gate_count"] == 0, report
    assert release_report["optimizer_family_coverage"]["present"] is False, report
    assert "native_update_release_review_supplemental_gates_not_present" in blockers, report
    assert "native_update_release_review_supplemental_gates_not_default_off" in blockers, report
    assert "native_update_release_review_optimizer_family_coverage_missing" in primary_blockers, report


def test_scorecard_rejects_optimizer_family_coverage_auto_wiring_claim() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    release = _release_review_package_ready()
    summary = dict(_optimizer_family_coverage_summary())
    summary["recommended_next_step"] = "wire optimizer native dispatch automatically"
    summary["priority_next_gates"] = ["auto-wire plugin selected-family native dispatch"]
    release["supplemental_gate_summaries"] = {"optimizer_family_coverage": summary}
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": release,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    optimizer_section = report["release_review_package"]["optimizer_family_coverage"]
    assert report["promotion_ready"] is False, report
    assert optimizer_section["recommended_next_step"] == "wire optimizer native dispatch automatically", report
    assert "native_update_release_review_optimizer_family_coverage_next_step_not_owner_release_hold" in blockers, report
    assert "native_update_release_review_optimizer_family_coverage_priority_gate_not_owner_release_hold" in blockers, report


def test_scorecard_rejects_optimizer_family_handoff_counts_mismatch() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    release = _release_review_package_ready()
    release["owner_release_review_handoff"]["supplemental_acknowledgement_counts"]["optimizer_family_coverage"][
        "plugin_selected_native_ready_count"
    ] = 1
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": release,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    optimizer_section = report["release_review_package"]["optimizer_family_coverage"]
    assert report["promotion_ready"] is False, report
    assert optimizer_section["handoff_counts_match"] is False, report
    assert "native_update_release_review_optimizer_family_handoff_counts_mismatch" in blockers, report


def test_scorecard_rejects_optimizer_family_source_payload_mismatch() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    release = _release_review_package_ready()
    release["supplemental_gate_summaries"]["optimizer_family_coverage"][
        "source_payload_digest_match"
    ] = False
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": release,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    optimizer_section = report["release_review_package"]["optimizer_family_coverage"]
    assert report["promotion_ready"] is False, report
    assert optimizer_section["source_payload_digest_match"] is False, report
    assert "native_update_release_review_optimizer_family_coverage_source_payload_digest_mismatch" in blockers, report


def test_scorecard_rejects_optimizer_family_handoff_sources_mismatch() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    release = _release_review_package_ready()
    release["owner_release_review_handoff"]["supplemental_acknowledgement_sources"][
        "optimizer_family_coverage"
    ]["source_names"] = ["wrong_optimizer_coverage_source.json"]
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": release,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    optimizer_section = report["release_review_package"]["optimizer_family_coverage"]
    assert report["promotion_ready"] is False, report
    assert optimizer_section["handoff_sources_match"] is False, report
    assert "native_update_release_review_optimizer_family_handoff_sources_mismatch" in blockers, report


def test_scorecard_rejects_handoff_template_digest_mismatch() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    release = _release_review_package_ready()
    release["owner_release_review_handoff"]["release_review_template_digest"] = "wrong_template_digest"
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": release,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    release_section = report["release_review_package"]
    assert report["promotion_ready"] is False, report
    assert release_section["handoff_release_review_template_digest_match"] is False, report
    assert "native_update_release_review_handoff_template_digest_mismatch" in blockers, report


def test_scorecard_preserves_real_optimizer_family_compact_counts() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    release = _real_release_review_package_default_off()
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": release,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    release_counts = release["supplemental_gate_summaries"]["optimizer_family_coverage"]["optimizer_family_counts"]
    compact = report["release_review_package"]["optimizer_family_coverage"]
    optimizer_counts = compact["optimizer_family_counts"]
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    assert "native_update_release_review_not_recorded" in report["promotion_blockers"], report
    assert compact["handoff_counts_match"] is True, report
    assert compact["handoff_sources_match"] is True, report
    assert compact["source_payload_digest_match"] is True, report
    assert optimizer_counts == release_counts, report
    assert len(optimizer_counts) >= 300, report
    _assert_optimizer_family_counts(optimizer_counts, allow_extra=True)
    _assert_summary_counts_preserved(
        {"summary": release_counts},
        optimizer_counts,
    )
    _assert_dispatch_closed(report)


def test_scorecard_rejects_real_optimizer_family_compact_count_tamper() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    release = _real_release_review_package_default_off()
    release["supplemental_gate_summaries"]["optimizer_family_coverage"]["optimizer_family_counts"][
        "plugin_selected_native_ready_count"
    ] = 1
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": release,
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    compact = report["release_review_package"]["optimizer_family_coverage"]
    assert report["promotion_ready"] is False, report
    assert compact["handoff_counts_match"] is False, report
    assert "native_update_release_review_optimizer_family_handoff_counts_mismatch" in blockers, report
    _assert_dispatch_closed(report)


def test_scorecard_requires_recorded_product_exposure_decision() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    exposure = _product_exposure_decision_ready()
    exposure["product_exposure_decision_recorded"] = False
    exposure["decision"] = "native_update_product_exposure_decision_hold_for_owner_review_default_off"
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        runtime_context={
            "native_update_product_exposure_decision": exposure,
            "native_update_release_review_package": _release_review_package_ready(),
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    blockers = set(report["promotion_blockers"])
    primary_blockers = set(report["primary_promotion_blockers"])
    assert report["promotion_ready"] is False, report
    assert report["product_exposure_decision"]["product_exposure_decision_recorded"] is False, report
    assert "native_update_product_exposure_decision_not_recorded" in blockers, report
    assert "native_update_product_exposure_decision_not_recorded" in primary_blockers, report


def test_scorecard_promotes_cuda_training_dispatch_runtime() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.0)
    param.grad = torch.tensor([0.25, -0.5], dtype=torch.float32, device=device)
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        performance_report=_performance_report(),
        readiness_report=_promotion_readiness_ok(),
        runtime_context={
            "native_update_executor_present": True,
            "native_update_runtime_execution_guard_enabled": True,
            "native_update_training_mutation_guard_enabled": True,
            "native_update_training_dispatch_enabled": True,
            "native_update_runtime_dispatch_available": True,
            "training_path_enabled": True,
            "native_update_owner_gradient_sync_guard_enabled": True,
            "native_update_owner_gradient_sync_bound": True,
            "native_update_direct_gradient_write_guard_enabled": True,
            "native_update_direct_gradient_write_bound": True,
            "native_update_flat_owner_training_guard_enabled": True,
            "native_update_flat_owner_bound": True,
            "native_update_training_dispatch_kernel_guard_enabled": True,
            "native_update_training_dispatch_kernel_bound": True,
            "native_update_stream_lifetime_ownership_guard_enabled": True,
            "native_update_stream_lifetime_ownership_bound": True,
            "native_update_training_executor_config": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "max_grad_norm": 0.0,
                "prefer_native_cuda": True,
                "prefer_triton": False,
            },
            "native_update_product_exposure_decision": _product_exposure_decision_ready(),
            "native_update_release_review_package": _release_review_package_ready(),
            "native_update_owner_release_review_record": _owner_release_review_record_ready(),
        },
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )
    result = report["dispatch_training_executor"]["result"]
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["product_exposure_decision"]["product_exposure_decision_recorded"] is True, report
    assert report["product_exposure_decision"]["post_product_exposure_request_fields"] == {}, report
    assert report["release_review_package"]["release_review_recorded"] is True, report
    assert report["release_review_package"]["owner_release_review_record"]["present"] is True, report
    assert report["release_review_package"]["owner_release_review_record"]["signed_review_valid"] is True, report
    assert report["release_review_package"]["owner_release_review_record"]["release_review_recorded"] is True, report
    assert report["release_review_package"]["expected_gate_count"] == 12, report
    assert report["release_review_package"]["supplemental_gate_count"] == 2, report
    assert report["release_review_package"]["present_supplemental_gate_count"] == 2, report
    assert report["release_review_package"]["default_off_supplemental_gate_count"] == 2, report
    multitensor_release_hold = report["release_review_package"]["native_update_optimizer_multitensor_release_hold"]
    assert multitensor_release_hold["present"] is True, report
    assert multitensor_release_hold["ok"] is True, report
    assert multitensor_release_hold["evidence_ready"] is True, report
    assert multitensor_release_hold["ready_for_review"] is True, report
    assert multitensor_release_hold["default_off"] is True, report
    assert multitensor_release_hold["unsafe_claims"] == [], report
    assert report["release_review_package"]["handoff_release_review_template_digest_match"] is True, report
    assert report["release_review_package"]["optimizer_family_coverage"]["default_off"] is True, report
    optimizer_counts = report["release_review_package"]["optimizer_family_coverage"]["optimizer_family_counts"]
    assert report["release_review_package"]["optimizer_family_coverage"]["handoff_counts"] == optimizer_counts, report
    assert report["release_review_package"]["optimizer_family_coverage"]["handoff_counts_match"] is True, report
    assert report["release_review_package"]["optimizer_family_coverage"]["source_payload_digest_match"] is True, report
    assert report["release_review_package"]["optimizer_family_coverage"]["source_count"] == 2, report
    assert report["release_review_package"]["optimizer_family_coverage"]["handoff_sources_match"] is True, report
    _assert_optimizer_family_counts(optimizer_counts)
    assert report["release_review_package"]["optimizer_family_coverage"]["recommended_next_step"] == (
        "keep native dispatch unwired until explicit owner/release approval is recorded"
    ), report
    assert report["release_review_package"]["post_release_request_fields"] == {}, report
    assert report["promotion_blockers"] == [], report
    assert report["primary_promotion_blockers"] == [], report
    assert report["derived_promotion_blockers"] == [], report
    assert report["native_step_executed"] is True, report
    assert report["should_call_pytorch_optimizer_step"] is False, report
    assert report["fallback_to_pytorch_required"] is False, report
    assert result["native_kernel_launched"] is True, report
    assert result["update_report"]["owner_backend"] == "rust_cuda_adamw_v0", report
    assert result["optimizer_state_sync"]["synced"] is True, report


def test_scorecard_surfaces_unsupported_context() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    report = build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_shadow_report(),
        runtime_context={"num_processes": 2, "deepspeed": True, "gradient_release_active": True},
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
    )
    reasons = set(report["blocked_reasons"])
    assert report["ok"] is True, report
    _assert_dispatch_closed(report)
    assert "optimizer_not_adamw" in reasons, report
    assert "distributed_not_supported" in reasons, report
    assert "deepspeed_not_supported" in reasons, report
    assert "gradient_release_not_supported" in reasons, report


def main() -> int:
    test_scorecard_is_coherent_but_not_promotion_ready()
    test_scorecard_accepts_representative_performance_gate_without_dispatch()
    test_scorecard_can_replay_diagnostic_executor_without_training_dispatch()
    test_scorecard_exercises_explicit_training_dispatch_without_promotion()
    test_scorecard_requires_recorded_release_review_package()
    test_scorecard_blocks_invalid_owner_release_review_record_when_supplied()
    test_scorecard_requires_release_review_package_ready_default_off_flags()
    test_scorecard_requires_optimizer_family_coverage_supplemental_gate()
    test_scorecard_rejects_optimizer_family_coverage_auto_wiring_claim()
    test_scorecard_rejects_optimizer_family_handoff_counts_mismatch()
    test_scorecard_rejects_optimizer_family_source_payload_mismatch()
    test_scorecard_rejects_optimizer_family_handoff_sources_mismatch()
    test_scorecard_rejects_handoff_template_digest_mismatch()
    test_scorecard_preserves_real_optimizer_family_compact_counts()
    test_scorecard_rejects_real_optimizer_family_compact_count_tamper()
    test_scorecard_requires_recorded_product_exposure_decision()
    test_scorecard_promotes_cuda_training_dispatch_runtime()
    test_scorecard_surfaces_unsupported_context()
    print("turbocore_native_update_promotion_scorecard_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
