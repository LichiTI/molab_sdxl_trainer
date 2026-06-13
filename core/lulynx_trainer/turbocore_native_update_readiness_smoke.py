"""Smoke checks for TurboCore native update readiness reporting."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_update_readiness import build_native_update_readiness_report  # noqa: E402


def _checkpoint_ready() -> dict[str, bool]:
    return {
        "checkpoint_metadata_integrated": True,
        "trainer_state_metadata_integrated": True,
        "trainer_state_save_sync_verified": True,
        "resume_owner_state_guard_verified": True,
        "checkpoint_owner_state_enabled": True,
    }


def test_readiness_reports_conservative_blockers() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    report = build_native_update_readiness_report(
        optimizer=optimizer,
        params=[param],
        runtime_context={"num_processes": 1},
        shadow_config={"mode": "shadow", "direct_grad": False},
        native_update_mode="profile",
    )
    reasons = set(report["blocked_reasons"])
    assert report["training_path_enabled"] is False, report
    assert report["training_dispatch_kernel_present"] is False, report
    assert report["diagnostic_runtime_available"] is True, report
    assert report["performance_test_ready"] is False, report
    assert report["stream_lifetime_bound"] is False, report
    assert report["static_checks"]["ok"] is True, report
    assert "native_training_flat_owner_default_off" in reasons, report
    assert "native_training_flat_owner_not_promoted" not in reasons, report
    assert "native_training_flat_owner_unavailable" not in reasons, report
    assert "native_training_dispatch_kernel_default_off" in reasons, report
    assert "native_training_dispatch_kernel_not_promoted" not in reasons, report
    assert "native_training_dispatch_kernel_missing" not in reasons, report
    assert "native_kernel_missing" not in reasons, report
    assert report["native_checks"]["diagnostic_runtime_available"] is True, report
    assert report["native_checks"]["diagnostic_runtime_probe_supported"] is True, report
    assert report["native_checks"]["current_process_tensor_object_session_supported"] is True, report
    assert report["native_checks"]["training_dispatch_kernel_present"] is False, report
    assert report["native_checks"]["flat_owner_contract_ready"] is True, report
    assert report["native_checks"]["training_flat_owner_boundary_ready"] is True, report
    assert report["native_checks"]["reference_flat_owner_ready"] is True, report
    assert report["native_checks"]["training_flat_owner_promoted"] is False, report
    assert report["native_checks"]["training_flat_owner_default_off"] is True, report
    assert report["native_checks"]["training_dispatch_kernel_contract_ready"] is True, report
    assert report["native_checks"]["training_dispatch_kernel_promoted"] is False, report
    assert report["native_checks"]["training_dispatch_kernel_default_off"] is True, report
    assert report["owner_checks"]["direct_gradient_write_boundary_ready"] is True, report
    assert report["owner_checks"]["direct_gradient_write_native_supported"] is False, report
    assert report["owner_checks"]["direct_gradient_write_default_off"] is True, report
    assert "direct_gradient_write_default_off" not in reasons, report
    assert "direct_gradient_write_not_native_supported" not in reasons, report
    assert "direct_gradient_write_default_off" in report["owner_checks"]["direct_gradient_write_optional_blocked_reasons"], report
    assert "owner_gradient_sync_default_off" in reasons, report
    assert "stream_lifetime_unbound" in reasons, report
    assert "native_runtime_error_recovery_missing" not in reasons, report
    assert "native_runtime_error_recovery_not_integrated" not in reasons, report
    assert report["native_checks"]["runtime_recovery_policy_defined"] is True, report
    assert report["native_checks"]["runtime_recovery_dispatch_integrated"] is True, report
    assert "trainer_checkpoint_integration_missing" in reasons, report
    assert "trainer_state_save_sync_guard_missing" in reasons, report
    assert "trainer_resume_owner_state_guard_missing" in reasons, report
    assert "parameter_owner_copyback_not_integrated" in reasons, report


def test_readiness_requires_explicit_checkpoint_guards_after_metadata() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    report = build_native_update_readiness_report(
        optimizer=optimizer,
        params=[param],
        runtime_context={
            "num_processes": 1,
            "native_update_training_dispatch_enabled": True,
            "training_path_enabled": True,
        },
        shadow_config={
            "mode": "shadow",
            "direct_grad": True,
            "direct_grad_lifecycle_integrated": True,
            "checkpoint_metadata_integrated": True,
            "checkpoint_owner_state_enabled": True,
            "copyback_scratch_probe_integrated": True,
            "copyback_dispatch_experimental_enabled": True,
            "copyback_dispatch_validated": True,
        },
        native_update_mode="native_experimental",
    )
    reasons = set(report["blocked_reasons"])
    owner = report["owner_checks"]
    assert owner["trainer_checkpoint_metadata_integrated"] is True, report
    assert owner["trainer_state_metadata_integrated"] is True, report
    assert owner["trainer_state_save_sync_verified"] is False, report
    assert owner["resume_owner_state_guard_verified"] is False, report
    assert owner["trainer_checkpoint_integration"] is False, report
    assert owner["owner_gradient_sync_training_integrated"] is False, report
    assert "trainer_checkpoint_integration_missing" not in reasons, report
    assert "trainer_state_save_sync_guard_missing" in reasons, report
    assert "trainer_resume_owner_state_guard_missing" in reasons, report
    assert "owner_gradient_sync_not_training_integrated" in reasons, report


def test_readiness_reports_copyback_probe_without_dispatch() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    report = build_native_update_readiness_report(
        optimizer=optimizer,
        params=[param],
        runtime_context={"num_processes": 1},
        shadow_config={
            "mode": "shadow",
            "direct_grad": True,
            "direct_grad_lifecycle_integrated": True,
            **_checkpoint_ready(),
            "copyback_scratch_probe_integrated": True,
            "native_tensor_binding_probe_integrated": True,
        },
        native_update_mode="profile",
    )
    reasons = set(report["blocked_reasons"])
    owner = report["owner_checks"]
    assert owner["parameter_owner_copyback_probe_integrated"] is True, report
    assert owner["parameter_owner_copyback_integrated"] is False, report
    assert owner["parameter_owner_copyback_dispatch_enabled"] is False, report
    assert owner["native_tensor_binding_probe_integrated"] is True, report
    assert "parameter_owner_copyback_dispatch_disabled" in reasons, report
    assert "parameter_owner_copyback_not_integrated" not in reasons, report


def test_readiness_reports_copyback_dispatch_experimental_contract() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    report = build_native_update_readiness_report(
        optimizer=optimizer,
        params=[param],
        runtime_context={"num_processes": 1},
        shadow_config={
            "mode": "shadow",
            "direct_grad": True,
            "direct_grad_lifecycle_integrated": True,
            **_checkpoint_ready(),
            "copyback_scratch_probe_integrated": True,
            "copyback_dispatch_experimental_enabled": True,
            "copyback_dispatch_validated": True,
            "native_tensor_binding_probe_integrated": True,
            "owner_native_event_chain_probe_requested": True,
        },
        native_update_mode="profile",
    )
    reasons = set(report["blocked_reasons"])
    owner = report["owner_checks"]
    assert owner["parameter_owner_copyback_dispatch_experimental_enabled"] is True, report
    assert owner["parameter_owner_copyback_dispatch_validated"] is True, report
    assert owner["parameter_owner_copyback_integrated"] is True, report
    assert owner["parameter_owner_copyback_dispatch_enabled"] is True, report
    assert owner["owner_native_event_chain_probe_requested"] is True, report
    assert report["native_checks"]["training_dispatch_kernel_contract_ready"] is True, report
    assert report["native_checks"]["training_dispatch_kernel_promoted"] is False, report
    assert report["native_checks"]["training_dispatch_kernel_default_off"] is True, report
    assert "parameter_owner_copyback_dispatch_disabled" not in reasons, report
    assert "parameter_owner_copyback_dispatch_not_validated" not in reasons, report


def test_readiness_keeps_promotion_blockers_when_training_context_is_explicit() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    report = build_native_update_readiness_report(
        optimizer=optimizer,
        params=[param],
        runtime_context={
            "num_processes": 1,
            "native_update_training_dispatch_enabled": True,
            "training_path_enabled": True,
        },
        shadow_config={
            "mode": "shadow",
            "direct_grad": True,
            "direct_grad_lifecycle_integrated": True,
            **_checkpoint_ready(),
            "copyback_scratch_probe_integrated": True,
            "copyback_dispatch_experimental_enabled": True,
            "copyback_dispatch_validated": True,
            "native_tensor_binding_probe_integrated": True,
        },
        native_update_mode="native_experimental",
    )
    reasons = set(report["blocked_reasons"])
    assert report["native_checks"]["explicit_training_context_requested"] is True, report
    assert report["owner_checks"]["direct_gradient_write_default_off"] is False, report
    assert report["native_checks"]["training_flat_owner_default_off"] is False, report
    assert report["native_checks"]["training_dispatch_kernel_default_off"] is False, report
    assert "direct_gradient_write_not_native_supported" not in reasons, report
    assert "direct_gradient_write_not_native_supported" in report["owner_checks"]["direct_gradient_write_optional_blocked_reasons"], report
    assert "direct_gradient_write_default_off" not in reasons, report
    assert "owner_gradient_sync_guard_disabled" in reasons, report
    assert "native_training_flat_owner_not_promoted" in reasons, report
    assert "native_training_dispatch_kernel_not_promoted" in reasons, report
    assert "native_training_flat_owner_default_off" not in reasons, report
    assert "native_training_dispatch_kernel_default_off" not in reasons, report


def test_readiness_refines_stream_lifetime_when_event_chain_is_verified() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    report = build_native_update_readiness_report(
        optimizer=optimizer,
        params=[param],
        runtime_context={"num_processes": 1},
        shadow_config={
            "mode": "shadow",
            "direct_grad": True,
            "direct_grad_lifecycle_integrated": True,
            **_checkpoint_ready(),
            "copyback_scratch_probe_integrated": True,
            "copyback_dispatch_experimental_enabled": True,
            "copyback_dispatch_validated": True,
            "native_tensor_binding_probe_integrated": True,
            "native_binding_stream_lifetime_bound": False,
            "owner_native_event_chain_probe_requested": True,
            "owner_native_event_chain_probe_attempted": True,
            "owner_native_event_chain_verified": True,
            "owner_native_pre_launch_ordering_verified": True,
            "owner_native_post_launch_ordering_verified": True,
            "owner_native_stream_wait_event_verified": True,
        },
        native_update_mode="profile",
    )
    reasons = set(report["blocked_reasons"])
    native = report["native_checks"]
    assert report["stream_lifetime_bound"] is True, report
    assert report["stream_lifetime_ownership_bound"] is False, report
    assert report["stream_ordering_verified"] is True, report
    assert report["event_chain_verified"] is True, report
    assert native["stream_lifetime_bound"] is True, report
    assert native["stream_lifetime_ownership_bound"] is False, report
    assert native["stream_ordering_verified"] is True, report
    assert native["event_chain_verified"] is True, report
    assert "stream_lifetime_unbound" not in reasons, report
    assert "stream_lifetime_ownership_not_promoted" in reasons, report


def test_readiness_blocks_unvalidated_copyback_dispatch() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    report = build_native_update_readiness_report(
        optimizer=optimizer,
        params=[param],
        runtime_context={"num_processes": 1},
        shadow_config={
            "mode": "shadow",
            "direct_grad_lifecycle_integrated": True,
            **_checkpoint_ready(),
            "copyback_scratch_probe_integrated": True,
            "copyback_dispatch_experimental_enabled": True,
            "copyback_dispatch_validated": False,
        },
        native_update_mode="profile",
    )
    assert "parameter_owner_copyback_dispatch_not_validated" in set(report["blocked_reasons"]), report


def test_readiness_blocks_bad_static_context() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    report = build_native_update_readiness_report(
        optimizer=optimizer,
        params=[param],
        runtime_context={"num_processes": 2, "deepspeed": True, "gradient_release_active": True},
        shadow_config={"mode": "off"},
        native_update_mode="native_experimental",
    )
    reasons = set(report["blocked_reasons"])
    assert "optimizer_not_adamw" in reasons, report
    assert "distributed_not_supported" in reasons, report
    assert "deepspeed_not_supported" in reasons, report
    assert "gradient_release_not_supported" in reasons, report
    assert "shadow_mode_not_enabled" in reasons, report


def main() -> int:
    test_readiness_reports_conservative_blockers()
    test_readiness_requires_explicit_checkpoint_guards_after_metadata()
    test_readiness_reports_copyback_probe_without_dispatch()
    test_readiness_reports_copyback_dispatch_experimental_contract()
    test_readiness_keeps_promotion_blockers_when_training_context_is_explicit()
    test_readiness_refines_stream_lifetime_when_event_chain_is_verified()
    test_readiness_blocks_unvalidated_copyback_dispatch()
    test_readiness_blocks_bad_static_context()
    print("turbocore_native_update_readiness_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
