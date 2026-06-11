"""Smoke checks for TurboCore native update kernel launch planning."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_update_kernel_launcher import build_native_update_kernel_launch_plan  # noqa: E402
from core.turbocore_native_update_kernel_launcher import build_native_update_adamw_launch_config  # noqa: E402


class _Config:
    lr = 1e-4
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.01
    block_size = 256


class _Owner:
    def __init__(self) -> None:
        import torch

        self.config = _Config()
        self.param_flat = torch.ones(4)
        self.step_index = 3


def test_kernel_launcher_without_request_is_blocked() -> None:
    report = build_native_update_kernel_launch_plan()
    reasons = set(report["blocked_reasons"])
    assert report["launcher"] == "turbocore_native_update_kernel_launcher_v0", report
    assert report["launch_allowed"] is False, report
    assert report["launch_attempted"] is False, report
    assert report["kernel_executed"] is False, report
    assert report["mutates_training_parameters"] is False, report
    assert "dispatch_request_missing" in reasons, report
    assert "owner_native_launch_probe_missing" in reasons, report
    assert "native_dispatch_runtime_not_implemented" in reasons, report


def test_kernel_launcher_maps_owner_probe_evidence() -> None:
    report = build_native_update_kernel_launch_plan(
        dispatch_request={"requested": True, "dispatch_allowed": False},
        dispatch_contract={"would_allow_native_dispatch": False},
        owner_native_launch_probe={
            "ok": True,
            "runtime_session_id": 11,
            "binding_session_id": 13,
            "kernel_executed": True,
            "parity_ok": True,
            "event_chain_verified": True,
            "elapsed_ms": 1.5,
        },
    )
    evidence = report["evidence"]
    reasons = set(report["blocked_reasons"])
    assert report["requested"] is True, report
    assert report["launch_allowed"] is False, report
    assert evidence["runtime_session_id"] == 11, report
    assert evidence["binding_session_id"] == 13, report
    assert evidence["diagnostic_kernel_executed"] is True, report
    assert evidence["diagnostic_parity_ok"] is True, report
    assert report["sequence"][4]["planned"] is True, report
    assert "dispatch_request_not_allowed" in reasons, report
    assert "dispatch_contract_not_allowing_launch" in reasons, report


def test_shared_launch_config_has_contract_fields() -> None:
    config = build_native_update_adamw_launch_config(_Owner(), max_numel=2, capture_stage="smoke")
    assert config["contract"] == "turbocore_native_update_adamw_launch_config_v0", config
    assert config["kernel"] == "adamw_flat_fp32_cuda_kernel_v0", config
    assert config["launch_plan"] == "adamw_flat_fp32_launch_plan_v0", config
    assert config["training_dispatch"] is False, config
    assert config["training_path_enabled"] is False, config
    assert config["step_index"] == 3, config
    assert config["max_numel"] == 4, config
    assert config["stream_guard_descriptor"]["stream_capture_stage"] == "smoke", config


def test_kernel_launcher_can_allow_launch_when_dispatch_and_probe_are_ready() -> None:
    report = build_native_update_kernel_launch_plan(
        dispatch_request={
            "requested": True,
            "dispatch_allowed": True,
            "training_path_enabled": True,
            "runtime_dispatch_available": True,
        },
        dispatch_contract={"would_allow_native_dispatch": True},
        owner_native_launch_probe={
            "ok": True,
            "runtime_session_id": 11,
            "binding_session_id": 13,
            "kernel_executed": True,
            "parity_ok": True,
            "event_chain_probe_requested": True,
            "event_chain_verified": True,
            "persistent_owner_mutated": False,
            "elapsed_ms": 0.25,
        },
    )
    assert report["requested"] is True, report
    assert report["training_dispatch"] is True, report
    assert report["training_path_enabled"] is True, report
    assert report["launch_allowed"] is True, report
    assert report["mutates_training_parameters"] is True, report
    assert report["blocked_reasons"] == [], report
    assert all(item["enabled"] is True for item in report["sequence"]), report
    assert report["evidence"]["diagnostic_kernel_executed"] is True, report
    assert report["evidence"]["diagnostic_parity_ok"] is True, report
    assert report["evidence"]["event_chain_verified"] is True, report


def main() -> int:
    test_kernel_launcher_without_request_is_blocked()
    test_kernel_launcher_maps_owner_probe_evidence()
    test_shared_launch_config_has_contract_fields()
    test_kernel_launcher_can_allow_launch_when_dispatch_and_probe_are_ready()
    print("turbocore_native_update_kernel_launcher_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
