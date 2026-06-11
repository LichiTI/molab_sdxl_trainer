"""Smoke checks for TurboCore native update dispatch request planning."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_update_dispatch_request import build_native_update_dispatch_request  # noqa: E402


def test_dispatch_request_defaults_to_not_requested() -> None:
    report = build_native_update_dispatch_request(mode="off", dispatch_enabled=False)
    reasons = set(report["blocked_reasons"])
    assert report["request"] == "turbocore_native_update_dispatch_request_v0", report
    assert report["requested"] is False, report
    assert report["training_dispatch"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["dispatch_allowed"] is False, report
    assert report["runtime_dispatch_available"] is False, report
    assert report["plan"]["call_pytorch_optimizer_step"] is True, report
    assert report["training_path_request"]["request_boundary_ready"] is True, report
    assert report["training_path_request"]["default_off"] is True, report
    assert report["evidence"]["training_path_default_off"] is True, report
    assert "native_dispatch_not_requested" in reasons, report
    assert "native_dispatch_runtime_not_implemented" in reasons, report


def test_explicit_request_is_still_runtime_blocked() -> None:
    contract = {
        "dispatch_rehearsal_ready": False,
        "would_allow_native_dispatch": False,
        "native_kernel_present": True,
        "stream_lifetime_bound": True,
        "performance_test_ready": True,
        "dispatch_sequence": [{"step": "launch_native_adamw_kernel", "planned": True, "enabled": False}],
        "blocked_reasons": ["native_dispatch_training_path_disabled"],
    }
    gate = {"would_enable_native_update": True, "native_kernel_present": True}
    report = build_native_update_dispatch_request(
        mode="native_experimental",
        dispatch_enabled=True,
        gate_report=gate,
        dispatch_contract=contract,
    )
    reasons = set(report["blocked_reasons"])
    assert report["requested"] is True, report
    assert report["dispatch_allowed"] is False, report
    assert report["plan"]["execute_native_step"] is False, report
    assert report["plan"]["call_pytorch_optimizer_step"] is True, report
    assert report["training_path_request"]["request_boundary_ready"] is True, report
    assert report["training_path_request"]["explicit_training_path_requested"] is False, report
    assert "native_dispatch_training_path_default_off" in report["training_path_request"]["blocked_reasons"], report
    assert report["plan"]["sequence"][0]["step"] == "launch_native_adamw_kernel", report
    assert report["evidence"]["gate_would_enable_native_update"] is True, report
    assert report["evidence"]["native_kernel_present"] is True, report
    assert "native_dispatch_runtime_not_implemented" in reasons, report
    assert "native_dispatch_contract_not_allowing_dispatch" in reasons, report
    assert "native_dispatch_training_path_disabled" in reasons, report


def test_explicit_training_path_request_remains_runtime_blocked() -> None:
    contract = {
        "dispatch_rehearsal_ready": False,
        "would_allow_native_dispatch": False,
        "blocked_reasons": ["native_dispatch_training_path_disabled"],
    }
    gate = {"would_enable_native_update": True}
    report = build_native_update_dispatch_request(
        mode="native_experimental",
        dispatch_enabled=True,
        gate_report=gate,
        dispatch_contract=contract,
        runtime_context={
            "native_update_training_dispatch_enabled": True,
            "training_path_enabled": True,
            "native_update_runtime_dispatch_available": False,
            "native_update_training_mutation_guard_enabled": False,
        },
    )
    path = report["training_path_request"]
    reasons = set(path["blocked_reasons"])
    assert path["request_boundary_ready"] is True, report
    assert path["explicit_training_path_requested"] is True, report
    assert path["default_off"] is False, report
    assert "native_dispatch_training_path_default_off" not in reasons, report
    assert "native_dispatch_runtime_not_implemented" in reasons, report
    assert "native_dispatch_training_mutation_guard_disabled" in reasons, report
    assert report["dispatch_allowed"] is False, report


def test_explicit_training_path_request_can_allow_dispatch_when_contract_allows() -> None:
    contract = {
        "dispatch_rehearsal_ready": True,
        "would_allow_native_dispatch": True,
        "native_kernel_present": True,
        "stream_lifetime_bound": True,
        "performance_test_ready": True,
        "dispatch_sequence": [{"step": "launch_native_adamw_kernel", "planned": True, "enabled": False}],
        "blocked_reasons": [],
    }
    gate = {"would_enable_native_update": True, "native_kernel_present": True}
    report = build_native_update_dispatch_request(
        mode="native_experimental",
        dispatch_enabled=True,
        gate_report=gate,
        dispatch_contract=contract,
        runtime_context={
            "native_update_training_dispatch_enabled": True,
            "training_path_enabled": True,
            "native_update_runtime_dispatch_available": True,
            "native_update_training_mutation_guard_enabled": True,
        },
    )
    assert report["requested"] is True, report
    assert report["dispatch_allowed"] is True, report
    assert report["training_dispatch"] is True, report
    assert report["training_path_enabled"] is True, report
    assert report["runtime_dispatch_available"] is True, report
    assert report["pytorch_optimizer_authoritative"] is False, report
    assert report["fallback_to_pytorch_required"] is False, report
    assert report["blocked_reasons"] == [], report
    assert report["plan"]["execute_native_step"] is True, report
    assert report["plan"]["call_pytorch_optimizer_step"] is False, report
    assert report["plan"]["sequence"][0]["enabled"] is True, report


def main() -> int:
    test_dispatch_request_defaults_to_not_requested()
    test_explicit_request_is_still_runtime_blocked()
    test_explicit_training_path_request_remains_runtime_blocked()
    test_explicit_training_path_request_can_allow_dispatch_when_contract_allows()
    print("turbocore_native_update_dispatch_request_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
