"""Smoke checks for the TurboCore optimizer performance gate."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.turbocore_optimizer_performance_gate import evaluate_optimizer_performance_gate  # noqa: E402


def _base_payload() -> dict:
    return {
        "iters": 24,
        "warmup": 6,
        "stateful_abi_gate": {"ok": True},
        "results": [
            {
                "optimizer": "torch_adamw_fused",
                "success": True,
                "step_ms": 1.0,
                "state_mb": 32.0,
                "parameter_mb": 16.0,
                "exact_adamw_candidate": True,
                "native_kernel_present": False,
                "parity_max_abs_diff": 0.0,
                "parity_max_rel_diff": 0.0,
            }
        ],
    }


def _native(step_ms: float, *, parity_abs: float = 0.0, parity_rel: float = 0.0) -> dict:
    return {
        "optimizer": "turbocore_cuda_adamw_v0",
        "success": True,
        "step_ms": step_ms,
        "state_mb": 32.0,
        "parameter_mb": 16.0,
        "exact_adamw_candidate": True,
        "native_kernel_present": True,
        "parity_max_abs_diff": parity_abs,
        "parity_max_rel_diff": parity_rel,
    }


def test_no_native_candidate_blocks() -> None:
    gate = evaluate_optimizer_performance_gate(_base_payload())
    assert gate["status"] == "no_native_candidate", gate
    assert gate["ok"] is False
    assert gate["runtime_dispatch_allowed"] is False
    assert gate["baseline_optimizer"] == "torch_adamw_fused"


def test_slow_native_candidate_blocks() -> None:
    payload = deepcopy(_base_payload())
    payload["results"].append(_native(0.95))
    gate = evaluate_optimizer_performance_gate(payload)
    assert gate["status"] == "blocked_performance", gate
    assert gate["ok"] is False
    assert gate["best_measured_candidate"]["speedup_vs_baseline"] < 1.10
    assert "speedup_below_threshold" in gate["best_measured_candidate"]["reasons"]


def test_fast_native_with_bad_parity_blocks() -> None:
    payload = deepcopy(_base_payload())
    payload["results"].append(_native(0.70, parity_abs=1e-2, parity_rel=1e-1))
    gate = evaluate_optimizer_performance_gate(payload)
    assert gate["status"] == "blocked_parity", gate
    assert gate["ok"] is False
    assert "parity_abs_rel_failed" in gate["best_measured_candidate"]["reasons"]


def test_tiny_abs_error_allows_large_relative_noise() -> None:
    payload = deepcopy(_base_payload())
    payload["results"].append(_native(0.75, parity_abs=1e-8, parity_rel=1e-1))
    gate = evaluate_optimizer_performance_gate(payload)
    assert gate["ok"] is True, gate
    assert gate["best_candidate"]["parity_policy"] == "pass_when_abs_or_rel_tolerance_passes"


def test_fast_native_candidate_is_report_only() -> None:
    payload = deepcopy(_base_payload())
    payload["results"].append(_native(0.75))
    gate = evaluate_optimizer_performance_gate(payload)
    assert gate["status"] == "promotion_candidate_needs_route_validation", gate
    assert gate["ok"] is True
    assert gate["promotion_gate_ok"] is True
    assert gate["best_candidate"]["optimizer"] == "turbocore_cuda_adamw_v0"
    assert gate["best_candidate"]["speedup_vs_baseline"] >= 1.20
    assert gate["training_activation_allowed"] is False
    assert gate["runtime_dispatch_allowed"] is False


def main() -> int:
    test_no_native_candidate_blocks()
    test_slow_native_candidate_blocks()
    test_fast_native_with_bad_parity_blocks()
    test_tiny_abs_error_allows_large_relative_noise()
    test_fast_native_candidate_is_report_only()
    print("PASS: turbocore optimizer performance gate smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
