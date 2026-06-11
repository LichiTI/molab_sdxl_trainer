"""Smoke checks for TurboCore native-update probe evidence retention."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_update_probe_cache import (  # noqa: E402
    can_retain_native_update_probe_evidence,
    probe_cache_summary,
    retain_native_update_probe_evidence,
)


def _promoted_gate() -> dict[str, object]:
    return {
        "mode": "native_experimental",
        "dispatch_request": {
            "requested": True,
            "dispatch_allowed": True,
            "training_path_enabled": True,
        },
        "dispatch_contract": {
            "dispatch_rehearsal_ready": True,
            "evidence": {
                "owner_native_launch_ok": True,
                "copyback_dispatch_validated": True,
                "native_binding_probe_present": True,
                "stream_ordering_verified": True,
                "event_chain_verified": True,
            },
        },
        "kernel_launch_plan": {
            "launch_allowed": True,
            "evidence": {
                "diagnostic_kernel_executed": True,
                "diagnostic_parity_ok": True,
            },
        },
        "fallback_policy": {
            "runtime_recovery": {
                "disable_native_update_for_run": False,
                "actions": [
                    "record_recovery_evidence_in_gate_report",
                    "require_shadow_parity_revalidation_after_recovery",
                ],
                "runtime": {"runtime_error_observed": False},
                "state_safety": {"state_mismatch_observed": False},
            }
        },
    }


def _autostop_shadow() -> dict[str, object]:
    return {
        "mode": "shadow",
        "reason": "auto_stopped_after_consecutive_passes",
        "after_optimizer": {
            "skipped": True,
            "reason": "auto_stopped_after_consecutive_passes",
        },
    }


def _native_runtime_step() -> dict[str, object]:
    return {
        "native_step_executed": True,
        "state": {"disabled_for_run": False},
        "blocked_reasons": [],
    }


def test_retains_previous_promoted_gate_after_shadow_autostop() -> None:
    assert can_retain_native_update_probe_evidence(
        previous_gate=_promoted_gate(),
        shadow_report=_autostop_shadow(),
        dispatch_runtime_report=_native_runtime_step(),
        defer_state_sync=True,
    )
    retained = retain_native_update_probe_evidence(_promoted_gate(), step=7)
    summary = probe_cache_summary(retained)
    assert retained["retained_probe_evidence"] is True, retained
    assert retained["probe_cache_reused_steps"] == 1, retained
    assert summary["retained"] is True, summary
    assert summary["reused_steps"] == 1, summary
    assert summary["evidence"]["copyback_dispatch_validated"] is True, summary


def test_retention_requires_explicit_deferred_state_sync() -> None:
    assert not can_retain_native_update_probe_evidence(
        previous_gate=_promoted_gate(),
        shadow_report=_autostop_shadow(),
        dispatch_runtime_report=_native_runtime_step(),
        defer_state_sync=False,
    )


def test_retention_blocks_runtime_recovery_revalidation() -> None:
    gate = _promoted_gate()
    recovery = gate["fallback_policy"]["runtime_recovery"]  # type: ignore[index]
    recovery["runtime"] = {"runtime_error_observed": True}  # type: ignore[index]
    assert not can_retain_native_update_probe_evidence(
        previous_gate=gate,
        shadow_report=_autostop_shadow(),
        dispatch_runtime_report=_native_runtime_step(),
        defer_state_sync=True,
    )


def test_retention_requires_current_native_step_execution() -> None:
    runtime = _native_runtime_step()
    runtime["native_step_executed"] = False
    assert not can_retain_native_update_probe_evidence(
        previous_gate=_promoted_gate(),
        shadow_report=_autostop_shadow(),
        dispatch_runtime_report=runtime,
        defer_state_sync=True,
    )


def main() -> int:
    test_retains_previous_promoted_gate_after_shadow_autostop()
    test_retention_requires_explicit_deferred_state_sync()
    test_retention_blocks_runtime_recovery_revalidation()
    test_retention_requires_current_native_step_execution()
    print("turbocore_native_update_probe_cache_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
