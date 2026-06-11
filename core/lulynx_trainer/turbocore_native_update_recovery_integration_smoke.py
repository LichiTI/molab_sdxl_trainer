"""Smoke checks for TurboCore native update recovery integration report."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_update_recovery_integration import (  # noqa: E402
    LEGACY_RECOVERY_BLOCKER,
    TRAINING_RECOVERY_BLOCKER,
    build_native_update_recovery_integration_report,
)


def test_integration_is_report_only_without_recovery_request() -> None:
    report = build_native_update_recovery_integration_report(
        mode="native_experimental",
        policy_defined=True,
    )
    reasons = set(report["blocked_reasons"])
    assert report["integration"] == "turbocore_native_update_recovery_integration_v0", report
    assert report["training_dispatch"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["policy_defined"] is True, report
    assert report["recovery_policy_observation_integrated"] is True, report
    assert report["run_disable_latch_integrated"] is True, report
    assert report["default_off_recovery_bridge_ready"] is True, report
    assert report["recovery_observation_bridge_ready"] is True, report
    assert report["training_dispatch_recovery_ready"] is False, report
    assert report["training_dispatch_recovery_blocked"] is True, report
    assert report["training_dispatch_recovery_blocker"] == TRAINING_RECOVERY_BLOCKER, report
    assert report["dispatch_integration_ready"] is True, report
    assert report["recovery_requested"] is False, report
    assert TRAINING_RECOVERY_BLOCKER in reasons, report
    assert LEGACY_RECOVERY_BLOCKER not in reasons, report


def test_integration_reports_active_run_latch() -> None:
    report = build_native_update_recovery_integration_report(
        mode="native_experimental",
        policy_defined=True,
        disable_native_update_for_run=True,
        runtime_error_observed=True,
        runtime_state={"disabled_for_run": True, "disable_reason": "native_runtime_error_observed"},
    )
    assert report["recovery_requested"] is True, report
    assert report["runtime_error_observed"] is True, report
    assert report["runtime_disabled_for_run"] is True, report
    assert report["runtime_disable_reason"] == "native_runtime_error_observed", report
    assert report["default_off_recovery_bridge_ready"] is True, report
    assert "native_runtime_recovery_latch_not_active" not in report["blocked_reasons"], report
    assert "require_shadow_parity_revalidation_after_latched_recovery" in report["actions"], report
    assert report["pytorch_optimizer_authoritative"] is True, report


def main() -> int:
    test_integration_is_report_only_without_recovery_request()
    test_integration_reports_active_run_latch()
    print("turbocore_native_update_recovery_integration_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
