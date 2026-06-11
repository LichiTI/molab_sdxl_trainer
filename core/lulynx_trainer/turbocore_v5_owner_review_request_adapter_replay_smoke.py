"""Smoke checks for V5 owner-review request adapter replay."""

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

from core.turbocore_v5_owner_review_evidence_package import (  # noqa: E402
    build_v5_owner_review_evidence_package,
)
from core.turbocore_v5_owner_review_request_adapter_replay import (  # noqa: E402
    build_v5_owner_review_request_adapter_replay,
    request_fields_from_owner_review_package,
)


def run_smoke() -> dict[str, Any]:
    pending_package = build_v5_owner_review_evidence_package(stability_gate=_stability_gate(), owner_review=None)
    signed_package = build_v5_owner_review_evidence_package(
        stability_gate=_stability_gate(),
        owner_review=_owner_review(),
    )
    report = build_v5_owner_review_request_adapter_replay(
        pending_package=pending_package,
        signed_package=signed_package,
    )
    assert report["ok"] is True, report
    gates = report["progress_gates"]
    assert gates["pending_unsigned_package_keeps_native_update_off"] is True, report
    assert gates["signed_package_maps_existing_native_update_fields"] is True, report
    assert gates["post_approval_fields_not_emitted_without_signed_review"] is True, report
    assert report["default_rollout_allowed"] is False, report
    pending = report["adapter_replay_cases"]["pending_unsigned_owner_review"]
    signed = report["adapter_replay_cases"]["signed_owner_review"]
    assert pending["package_signed"] is False, pending
    assert pending["request_fields_ready"] is False, pending
    assert pending["native_update_enabled"] is False, pending
    assert pending["resolved_fields"]["turbocore_native_update_mode"] == "off", pending
    assert signed["package_signed"] is True, signed
    assert signed["request_fields_ready"] is True, signed
    assert signed["native_update_enabled"] is True, signed
    assert signed["resolved_fields"]["turbocore_native_update_mode"] == "native_experimental", signed

    unsafe_pending_fields = request_fields_from_owner_review_package(pending_package)
    assert "turbocoreNativeUpdateManualWiderCanaryReviewReady" not in unsafe_pending_fields
    assert request_fields_from_owner_review_package(signed_package)[
        "turbocoreNativeUpdateManualWiderCanaryReviewReady"
    ] is True
    return {
        "schema_version": 1,
        "probe": "turbocore_v5_owner_review_request_adapter_replay_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["recommended_next_step"],
    }


def _stability_gate() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_replicate_stability_gate_v0",
        "stability_gate_ready": True,
        "run_count": 3,
        "ready_run_count": 3,
        "min_replicate_runs": 3,
        "aggregate": {
            "speedup_samples": [1.2151, 1.088, 1.196],
            "min_speedup": 1.088,
            "mean_speedup": 1.1664,
            "median_speedup": 1.196,
            "speedup_spread_ratio": 0.1063,
        },
        "blocked_reasons": [],
    }


def _owner_review() -> dict[str, Any]:
    return {
        "reviewer": "owner_fixture",
        "reviewed_at": "2026-06-01",
        "requested_scope": "manual_wider_canary",
        "approve_manual_wider_canary": True,
        "confirm_default_training_path_enabled": False,
        "confirm_training_path_enabled": False,
        "confirm_default_rollout_allowed": False,
        "confirm_auto_rollout_allowed": False,
        "acknowledge_runtime_synchronization": True,
        "rollback_policy": {
            "fallback_authoritative": True,
            "fallback_backend": "pytorch_adamw",
            "disable_for_run_on_native_error": True,
            "disable_for_run_on_state_sync_failure": True,
            "disable_for_run_on_checkpoint_resume_mismatch": True,
            "rollback_on_resume_mismatch": True,
            "rollback_on_performance_regression": True,
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
