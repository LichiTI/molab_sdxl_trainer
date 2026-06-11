"""Smoke checks for V5 stream/event-chain sync policy."""

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

from core.turbocore_v5_stream_sync_policy import build_v5_stream_sync_policy  # noqa: E402


def run_smoke() -> dict[str, Any]:
    default = build_v5_stream_sync_policy(timing_triage=_triage())
    assert default["ok"] is False, default
    assert default["requested_mode"] == "off", default
    assert "v5_p8_stream_sync_policy_default_off" in default["blocked_reasons"], default
    assert default["default_rollout_allowed"] is False, default
    assert default["auto_rollout_allowed"] is False, default

    current_blocked = build_v5_stream_sync_policy(
        timing_triage=_triage(),
        stream_guard={},
        native_runtime={},
        requested_mode="event_chain_experimental",
    )
    assert current_blocked["ok"] is False, current_blocked
    assert "v5_p8_native_runtime_borrowed_stream_launch_supported_missing" in current_blocked["blocked_reasons"], current_blocked
    assert "v5_p8_stream_guard_event_chain_verified_missing" in current_blocked["blocked_reasons"], current_blocked
    assert current_blocked["sync_fast_path_allowed"] is False, current_blocked

    rollback_blocked = build_v5_stream_sync_policy(
        timing_triage=_triage(),
        stream_guard=_stream_guard(),
        native_runtime=_runtime_caps(),
        rollback_policy={"disable_for_run_on_stream_ordering_failure": False},
        requested_mode="event_chain_experimental",
    )
    assert rollback_blocked["ok"] is False, rollback_blocked
    assert "v5_p8_rollback_policy_ready_missing" in rollback_blocked["blocked_reasons"], rollback_blocked

    allowed = build_v5_stream_sync_policy(
        timing_triage=_triage(),
        stream_guard=_stream_guard(),
        native_runtime=_runtime_caps(),
        requested_mode="event_chain_experimental",
    )
    assert allowed["ok"] is True, allowed
    assert allowed["sync_fast_path_allowed"] is True, allowed
    assert allowed["training_path_enabled"] is False, allowed
    assert allowed["manual_wider_canary_allowed"] is False, allowed
    assert allowed["blocked_reasons"] == [], allowed

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_stream_sync_policy_smoke",
        "ok": True,
        "default_blocker": default["blocked_reasons"][0],
        "current_blocker_count": len(current_blocked["blocked_reasons"]),
        "allowed_next_step": allowed["recommended_next_step"],
    }


def _triage() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "triage": "turbocore_v5_timing_bottleneck_triage_v0",
        "timing_triage_ready": True,
        "primary_bottleneck": "stream_event_chain_sync_fast_path",
        "metrics": {
            "runtime_synchronization": "cuCtxSynchronize_after_native_step",
            "runtime_stream_binding": "cuda_driver_default_stream_null_synchronized",
        },
    }


def _stream_guard() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stream_handle_kind": "external_cuda_stream_handle",
        "stream_handle_reported": True,
        "stream_handle_nonzero": True,
        "event_chain_verified": True,
        "pre_launch_ordering_verified": True,
        "post_launch_ordering_verified": True,
        "stream_wait_event_verified": True,
        "stream_lifetime_bound": True,
        "blocked_reasons": [],
    }


def _runtime_caps() -> dict[str, Any]:
    return {
        "adamw_launch_on_borrowed_stream_supported": True,
        "ctx_synchronize_free_training_step_supported": True,
        "event_chain_synchronization_supported": True,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
