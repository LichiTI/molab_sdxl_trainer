"""Smoke checks for V5-P15 runtime-managed borrowed stream descriptors."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_flat_adamw_state import FlatAdamWConfig  # noqa: E402
from core.turbocore_native_adamw_runtime_backend import (  # noqa: E402
    _launch_config,
    _managed_borrowed_stream_descriptor,
    _managed_borrowed_stream_lease_evidence,
)
from core.turbocore_native_update_timing_summary import summarize_native_update_timing  # noqa: E402


def run_smoke() -> dict[str, Any]:
    default_payload = _launch_config(_owner(FlatAdamWConfig()), step_number=1, training_dispatch=True)
    assert default_payload["runtime_synchronization_policy"] == "context_synchronize", default_payload
    assert "stream_guard_descriptor" not in default_payload, default_payload

    descriptor = _managed_borrowed_stream_descriptor(
        device_index=0,
        cuda_stream_handle=123456,
        training_dispatch=True,
    )
    lease = _managed_borrowed_stream_lease_evidence(training_dispatch=True)
    descriptor["stream_lifetime_lease_evidence"] = lease
    payload = _launch_config(
        _owner(FlatAdamWConfig(native_runtime_synchronization_policy="borrowed_stream_event_chain")),
        step_number=2,
        training_dispatch=True,
        runtime_stream_guard_descriptor=descriptor,
        runtime_stream_lifetime_lease_evidence=lease,
    )
    evidence = payload["runtime_stream_guard_evidence"]
    launch_descriptor = payload["stream_guard_descriptor"]
    assert payload["runtime_synchronization_policy"] == "borrowed_stream_event_chain", payload
    assert evidence["source"] == "configured_descriptor", evidence
    assert evidence["ready_for_borrowed_stream_launch"] is True, evidence
    assert evidence["blocked_reasons"] == [], evidence
    assert launch_descriptor["runtime_stream_guard_evidence_ready"] is True, launch_descriptor
    assert launch_descriptor["cuda_stream_handle"] == 123456, launch_descriptor
    assert launch_descriptor["event_chain_verified"] is True, launch_descriptor
    assert launch_descriptor["stream_lifetime_bound"] is True, launch_descriptor

    zero_descriptor = _managed_borrowed_stream_descriptor(
        device_index=0,
        cuda_stream_handle=0,
        training_dispatch=True,
    )
    blocked_payload = _launch_config(
        _owner(FlatAdamWConfig(native_runtime_synchronization_policy="borrowed_stream_event_chain")),
        step_number=3,
        training_dispatch=True,
        runtime_stream_guard_descriptor=zero_descriptor,
        runtime_stream_lifetime_lease_evidence=lease,
    )
    blocked = blocked_payload["runtime_stream_guard_evidence"]
    assert blocked["ready_for_borrowed_stream_launch"] is False, blocked
    assert "v5_p10_stream_handle_nonzero_missing" in blocked["blocked_reasons"], blocked

    timing = summarize_native_update_timing([_executed_runtime_report()])
    assert timing["native_dispatch_owner_native_runtime_synchronization"] == (
        "borrowed_stream_event_chain_no_ctx_sync"
    ), timing
    assert timing["native_dispatch_owner_native_ctx_synchronize_skipped"] is True, timing
    assert timing["native_dispatch_owner_native_borrowed_stream_policy_allowed"] is True, timing
    assert timing["native_dispatch_owner_native_borrowed_stream_handle_nonzero"] is True, timing
    assert timing["native_dispatch_owner_native_borrowed_stream_runtime_lease_ok"] is True, timing

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_runtime_managed_borrowed_stream_smoke",
        "ok": True,
        "default_training_path_enabled": False,
        "auto_rollout_allowed": False,
    }


def _owner(config: FlatAdamWConfig) -> SimpleNamespace:
    return SimpleNamespace(config=config, param_flat=torch.empty(4, dtype=torch.float32))


def _executed_runtime_report() -> dict[str, Any]:
    return {
        "training_executor": {
            "result": {
                "native_step_executed": True,
                "update_report": {
                    "owner_step": {
                        "native_report": {
                            "runtime_synchronization": "borrowed_stream_event_chain_no_ctx_sync",
                            "runtime_launch_stream_binding": "borrowed_cuda_stream_event_chain",
                            "stream_lifetime_bound": True,
                            "stream_synchronization_bound": True,
                            "ctx_synchronize_skipped": True,
                            "borrowed_stream_policy": {
                                "allowed": True,
                                "stream_handle_nonzero": True,
                            },
                            "borrowed_stream_runtime_lease": {"ok": True},
                        }
                    }
                },
            }
        }
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
