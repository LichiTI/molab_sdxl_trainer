"""Smoke checks for V5 runtime stream-guard evidence normalization."""

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

from core.turbocore_v5_runtime_stream_guard_evidence import (  # noqa: E402
    build_runtime_stream_guard_evidence,
    stream_guard_descriptor_for_runtime_launch,
)


def run_smoke() -> dict[str, Any]:
    missing = build_runtime_stream_guard_evidence(owner=_cpu_owner())
    assert missing["ready_for_borrowed_stream_launch"] is False, missing
    assert "v5_p10_stream_lifetime_bound_missing" in missing["blocked_reasons"], missing
    assert "v5_p10_event_chain_verified_missing" in missing["blocked_reasons"], missing

    verified = build_runtime_stream_guard_evidence(configured_descriptor=_verified_stream_guard())
    assert verified["ready_for_borrowed_stream_launch"] is True, verified
    assert verified["blocked_reasons"] == [], verified
    assert verified["stream_lifetime_lease_evidence"]["ready_for_runtime_stream_guard"] is True, verified
    launch_descriptor = stream_guard_descriptor_for_runtime_launch(verified)
    assert launch_descriptor["runtime_stream_guard_evidence_ready"] is True, launch_descriptor
    assert launch_descriptor["cuda_stream_handle"] == 123456, launch_descriptor

    lying_handle = build_runtime_stream_guard_evidence(
        configured_descriptor={**_verified_stream_guard(), "cuda_stream_handle": 0}
    )
    assert lying_handle["ready_for_borrowed_stream_launch"] is False, lying_handle
    assert "v5_p10_stream_handle_nonzero_missing" in lying_handle["blocked_reasons"], lying_handle

    nested_probe = build_runtime_stream_guard_evidence(
        native_binding_probe={
            "ok": True,
            "native_session": {
                "stream_guard_probe": {
                    **_verified_stream_guard_without_lease(),
                    "stream_lifetime_bound": False,
                    "blocked_reasons": ["stream_lifetime_not_bound"],
                }
            },
        }
    )
    assert nested_probe["source"] == "native_binding_probe", nested_probe
    assert nested_probe["ready_for_borrowed_stream_launch"] is False, nested_probe
    assert "v5_p10_stream_lifetime_bound_missing" in nested_probe["blocked_reasons"], nested_probe

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_runtime_stream_guard_evidence_smoke",
        "ok": True,
        "default_behavior_changed": False,
        "missing_evidence_ready": missing["ready_for_borrowed_stream_launch"],
        "verified_evidence_ready": verified["ready_for_borrowed_stream_launch"],
        "nested_probe_ready": nested_probe["ready_for_borrowed_stream_launch"],
    }


def _cpu_owner() -> SimpleNamespace:
    return SimpleNamespace(param_flat=torch.empty(4, dtype=torch.float32))


def _verified_stream_guard() -> dict[str, Any]:
    payload = _verified_stream_guard_without_lease()
    payload["stream_lifetime_lease_evidence"] = {
        "lease_scope": "native_adamw_runtime_step",
        "lease_active_for_current_step": True,
        "explicit_training_context_requested": True,
        "ownership_guard_enabled": True,
        "ownership_binding_enabled": True,
        "runtime_recovery_ready": True,
        "native_error_recovery_verified": True,
    }
    return payload


def _verified_stream_guard_without_lease() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "descriptor": "turbocore_borrowed_cuda_stream_descriptor_v0",
        "device_type": "cuda",
        "device_index": 0,
        "stream_handle_kind": "external_cuda_stream_handle",
        "stream_handle_reported": True,
        "stream_handle_nonzero": True,
        "cuda_stream_handle": 123456,
        "event_chain_verified": True,
        "pre_launch_ordering_verified": True,
        "post_launch_ordering_verified": True,
        "stream_wait_event_verified": True,
        "stream_lifetime_bound": True,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
