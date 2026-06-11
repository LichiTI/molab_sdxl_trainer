"""Smoke checks for V5 borrowed-stream AdamW launch config plumbing."""

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
from core.turbocore_native_adamw_runtime_backend import _launch_config  # noqa: E402
from core.turbocore_native_update_training_executor import NativeUpdateTrainingExecutorConfig  # noqa: E402
from core.turbocore_update_executor import TurboCoreUpdateExecutorConfig  # noqa: E402


def run_smoke() -> dict[str, Any]:
    default_owner = _owner(FlatAdamWConfig())
    default_payload = _launch_config(default_owner, step_number=1, training_dispatch=True)
    assert default_payload["runtime_synchronization_policy"] == "context_synchronize", default_payload
    assert "stream_guard_descriptor" not in default_payload, default_payload

    missing_evidence_cfg = FlatAdamWConfig(native_runtime_synchronization_policy="borrowed_stream_event_chain")
    missing_evidence_payload = _launch_config(_owner(missing_evidence_cfg), step_number=2, training_dispatch=True)
    missing_evidence = missing_evidence_payload["runtime_stream_guard_evidence"]
    assert missing_evidence_payload["runtime_synchronization_policy"] == "borrowed_stream_event_chain", missing_evidence_payload
    assert missing_evidence["ready_for_borrowed_stream_launch"] is False, missing_evidence_payload
    assert missing_evidence_payload["stream_guard_descriptor"]["runtime_stream_guard_evidence_ready"] is False, missing_evidence_payload
    assert "v5_p10_stream_lifetime_bound_missing" in missing_evidence["blocked_reasons"], missing_evidence_payload
    assert "v5_p11_stream_lifetime_lease_missing" in missing_evidence["blocked_reasons"], missing_evidence_payload

    guard = _verified_stream_guard()
    flat_cfg = FlatAdamWConfig(
        native_runtime_synchronization_policy="borrowed_stream_event_chain",
        native_runtime_stream_guard_descriptor=guard,
        native_runtime_stream_lifetime_lease_evidence=_verified_lifetime_lease(),
    )
    borrowed_payload = _launch_config(_owner(flat_cfg), step_number=2, training_dispatch=True)
    assert borrowed_payload["runtime_synchronization_policy"] == "borrowed_stream_event_chain", borrowed_payload
    assert borrowed_payload["runtime_stream_guard_evidence"]["ready_for_borrowed_stream_launch"] is True, borrowed_payload
    assert borrowed_payload["stream_guard_descriptor"]["event_chain_verified"] is True, borrowed_payload
    assert borrowed_payload["stream_guard_descriptor"]["stream_lifetime_bound"] is True, borrowed_payload

    update_cfg = TurboCoreUpdateExecutorConfig(
        native_runtime_synchronization_policy="borrowed_stream_event_chain",
        native_runtime_stream_guard_descriptor=guard,
        native_runtime_stream_lifetime_lease_evidence=_verified_lifetime_lease(),
    )
    update_flat = update_cfg.flat_adamw_config()
    assert update_flat.native_runtime_synchronization_policy == "borrowed_stream_event_chain", update_flat
    assert update_flat.native_runtime_stream_guard_descriptor["stream_handle_nonzero"] is True, update_flat

    training_cfg = NativeUpdateTrainingExecutorConfig(
        native_runtime_synchronization_policy="borrowed_stream_event_chain",
        native_runtime_stream_guard_descriptor=guard,
        native_runtime_stream_lifetime_lease_evidence=_verified_lifetime_lease(),
    )
    training_flat = training_cfg.executor_config().flat_adamw_config()
    assert training_flat.native_runtime_synchronization_policy == "borrowed_stream_event_chain", training_flat
    assert training_flat.native_runtime_stream_guard_descriptor["stream_wait_event_verified"] is True, training_flat

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_borrowed_stream_launch_config_smoke",
        "ok": True,
        "default_policy": default_payload["runtime_synchronization_policy"],
        "borrowed_policy": borrowed_payload["runtime_synchronization_policy"],
        "default_training_path_enabled": False,
        "auto_rollout_allowed": False,
    }


def _owner(config: FlatAdamWConfig) -> SimpleNamespace:
    return SimpleNamespace(config=config, param_flat=torch.empty(4, dtype=torch.float32))


def _verified_stream_guard() -> dict[str, Any]:
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


def _verified_lifetime_lease() -> dict[str, Any]:
    return {
        "lease_scope": "native_adamw_runtime_step",
        "lease_active_for_current_step": True,
        "explicit_training_context_requested": True,
        "ownership_guard_enabled": True,
        "ownership_binding_enabled": True,
        "runtime_recovery_ready": True,
        "native_error_recovery_verified": True,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
