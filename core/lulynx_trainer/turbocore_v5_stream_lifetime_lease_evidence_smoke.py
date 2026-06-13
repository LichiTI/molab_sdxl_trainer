"""Smoke checks for V5 stream lifetime lease evidence."""

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

from core.turbocore_v5_stream_lifetime_lease_evidence import (  # noqa: E402
    build_single_step_lifetime_lease_request,
    build_stream_lifetime_lease_evidence,
)


def run_smoke() -> dict[str, Any]:
    missing = build_stream_lifetime_lease_evidence(
        stream_guard=_stream_guard(),
        lease_evidence={},
        requested_policy="borrowed_stream_event_chain",
    )
    assert missing["ready_for_runtime_stream_guard"] is False, missing
    assert "v5_p11_explicit_training_context_missing" in missing["blocked_reasons"], missing
    assert "v5_p11_lease_not_active_for_current_step" in missing["blocked_reasons"], missing

    wrong_policy = build_stream_lifetime_lease_evidence(
        stream_guard=_stream_guard(),
        lease_evidence=_lease(),
        requested_policy="context_synchronize",
    )
    assert wrong_policy["ready_for_runtime_stream_guard"] is False, wrong_policy
    assert "v5_p11_borrowed_stream_policy_missing" in wrong_policy["blocked_reasons"], wrong_policy

    ready = build_stream_lifetime_lease_evidence(
        stream_guard=_stream_guard(),
        lease_evidence=_lease(),
        requested_policy="borrowed_stream_event_chain",
    )
    assert ready["ready_for_runtime_stream_guard"] is True, ready
    assert ready["stream_lifetime_bound"] is True, ready
    assert ready["default_behavior_changed"] is False, ready
    assert ready["requires_explicit_opt_in"] is True, ready
    assert ready["training_path_enabled"] is False, ready
    assert ready["blocked_reasons"] == [], ready

    request = build_single_step_lifetime_lease_request(
        explicit_training_context=True,
        recovery_ready=True,
    )
    assert request["lease_active_for_current_step"] is True, request
    assert request["native_error_recovery_verified"] is True, request

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_stream_lifetime_lease_evidence_smoke",
        "ok": True,
        "missing_ready": missing["ready_for_runtime_stream_guard"],
        "wrong_policy_ready": wrong_policy["ready_for_runtime_stream_guard"],
        "ready": ready["ready_for_runtime_stream_guard"],
        "summary": {
            "stream_lifetime_lease_ready_count": int(bool(ready["ready_for_runtime_stream_guard"])),
            "stream_lifetime_lease_blocked_case_count": int(not missing["ready_for_runtime_stream_guard"])
            + int(not wrong_policy["ready_for_runtime_stream_guard"]),
            "stream_lifetime_default_behavior_changed_count": int(bool(ready["default_behavior_changed"])),
            "stream_lifetime_requires_explicit_opt_in_count": int(bool(ready["requires_explicit_opt_in"])),
            "stream_lifetime_training_path_enabled_count": int(bool(ready["training_path_enabled"])),
            "stream_lifetime_request_training_path_enabled_count": int(bool(request["training_path_enabled"])),
        },
    }


def _stream_guard() -> dict[str, Any]:
    return {
        "stream_handle_kind": "external_cuda_stream_handle",
        "stream_handle_reported": True,
        "stream_handle_nonzero": True,
        "event_chain_verified": True,
        "pre_launch_ordering_verified": True,
        "post_launch_ordering_verified": True,
        "stream_wait_event_verified": True,
    }


def _lease() -> dict[str, Any]:
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
